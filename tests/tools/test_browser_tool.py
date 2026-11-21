# -*- coding: utf-8 -*-
"""Tests for the strong-typed browser tool surface and its CLI driver."""
import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.tools.base import FunctionCall
from agentica.tools.browser_tool import AgentBrowserDriver, BrowserTool

SNAPSHOT = "\n".join([
    '- link "Learn more" [ref=e1]',
    '- button "Go" [ref=e2]',
    '- textbox "Query" [ref=e3]',
    '- combobox [expanded=false, ref=e4]: One',
    '  - option "One" [selected, ref=e5]',
])


class FakeDriver:
    """Records argv commands and returns canned CLI output."""

    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {}
        self.closed = False

    async def run(self, *args):
        self.calls.append(args)
        return self.outputs.get(args[0], "ok")

    async def close(self):
        self.closed = True


def make_tool(cap=240):
    driver = FakeDriver({"snapshot": SNAPSHOT, "open": "navigated", "get": "http://example.com"})
    return BrowserTool(driver=driver, max_elements=cap), driver


class TestBrowserTool(unittest.TestCase):
    def test_observe_parses_refs_caps_elements_and_marks_generation(self):
        tool, driver = make_tool(cap=3)
        out = asyncio.run(tool.browser_observe("https://example.com"))
        self.assertIn("generation=1", out)
        self.assertIn("elements=3", out)
        self.assertIn("ref=e1", out)
        self.assertNotIn("ref=e4", out)
        self.assertEqual(("open", "https://example.com"), driver.calls[0])
        self.assertEqual(("snapshot", "-i"), driver.calls[1])
        self.assertEqual({"e1", "e2", "e3"}, tool._refs)

    def test_observe_rejects_non_http_url(self):
        tool, driver = make_tool()
        out = asyncio.run(tool.browser_observe("file:///etc/passwd"))
        self.assertTrue(out.startswith("Error:"))
        self.assertEqual([], driver.calls)

    def test_act_composes_action_wait_and_url_in_one_call(self):
        tool, driver = make_tool()
        asyncio.run(tool.browser_observe())
        driver.calls.clear()
        out = asyncio.run(tool.browser_act("e1", wait_for="#done"))
        self.assertEqual(("click", "@e1"), driver.calls[0])
        self.assertEqual(("wait", "#done"), driver.calls[1])
        self.assertEqual(("get", "url"), driver.calls[2])
        self.assertIn("Refs invalidated", out)
        self.assertEqual(set(), tool._refs)

    def test_stale_ref_is_refused(self):
        tool, _ = make_tool()
        asyncio.run(tool.browser_observe())
        asyncio.run(tool.browser_act("e1"))
        out = asyncio.run(tool.browser_act("e2"))
        self.assertIn("stale or unknown ref", out)

    def test_act_validates_action_and_required_text(self):
        tool, _ = make_tool()
        asyncio.run(tool.browser_observe())
        self.assertIn("action must be one of", asyncio.run(tool.browser_act("e1", action="nuke")))
        self.assertIn("requires text", asyncio.run(tool.browser_act("e3", action="fill")))

    def test_press_inserts_whole_text_and_press_uses_key(self):
        tool, driver = make_tool()
        asyncio.run(tool.browser_press(text="hello world"))
        self.assertEqual(("keyboard", "inserttext", "hello world"), driver.calls[-1])
        asyncio.run(tool.browser_press(key="Enter"))
        self.assertEqual(("press", "Enter"), driver.calls[-1])
        self.assertIn("provide either", asyncio.run(tool.browser_press()))

    def test_upload_and_close(self):
        tool, driver = make_tool()
        asyncio.run(tool.browser_observe())
        asyncio.run(tool.browser_upload("@e3", ["/tmp/a.png", "/tmp/b.png"]))
        self.assertEqual(("upload", "@e3", "/tmp/a.png", "/tmp/b.png"), driver.calls[-1])
        out = asyncio.run(tool.browser_close())
        self.assertIn("closed", out)
        self.assertTrue(driver.closed)

    def test_schema_and_policy_prompt(self):
        tool, _ = make_tool(cap=7)
        act_fn = tool.functions["browser_act"]
        self.assertEqual(
            ["click", "fill", "type", "hover", "check", "select"],
            act_fn.parameters["properties"]["action"]["enum"],
        )
        prompt = tool.get_system_prompt()
        self.assertIn("browser_observe first", prompt)
        self.assertIn("ONE call", prompt)
        self.assertIn("at most 7", prompt)
        self.assertIn("browser_close", prompt)

    def test_driver_builds_argv_without_a_shell(self):
        driver = AgentBrowserDriver(session="work", profile="/tmp/p", restore=True)
        self.assertEqual(
            ["agent-browser", "--session", "work", "--profile", "/tmp/p", "--restore",
             "open", "https://example.com"],
            driver._argv("open", "https://example.com"),
        )
        plain = AgentBrowserDriver(restore=False)
        self.assertEqual(["agent-browser", "--session", "agentica", "close"], plain._argv("close"))

    def test_gateway_driver_ignores_ambient_browser_config(self):
        driver = AgentBrowserDriver(session="ag-gw-isolated", restore=False, isolated=True)
        proc = AsyncMock()
        proc.communicate.return_value = (b"https://example.test/", b"")
        proc.returncode = 0
        with patch.dict("os.environ", {"AGENT_BROWSER_PROFILE": "/personal", "AGENT_BROWSER_RESTORE": "personal"}), \
             patch("asyncio.create_subprocess_exec", return_value=proc) as spawn:
            self.assertEqual(asyncio.run(driver.run("get", "url")), "https://example.test/")
        args, kwargs = spawn.call_args
        config = Path(args[args.index("--config") + 1])
        self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"headed": False})
        self.assertEqual(args[-2:], ("get", "url"))
        self.assertFalse(any(key.startswith("AGENT_BROWSER_") for key in kwargs["env"]))

    def test_function_call_layer_executes_tool(self):
        tool, driver = make_tool()
        fn = tool.functions["browser_observe"]
        fn.process_entrypoint()
        fc = FunctionCall(function=fn, arguments={"url": ""})
        self.assertTrue(asyncio.run(fc.execute()))
        self.assertIn("generation=1", fc.result)
        self.assertIn(("snapshot", "-i"), driver.calls)

    def test_preview_metadata_uses_the_actual_browser_url(self):
        tool, driver = make_tool()
        driver.outputs["get"] = "https://example.com/redirected"
        observed = asyncio.run(tool.browser_observe("https://example.com/start"))
        self.assertIsInstance(observed, str)
        self.assertEqual(observed.display_meta, {"browser_url": "https://example.com/redirected"})
        driver.outputs["get"] = "https://example.com/done"
        acted = asyncio.run(tool.browser_act("e1", wait_for="#done"))
        self.assertEqual(acted.display_meta, {"browser_url": "https://example.com/done"})

    def test_observe_caps_nested_rows_without_refs(self):
        nested = "\n".join([
            '- combobox [ref=e1]: One',
            *[f'  - option "Item {i}"' for i in range(200)],
            '- link "Next" [ref=e2]',
        ])
        tool, driver = make_tool(cap=5)
        driver.outputs["snapshot"] = nested
        out = asyncio.run(tool.browser_observe())
        body = [line for line in str(out).splitlines() if line.startswith(("-", " ", "\t"))]
        self.assertEqual(len(body), 5)
        self.assertIn("elements=1", out)
        self.assertNotIn("ref=e2", out)

    def test_close_propagates_cli_failures(self):
        driver = AgentBrowserDriver()
        with patch.object(driver, "run", new_callable=AsyncMock) as run:
            run.side_effect = RuntimeError("browser command failed (1): daemon gone")
            with self.assertRaisesRegex(RuntimeError, "daemon gone"):
                asyncio.run(driver.close())

    def test_run_reaps_the_child_when_a_command_fails(self):
        driver = AgentBrowserDriver()
        proc = MagicMock()
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=RuntimeError("pipe broke"))
        proc.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with self.assertRaisesRegex(RuntimeError, "pipe broke"):
                asyncio.run(driver.run("get", "url"))
        proc.kill.assert_called_once_with()
        proc.wait.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
