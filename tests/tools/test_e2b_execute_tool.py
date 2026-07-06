# -*- coding: utf-8 -*-
"""Tests for E2BExecuteTool — remote sandboxed code/shell execution."""
import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def _fake_execution(stdout="", stderr="", text=None, error=None, results=None):
    """Build a fake e2b_code_interpreter Execution-shaped object."""
    logs = SimpleNamespace(
        stdout=[stdout] if isinstance(stdout, str) else list(stdout),
        stderr=[stderr] if isinstance(stderr, str) else list(stderr),
    )
    return SimpleNamespace(
        results=results or [],
        logs=logs,
        text=text,
        error=error,
        execution_count=1,
    )


def _fake_command_result(stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


class TestE2BExecuteToolImport(unittest.TestCase):
    def test_class_importable(self):
        from agentica.tools.e2b_tool import E2BExecuteTool  # noqa: F401

    def test_exported_from_tools_package(self):
        from agentica.tools import E2BExecuteTool  # noqa: F401


class TestE2BExecuteToolInit(unittest.TestCase):
    def test_uses_explicit_api_key_over_env(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        with patch.dict(os.environ, {"E2B_API_KEY": "env-key"}, clear=False):
            tool = E2BExecuteTool(api_key="explicit-key")
        self.assertEqual(tool.api_key, "explicit-key")

    def test_falls_back_to_env_api_key(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        with patch.dict(os.environ, {"E2B_API_KEY": "env-key"}, clear=False):
            tool = E2BExecuteTool()
        self.assertEqual(tool.api_key, "env-key")

    def test_registers_run_python_and_execute(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        tool = E2BExecuteTool(api_key="k")
        names = set(tool.functions.keys())
        self.assertIn("run_python", names)
        self.assertIn("execute", names)


class TestE2BRunPython(unittest.IsolatedAsyncioTestCase):
    async def test_returns_stdout_and_main_result_text(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        fake_sbx = MagicMock()
        fake_sbx.run_code.return_value = _fake_execution(
            stdout="hello\n", text="42"
        )
        tool = E2BExecuteTool(api_key="k")
        with patch.object(tool, "_get_sandbox", return_value=fake_sbx):
            out = await tool.run_python("print('hello'); 42")
        self.assertIn("hello", out)
        self.assertIn("42", out)

    async def test_propagates_execution_error(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        err = SimpleNamespace(
            name="ZeroDivisionError",
            value="division by zero",
            traceback="Traceback...\nZeroDivisionError: division by zero",
        )
        fake_sbx = MagicMock()
        fake_sbx.run_code.return_value = _fake_execution(error=err)
        tool = E2BExecuteTool(api_key="k")
        with patch.object(tool, "_get_sandbox", return_value=fake_sbx):
            out = await tool.run_python("1/0")
        self.assertIn("ZeroDivisionError", out)
        self.assertIn("division by zero", out)


class TestE2BExecuteShell(unittest.IsolatedAsyncioTestCase):
    async def test_returns_combined_output_with_exit_code(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        fake_sbx = MagicMock()
        fake_sbx.commands.run.return_value = _fake_command_result(
            stdout="ok\n", stderr="warn", exit_code=0
        )
        tool = E2BExecuteTool(api_key="k")
        with patch.object(tool, "_get_sandbox", return_value=fake_sbx):
            out = await tool.execute("echo ok")
        self.assertIn("ok", out)
        self.assertIn("warn", out)

    async def test_non_zero_exit_code_appended(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        fake_sbx = MagicMock()
        fake_sbx.commands.run.return_value = _fake_command_result(
            stdout="", stderr="boom", exit_code=2
        )
        tool = E2BExecuteTool(api_key="k")
        with patch.object(tool, "_get_sandbox", return_value=fake_sbx):
            out = await tool.execute("false")
        self.assertIn("Exit code: 2", out)
        self.assertIn("boom", out)

    async def test_truncates_long_output(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        fake_sbx = MagicMock()
        fake_sbx.commands.run.return_value = _fake_command_result(
            stdout="x" * 100_000, stderr="", exit_code=0
        )
        tool = E2BExecuteTool(api_key="k", max_output_length=2000)
        with patch.object(tool, "_get_sandbox", return_value=fake_sbx):
            out = await tool.execute("seq 1 1000000")
        self.assertLessEqual(len(out), 2200)
        self.assertIn("truncated", out)


class TestE2BLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_close_kills_sandbox_and_clears_handle(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        fake_sbx = MagicMock()
        tool = E2BExecuteTool(api_key="k")
        tool._sandbox = fake_sbx
        tool.close()
        fake_sbx.kill.assert_called_once()
        self.assertIsNone(tool._sandbox)

    async def test_get_sandbox_lazy_constructs_once(self):
        from agentica.tools import e2b_tool as et

        tool = et.E2BExecuteTool(api_key="k", template="custom")
        with patch.object(et, "_load_sandbox_class") as load_cls:
            FakeSandbox = MagicMock()
            load_cls.return_value = FakeSandbox
            sbx1 = tool._get_sandbox()
            sbx2 = tool._get_sandbox()
        self.assertIs(sbx1, sbx2)
        FakeSandbox.assert_called_once()
        kwargs = FakeSandbox.call_args.kwargs
        self.assertEqual(kwargs.get("api_key"), "k")
        self.assertEqual(kwargs.get("template"), "custom")


class TestE2BMissingSdk(unittest.TestCase):
    def test_missing_sdk_raises_clear_install_hint(self):
        from agentica.tools import e2b_tool as et

        tool = et.E2BExecuteTool(api_key="k")

        def _raise(*a, **kw):
            raise ImportError("No module named 'e2b_code_interpreter'")

        with patch.object(et, "_load_sandbox_class", side_effect=_raise):
            with self.assertRaises(ImportError) as cm:
                tool._get_sandbox()
        msg = str(cm.exception)
        self.assertIn("e2b_code_interpreter", msg)
        self.assertIn("agentica[e2b]", msg)


class TestE2BOriginIsFunction(unittest.TestCase):
    def test_registered_functions_have_function_origin(self):
        from agentica.tools.e2b_tool import E2BExecuteTool

        tool = E2BExecuteTool(api_key="k")
        for fn in tool.functions.values():
            self.assertIsNotNone(fn.origin)
            self.assertEqual(fn.origin.type, "function")


if __name__ == "__main__":
    unittest.main()
