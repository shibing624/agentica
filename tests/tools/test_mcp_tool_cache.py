# -*- coding: utf-8 -*-
"""MCP tool lists at agent build: cached per server config, discovered in parallel.

Startup used to connect to every configured MCP server one after another just
to list tools (~1 s each), on every CLI start and gateway agent build.
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("mcp", reason="MCP tests require the mcp extra")

from agentica.tools import mcp_tool as m  # noqa: E402
from agentica.tools.mcp_tool import McpTool, prepare_mcp_servers  # noqa: E402

LISTS = {
    "alpha": [{"name": "a_search", "description": "search", "input_schema": {
        "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}],
    "beta": [{"name": "b_echo", "description": "echo", "input_schema": {"type": "object", "properties": {}}}],
}


def _server(name, command="python -m fake_server"):
    tool = McpTool(command=command)
    tool._cache_identity = {"version": m.MCP_TOOLS_CACHE_VERSION, "name": name, "command": command, "args": None,
                            "env": None, "url": None, "headers": None}
    tool._config_server_name = name
    return tool


class _FakeConnect:
    """Stands in for a real connect + list_tools; records calls and overlap."""

    def __init__(self, delay=0.3, fail=()):
        self.delay, self.fail, self.calls, self.active, self.max_active = delay, set(fail), [], 0, 0

    def install(self):
        fake = self

        async def aenter(tool):
            fake.calls.append(tool._config_server_name)
            fake.active += 1
            fake.max_active = max(fake.max_active, fake.active)
            try:
                await asyncio.sleep(fake.delay)
                if tool._config_server_name in fake.fail:
                    raise ConnectionError("server down")
                tool._register_tool_specs(LISTS[tool._config_server_name])
                return tool
            finally:
                fake.active -= 1

        async def aexit(tool, *a):
            return None

        return patch.multiple(McpTool, __aenter__=aenter, __aexit__=aexit)


class TestPrepareMcpServers(unittest.TestCase):
    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"AGENTICA_HOME": self._home.name, "AGENTICA_MCP_TOOLS_CACHE": "1"})
        self._env.start()
        self._bg = patch.object(m, "_refresh_in_background")
        self.refresh = self._bg.start()

    def tearDown(self):
        self._bg.stop()
        self._env.stop()
        self._home.cleanup()

    def test_cold_start_discovers_servers_concurrently_and_writes_the_cache(self):
        fake = _FakeConnect(delay=0.3)
        with fake.install():
            t0 = time.perf_counter()
            ready = prepare_mcp_servers([_server("alpha"), _server("beta")])
            elapsed = time.perf_counter() - t0
        self.assertEqual([t._config_server_name for t in ready], ["alpha", "beta"])
        self.assertEqual(fake.max_active, 2)
        self.assertLess(elapsed, 0.55)  # one delay, not two
        self.assertEqual(ready[0].functions["a_search"].parameters["required"], ["q"])
        files = os.listdir(os.path.join(self._home.name, "cache", "mcp_tools"))
        self.assertEqual(len(files), 2)

    def test_warm_start_connects_to_nothing_and_schedules_a_refresh(self):
        with _FakeConnect(delay=0).install():
            prepare_mcp_servers([_server("alpha"), _server("beta")])
        fake = _FakeConnect(delay=0)
        with fake.install():
            ready = prepare_mcp_servers([_server("alpha"), _server("beta")])
        self.assertEqual(fake.calls, [])
        self.assertEqual(sorted(f for t in ready for f in t.functions), ["a_search", "b_echo"])
        self.assertEqual(ready[1].functions["b_echo"].description, "echo")
        refreshed = self.refresh.call_args_list[-1].args[0]
        self.assertEqual([t._config_server_name for t in refreshed], ["alpha", "beta"])

    def test_a_cached_tool_is_callable_without_a_startup_session(self):
        with _FakeConnect(delay=0).install():
            prepare_mcp_servers([_server("alpha")])
        ready = prepare_mcp_servers([_server("alpha")])
        fn = ready[0].functions["a_search"]
        self.assertIsNone(ready[0].session)
        self.assertTrue(callable(fn.entrypoint))

    def test_an_unreachable_server_is_left_out_and_not_cached(self):
        with _FakeConnect(delay=0, fail={"beta"}).install():
            ready = prepare_mcp_servers([_server("alpha"), _server("beta")])
        self.assertEqual([t._config_server_name for t in ready], ["alpha"])
        fake = _FakeConnect(delay=0)
        with fake.install():
            prepare_mcp_servers([_server("alpha"), _server("beta")])
        self.assertEqual(fake.calls, ["beta"])  # alpha from cache, beta retried

    def test_changing_the_server_config_misses_the_cache(self):
        with _FakeConnect(delay=0).install():
            prepare_mcp_servers([_server("alpha", command="python -m srv --v1")])
        fake = _FakeConnect(delay=0)
        with fake.install():
            prepare_mcp_servers([_server("alpha", command="python -m srv --v2")])
        self.assertEqual(fake.calls, ["alpha"])

    def test_cache_can_be_turned_off(self):
        with patch.dict(os.environ, {"AGENTICA_MCP_TOOLS_CACHE": "0"}):
            for _ in range(2):
                fake = _FakeConnect(delay=0)
                with fake.install():
                    prepare_mcp_servers([_server("alpha")])
                self.assertEqual(fake.calls, ["alpha"])

    def test_a_corrupt_cache_file_is_ignored(self):
        with _FakeConnect(delay=0).install():
            prepare_mcp_servers([_server("alpha")])
        path = m._tools_cache_path(_server("alpha"))
        with open(path, "w") as f:
            f.write("{not json")
        fake = _FakeConnect(delay=0)
        with fake.install():
            ready = prepare_mcp_servers([_server("alpha")])
        self.assertEqual(fake.calls, ["alpha"])
        self.assertIn("a_search", ready[0].functions)
        self.assertEqual(json.load(open(path))["tools"][0]["name"], "a_search")


class TestBackgroundRefresh(unittest.TestCase):
    def test_refresh_rewrites_the_cache_on_fresh_instances_once_per_process(self):
        with tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, {"AGENTICA_HOME": home}), \
                patch.object(m, "MCP_REFRESH_DELAY_S", 0), \
                patch.object(m, "_refresh_started", set()):
            live = _server("alpha")
            live._register_tool_specs(LISTS["alpha"])
            fresh = _server("alpha")
            built = []

            def from_config(server_names=None, config_path=None):
                built.append(server_names)
                return fresh

            changed = [{"name": "a_new", "description": "new", "input_schema": {}}]

            async def aenter(tool):
                tool._register_tool_specs(changed)
                return tool

            async def aexit(tool, *a):
                return None

            with patch.object(McpTool, "from_config", side_effect=from_config), \
                    patch.multiple(McpTool, __aenter__=aenter, __aexit__=aexit):
                m._refresh_in_background([live])
                m._refresh_in_background([live])  # second call: already refreshed
                deadline = time.time() + 5
                while m.load_cached_tool_specs(live) is None and time.time() < deadline:
                    time.sleep(0.02)
            self.assertEqual(built, ["alpha"])
            self.assertEqual(m.load_cached_tool_specs(live), changed)
            # The running agent's tools are untouched; the change is for next time.
            self.assertEqual(sorted(live.functions), ["a_search"])


class TestNoMcpImportOnTheCachedPath(unittest.TestCase):
    def test_import_agentica_and_mcp_tool_do_not_load_the_mcp_sdk(self):
        code = ("import sys, agentica, agentica.tools.mcp_tool, agentica.mcp.config; "
                "from agentica import Agent, DeepAgent; "
                "print(sorted(k for k in ('mcp', 'bs4', 'requests', 'markdownify') if k in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "OPENAI_API_KEY": "fake"}, timeout=60)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]", out.stderr[-2000:])

    def test_agentica_mcp_names_still_resolve(self):
        from agentica.mcp import MCPClient, MCPServerStdio

        self.assertEqual(MCPClient.__name__, "MCPClient")
        self.assertEqual(MCPServerStdio.__name__, "MCPServerStdio")
        with self.assertRaises(AttributeError):
            __import__("agentica.mcp", fromlist=["nope"]).nope


if __name__ == "__main__":
    unittest.main()
