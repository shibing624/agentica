# -*- coding: utf-8 -*-
"""
Tests for ToolOrigin metadata (P1-7).

Each tool registered with an Agent must be tagged with a ToolOrigin
indicating where it came from: builtin / function / mcp / agent / model.
This metadata flows down into session log entries and tool events.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agentica.tools.base import Function, ModelTool, Tool
from agentica.tools.origin import ToolOrigin


class TestToolOriginDataclass(unittest.TestCase):
    def test_origin_type_field(self):
        o = ToolOrigin(type="agent", agent_name="Planner", source_tool_name="planner")
        self.assertEqual(o.type, "agent")
        self.assertEqual(o.agent_name, "Planner")
        self.assertIsNone(o.provider_name)

    def test_origin_is_frozen(self):
        o = ToolOrigin(type="function")
        with self.assertRaises(Exception):
            o.type = "agent"  # type: ignore[misc]


class TestFunctionOriginField(unittest.TestCase):
    def test_function_default_origin_is_none_until_registered(self):
        fn = Function(name="x")
        self.assertIsNone(fn.origin)

    def test_function_accepts_origin_init_arg(self):
        origin = ToolOrigin(type="function")
        fn = Function(name="x", origin=origin)
        self.assertIs(fn.origin, origin)


class TestToolRegisterAssignsOrigin(unittest.TestCase):
    def test_plain_toolkit_register_marks_function(self):
        tk = Tool(name="my_tk", description="d")

        def my_fn():
            return None

        tk.register(my_fn)
        registered = list(tk.functions.values())[0]
        self.assertIsNotNone(registered.origin)
        self.assertEqual(registered.origin.type, "function")

    def test_builtin_named_toolkit_marks_builtin(self):
        tk = Tool(name="builtin_demo_tool", description="d")

        def my_fn():
            return None

        tk.register(my_fn)
        registered = list(tk.functions.values())[0]
        self.assertEqual(registered.origin.type, "builtin")


class TestAsToolOrigin(unittest.TestCase):
    def test_agent_as_tool_origin_is_agent(self):
        from agentica.agent.base import Agent

        agent = Agent(name="Planner")
        fn = agent.as_tool()
        self.assertIsNotNone(fn.origin)
        self.assertEqual(fn.origin.type, "agent")
        self.assertEqual(fn.origin.agent_name, "Planner")


class TestMcpToolOrigin(unittest.TestCase):
    def test_mcp_wrapped_function_marked_mcp(self):
        # Build a Function "as if" McpTool created it. Validates the field
        # accepts the mcp origin shape; actual McpTool wiring is exercised
        # in the McpTool.initialize() change.
        origin = ToolOrigin(
            type="mcp",
            provider_name="my-mcp-server",
            source_tool_name="raw_tool_name",
        )
        fn = Function(name="my_mcp_server__raw_tool_name", origin=origin)
        self.assertEqual(fn.origin.type, "mcp")
        self.assertEqual(fn.origin.provider_name, "my-mcp-server")


class TestModelToolOrigin(unittest.TestCase):
    def test_model_tool_accepts_origin(self):
        tool = ModelTool(
            type="function",
            function={"name": "web_search"},
            origin=ToolOrigin(type="model", provider_name="openai"),
        )
        self.assertEqual(tool.origin.type, "model")
        self.assertEqual(tool.origin.provider_name, "openai")


class TestSessionLogPersistsOrigin(unittest.TestCase):
    def test_session_log_append_stores_origin_meta(self):
        from agentica.memory.session_log import SessionLog
        import json

        with tempfile.TemporaryDirectory() as td:
            log = SessionLog(session_id="sess-1", base_dir=td)
            log.append(
                "tool",
                "result text",
                tool_name="my_mcp_server__lookup",
                tool_call_id="call_1",
                origin_type="mcp",
                origin_provider_name="my-mcp-server",
            )
            data = (Path(td) / "sess-1.jsonl").read_text(encoding="utf-8").splitlines()
            entry = json.loads(data[-1])
            self.assertEqual(entry["origin_type"], "mcp")
            self.assertEqual(entry["origin_provider_name"], "my-mcp-server")


if __name__ == "__main__":
    unittest.main()
