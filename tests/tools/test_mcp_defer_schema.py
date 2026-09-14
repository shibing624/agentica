"""Tests for McpTool.defer_schema: optional MCP tools stay executable but do
not expand into the top-level provider schema, so MCP inventory churn no longer
cold-starts the prompt cache (Reasonix use_capability 移植的 MCP 侧)。

Also covers the parameter schema an MCP tool hands to us: mcp 2.x renamed
``Tool.inputSchema`` to ``input_schema``, and reading the old name is a *silent*
failure — an empty ``properties`` dict, no exception — so the LLM would be told
the tool takes no arguments. The assertions below pin the contents, not just
that a dict came out.
"""
import unittest

import pytest

pytest.importorskip("mcp", reason="MCP tests require the mcp extra")


class _FakeTool:
    """Mirrors mcp 2.x: attribute access is snake_case only."""

    def __init__(self, name, input_schema=None):
        self.name = name
        self.description = f"Tool {name}"
        self.input_schema = input_schema if input_schema is not None else {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }


class _FakeToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeSession:
    def __init__(self, tools):
        self._tools = tools

    async def initialize(self):
        return None

    async def list_tools(self):
        return _FakeToolsResult(self._tools)


class TestMcpToolDeferSchema(unittest.IsolatedAsyncioTestCase):
    async def test_defer_schema_marks_functions_deferred(self):
        from agentica.tools.mcp_tool import McpTool

        mcp = McpTool(session=_FakeSession([_FakeTool("search"), _FakeTool("echo")]), defer_schema=True)
        await mcp.initialize()
        self.assertIn("search", mcp.functions)
        self.assertTrue(mcp.functions["search"].deferred)
        self.assertTrue(mcp.functions["echo"].deferred)

    async def test_default_expands_schema(self):
        from agentica.tools.mcp_tool import McpTool

        mcp = McpTool(session=_FakeSession([_FakeTool("search")]))
        await mcp.initialize()
        self.assertFalse(mcp.functions["search"].deferred)


class TestMcpToolParameterSchema(unittest.IsolatedAsyncioTestCase):
    """The tool's JSON Schema must reach Function.parameters verbatim."""

    async def test_properties_survive_onto_the_function(self):
        from agentica.tools.mcp_tool import McpTool

        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        }
        mcp = McpTool(session=_FakeSession([_FakeTool("forecast", input_schema=schema)]))
        await mcp.initialize()

        params = mcp.functions["forecast"].parameters
        self.assertEqual(params, schema)
        self.assertEqual(set(params["properties"]), {"city", "days"})
        self.assertEqual(params["properties"]["city"]["description"], "City name")
        self.assertEqual(params["required"], ["city"])

    async def test_bare_properties_dict_is_wrapped_as_object(self):
        """A server that sends only the property map still yields an object schema."""
        from agentica.tools.mcp_tool import McpTool

        mcp = McpTool(session=_FakeSession([
            _FakeTool("emit", input_schema={"payload": {"type": "string"}}),
        ]))
        await mcp.initialize()

        params = mcp.functions["emit"].parameters
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"], {"payload": {"type": "string"}})

    async def test_real_sdk_tool_schema_flows_through(self):
        """Against the installed SDK, not a stand-in: this is the shape that moved."""
        from mcp import Tool as SDKTool

        from agentica.tools.mcp_tool import McpTool

        sdk_tool = SDKTool(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        )
        mcp = McpTool(session=_FakeSession([sdk_tool]))
        await mcp.initialize()

        params = mcp.functions["add"].parameters
        self.assertEqual(set(params["properties"]), {"a", "b"})
        self.assertEqual(params["required"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
