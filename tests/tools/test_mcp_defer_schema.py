"""Tests for McpTool.defer_schema: optional MCP tools stay executable but do
not expand into the top-level provider schema, so MCP inventory churn no longer
cold-starts the prompt cache (Reasonix use_capability 移植的 MCP 侧)。
"""
import unittest


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"Tool {name}"
        self.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}


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


if __name__ == "__main__":
    unittest.main()
