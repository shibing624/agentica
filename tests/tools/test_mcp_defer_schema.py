"""Tests for McpTool.defer_schema: optional MCP tools stay executable but do
not expand into the top-level provider schema, so MCP inventory churn no longer
cold-starts the prompt cache (Reasonix use_capability 移植的 MCP 侧)。

Also covers the parameter schema an MCP tool hands to us: mcp 2.x renamed
``Tool.inputSchema`` to ``input_schema``, and reading the old name is a *silent*
failure — an empty ``properties`` dict, no exception — so the LLM would be told
the tool takes no arguments. The assertions below pin the contents, not just
that a dict came out.
"""
import json
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


def test_config_opt_in_keeps_optional_mcp_catalog_out_of_wire_schema(tmp_path, monkeypatch):
    from agentica.agent import Agent
    from agentica.agent.config import ToolConfig
    from agentica.model.openai import OpenAIChat
    from agentica.tools import mcp_tool

    config = tmp_path / "mcp_config.json"
    config.write_text(json.dumps({"mcpServers": {
        "always": {"command": "test-always"},
        "optional": {"command": "test-optional", "defer_schema": True},
    }}))
    monkeypatch.chdir(tmp_path)
    catalog = ["one", "two"]

    def discover(servers):
        for server in servers:
            names = catalog if server._config_server_name == "optional" else ["always"]
            server._register_tool_specs([{"name": name} for name in names])
        return servers

    monkeypatch.setattr(mcp_tool, "prepare_mcp_servers", discover)

    def build():
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
            tool_config=ToolConfig(auto_load_mcp=True),
        )
        agent.update_model()
        return agent

    one = build()
    wire = one.model.get_tools_for_api()
    names = {entry["function"]["name"] for entry in wire}
    assert {"always", "use_capability"} <= names
    assert not names.intersection(catalog)
    assert one.model.functions["one"].deferred
    catalog[:] = ["three"]
    two = build()
    assert "three" in two.model.functions and "one" not in two.model.functions
    assert two.model.get_tools_for_api() == wire


def test_mcp_image_result_preserves_text_and_images_without_text_base64():
    from mcp.types import CallToolResult, ImageContent, TextContent

    from agentica.mcp.client import to_agent_tool_result

    result = to_agent_tool_result(CallToolResult(content=[
        TextContent(type="text", text="window captured"),
        ImageContent(type="image", mime_type="image/png", data="YQ=="),
    ]))
    assert result["text_summary"] == "window captured"
    assert result["images"] == [{"url": "data:image/png;base64,YQ=="}]


@pytest.mark.asyncio
async def test_async_mcp_call_cancellation_closes_its_transport(monkeypatch):
    import asyncio

    from agentica.mcp import client
    from agentica.tools.base import FunctionCall
    from agentica.tools.mcp_tool import McpTool

    waiting = asyncio.Event()
    exited = asyncio.Event()

    class FakeClient:
        def __init__(self, server):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            exited.set()

        async def call_tool(self, *_):
            waiting.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(client, "MCPClient", FakeClient)
    server = McpTool(command="test-stdio")
    server._register_tool_specs([{"name": "wait_forever"}])
    assert asyncio.iscoroutinefunction(server.functions["wait_forever"].entrypoint)
    call = FunctionCall(function=server.functions["wait_forever"])
    running = asyncio.create_task(call.execute())
    await asyncio.wait_for(waiting.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=5)
    assert exited.is_set()


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
