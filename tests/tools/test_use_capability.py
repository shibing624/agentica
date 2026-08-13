"""Tests for the stable `use_capability` proxy (Reasonix TOOL_CONTRACT.zh-CN.md
use_capability 移植) and McpTool defer_schema.

The proxy's provider-visible schema must be fixed (action + name + arguments),
so MCP/dynamic-skill inventory churn no longer cold-starts the prompt cache —
the whole point of extracting optional tools out of the top-level schema.
"""
import unittest

from agentica.model.openai import OpenAIChat
from agentica.tools.base import FunctionCall
from agentica.tools.decorators import tool


def _build_agent_with_deferred():
    from agentica.agent import Agent
    from agentica.tools.use_capability_tool import UseCapabilityTool

    @tool(deferred=True)
    def get_weather(city: str) -> str:
        """Get the weather for a city.

        Args:
            city: City name.
        """
        return f"sunny in {city}"

    agent = Agent(
        name="CapabilityProxy",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tools=[get_weather, UseCapabilityTool()],
    )
    agent.update_model()
    return agent


class TestUseCapabilityProxySchema(unittest.TestCase):
    def test_proxy_schema_is_fixed_three_fields(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        self.assertEqual(set(fn.parameters["properties"].keys()), {"action", "name", "arguments"})
        # action is a closed enum -> schema cannot drift with inventory.
        self.assertEqual(
            fn.parameters["properties"]["action"]["enum"],
            ["list", "inspect", "call", "decline"],
        )

    def test_deferred_tool_not_in_top_level_schema(self):
        agent = _build_agent_with_deferred()
        tool_names = {t["function"]["name"] for t in agent.model.get_tools_for_api()}
        self.assertIn("use_capability", tool_names)
        self.assertNotIn("get_weather", tool_names)
        # but it stays executable via the host registry
        self.assertIn("get_weather", agent.model.functions)


class TestUseCapabilityDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_list_returns_deferred_tool_names(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        fc = FunctionCall(function=fn, arguments={"action": "list"})
        ok = await fc.execute()
        self.assertTrue(ok)
        self.assertIn("get_weather", str(fc.result))

    async def test_inspect_returns_deferred_tool_schema(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        fc = FunctionCall(function=fn, arguments={"action": "inspect", "name": "get_weather"})
        ok = await fc.execute()
        self.assertTrue(ok)
        self.assertIn("city", str(fc.result))

    async def test_call_dispatches_to_deferred_tool(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        fc = FunctionCall(
            function=fn,
            arguments={"action": "call", "name": "get_weather", "arguments": {"city": "Beijing"}},
        )
        ok = await fc.execute()
        self.assertTrue(ok)
        self.assertIn("sunny in Beijing", str(fc.result))

    async def test_call_unknown_name_reports_error(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        fc = FunctionCall(
            function=fn,
            arguments={"action": "call", "name": "no_such_tool", "arguments": {}},
        )
        ok = await fc.execute()
        self.assertTrue(ok)  # proxy itself succeeded; it reports the dispatch error in text
        self.assertIn("no_such_tool", str(fc.result))

    async def test_decline_acknowledges(self):
        agent = _build_agent_with_deferred()
        fn = agent.model.functions["use_capability"]
        fc = FunctionCall(function=fn, arguments={"action": "decline", "name": "get_weather"})
        ok = await fc.execute()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
