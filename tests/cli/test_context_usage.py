# -*- coding: utf-8 -*-
"""Context breakdown behind the status bar and /usage.

All tests mock LLM API keys — no real API usage.
"""
import asyncio
import unittest

from agentica.agent import Agent
from agentica.cli.context_usage import COMPACT_SUMMARY_PREFIX, measure_context
from agentica.memory.models import AgentRun
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.run_response import RunResponse


def _agent(**kwargs) -> Agent:
    return Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"), **kwargs)


def _measure(agent):
    return asyncio.run(measure_context(agent))


def _row(breakdown, label: str) -> int:
    return dict(breakdown.sections)[label]


class TestMeasureContext(unittest.TestCase):

    def test_sections_sum_to_the_reported_total(self):
        """A breakdown whose rows don't add up to the headline is a lie."""
        b = _measure(_agent())
        self.assertEqual(b.total, sum(t for _, t in b.sections))

    def test_system_prompt_is_present_before_any_turn(self):
        b = _measure(_agent())
        self.assertGreater(_row(b, "System prompt"), 0)
        self.assertGreater(b.total, 0)

    def test_tool_definitions_are_counted_separately(self):
        def sample_tool(path: str) -> str:
            """Read a file at the given path and return its contents."""
            return path

        bare = _measure(_agent())
        withtool = _measure(_agent(tools=[sample_tool]))
        self.assertEqual(_row(bare, "Tool definitions"), 0)
        self.assertGreater(_row(withtool, "Tool definitions"), 0)
        self.assertGreater(withtool.total, bare.total)

    def test_compacted_summary_is_split_from_ordinary_conversation(self):
        agent = _agent(add_history_to_context=True)
        agent.working_memory.add_run(AgentRun(response=RunResponse(messages=[
            Message(role="user", content=f"{COMPACT_SUMMARY_PREFIX}\n\nearlier turns " * 50),
            Message(role="assistant", content="Understood."),
            Message(role="user", content="and now the live question " * 50),
        ])))
        b = _measure(agent)
        self.assertGreater(_row(b, "Summarized conversation"), 0)
        self.assertGreater(_row(b, "Conversation"), 0)

    def test_history_only_counts_when_the_agent_replays_it(self):
        run = AgentRun(response=RunResponse(
            messages=[Message(role="user", content="ignored " * 200)]
        ))
        off = _agent(add_history_to_context=False)
        off.working_memory.add_run(run)
        self.assertEqual(_row(_measure(off), "Conversation"), 0)

        on = _agent(add_history_to_context=True)
        on.working_memory.add_run(run)
        self.assertGreater(_row(_measure(on), "Conversation"), 0)

    def test_empty_rows_are_hidden_from_display(self):
        b = _measure(_agent())
        self.assertTrue(all(t > 0 for _, t in b.visible_sections()))
        self.assertLess(len(b.visible_sections()), len(b.sections))

    def test_percent_full_uses_the_model_window(self):
        agent = _agent()
        agent.model.context_window = 1000
        b = _measure(agent)
        self.assertEqual(b.window, 1000)
        self.assertAlmostEqual(b.percent_full, b.total / 10, places=4)


class TestMcpToolSplit(unittest.TestCase):
    """MCP schemas are billed to the context like any other tool."""

    def test_mcp_sourced_tools_land_in_their_own_row(self):
        from agentica.tools.base import Function, Tool
        from agentica.tools.origin import ToolOrigin

        def local_tool(a: str) -> str:
            """A locally defined tool that echoes its argument."""
            return a

        def remote_tool(b: str) -> str:
            """A tool served by an MCP server, registered the way McpTool does."""
            return b

        # Mirrors McpTool._register_tools: a Function carrying an mcp origin is
        # placed straight into the toolkit.
        mcp_toolkit = Tool(name="fake_mcp")
        fn = Function.from_callable(remote_tool)
        fn.origin = ToolOrigin(type="mcp", provider_name="fake_mcp")
        mcp_toolkit.functions[fn.name] = fn

        agent = _agent(tools=[local_tool, mcp_toolkit])
        b = _measure(agent)
        self.assertGreater(_row(b, "MCP tools"), 0)
        self.assertGreater(_row(b, "Tool definitions"), 0)


if __name__ == "__main__":
    unittest.main()
