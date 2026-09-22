# -*- coding: utf-8 -*-
"""Context breakdown behind the status bar and /usage.

All tests mock LLM API keys — no real API usage.
"""
import asyncio
import unittest

from agentica.agent import Agent
from agentica.agent.config import ToolConfig
from agentica.cli.context_usage import COMPACT_SUMMARY_PREFIX, measure_context
from agentica.memory.models import AgentRun
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.run.response import RunResponse
from agentica.utils.tokens import count_tokens


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
        self.assertGreater(_row(bare, "Tool definitions"), 0, "BuiltinContextTool is always on")
        self.assertGreater(_row(withtool, "Tool definitions"), _row(bare, "Tool definitions"))
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

    def test_new_window_preamble_is_split_from_ordinary_conversation(self):
        agent = _agent(add_history_to_context=True)
        agent.working_memory.add_run(AgentRun(response=RunResponse(messages=[
            Message(
                role="user",
                content="<context_window>\nCurrent context window 1.\n</context_window>\n\n"
                + ("earlier turns " * 50),
            ),
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

    def test_measure_reflects_what_eviction_will_actually_send(self):
        """The idle bar must show the next request's real size, not the
        pre-compression history.

        Layer 1 evicts old tool results before every request once the request
        crosses the pressure threshold. A bar that ignores that showed 144%
        (284K/192K) on a session whose next request actually shipped ~75% —
        the user cannot tell "about to die" from "healthy, will be evicted".
        """
        from agentica.model.message import Message

        agent = _agent(add_history_to_context=True)
        agent.model.context_window = 8_000  # pressure threshold = 6_400
        msgs = [Message(role="user", content="question " * 50)]
        # Old tool rounds: exactly what Layer 1 replaces with placeholders.
        for i in range(20):
            msgs.append(Message(
                role="assistant", content="", tool_calls=[{
                    "id": f"call_{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            ))
            msgs.append(Message(
                role="tool", tool_call_id=f"call_{i}",
                content="file body " * 400,
            ))
        agent.working_memory.add_run(AgentRun(response=RunResponse(messages=msgs)))

        from agentica.utils.tokens import count_tokens
        raw = count_tokens(msgs, None, "gpt-4o-mini")
        # The raw history alone is far over the pressure line...
        self.assertGreater(raw, 6_400)
        b = _measure(agent)
        # ...and the eviction the runner would run is already reflected: the
        # plain-history section shrinks below its raw size.
        self.assertLess(_row(b, "Conversation") + _row(b, "Summarized conversation"), raw)


class TestWorkingCapIsTheCompressionReference(unittest.TestCase):
    """The bar and /usage must judge occupancy by the same window the
    compression policy uses, or they report "61%, healthy" for a session the
    runner has already been evicting for hours.

    ``compact_token_limit`` is an absolute working budget (Codex-style 300k on
    a 1M model). ``model.context_window`` stays the provider hard limit and
    must keep being reported, because that is what a provider error would say.
    """

    def _capped_agent(self, *, cap: int, window: int = 1_000_000) -> Agent:
        agent = _agent(
            add_history_to_context=True,
            tool_config=ToolConfig(compact_token_limit=cap),
        )
        agent.model.context_window = window
        # One old tool round: exactly what Layer 1 replaces with placeholders.
        msgs = [Message(role="user", content="question " * 50)]
        for i in range(30):
            msgs.append(Message(
                role="assistant", content="", tool_calls=[{
                    "id": f"call_{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            ))
            msgs.append(Message(
                role="tool", tool_call_id=f"call_{i}",
                content="file body " * 400,
            ))
        agent.working_memory.add_run(AgentRun(response=RunResponse(messages=msgs)))
        return agent, msgs

    def test_cap_is_the_denominator_not_the_provider_window(self):
        """Reporting against 1M is what produced "61%" on a session the runner
        was already evicting: the cap is the budget that actually fills."""
        agent, _ = self._capped_agent(cap=8_000)
        b = _measure(agent)
        self.assertEqual(b.window, 8_000, "bar must divide by the working cap")
        self.assertEqual(b.provider_window, 1_000_000, "hard limit stays visible")

    def test_no_cap_keeps_the_provider_window_as_the_reference(self):
        """Unset cap means no working budget — behaviour must not change."""
        agent = _agent(add_history_to_context=True)
        agent.model.context_window = 128_000
        b = _measure(agent)
        self.assertEqual(b.window, 128_000)
        self.assertEqual(b.provider_window, 128_000)

    def test_cap_above_the_window_cannot_raise_the_reference(self):
        """A cap larger than the provider limit is not a bigger budget."""
        agent = _agent(add_history_to_context=True, tool_config=ToolConfig(
            compact_token_limit=2_000_000))
        agent.model.context_window = 128_000
        self.assertEqual(_measure(agent).window, 128_000)

    def test_eviction_is_reflected_at_the_cap_threshold(self):
        """The whole point: the reported figure drops to what will ship.

        With the cap in play the pressure line is 0.8 x cap, so a transcript
        far above it must be reported already-evicted, not raw.
        """
        agent, msgs = self._capped_agent(cap=8_000)
        raw = count_tokens(msgs, None, "gpt-4o-mini")
        b = _measure(agent)
        self.assertGreater(raw, 8_000 * 0.8, "fixture must be under real pressure")
        self.assertLess(b.total, raw, "reported total must be the post-evict size")
        self.assertLess(b.percent_full, 100.0, "an evicted request is not over budget")


class TestReportedHistoryWindow(unittest.TestCase):
    """The turn window is configurable, so the reported figure must follow the
    agent's ``num_history_turns`` rather than assume the built-in 20."""

    def _agent_with_runs(self, num_runs: int, **kwargs) -> Agent:
        agent = _agent(add_history_to_context=True, **kwargs)
        for i in range(num_runs):
            agent.working_memory.add_run(AgentRun(response=RunResponse(messages=[
                Message(role="user", content=f"question {i} " * 20),
                Message(role="assistant", content=f"answer {i} " * 20),
            ])))
        return agent

    def test_reported_turns_is_the_configured_window(self):
        agent = self._agent_with_runs(30, num_history_turns=7)
        self.assertEqual(_measure(agent).history_turns, 7)

    def test_a_window_larger_than_the_session_reports_only_what_exists(self):
        """Claiming "last 50 turns" over six stored runs would overstate."""
        agent = self._agent_with_runs(6, num_history_turns=50)
        self.assertEqual(_measure(agent).history_turns, 6)

    def test_reported_messages_are_the_ones_the_prompt_carries(self):
        """Count and tokens must come from one set: 3 runs x 2 messages."""
        agent = self._agent_with_runs(3, num_history_turns=20)
        self.assertEqual(_measure(agent).history_messages, 6)

    def test_none_means_every_run_not_zero(self):
        """More runs than the built-in default, so "all" and "20" differ."""
        agent = self._agent_with_runs(25, num_history_turns=None)
        self.assertEqual(_measure(agent).history_turns, 25)

    def test_history_off_reports_no_turns_however_many_are_stored(self):
        """Nothing is replayed, so the window covered nothing."""
        agent = _agent(add_history_to_context=False)
        agent.working_memory.add_run(AgentRun(response=RunResponse(
            messages=[Message(role="user", content="stored but not replayed " * 20)]
        )))
        b = _measure(agent)
        self.assertEqual(b.history_turns, 0)
        self.assertEqual(b.history_messages, 0)


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
