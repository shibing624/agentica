# -*- coding: utf-8 -*-
"""Status-bar context seeding + the compaction spinner hint.

All tests mock LLM API keys — no real API usage.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agentica.cli.interactive.stream_loop import (
    _make_compact_phase_handler,
    _read_git_branch,
    _record_main_auto_compaction,
    _record_main_context_usage,
    _render_spinner_text,
    _seed_context_tokens,
    _status_thinking_mode,
)
from agentica.model.message import Message


def _make_agent():
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))


class TestStaticContextSeed(unittest.TestCase):
    """The bar must never claim an empty context: the prefix is always loaded.

    Before the first API call there is no provider figure, but the system prompt
    and tool definitions already occupy the window.
    """

    def test_seed_is_not_zero_before_any_api_call(self):
        agent = _make_agent()
        tui_state = {"context_tokens": 0}
        _seed_context_tokens(agent, tui_state)
        self.assertGreater(tui_state["context_tokens"], 0)

    def test_seed_counts_tool_definitions_on_top_of_the_system_prompt(self):
        """Tool schemas are the larger half of the prefix; missing them under-reports."""
        def sample_tool(path: str) -> str:
            """Read a file at the given path and return its contents."""
            return path

        bare = _make_agent()
        tui_bare = {}
        _seed_context_tokens(bare, tui_bare)

        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        with_tool = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[sample_tool],
        )
        tui_tool = {}
        _seed_context_tokens(with_tool, tui_tool)

        self.assertGreater(tui_tool["context_tokens"], tui_bare["context_tokens"])
        self.assertTrue(with_tool.model.tools, "update_model() must attach tool schemas")

    def test_no_agent_leaves_state_untouched(self):
        tui_state = {"context_tokens": 123}
        _seed_context_tokens(None, tui_state)
        self.assertEqual(tui_state["context_tokens"], 123)

    def test_resumed_history_counts_on_top_of_the_prefix(self):
        """/resume hydrates a whole conversation before the bar is reseeded."""
        from agentica.agent import Agent
        from agentica.memory.models import AgentRun
        from agentica.model.openai import OpenAIChat
        from agentica.run_response import RunResponse

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            add_history_to_context=True,
        )
        empty = {}
        _seed_context_tokens(agent, empty)

        agent.working_memory.add_run(AgentRun(response=RunResponse(messages=[
            Message(role="user", content="a long resumed question " * 200),
            Message(role="assistant", content="a long resumed answer " * 200),
        ])))
        resumed = {}
        _seed_context_tokens(agent, resumed)
        self.assertGreater(resumed["context_tokens"], empty["context_tokens"] * 2)

    def test_history_ignored_when_the_agent_does_not_replay_it(self):
        """The seed must mirror the prompt builder, which gates on this flag."""
        from agentica.agent import Agent
        from agentica.memory.models import AgentRun
        from agentica.model.openai import OpenAIChat
        from agentica.run_response import RunResponse

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            add_history_to_context=False,
        )
        before = {}
        _seed_context_tokens(agent, before)
        agent.working_memory.add_run(AgentRun(response=RunResponse(
            messages=[Message(role="user", content="ignored " * 500)]
        )))
        after = {}
        _seed_context_tokens(agent, after)
        self.assertEqual(after["context_tokens"], before["context_tokens"])

    def test_failure_leaves_the_previous_figure_alone(self):
        """A cosmetic estimate must never abort CLI startup or a command."""
        agent = _make_agent()
        tui_state = {"context_tokens": 4321}
        with patch.object(type(agent), "get_system_message",
                          side_effect=OSError("git exploded")):
            _seed_context_tokens(agent, tui_state)
        self.assertEqual(tui_state["context_tokens"], 4321)


class TestLiveContextUsage(unittest.TestCase):
    """Only main-agent request context may update the session status bar."""

    def test_main_request_replaces_context_and_window(self):
        state = {"context_tokens": 120000, "context_window": 128000}

        _record_main_context_usage(
            {
                "type": "context.usage",
                "is_main_agent": True,
                "context_tokens": 18000,
                "context_window": 200000,
            },
            state,
        )

        self.assertEqual(state["context_tokens"], 18000)
        self.assertEqual(state["context_window"], 200000)

    def test_subagent_and_unrelated_events_do_not_pollute_main_context(self):
        state = {"context_tokens": 42000, "context_window": 128000}
        _record_main_context_usage(
            {
                "type": "context.usage",
                "is_main_agent": False,
                "context_tokens": 900,
                "context_window": 1000,
            },
            state,
        )
        _record_main_context_usage(
            {"type": "compact.auto", "is_main_agent": True},
            state,
        )

        self.assertEqual(state, {"context_tokens": 42000, "context_window": 128000})


class TestStatusProjectIdentity(unittest.TestCase):
    def test_configured_effort_is_the_concise_thinking_label(self):
        agent = _make_agent()
        self.assertEqual(
            _status_thinking_mode(agent, {"reasoning_effort": "high"}),
            "high",
        )

    def test_unset_effort_is_default_not_off(self):
        agent = _make_agent()
        self.assertEqual(_status_thinking_mode(agent, {}), "default")

    def test_explicit_off_in_config_still_shows_off(self):
        agent = _make_agent()
        self.assertEqual(
            _status_thinking_mode(agent, {"reasoning_effort": "off"}),
            "off",
        )

    def test_extra_body_effort_is_just_the_intensity(self):
        agent = _make_agent()
        agent.model.extra_body = {
            "thinking_enabled": True,
            "reasoning_effort": "high",
            "thinking_display": "omitted",
        }
        self.assertEqual(_status_thinking_mode(agent, {}), "high")

    def test_git_branch_is_empty_outside_a_repository(self):
        import tempfile

        with tempfile.TemporaryDirectory() as work_dir:
            self.assertEqual(_read_git_branch(work_dir), "")


class TestCompactingSpinner(unittest.TestCase):
    """Auto-compact blocks the turn on an LLM call; the spinner must say so."""

    def test_compacting_phase_renders_its_own_label(self):
        text = _render_spinner_text(0, "compacting", "", 7.0)
        self.assertIn("compacting", text)
        self.assertIn("7s", text)

    def test_unknown_phase_still_falls_back_to_thinking(self):
        self.assertIn("thinking", _render_spinner_text(0, "thinking", "", 1.0))


class TestAutoCompactEmitsSpinnerEvents(unittest.TestCase):
    """compact.start/end bracket the summarisation so the CLI can react."""

    def _run_compact(self, summariser):
        from agentica.compression.manager import CompressionManager

        events = []
        agent = SimpleNamespace(
            _event_callback=lambda e: events.append(e),
            name="Agent",
            _session_log=None,
        )
        model = SimpleNamespace(id="gpt-4o", context_window=200_000,
                                _agent_ref=lambda: agent)
        cm = CompressionManager()
        msgs = [Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2")]
        with patch.object(cm, "_summarise_conversation", new=summariser):
            try:
                asyncio.run(cm.auto_compact(msgs, model=model, force=True))
            except RuntimeError:
                pass
        return [e["type"] for e in events]

    def test_brackets_the_summarisation(self):
        types = self._run_compact(AsyncMock(return_value="a summary"))
        self.assertIn("compact.start", types)
        self.assertIn("compact.end", types)
        self.assertLess(types.index("compact.start"), types.index("compact.end"))

    def test_end_fires_when_summarisation_returns_nothing(self):
        """The realistic failure: the LLM call is swallowed and yields None."""
        types = self._run_compact(AsyncMock(return_value=None))
        self.assertIn("compact.end", types)

    def test_end_fires_when_summarisation_raises(self):
        """Cancellation propagates out; a stuck 'compacting' spinner must not."""
        types = self._run_compact(AsyncMock(side_effect=RuntimeError("boom")))
        self.assertIn("compact.end", types)

    def test_concurrent_compactions_keep_the_notice_up(self):
        """Subagents share the callback; the first to finish must not clear it."""
        phases = []

        def set_phase(phase, base=""):
            phases.append(phase)

        handler = _make_compact_phase_handler(set_phase, {"_phase": "tool",
                                                          "_spinner_base": "🔧 task"})
        handler({"type": "compact.start"})
        handler({"type": "compact.start"})
        handler({"type": "compact.end"})
        self.assertEqual(phases, ["compacting"], "notice cleared while one still runs")
        handler({"type": "compact.end"})
        self.assertEqual(phases, ["compacting", "tool"])

    def test_interrupted_tool_phase_is_restored_not_assumed(self):
        """A subagent compaction must not cost the parent its tool label."""
        restored = []

        def set_phase(phase, base=""):
            restored.append((phase, base))

        handler = _make_compact_phase_handler(set_phase, {"_phase": "tool",
                                                          "_spinner_base": "🔧 task"})
        handler({"type": "compact.start"})
        handler({"type": "compact.end"})
        self.assertEqual(restored[-1], ("tool", "🔧 task"))

    def test_sm_compact_path_stays_silent(self):
        """Reusing the stored summary is instant — no spinner churn for it."""
        from agentica.compression.manager import CompressionManager

        events = []
        agent = SimpleNamespace(_event_callback=lambda e: events.append(e),
                                name="Agent", _session_log=None)
        model = SimpleNamespace(id="gpt-4o", context_window=200_000,
                                _agent_ref=lambda: agent)
        wm = SimpleNamespace(summary=SimpleNamespace(summary="stored", topics=[]))
        cm = CompressionManager()
        msgs = [Message(role="user", content="q1"), Message(role="assistant", content="a1")]
        asyncio.run(cm.auto_compact(msgs, model=model, force=True, working_memory=wm))
        self.assertNotIn("compact.start", [e["type"] for e in events])


class TestMainAutoCompactionCount(unittest.TestCase):
    """Only successful full compactions of the active main agent are counted."""

    def test_ignores_lifecycle_failures_reactive_and_subagent_events(self):
        state = {"compaction_count": 0}
        ignored = [
            {"type": "compact.end", "is_main_agent": True},
            {"type": "compact.reactive", "is_main_agent": True},
            {"type": "compact.auto", "is_main_agent": False},
        ]

        for event in ignored:
            _record_main_auto_compaction(event, state)

        self.assertEqual(state["compaction_count"], 0)
        self.assertTrue(all("compaction_count" not in event for event in ignored))

    def test_counts_successful_main_auto_compactions_and_annotates_event(self):
        state = {"compaction_count": 0}
        first = {"type": "compact.auto", "is_main_agent": True}
        second = {"type": "compact.auto", "is_main_agent": True}

        _record_main_auto_compaction(first, state)
        _record_main_auto_compaction(second, state)

        self.assertEqual(state["compaction_count"], 2)
        self.assertEqual(first["compaction_count"], 1)
        self.assertEqual(second["compaction_count"], 2)


if __name__ == "__main__":
    unittest.main()
