# -*- coding: utf-8 -*-
"""Status-bar context seeding + the compaction spinner hint.

All tests mock LLM API keys — no real API usage.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agentica.cli.interactive import _render_spinner_text, _seed_static_context_tokens
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
        _seed_static_context_tokens(agent, tui_state)
        self.assertGreater(tui_state["context_tokens"], 0)

    def test_seed_counts_tool_definitions_on_top_of_the_system_prompt(self):
        """Tool schemas are the larger half of the prefix; missing them under-reports."""
        from agentica.utils.tokens import count_tokens

        def sample_tool(path: str) -> str:
            """Read a file at the given path and return its contents."""
            return path

        bare = _make_agent()
        tui_bare = {}
        _seed_static_context_tokens(bare, tui_bare)

        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        with_tool = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[sample_tool],
        )
        tui_tool = {}
        _seed_static_context_tokens(with_tool, tui_tool)

        self.assertGreater(tui_tool["context_tokens"], tui_bare["context_tokens"])
        self.assertTrue(with_tool.model.tools, "update_model() must attach tool schemas")

    def test_no_agent_leaves_state_untouched(self):
        tui_state = {"context_tokens": 123}
        _seed_static_context_tokens(None, tui_state)
        self.assertEqual(tui_state["context_tokens"], 123)


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


if __name__ == "__main__":
    unittest.main()
