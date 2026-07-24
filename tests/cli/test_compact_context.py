# -*- coding: utf-8 -*-
"""Tests for /compact shrinking the context the next request actually carries.

The prompt builder reads history from ``working_memory.runs`` via
``get_messages_from_last_n_runs()``, not from ``working_memory.messages``.
Compaction that rewrites only the latter shrinks the archive while leaving the
next request as large as before, so these tests assert on the ``runs``-derived
list rather than on ``messages``.
"""
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.cli.commands import (
    CommandContext,
    _apply_context_shrink,
    _cmd_compact,
    _history_tokens,
)
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.run_response import RunResponse


def _fat_run(turn: int) -> AgentRun:
    """An AgentRun roughly the shape of a real tool-using turn."""
    user = Message(role="user", content=f"question {turn} " + "detail " * 200)
    assistant = Message(role="assistant", content=f"answer {turn} " + "prose " * 200)
    tool = Message(role="tool", tool_call_id=f"c{turn}", content="tool output " * 400)
    return AgentRun(
        message=user,
        messages=[user],
        response=RunResponse(messages=[user, assistant, tool]),
    )


def _build_agent(num_runs: int = 5):
    from agentica import Agent

    agent = Agent(model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"))
    wm = WorkingMemory()
    for i in range(num_runs):
        run = _fat_run(i)
        wm.add_run(run)
        wm.add_messages(run.response.messages)
    agent.working_memory = wm
    return agent


class TestCollapseRuns(unittest.TestCase):
    """WorkingMemory.collapse_runs replaces runs with one summarised run."""

    def test_collapse_shrinks_prompt_history(self):
        wm = WorkingMemory()
        for i in range(4):
            wm.add_run(_fat_run(i))
        before = len(wm.get_messages_from_last_n_runs())

        wm.collapse_runs([Message(role="user", content="[Context compressed] summary")])

        after = wm.get_messages_from_last_n_runs()
        self.assertEqual(len(wm.runs), 1)
        self.assertLess(len(after), before)
        self.assertIn("[Context compressed]", after[0].content)

    def test_collapse_deep_copies_messages(self):
        """Later mutation of the caller's list must not rewrite stored history."""
        wm = WorkingMemory()
        wm.add_run(_fat_run(0))
        source = [Message(role="user", content="summary")]

        wm.collapse_runs(source)
        source[0].content = "mutated"

        self.assertEqual(wm.get_messages_from_last_n_runs()[0].content, "summary")

    def test_collapse_with_empty_messages_clears_runs(self):
        wm = WorkingMemory()
        wm.add_run(_fat_run(0))
        wm.collapse_runs([])
        self.assertEqual(wm.runs, [])
        self.assertEqual(wm.get_messages_from_last_n_runs(), [])


class TestCmdCompactShrinksNextRequest(unittest.TestCase):
    """/compact must reduce the history the next request carries."""

    def _run_compact(self, agent):
        ctx = CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=agent,
            tui_state={"context_tokens": 120000, "context_window": 128000},
        )
        _cmd_compact(ctx)
        return ctx

    def test_rule_based_compact_shrinks_runs_history(self):
        agent = _build_agent(num_runs=5)
        before = _history_tokens(agent)

        self._run_compact(agent)

        self.assertLess(_history_tokens(agent), before)

    def test_rule_based_compact_lowers_status_bar_context(self):
        agent = _build_agent(num_runs=5)
        ctx = self._run_compact(agent)
        self.assertLess(ctx.tui_state["context_tokens"], 120000)

    def test_compact_is_noop_on_empty_history(self):
        from agentica import Agent

        agent = Agent(model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"))
        agent.working_memory = WorkingMemory()
        ctx = self._run_compact(agent)
        self.assertEqual(ctx.tui_state["context_tokens"], 120000)


class TestApplyContextShrink(unittest.TestCase):
    """The status bar drops by the measured delta, keeping fixed overhead."""

    def _ctx(self, tokens):
        return CommandContext(agent_config={}, current_agent=None,
                              tui_state={"context_tokens": tokens})

    def test_subtracts_removed_tokens_not_absolute_estimate(self):
        """127K with 30K of history removed lands at 97K, not at the 5K estimate."""
        ctx = self._ctx(127000)
        _apply_context_shrink(ctx, before_tokens=35000, after_tokens=5000)
        self.assertEqual(ctx.tui_state["context_tokens"], 97000)

    def test_never_falls_below_remaining_history(self):
        ctx = self._ctx(6000)
        _apply_context_shrink(ctx, before_tokens=50000, after_tokens=5000)
        self.assertEqual(ctx.tui_state["context_tokens"], 5000)

    def test_no_change_when_nothing_was_removed(self):
        ctx = self._ctx(127000)
        _apply_context_shrink(ctx, before_tokens=5000, after_tokens=5000)
        self.assertEqual(ctx.tui_state["context_tokens"], 127000)

    def test_tolerates_missing_tui_state(self):
        ctx = CommandContext(agent_config={}, current_agent=None, tui_state=None)
        _apply_context_shrink(ctx, before_tokens=50000, after_tokens=5000)


if __name__ == "__main__":
    unittest.main()
