# -*- coding: utf-8 -*-
"""Tests for /compact shrinking the context the next request actually carries.

The prompt builder reads history from ``working_memory.runs`` via
``get_messages_from_last_n_runs()``, not from ``working_memory.messages``.
"""
import os
import asyncio
import unittest
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.session import _cmd_compact
from agentica.cli.context_usage import measure_context
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.run_response import RunResponse


def _fat_run(turn: int) -> AgentRun:
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

    agent = Agent(
        model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"),
        add_history_to_context=True,
    )
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

        wm.collapse_runs([Message(role="user", content="<context_window>\nCurrent context window 1.\n</context_window>")])

        after = wm.get_messages_from_last_n_runs()
        self.assertEqual(len(wm.runs), 1)
        self.assertLess(len(after), before)
        self.assertIn("<context_window>", after[0].content)

    def test_collapse_deep_copies_messages(self):
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

    def _run_compact(self, agent, args=""):
        ctx = CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=agent,
            tui_state={"context_tokens": 120000, "context_window": 128000},
        )
        _cmd_compact(ctx, args)
        return ctx

    def test_new_window_shrinks_runs_history(self):
        agent = _build_agent(num_runs=5)
        before = asyncio.run(measure_context(agent)).total
        self._run_compact(agent)
        self.assertLess(asyncio.run(measure_context(agent)).total, before)
        history = agent.working_memory.get_messages_from_last_n_runs()
        joined = " ".join(str(m.content) for m in history)
        self.assertIn("<context_window>", joined)
        self.assertNotIn("[Context compressed]", joined)
        self.assertTrue(
            any("New context window started" in str(m.content) for m in history),
            "idle /compact must leave a preamble for the next request",
        )
        self.assertFalse(
            any(m.role == "assistant" for m in history),
            "idle /compact must drop the last answered turn",
        )

    def test_next_request_folds_preamble_into_the_user_turn(self):
        agent = _build_agent(num_runs=3)
        self._run_compact(agent)
        agent.run_response = RunResponse()
        _, user_messages, messages_for_model = asyncio.run(
            agent.get_messages_for_run(message="what is the ticket id?")
        )
        self.assertEqual(len(user_messages), 1)
        self.assertIn("<context_window>", user_messages[0].content)
        self.assertIn("what is the ticket id?", user_messages[0].content)
        roles = [m.role for m in messages_for_model if m.role != "system"]
        self.assertNotIn(("user", "user"), list(zip(roles, roles[1:])))
        self.assertFalse(
            any(
                "New context window started" in str(m.content)
                for m in agent.working_memory.get_messages_from_last_n_runs()
            )
        )

    def test_new_window_lowers_status_bar_context(self):
        agent = _build_agent(num_runs=5)
        ctx = self._run_compact(agent)
        self.assertLess(ctx.tui_state["context_tokens"], 120000)

    def test_compact_still_runs_when_auto_compact_is_off(self):
        agent = _build_agent(num_runs=5)
        agent.tool_config.enable_auto_compact = False
        before = asyncio.run(measure_context(agent)).total
        self._run_compact(agent)
        self.assertLess(asyncio.run(measure_context(agent)).total, before)

    def test_compact_is_noop_on_empty_history(self):
        from agentica import Agent

        agent = Agent(model=OpenAIChat(id="gpt-4o", api_key="fake_openai_key"))
        agent.working_memory = WorkingMemory()
        ctx = self._run_compact(agent)
        self.assertEqual(ctx.tui_state["context_tokens"], 120000)

    def test_compact_ignores_native_endpoint_and_instructions(self):
        from agentica.model.openai import OpenAIResponses

        agent = _build_agent(num_runs=2)
        model = OpenAIResponses(id="gpt-5.6-sol", api_key="fake_openai_key")
        model.compact_context = AsyncMock(side_effect=AssertionError("native compact is deleted"))
        agent.model = model
        ctx = self._run_compact(agent, "Keep decisions")
        model.compact_context.assert_not_called()
        self.assertLess(ctx.tui_state["context_tokens"], 120000)


if __name__ == "__main__":
    unittest.main()
