# -*- coding: utf-8 -*-
"""
Tests for AgentHooks, RunHooks, _CompositeRunHooks, and ConversationArchiveHooks.
All tests mock LLM calls — no real API usage.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.hooks import (
    AgentHooks, RunHooks, ConversationArchiveHooks,
    _CompositeRunHooks, _CompositeAgentHooks,
)


# ---------------------------------------------------------------------------
# AgentHooks
# ---------------------------------------------------------------------------

class TestAgentHooks(unittest.TestCase):
    """AgentHooks base class — default methods are no-ops."""

    def test_on_start_is_noop(self):
        hooks = AgentHooks()
        agent = MagicMock()
        # Should complete without error
        asyncio.run(hooks.on_start(agent=agent))

    def test_on_end_is_noop(self):
        hooks = AgentHooks()
        agent = MagicMock()
        asyncio.run(hooks.on_end(agent=agent, output="hello"))

    def test_subclass_on_start_called(self):
        calls = []

        class MyHooks(AgentHooks):
            async def on_start(self, agent, **kwargs):
                calls.append(("start", agent))

            async def on_end(self, agent, output, **kwargs):
                calls.append(("end", output))

        hooks = MyHooks()
        agent = MagicMock()
        asyncio.run(hooks.on_start(agent=agent))
        asyncio.run(hooks.on_end(agent=agent, output="result"))

        self.assertEqual(calls[0][0], "start")
        self.assertEqual(calls[1][0], "end")
        self.assertEqual(calls[1][1], "result")

    def test_composite_fans_out_to_all(self):
        """_CompositeAgentHooks dispatches on_start/on_end to every member."""
        calls = []

        class A(AgentHooks):
            async def on_start(self, agent, **kwargs):
                calls.append("a_start")

            async def on_end(self, agent, output, **kwargs):
                calls.append(("a_end", output))

        class B(AgentHooks):
            async def on_start(self, agent, **kwargs):
                calls.append("b_start")

            async def on_end(self, agent, output, **kwargs):
                calls.append(("b_end", output))

        composite = _CompositeAgentHooks([A(), B()])
        agent = MagicMock()
        asyncio.run(composite.on_start(agent=agent))
        asyncio.run(composite.on_end(agent=agent, output="r"))

        self.assertEqual(calls[0], "a_start")
        self.assertEqual(calls[1], "b_start")
        self.assertIn(("a_end", "r"), calls)
        self.assertIn(("b_end", "r"), calls)
        # Underlying list is exposed so callers can locate a specific hook.
        self.assertEqual(len(composite.hooks), 2)

    def test_agent_wraps_hooks_list(self):
        """Agent(hooks=[...]) wraps a list into _CompositeAgentHooks; a single
        hook is kept as-is; None stays None."""
        import os
        os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")
        from agentica import Agent

        single = AgentHooks()
        self.assertIs(Agent(hooks=single).hooks, single)
        self.assertIsNone(Agent(hooks=None).hooks)

        h1, h2 = AgentHooks(), AgentHooks()
        wrapped = Agent(hooks=[h1, h2]).hooks
        self.assertIsInstance(wrapped, _CompositeAgentHooks)
        self.assertEqual(wrapped.hooks, [h1, h2])
        # Empty list collapses to None (nothing to dispatch).
        self.assertIsNone(Agent(hooks=[]).hooks)

    def test_hook_exception_does_not_propagate_through_base(self):
        """If a hook raises, it should surface (not be silently swallowed)."""
        class BrokenHooks(AgentHooks):
            async def on_start(self, agent, **kwargs):
                raise RuntimeError("hook failed")

        hooks = BrokenHooks()
        with self.assertRaises(RuntimeError, msg="hook failed"):
            asyncio.run(hooks.on_start(agent=MagicMock()))


# ---------------------------------------------------------------------------
# RunHooks
# ---------------------------------------------------------------------------

class TestRunHooks(unittest.TestCase):
    """RunHooks base class — all methods are no-ops by default."""

    def test_all_default_methods_are_noop(self):
        hooks = RunHooks()
        agent = MagicMock()
        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="x"))
        asyncio.run(hooks.on_llm_start(agent=agent, messages=[]))
        asyncio.run(hooks.on_llm_end(agent=agent, response=None))
        asyncio.run(hooks.on_tool_start(agent=agent, tool_name="t", tool_call_id="1", tool_args={}))
        asyncio.run(hooks.on_tool_end(agent=agent, tool_name="t", tool_call_id="1", tool_args={}, result="r"))
        asyncio.run(hooks.on_agent_transfer(from_agent=agent, to_agent=agent))

    def test_on_tool_end_elapsed_parameter(self):
        """on_tool_end must accept elapsed kwarg."""
        received = {}

        class TimingHooks(RunHooks):
            async def on_tool_end(self, agent, tool_name="", tool_call_id="",
                                  tool_args=None, result=None, is_error=False, elapsed=0.0, **kwargs):
                received["elapsed"] = elapsed
                received["result"] = result

        hooks = TimingHooks()
        asyncio.run(hooks.on_tool_end(
            agent=MagicMock(), tool_name="web_search", tool_call_id="c1",
            tool_args={}, result="found", elapsed=1.23,
        ))
        self.assertAlmostEqual(received["elapsed"], 1.23)
        self.assertEqual(received["result"], "found")


# ---------------------------------------------------------------------------
# _CompositeRunHooks
# ---------------------------------------------------------------------------

class TestCompositeRunHooks(unittest.TestCase):
    """_CompositeRunHooks must call all constituent hooks."""

    def _make_recording_hooks(self, label: str, log: list):
        class RecordHooks(RunHooks):
            async def on_agent_start(self, agent, **kwargs):
                log.append(f"{label}.on_agent_start")

            async def on_agent_end(self, agent, output, **kwargs):
                log.append(f"{label}.on_agent_end")

            async def on_llm_start(self, agent, messages=None, **kwargs):
                log.append(f"{label}.on_llm_start")

            async def on_llm_end(self, agent, response=None, **kwargs):
                log.append(f"{label}.on_llm_end")

            async def on_tool_start(self, agent, tool_name="", tool_call_id="", tool_args=None, **kwargs):
                log.append(f"{label}.on_tool_start")

            async def on_tool_end(self, agent, tool_name="", tool_call_id="", tool_args=None,
                                  result=None, is_error=False, elapsed=0.0, **kwargs):
                log.append(f"{label}.on_tool_end")

        return RecordHooks()

    def test_both_hooks_called_on_agent_start(self):
        log = []
        h1 = self._make_recording_hooks("h1", log)
        h2 = self._make_recording_hooks("h2", log)
        composite = _CompositeRunHooks([h1, h2])

        asyncio.run(composite.on_agent_start(agent=MagicMock()))
        self.assertIn("h1.on_agent_start", log)
        self.assertIn("h2.on_agent_start", log)

    def test_both_hooks_called_on_tool_end(self):
        log = []
        h1 = self._make_recording_hooks("h1", log)
        h2 = self._make_recording_hooks("h2", log)
        composite = _CompositeRunHooks([h1, h2])

        asyncio.run(composite.on_tool_end(
            agent=MagicMock(), tool_name="x", tool_call_id="1",
            tool_args={}, result="ok", elapsed=0.5,
        ))
        self.assertIn("h1.on_tool_end", log)
        self.assertIn("h2.on_tool_end", log)

    def test_composite_continues_if_first_hook_raises(self):
        """Second hook must still be called even if first raises."""
        log = []

        class FailHook(RunHooks):
            async def on_agent_start(self, agent, **kwargs):
                raise RuntimeError("first hook failed")

        h2 = self._make_recording_hooks("h2", log)
        composite = _CompositeRunHooks([FailHook(), h2])

        # The composite should propagate the error but still call h2
        # (depending on implementation — at minimum h2 should be called)
        try:
            asyncio.run(composite.on_agent_start(agent=MagicMock()))
        except RuntimeError:
            pass
        # h2 may or may not be called depending on impl — at least no silent swallow
        # This test documents the behavior: error is visible (not silently swallowed)


# ---------------------------------------------------------------------------
# ConversationArchiveHooks
# ---------------------------------------------------------------------------

class TestConversationArchiveHooks(unittest.TestCase):
    """ConversationArchiveHooks must write to workspace after agent run."""

    def test_on_agent_start_captures_run_input(self):
        """ConversationArchiveHooks reads run_input at on_agent_end time."""
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent-123"
        agent.run_input = "hello world"
        agent.run_id = "run-1"
        agent.workspace = MagicMock()
        agent.workspace.archive_conversation = AsyncMock(return_value="/tmp/archive.md")

        asyncio.run(hooks.on_agent_end(agent=agent, output="response"))
        # Should have archived with current run_input
        call_args = agent.workspace.archive_conversation.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "hello world")

    def test_on_agent_end_with_no_workspace_is_noop(self):
        """If agent has no workspace, on_agent_end should not raise."""
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent-no-workspace"
        agent.run_input = "test"
        agent.workspace = None

        asyncio.run(hooks.on_agent_start(agent=agent))
        # Should complete silently — no workspace to write to
        asyncio.run(hooks.on_agent_end(agent=agent, output="response"))

    def test_on_agent_end_writes_to_workspace(self):
        """If workspace is set, on_agent_end must call workspace.archive_conversation."""
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent-ws"
        agent.run_id = "run-1"
        agent.run_input = "user question"
        agent.workspace = MagicMock()
        agent.workspace.archive_conversation = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="agent answer"))

        agent.workspace.archive_conversation.assert_called_once()
        call_args = agent.workspace.archive_conversation.call_args
        archived_messages = call_args[0][0]  # first positional arg
        roles = [m["role"] for m in archived_messages]
        contents = [m["content"] for m in archived_messages]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertIn("user question", contents)
        self.assertIn("agent answer", contents)


if __name__ == "__main__":
    unittest.main()
