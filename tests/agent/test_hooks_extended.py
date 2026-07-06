# -*- coding: utf-8 -*-
"""Tests for agentica.hooks — new hook methods: on_user_prompt, on_pre_compact, on_post_compact,
and _CompositeRunHooks chaining behavior."""
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

from agentica.hooks import (
    AgentHooks,
    RunHooks,
    ConversationArchiveHooks,
    _CompositeRunHooks,
)


class TestRunHooksOnUserPrompt(unittest.TestCase):
    """on_user_prompt: returns modified message or None."""

    def test_default_returns_none(self):
        hooks = RunHooks()
        result = asyncio.run(hooks.on_user_prompt(agent=MagicMock(), message="hello"))
        self.assertIsNone(result)

    def test_subclass_modifies_message(self):
        class UpperHooks(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return message.upper()

        hooks = UpperHooks()
        result = asyncio.run(hooks.on_user_prompt(agent=MagicMock(), message="hello"))
        self.assertEqual(result, "HELLO")


class TestRunHooksCompactCallbacks(unittest.TestCase):
    """on_pre_compact and on_post_compact: lifecycle hooks for compression."""

    def test_on_pre_compact_default_noop(self):
        hooks = RunHooks()
        # Should not raise
        asyncio.run(hooks.on_pre_compact(agent=MagicMock(), messages=[]))

    def test_on_post_compact_default_noop(self):
        hooks = RunHooks()
        asyncio.run(hooks.on_post_compact(agent=MagicMock(), messages=[]))

    def test_on_pre_compact_subclass(self):
        captured = {}

        class MyHooks(RunHooks):
            async def on_pre_compact(self, agent, messages=None, **kwargs):
                captured["pre"] = len(messages or [])

        hooks = MyHooks()
        asyncio.run(hooks.on_pre_compact(agent=MagicMock(), messages=[1, 2, 3]))
        self.assertEqual(captured["pre"], 3)

    def test_on_post_compact_subclass(self):
        captured = {}

        class MyHooks(RunHooks):
            async def on_post_compact(self, agent, messages=None, **kwargs):
                captured["post"] = len(messages or [])

        hooks = MyHooks()
        asyncio.run(hooks.on_post_compact(agent=MagicMock(), messages=[1]))
        self.assertEqual(captured["post"], 1)


class TestCompositeRunHooksOnUserPrompt(unittest.TestCase):
    """_CompositeRunHooks chains on_user_prompt: each hook sees the modified message."""

    def test_single_hook_returns_modified(self):
        class AddPrefix(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return "[prefix] " + message

        composite = _CompositeRunHooks([AddPrefix()])
        result = asyncio.run(composite.on_user_prompt(agent=MagicMock(), message="hello"))
        self.assertEqual(result, "[prefix] hello")

    def test_chained_hooks(self):
        """Second hook sees the modified message from the first."""

        class AddA(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return message + " [A]"

        class AddB(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return message + " [B]"

        composite = _CompositeRunHooks([AddA(), AddB()])
        result = asyncio.run(composite.on_user_prompt(agent=MagicMock(), message="start"))
        self.assertEqual(result, "start [A] [B]")

    def test_hook_returning_none_preserves_message(self):
        class NoOp(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return None  # no modification

        class AddTag(RunHooks):
            async def on_user_prompt(self, agent, message, **kwargs):
                return message + " [tagged]"

        composite = _CompositeRunHooks([NoOp(), AddTag()])
        result = asyncio.run(composite.on_user_prompt(agent=MagicMock(), message="msg"))
        self.assertEqual(result, "msg [tagged]")

    def test_all_noop_returns_none(self):
        composite = _CompositeRunHooks([RunHooks(), RunHooks()])
        result = asyncio.run(composite.on_user_prompt(agent=MagicMock(), message="msg"))
        self.assertIsNone(result)


class TestCompositeRunHooksCompact(unittest.TestCase):
    """_CompositeRunHooks dispatches on_pre_compact/on_post_compact to all hooks."""

    def test_pre_compact_dispatched(self):
        called = []

        class H1(RunHooks):
            async def on_pre_compact(self, agent, messages=None, **kwargs):
                called.append("h1")

        class H2(RunHooks):
            async def on_pre_compact(self, agent, messages=None, **kwargs):
                called.append("h2")

        composite = _CompositeRunHooks([H1(), H2()])
        asyncio.run(composite.on_pre_compact(agent=MagicMock()))
        self.assertEqual(called, ["h1", "h2"])

    def test_post_compact_dispatched(self):
        called = []

        class H1(RunHooks):
            async def on_post_compact(self, agent, messages=None, **kwargs):
                called.append("h1")

        class H2(RunHooks):
            async def on_post_compact(self, agent, messages=None, **kwargs):
                called.append("h2")

        composite = _CompositeRunHooks([H1(), H2()])
        asyncio.run(composite.on_post_compact(agent=MagicMock()))
        self.assertEqual(called, ["h1", "h2"])


class TestCompositeRunHooksAllMethods(unittest.TestCase):
    """_CompositeRunHooks dispatches all methods to all hooks."""

    def test_on_agent_start_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_agent_start(self, agent, **kwargs):
                called.append(1)

        composite = _CompositeRunHooks([H(), H()])
        asyncio.run(composite.on_agent_start(agent=MagicMock()))
        self.assertEqual(called, [1, 1])

    def test_on_agent_end_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_agent_end(self, agent, output, **kwargs):
                called.append(output)

        composite = _CompositeRunHooks([H(), H()])
        asyncio.run(composite.on_agent_end(agent=MagicMock(), output="done"))
        self.assertEqual(called, ["done", "done"])

    def test_on_llm_start_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_llm_start(self, agent, messages=None, **kwargs):
                called.append(len(messages or []))

        composite = _CompositeRunHooks([H()])
        asyncio.run(composite.on_llm_start(agent=MagicMock(), messages=[1, 2]))
        self.assertEqual(called, [2])

    def test_on_llm_end_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_llm_end(self, agent, response=None, **kwargs):
                called.append(response)

        composite = _CompositeRunHooks([H()])
        asyncio.run(composite.on_llm_end(agent=MagicMock(), response="resp"))
        self.assertEqual(called, ["resp"])

    def test_on_tool_start_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_tool_start(self, agent, tool_name="", **kwargs):
                called.append(tool_name)

        composite = _CompositeRunHooks([H()])
        asyncio.run(composite.on_tool_start(agent=MagicMock(), tool_name="read_file"))
        self.assertEqual(called, ["read_file"])

    def test_on_tool_end_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_tool_end(self, agent, tool_name="", elapsed=0.0, **kwargs):
                called.append((tool_name, elapsed))

        composite = _CompositeRunHooks([H()])
        asyncio.run(composite.on_tool_end(agent=MagicMock(), tool_name="exec", elapsed=1.5))
        self.assertEqual(called, [("exec", 1.5)])

    def test_on_agent_transfer_dispatched(self):
        called = []

        class H(RunHooks):
            async def on_agent_transfer(self, from_agent, to_agent, **kwargs):
                called.append((from_agent, to_agent))

        composite = _CompositeRunHooks([H()])
        asyncio.run(composite.on_agent_transfer(from_agent="a1", to_agent="a2"))
        self.assertEqual(called, [("a1", "a2")])


class TestConversationArchiveHooksCapture(unittest.TestCase):
    """ConversationArchiveHooks reads run_input at on_agent_end time."""

    def test_captures_input_at_end(self):
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent_1"
        agent.run_input = "Hello, how are you?"
        agent.run_id = "run-1"
        agent.workspace = MagicMock()
        agent.workspace.archive_conversation = AsyncMock(return_value="/path")

        asyncio.run(hooks.on_agent_end(agent=agent, output="response"))
        call_args = agent.workspace.archive_conversation.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        self.assertEqual(user_msg["content"], "Hello, how are you?")

    def test_non_string_input_skipped(self):
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent_1"
        agent.run_input = 42  # not a string
        agent.run_id = "run-1"
        agent.workspace = MagicMock()
        agent.workspace.archive_conversation = AsyncMock(return_value="/path")

        asyncio.run(hooks.on_agent_end(agent=agent, output="response"))
        # Only assistant message should be archived (no user message from non-string input)
        call_args = agent.workspace.archive_conversation.call_args
        messages = call_args[0][0]
        user_msgs = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 0)

    def test_end_with_no_workspace_is_noop(self):
        hooks = ConversationArchiveHooks()
        agent = MagicMock()
        agent.agent_id = "agent_1"
        agent.workspace = None
        # Should not raise
        asyncio.run(hooks.on_agent_end(agent=agent, output="done"))


class TestAgentHooksDefaults(unittest.TestCase):
    """AgentHooks default methods are noops."""

    def test_on_start_noop(self):
        hooks = AgentHooks()
        asyncio.run(hooks.on_start(agent=MagicMock()))

    def test_on_end_noop(self):
        hooks = AgentHooks()
        asyncio.run(hooks.on_end(agent=MagicMock(), output="done"))


if __name__ == "__main__":
    unittest.main()
