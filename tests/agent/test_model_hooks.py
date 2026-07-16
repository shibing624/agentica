# -*- coding: utf-8 -*-
"""
Tests for Model-layer pre/post tool hooks:
- _build_pre_tool_hook() injection via update_model()
- Context overflow handling (context_overflow_threshold) — compress then evict
- Fast path: feature disabled → hook is None
All tests mock LLM API keys — no real API calls.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from agentica.agent.config import ToolConfig
from agentica.model.message import Message


def _make_agent(tool_config=None):
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tool_config=tool_config or ToolConfig(),
    )


class TestHookInjection(unittest.TestCase):
    """_build_pre_tool_hook() must return None or callable based on ToolConfig."""

    def test_disabled_hook_is_none(self):
        agent = _make_agent(ToolConfig())
        self.assertIsNone(agent._build_pre_tool_hook())

    def test_overflow_enabled_hook_is_set(self):
        agent = _make_agent(ToolConfig(context_overflow_threshold=0.8))
        self.assertIsNotNone(agent._build_pre_tool_hook())

    def test_hook_cleared_on_subsequent_config_change(self):
        """Switching to disabled config makes _build_pre_tool_hook() return None."""
        agent = _make_agent(ToolConfig(context_overflow_threshold=0.8))
        self.assertIsNotNone(agent._build_pre_tool_hook())
        # Switch to disabled
        agent.tool_config = ToolConfig()
        self.assertIsNone(agent._build_pre_tool_hook())


class TestContextOverflowHandling(unittest.TestCase):
    """_build_pre_tool_hook must handle context overflow without dropping system msg."""

    def _make_hook(self, threshold=0.5, window=200):
        agent = _make_agent(ToolConfig(context_overflow_threshold=threshold))
        agent.update_model()
        agent.model.context_window = window
        hook = agent._build_pre_tool_hook()
        return hook

    def test_overflow_evicts_oldest_non_system_message(self):
        hook = self._make_hook(threshold=0.5, window=200)
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="A" * 150),
            Message(role="assistant", content="B" * 150),
            Message(role="user", content="C" * 150),
        ]
        result = asyncio.run(hook(messages, []))
        self.assertFalse(result, "Overflow handling does not skip tool batch")
        self.assertLess(len(messages), 4, "At least one message should be dropped")

    def test_system_message_always_preserved(self):
        hook = self._make_hook(threshold=0.1, window=50)
        messages = [
            Message(role="system", content="System prompt."),
            Message(role="user", content="X" * 200),
        ]
        asyncio.run(hook(messages, []))
        self.assertEqual(messages[0].role, "system")

    def test_no_overflow_no_change(self):
        hook = self._make_hook(threshold=0.9, window=100000)
        messages = [
            Message(role="system", content="System."),
            Message(role="user", content="short message"),
        ]
        original_len = len(messages)
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), original_len)

    def test_overflow_returns_false_not_true(self):
        """Context overflow handling does NOT skip the tool batch."""
        hook = self._make_hook(threshold=0.1, window=10)
        messages = [
            Message(role="system", content="Sys"),
            Message(role="user", content="A" * 50),
        ]
        result = asyncio.run(hook(messages, []))
        self.assertFalse(result)

    def test_compression_tried_before_eviction(self):
        """When a compression_manager is wired, compress() must run before FIFO evict.

        If compression alone brings usage below the hard limit, no messages are
        evicted. This preserves information when possible.
        """
        agent = _make_agent(ToolConfig(context_overflow_threshold=0.5))
        agent.update_model()
        agent.model.context_window = 200

        # Wire a mock compression manager that "compresses" by replacing content
        # with a short stub — dropping total chars below the hard limit.
        async def fake_compress(msgs, tools=None, model=None, response_format=None, **_kw):
            for m in msgs:
                if m.role != "system":
                    m.content = "x"
        cm = MagicMock()
        cm.compress = AsyncMock(side_effect=fake_compress)
        agent.tool_config.compression_manager = cm

        hook = agent._build_pre_tool_hook()
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="A" * 200),
            Message(role="assistant", content="B" * 200),
            Message(role="user", content="C" * 200),
        ]
        n_before = len(messages)
        result = asyncio.run(hook(messages, []))
        self.assertFalse(result)
        cm.compress.assert_awaited_once()
        # With effective compression, no eviction should occur.
        self.assertEqual(len(messages), n_before, "Compression alone should suffice")

    def test_overflow_eviction_appends_context_maintenance_notice(self):
        """FIFO eviction must make file-read invalidation visible to the model.

        Regression for the silent-stale path: the overflow handler used to
        mark reads stale without appending any notice, leaving the model to
        discover staleness only when an edit was rejected.
        """
        import os
        import tempfile
        from pathlib import Path
        from agentica.tools.buildin_tools import BuiltinFileTool

        agent = _make_agent(ToolConfig(context_overflow_threshold=0.5))
        agent.update_model()
        agent.model.context_window = 200

        file_tool = BuiltinFileTool()
        agent.tools = [file_tool]
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "read_then_evicted.py")
            Path(fp).write_text("value = 1\n", encoding="utf-8")
            asyncio.run(file_tool.read_file(fp))
            # Sanity: the read is recorded as context-fresh.
            self.assertTrue(
                file_tool._file_read_state[str(Path(fp).resolve())].context_available
            )

            hook = agent._build_pre_tool_hook()
            messages = [
                Message(role="system", content="sys"),
                Message(role="user", content="A" * 200),
                Message(role="assistant", content="B" * 200),
                Message(role="user", content="C" * 200),
            ]
            asyncio.run(hook(messages, []))

            # The read was marked stale and a [Context maintenance] notice appended.
            self.assertFalse(
                file_tool._file_read_state[str(Path(fp).resolve())].context_available
            )
            notices = [
                m for m in messages
                if m.role == "user" and "[Context maintenance]" in str(m.content)
            ]
            self.assertEqual(len(notices), 1, "Expected one context-maintenance notice")
            self.assertIn(str(Path(fp).resolve()), notices[0].content)


class TestPostToolHook(unittest.TestCase):
    """_build_post_tool_hook: None when no TodoTool, set when TodoTool is present."""

    def test_post_tool_hook_is_none_without_todo_tool(self):
        """Without BuiltinTodoTool, _build_post_tool_hook should return None."""
        agent = _make_agent()
        self.assertIsNone(agent._build_post_tool_hook())

    def test_post_tool_hook_is_set_with_todo_tool(self):
        """With BuiltinTodoTool, _build_post_tool_hook should return an async callable."""
        agent = _make_agent_with_todo_tool()
        self.assertIsNotNone(agent._build_post_tool_hook())

    def test_post_tool_hook_is_none_when_reminder_disabled(self):
        """With todo_reminder_interval=0, _build_post_tool_hook should return None."""
        from agentica.tools.buildin_tools import BuiltinTodoTool
        from agentica.agent.config import PromptConfig
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[BuiltinTodoTool()],
            prompt_config=PromptConfig(todo_reminder_interval=0),
        )
        self.assertIsNone(agent._build_post_tool_hook())


class TestTodoReminder(unittest.TestCase):
    """Integration tests for _post_tool_hook todo reminder injection."""

    def _make_agent_with_todos(self, interval=3, todos=None):
        agent = _make_agent_with_todo_tool(reminder_interval=interval)
        agent.update_model()
        if todos:
            agent.todos = todos
        return agent

    def test_no_reminder_when_write_todos_recent(self):
        """If write_todos was called recently, no reminder should be injected."""
        agent = self._make_agent_with_todos(
            interval=3,
            todos=[{"content": "Task A", "status": "pending"}],
        )
        hook = agent._build_post_tool_hook()
        # Simulate: write_todos was 1 assistant turn ago
        messages = [
            Message(role="user", content="do task"),
            Message(role="assistant", content="ok"),
            Message(role="tool", tool_name="write_todos", content="{}"),
            Message(role="assistant", content="updated todos"),
        ]
        original_len = len(messages)
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), original_len, "No reminder should be injected")

    def test_reminder_injected_after_enough_turns(self):
        """After enough assistant turns without write_todos, reminder is injected."""
        agent = self._make_agent_with_todos(
            interval=2,
            todos=[{"content": "Task A", "status": "in_progress"}],
        )
        hook = agent._build_post_tool_hook()
        # Simulate: no write_todos call, 3 assistant turns
        messages = [
            Message(role="user", content="do stuff"),
            Message(role="assistant", content="doing 1"),
            Message(role="user", content="next"),
            Message(role="assistant", content="doing 2"),
            Message(role="user", content="next"),
            Message(role="assistant", content="doing 3"),
        ]
        asyncio.run(hook(messages, []))
        # Should have injected a reminder
        self.assertEqual(len(messages), 7)
        injected = messages[-1]
        self.assertEqual(injected.role, "user")
        self.assertIn("[Todo Reminder]", injected.content)
        self.assertIn("Task A", injected.content)

    def test_no_reminder_when_todos_empty(self):
        """No reminder when there are no active todos."""
        agent = self._make_agent_with_todos(interval=1, todos=[])
        hook = agent._build_post_tool_hook()
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="assistant", content="hello2"),
        ]
        original_len = len(messages)
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), original_len)

    def test_no_double_reminder(self):
        """Should not inject reminder if a recent reminder already exists within interval."""
        agent = self._make_agent_with_todos(
            interval=3,
            todos=[{"content": "Task A", "status": "pending"}],
        )
        hook = agent._build_post_tool_hook()
        # Simulate: reminder was injected, then only 2 assistant turns (< interval=3)
        messages = [
            Message(role="user", content="[Todo Reminder] ..."),
            Message(role="assistant", content="ok noted"),
            Message(role="assistant", content="working..."),
        ]
        original_len = len(messages)
        asyncio.run(hook(messages, []))
        self.assertEqual(len(messages), original_len, "No double reminder")


def _make_agent_with_todo_tool(reminder_interval=10):
    """Helper: create agent with BuiltinTodoTool registered."""
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    from agentica.tools.buildin_tools import BuiltinTodoTool
    from agentica.agent.config import PromptConfig
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tools=[BuiltinTodoTool()],
        prompt_config=PromptConfig(todo_reminder_interval=reminder_interval),
    )


if __name__ == "__main__":
    unittest.main()
