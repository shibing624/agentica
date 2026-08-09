# -*- coding: utf-8 -*-
"""
Tests for the post-tool hook (todo reminder injection).

There is no pre-tool hook: context-overflow FIFO dropping was a third
compression layer with its own threshold and its own char/4 token estimate,
and it now lives nowhere — the Runner's evict → summarise pipeline is the
only thing that shrinks a request.

All tests mock LLM API keys — no real API calls.
"""
import asyncio
import unittest

from agentica.agent.config import ToolConfig
from agentica.model.message import Message


def _make_agent(tool_config=None):
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tool_config=tool_config or ToolConfig(),
    )


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
        from agentica.tools.builtin import BuiltinTodoTool
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
    from agentica.tools.builtin import BuiltinTodoTool
    from agentica.agent.config import PromptConfig
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tools=[BuiltinTodoTool()],
        prompt_config=PromptConfig(todo_reminder_interval=reminder_interval),
    )


if __name__ == "__main__":
    unittest.main()
