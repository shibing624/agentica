# -*- coding: utf-8 -*-
"""Tests for BuiltinTodoTool, BuiltinTaskTool, and Agent auto-wiring."""
import asyncio
import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from agentica.tools.builtin import BuiltinTaskTool, BuiltinTodoTool
from agentica.tools.builtin.task_state_tools import (
    BuiltinTodoTool as CanonicalBuiltinTodoTool,
)


def test_task_state_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinTodoTool is CanonicalBuiltinTodoTool


class TestBuiltinTodoTool:
    def test_write_todos_basic(self, todo_tool):
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
        ])
        assert result == "Todos updated (2 items: 1 in progress, 1 pending)."
        assert len(todo_tool.todos) == 2

    def test_write_todos_result_does_not_echo_the_list(self, todo_tool):
        """The model just sent this list; echoing it back is pure context cost,
        and at one update per finished step that cost repeats all session."""
        result = todo_tool.write_todos([
            {"content": "Review the model layer", "status": "completed"},
            {"content": "Review the runner", "status": "in_progress"},
            {"content": "Summarise findings", "status": "pending"},
        ])
        assert "Review the model layer" not in result
        assert len(result) < 120

    def test_write_todos_description_disambiguates_steps_from_tool_calls(self, todo_tool):
        """"3+ steps" alone reads as "3+ tool calls", which opens a todo list for
        almost every request. The description must draw that distinction and must
        not carry a bias toward calling the tool when unsure."""
        description = todo_tool.functions["write_todos"].description or ""

        assert "not 3 tool calls" in description
        assert "in_progress" in description
        assert "when in doubt" not in description.lower()

    def test_write_todos_invalid_status(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"content": "Bad", "status": "unknown"}])

    def test_write_todos_missing_content(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"status": "pending"}])

    def test_write_todos_none(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos(None)

    def test_write_todos_empty_list(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([])

    def test_write_todos_overwrites(self, todo_tool):
        """Writing new todos replaces old ones entirely."""
        todo_tool.write_todos([{"content": "Old", "status": "pending"}])
        todo_tool.write_todos([{"content": "New1", "status": "pending"}, {"content": "New2", "status": "pending"}])
        assert len(todo_tool.todos) == 2
        contents = [t["content"] for t in todo_tool.todos]
        assert "Old" not in contents
        assert "New1" in contents

    def test_set_agent_stores_on_agent(self):
        """When set_agent is called, todos are stored on agent.todos."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = []
        tool.set_agent(mock_agent)

        tool.write_todos([
            {"content": "Task X", "status": "pending"},
            {"content": "Task Y", "status": "in_progress"},
        ])
        # Todos should be stored on mock_agent.todos
        assert len(mock_agent.todos) == 2
        assert mock_agent.todos[0]["content"] == "Task X"
        assert mock_agent.todos[1]["content"] == "Task Y"

    def test_standalone_mode_uses_local_todos(self):
        """Without set_agent, todos are stored locally on the tool."""
        tool = BuiltinTodoTool()
        tool.write_todos([{"content": "Local task", "status": "pending"}])
        assert len(tool.todos) == 1
        assert tool.todos[0]["content"] == "Local task"
        # _agent should be None
        assert tool._agent is None

    def test_todos_property_reads_from_agent(self):
        """The todos property should read from agent when agent is set."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = [{"id": "1", "content": "Agent task", "status": "completed"}]
        tool.set_agent(mock_agent)
        assert tool.todos == mock_agent.todos

    # ---- Auto-clear tests (mirrors CC allDone logic) ----

    def test_auto_clear_when_all_completed(self, todo_tool):
        """All-completed todos should auto-clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        assert result == "All 2 todos completed; list cleared."
        assert len(todo_tool.todos) == 0

    def test_no_auto_clear_when_not_all_completed(self, todo_tool):
        """Partial completion should NOT clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "in_progress"},
        ])
        assert result == "Todos updated (2 items: 1 done, 1 in progress)."
        assert len(todo_tool.todos) == 2

    # ---- Verification nudge tests (mirrors CC structural nudge) ----

    def test_verification_nudge_3plus_all_completed_no_verify(self, todo_tool):
        """3+ all-completed tasks with no verification keyword -> nudge fires."""
        result = todo_tool.write_todos([
            {"content": "Implement feature A", "status": "completed"},
            {"content": "Implement feature B", "status": "completed"},
            {"content": "Implement feature C", "status": "completed"},
        ])
        assert "NOTE:" in result

    def test_no_nudge_when_less_than_3_tasks(self, todo_tool):
        """< 3 tasks all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_not_all_completed(self, todo_tool):
        """3+ tasks but not all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
            {"content": "Task C", "status": "in_progress"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_verification_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'verify' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Verify implementation", "status": "completed"},
            {"content": "Deploy to staging", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_test_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'test' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Write unit tests", "status": "completed"},
            {"content": "Update docs", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_lint_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'lint' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Refactor module", "status": "completed"},
            {"content": "Run linting", "status": "completed"},
            {"content": "Deploy", "status": "completed"},
        ])
        assert "NOTE:" not in result

    # ---- _needs_verification_nudge static method tests ----

    def test_needs_verification_nudge_static(self):
        """Direct test of the static nudge detection method."""
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is True

        # Has 'check' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Check results", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'review' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Code review", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'validate' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Validate output", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

    # ---- Tool result message format tests ----

    def test_tool_result_message_is_neutral(self, todo_tool):
        """Tool result message confirms the update without nudging re-calls."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
        ])
        assert result == "Todos updated (1 items: 1 pending)."


# ===========================================================================
# BuiltinTaskTool tests (requires mocking LLM / Agent)
# ===========================================================================

class TestBuiltinTaskTool:
    """``BuiltinTaskTool`` is a thin LLM-facing adapter around
    ``SubagentRegistry.spawn``. Tests focus on the adapter contract; the
    runtime behavior of ``spawn`` itself is covered by ``test_subagent.py``.
    """

    def test_task_without_parent_returns_error(self):
        """Unbound tool (no parent agent) cannot spawn anything."""
        tool = BuiltinTaskTool()
        result = asyncio.run(tool.task("do something"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "not bound" in parsed["error"]

    def test_task_forwards_to_spawn_and_serializes_completed(self):
        """Adapter calls ``SubagentRegistry().spawn`` and JSON-serializes the result."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        spawn_result = {
            "status": "completed",
            "agent_type": "code",
            "subagent_name": "Code Agent",
            "content": "answer is 42",
            "tool_calls_summary": [{"name": "read_file", "info": "x.py"}],
            "tool_count": 1,
            "execution_time": 0.123,
            "run_id": "abc",
        }

        async def fake_spawn(self, **kwargs):
            assert kwargs["task"] == "compute 6 * 7"
            assert kwargs["agent_type"] == "code"
            return spawn_result

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("compute 6 * 7", subagent_type="code"))

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["subagent_type"] == "code"
        assert parsed["subagent_name"] == "Code Agent"
        assert parsed["result"] == "answer is 42"
        assert parsed["tool_count"] == 1
        assert parsed["execution_time"] == 0.123

    def test_task_serializes_error_result(self):
        """Adapter surfaces spawn errors through the LLM-facing JSON envelope."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        async def fake_spawn(self, **kwargs):
            return {
                "status": "error",
                "error": "Subagent timed out after 5 seconds",
                "agent_type": "code",
                "content": "",
            }

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("slow"))

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "timed out" in parsed["error"]
        assert parsed["subagent_type"] == "code"

    def test_format_tool_brief_read_file(self):
        brief = BuiltinTaskTool._format_tool_brief("read_file", {"file_path": "/a/b/c.py"})
        assert "c.py" in brief

    def test_format_tool_brief_grep(self):
        brief = BuiltinTaskTool._format_tool_brief("grep", {"pattern": "hello"}, "found 3 matches")
        assert "hello" in brief

    def test_format_tool_brief_execute(self):
        brief = BuiltinTaskTool._format_tool_brief("execute", {"command": "ls -la /tmp"})
        assert "ls -la" in brief

    def test_format_tool_brief_default(self):
        brief = BuiltinTaskTool._format_tool_brief("unknown_tool", {"key": "value"})
        assert "key=" in brief

    def test_set_parent_agent(self):
        tool = BuiltinTaskTool()
        mock_agent = MagicMock()
        tool.set_parent_agent(mock_agent)
        assert tool._parent_agent is mock_agent

    def test_task_declares_own_timeout_management(self):
        tool = BuiltinTaskTool()
        assert tool.functions["task"].manages_own_timeout is True

    def test_task_is_registered_as_a_parallelizable_read_only_tool(self):
        # The executor only gathers ``concurrency_safe`` calls; without the flag
        # a batch of subagents runs one after another.
        tool = BuiltinTaskTool()
        assert tool.functions["task"].concurrency_safe is True
        assert tool.functions["task"].is_read_only is True

    def test_task_passes_auxiliary_model_to_spawn(self):
        """When ``auxiliary_model`` is set, the adapter forwards it to spawn as
        the cheap-tier model (main-tier types ignore it)."""
        custom_model = MagicMock()
        tool = BuiltinTaskTool(auxiliary_model=custom_model)
        tool.set_parent_agent(MagicMock())

        captured: Dict[str, Any] = {}

        async def fake_spawn(self, **kwargs):
            captured.update(kwargs)
            return {"status": "completed", "agent_type": "code", "content": "ok",
                    "tool_calls_summary": [], "tool_count": 0, "execution_time": 0}

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            asyncio.run(tool.task("test", subagent_type="code"))

        assert captured["auxiliary_model_override"] is custom_model


# ===========================================================================
# Agent auto-wire tests (Agent.__init__ wires TodoTool / TaskTool)
# ===========================================================================

class TestAgentAutoWire:
    """Agent.__init__ clones stateful tools per-agent (so the user's original
    instance is never overwritten when the same logical tool is reused across
    multiple agents) and wires the per-agent clone to ``self``."""

    def test_agent_wires_todo_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTodoTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        todo_tool = BuiltinTodoTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[todo_tool],
        )
        # User's original tool is left untouched (isolation contract)
        assert todo_tool._agent is None
        # Agent owns its own clone, wired to itself
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        assert wired is not todo_tool
        assert wired._agent is agent

    def test_agent_wires_task_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTaskTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        task_tool = BuiltinTaskTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[task_tool],
        )
        assert task_tool._parent_agent is None
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTaskTool))
        assert wired is not task_tool
        assert wired._parent_agent is agent

    def test_todo_tool_stores_on_agent(self):
        """After wiring, write_todos on the agent's clone stores todos on agent.todos."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[BuiltinTodoTool()],
        )
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        wired.write_todos([
            {"content": "Test task", "status": "pending"},
        ])
        assert len(agent.todos) == 1
        assert agent.todos[0]["content"] == "Test task"
