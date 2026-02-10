# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Memory system.
"""
import sys
import unittest
from unittest.mock import Mock, patch
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.memory import (
    AgentMemory,
    Memory,
    MemoryManager,
    AgentRun,
    SessionSummary,
    WorkflowMemory,
    WorkflowRun,
)
from agentica.memory.agent_memory import (
    _clean_message_for_history,
    _is_conversation_message,
    _truncate_tool_content,
)
from agentica.model.message import Message
from agentica.run_response import RunResponse


class TestMemory(unittest.TestCase):
    """Test cases for Memory model."""

    def test_memory_creation(self):
        """Test Memory creation."""
        memory = Memory(memory="User likes Python")
        self.assertEqual(memory.memory, "User likes Python")
        self.assertIsNone(memory.input_text)

    def test_memory_with_input_text(self):
        """Test Memory with input_text."""
        memory = Memory(memory="User likes Python", input_text="What do you like?")
        self.assertEqual(memory.memory, "User likes Python")
        self.assertEqual(memory.input_text, "What do you like?")

    def test_memory_to_dict(self):
        """Test Memory to_dict method."""
        memory = Memory(memory="Test memory")
        result = memory.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["memory"], "Test memory")

    def test_memory_to_str(self):
        """Test Memory to_str method."""
        memory = Memory(memory="Test memory")
        result = memory.to_str()
        self.assertIsInstance(result, str)
        self.assertIn("memory", result)


class TestSessionSummary(unittest.TestCase):
    """Test cases for SessionSummary model."""

    def test_session_summary_creation(self):
        """Test SessionSummary creation."""
        summary = SessionSummary(summary="User discussed Python programming")
        self.assertEqual(summary.summary, "User discussed Python programming")
        self.assertIsNone(summary.topics)

    def test_session_summary_with_topics(self):
        """Test SessionSummary with topics."""
        summary = SessionSummary(
            summary="User discussed programming",
            topics=["Python", "AI", "Machine Learning"]
        )
        self.assertEqual(len(summary.topics), 3)
        self.assertIn("Python", summary.topics)

    def test_session_summary_to_dict(self):
        """Test SessionSummary to_dict method."""
        summary = SessionSummary(summary="Test summary", topics=["topic1"])
        result = summary.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Test summary")

    def test_session_summary_to_json(self):
        """Test SessionSummary to_json method."""
        summary = SessionSummary(summary="Test summary")
        result = summary.to_json()
        self.assertIsInstance(result, str)
        self.assertIn("Test summary", result)


class TestAgentRun(unittest.TestCase):
    """Test cases for AgentRun model."""

    def test_agent_run_creation(self):
        """Test AgentRun creation."""
        run = AgentRun()
        self.assertIsNone(run.message)
        self.assertIsNone(run.messages)
        self.assertIsNone(run.response)

    def test_agent_run_with_message(self):
        """Test AgentRun with message."""
        message = Message(role="user", content="Hello")
        run = AgentRun(message=message)
        self.assertEqual(run.message.content, "Hello")

    def test_agent_run_with_response(self):
        """Test AgentRun with response."""
        response = RunResponse(content="Hi there!")
        run = AgentRun(response=response)
        self.assertEqual(run.response.content, "Hi there!")


class TestAgentMemory(unittest.TestCase):
    """Test cases for AgentMemory class."""

    def test_default_initialization(self):
        """Test AgentMemory default initialization."""
        memory = AgentMemory()
        self.assertEqual(len(memory.runs), 0)
        self.assertEqual(len(memory.messages), 0)
        self.assertFalse(memory.create_session_summary)
        self.assertFalse(memory.create_user_memories)

    def test_add_message(self):
        """Test adding message to AgentMemory."""
        memory = AgentMemory()
        message = Message(role="user", content="Hello")
        memory.add_message(message)
        self.assertEqual(len(memory.messages), 1)
        self.assertEqual(memory.messages[0].content, "Hello")

    def test_add_run(self):
        """Test adding run to AgentMemory."""
        memory = AgentMemory()
        run = AgentRun(
            message=Message(role="user", content="Hello"),
            response=RunResponse(content="Hi!")
        )
        memory.add_run(run)
        self.assertEqual(len(memory.runs), 1)

    def test_get_messages_from_last_n_runs(self):
        """Test getting messages from last n runs."""
        memory = AgentMemory()

        # Add some runs
        for i in range(5):
            run = AgentRun(
                message=Message(role="user", content=f"Message {i}"),
                messages=[Message(role="user", content=f"Message {i}")],
                response=RunResponse(content=f"Response {i}")
            )
            memory.add_run(run)

        messages = memory.get_messages_from_last_n_runs(last_n=2)
        self.assertIsInstance(messages, list)

    def test_clear_memory(self):
        """Test clearing AgentMemory."""
        memory = AgentMemory()
        memory.add_message(Message(role="user", content="Hello"))
        memory.clear()
        self.assertEqual(len(memory.messages), 0)
        self.assertEqual(len(memory.runs), 0)


class TestAgentMemoryWithDb(unittest.TestCase):
    """Test cases for AgentMemory with database."""

    def test_memory_db_attribute(self):
        """Test AgentMemory db attribute can be set."""
        memory = AgentMemory()
        self.assertIsNone(memory.db)


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager class."""

    def test_default_initialization(self):
        """Test MemoryManager default initialization."""
        manager = MemoryManager()
        self.assertEqual(manager.mode, "rule")
        self.assertIsNone(manager.model)
        self.assertIsNone(manager.db)

    def test_mode_setting(self):
        """Test MemoryManager mode setting."""
        manager = MemoryManager(mode="model")
        self.assertEqual(manager.mode, "model")


class TestWorkflowMemory(unittest.TestCase):
    """Test cases for WorkflowMemory class."""

    def test_default_initialization(self):
        """Test WorkflowMemory default initialization."""
        memory = WorkflowMemory()
        self.assertEqual(len(memory.runs), 0)

    def test_add_run(self):
        """Test adding run to WorkflowMemory."""
        memory = WorkflowMemory()
        run = WorkflowRun(
            input={"message": "Hello"},
            response=RunResponse(content="Hi!")
        )
        memory.runs.append(run)
        self.assertEqual(len(memory.runs), 1)


class TestWorkflowRun(unittest.TestCase):
    """Test cases for WorkflowRun model."""

    def test_workflow_run_creation(self):
        """Test WorkflowRun creation."""
        run = WorkflowRun()
        self.assertIsNone(run.input)
        self.assertIsNone(run.response)

    def test_workflow_run_with_data(self):
        """Test WorkflowRun with data."""
        run = WorkflowRun(
            input={"key": "value"},
            response=RunResponse(content="Result")
        )
        self.assertEqual(run.input["key"], "value")
        self.assertEqual(run.response.content, "Result")


class TestHelperFunctions(unittest.TestCase):
    """Test cases for module-level helper functions."""

    def test_is_conversation_message_user(self):
        """Test _is_conversation_message keeps user messages."""
        msg = Message(role="user", content="Hello")
        self.assertTrue(_is_conversation_message(msg))

    def test_is_conversation_message_assistant(self):
        """Test _is_conversation_message keeps assistant messages."""
        msg = Message(role="assistant", content="Hi there")
        self.assertTrue(_is_conversation_message(msg))

    def test_is_conversation_message_tool(self):
        """Test _is_conversation_message filters tool messages."""
        msg = Message(role="tool", content="result")
        self.assertFalse(_is_conversation_message(msg))

    def test_is_conversation_message_assistant_with_tool_calls(self):
        """Test _is_conversation_message filters assistant with tool_calls."""
        msg = Message(role="assistant", content="calling", tool_calls=[{"id": "1"}])
        self.assertFalse(_is_conversation_message(msg))

    def test_truncate_tool_content_short(self):
        """Test _truncate_tool_content keeps short content unchanged."""
        msg = Message(role="assistant", content="short text")
        result = _truncate_tool_content(msg, max_chars=500)
        self.assertEqual(result.content, "short text")

    def test_truncate_tool_content_long(self):
        """Test _truncate_tool_content truncates long content."""
        long_text = "a" * 1000
        msg = Message(role="assistant", content=long_text)
        result = _truncate_tool_content(msg, max_chars=200)
        self.assertIn("truncated", result.content)
        self.assertLess(len(result.content), len(long_text))

    def test_truncate_tool_content_preserves_head_tail(self):
        """Test _truncate_tool_content keeps head and tail of content."""
        content = "HEAD_" + "x" * 1000 + "_TAIL"
        msg = Message(role="assistant", content=content)
        result = _truncate_tool_content(msg, max_chars=200)
        self.assertTrue(result.content.startswith("HEAD_"))
        self.assertTrue(result.content.endswith("_TAIL"))

    def test_truncate_tool_content_non_string(self):
        """Test _truncate_tool_content handles non-string content."""
        msg = Message(role="assistant", content=[{"type": "text", "text": "data"}])
        result = _truncate_tool_content(msg, max_chars=10)
        self.assertEqual(result.content, msg.content)


class TestGetMessagesTokenBudget(unittest.TestCase):
    """Test token budget and truncation in get_messages_from_last_n_runs."""

    def _make_run(self, user_msg: str, assistant_msg: str) -> AgentRun:
        """Helper to create an AgentRun with user + assistant messages."""
        return AgentRun(
            message=Message(role="user", content=user_msg),
            response=RunResponse(
                content=assistant_msg,
                messages=[
                    Message(role="user", content=user_msg),
                    Message(role="assistant", content=assistant_msg),
                ]
            )
        )

    def test_basic_last_n(self):
        """Test basic last_n still works as before."""
        memory = AgentMemory()
        for i in range(5):
            memory.add_run(self._make_run(f"Q{i}", f"A{i}"))

        msgs = memory.get_messages_from_last_n_runs(last_n=2)
        # Should get messages from last 2 runs (each has user + assistant)
        self.assertGreater(len(msgs), 0)
        contents = [m.content for m in msgs]
        self.assertIn("Q3", contents)
        self.assertIn("A4", contents)

    def test_no_last_n_returns_all(self):
        """Test None last_n returns all runs."""
        memory = AgentMemory()
        for i in range(3):
            memory.add_run(self._make_run(f"Q{i}", f"A{i}"))

        msgs = memory.get_messages_from_last_n_runs()
        contents = [m.content for m in msgs]
        self.assertIn("Q0", contents)
        self.assertIn("A2", contents)

    def test_max_tokens_limits_history(self):
        """Test max_tokens limits the number of history messages."""
        memory = AgentMemory()
        # Each message has ~10 tokens, so 10 runs * 2 messages * ~10 tokens = ~200
        for i in range(10):
            memory.add_run(self._make_run(f"Question {i} about topic", f"Answer {i} about topic"))

        # Very small token budget should return fewer messages
        msgs_limited = memory.get_messages_from_last_n_runs(max_tokens=50)
        msgs_unlimited = memory.get_messages_from_last_n_runs()
        self.assertLessEqual(len(msgs_limited), len(msgs_unlimited))

    def test_max_tokens_always_includes_newest(self):
        """Test max_tokens always includes at least the newest run."""
        memory = AgentMemory()
        for i in range(5):
            memory.add_run(self._make_run(f"Q{i}", f"A{i}"))

        # Even with tiny budget, should get at least the newest run
        msgs = memory.get_messages_from_last_n_runs(max_tokens=1)
        self.assertGreater(len(msgs), 0)

    def test_truncate_tool_results_in_older_runs(self):
        """Test that older runs get tool result truncation."""
        memory = AgentMemory()
        # First run with long content
        long_answer = "detailed " * 200
        memory.add_run(self._make_run("Q0", long_answer))
        # Recent run with short content
        memory.add_run(self._make_run("Q1", "short"))

        msgs = memory.get_messages_from_last_n_runs(
            truncate_tool_results=True,
            tool_result_max_chars=50,
        )
        # The older run's long assistant message should be truncated
        for m in msgs:
            if "detailed" in str(m.content):
                self.assertIn("truncated", m.content)

    def test_skip_role(self):
        """Test skip_role parameter still works."""
        memory = AgentMemory()
        memory.add_run(self._make_run("Q0", "A0"))

        msgs = memory.get_messages_from_last_n_runs(skip_role="user")
        roles = [m.role for m in msgs]
        self.assertNotIn("user", roles)

    def test_empty_runs(self):
        """Test with no runs returns empty list."""
        memory = AgentMemory()
        msgs = memory.get_messages_from_last_n_runs(last_n=5, max_tokens=1000)
        self.assertEqual(len(msgs), 0)


if __name__ == "__main__":
    unittest.main()
