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


if __name__ == "__main__":
    unittest.main()
