# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Workflow class.
"""
import sys
import unittest
from unittest.mock import Mock, patch
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.workflow import Workflow
from agentica.memory import WorkflowMemory
from agentica.run_response import RunResponse


class SimpleWorkflow(Workflow):
    """A simple workflow for testing."""

    def run(self, message: str) -> RunResponse:
        """Simple run implementation."""
        return RunResponse(content=f"Processed: {message}")


class TestWorkflowInitialization(unittest.TestCase):
    """Test cases for Workflow initialization."""

    def test_workflow_has_name(self):
        """Test Workflow name attribute."""
        workflow = SimpleWorkflow(name="TestWorkflow")
        self.assertEqual(workflow.name, "TestWorkflow")

    def test_workflow_has_description(self):
        """Test Workflow description attribute."""
        workflow = SimpleWorkflow(description="A test workflow")
        self.assertEqual(workflow.description, "A test workflow")

    def test_workflow_has_workflow_id(self):
        """Test Workflow has workflow_id."""
        workflow = SimpleWorkflow()
        self.assertIsNotNone(workflow.workflow_id)

    def test_workflow_has_session_id(self):
        """Test Workflow has session_id."""
        workflow = SimpleWorkflow()
        self.assertIsNotNone(workflow.session_id)

    def test_workflow_custom_ids(self):
        """Test Workflow with custom IDs."""
        workflow = SimpleWorkflow(
            workflow_id="wf-123",
            session_id="sess-456"
        )
        self.assertEqual(workflow.workflow_id, "wf-123")
        self.assertEqual(workflow.session_id, "sess-456")

    def test_workflow_user_id(self):
        """Test Workflow with user_id."""
        workflow = SimpleWorkflow(user_id="user-789")
        self.assertEqual(workflow.user_id, "user-789")


class TestWorkflowMemory(unittest.TestCase):
    """Test cases for Workflow memory management."""

    def test_default_memory(self):
        """Test Workflow with default memory."""
        workflow = SimpleWorkflow()
        self.assertIsInstance(workflow.memory, WorkflowMemory)
        self.assertEqual(len(workflow.memory.runs), 0)

    def test_custom_memory(self):
        """Test Workflow with custom memory."""
        memory = WorkflowMemory()
        workflow = SimpleWorkflow(memory=memory)
        self.assertIsInstance(workflow.memory, WorkflowMemory)


class TestWorkflowRun(unittest.TestCase):
    """Test cases for Workflow run method."""

    def test_simple_run(self):
        """Test simple workflow run."""
        workflow = SimpleWorkflow()
        response = workflow.run("Hello")
        self.assertIsInstance(response, RunResponse)
        self.assertEqual(response.content, "Processed: Hello")

    def test_run_workflow_wrapper(self):
        """Test run_workflow wrapper method."""
        workflow = SimpleWorkflow()
        response = workflow.run_workflow("Test message")
        self.assertIsInstance(response, RunResponse)


class TestWorkflowSessionState(unittest.TestCase):
    """Test cases for Workflow session state."""

    def test_default_session_state(self):
        """Test default session state is empty dict."""
        workflow = SimpleWorkflow()
        self.assertEqual(workflow.session_state, {})

    def test_session_state_modification(self):
        """Test session state can be modified."""
        workflow = SimpleWorkflow()
        workflow.session_state["key"] = "value"
        self.assertEqual(workflow.session_state["key"], "value")


class TestWorkflowDeepCopy(unittest.TestCase):
    """Test cases for Workflow deep copy functionality."""

    def test_deep_copy_basic(self):
        """Test basic deep copy of Workflow."""
        workflow = SimpleWorkflow(name="Original", user_id="user-1")
        copied = workflow.deep_copy()

        self.assertEqual(copied.name, "Original")
        self.assertEqual(copied.user_id, "user-1")
        self.assertIsNot(workflow, copied)

    def test_deep_copy_with_update(self):
        """Test deep copy with updates."""
        workflow = SimpleWorkflow(name="Original")
        copied = workflow.deep_copy(update={"name": "Copied"})

        self.assertEqual(workflow.name, "Original")
        self.assertEqual(copied.name, "Copied")


class TestWorkflowDebugMode(unittest.TestCase):
    """Test cases for Workflow debug mode."""

    def test_debug_mode_default(self):
        """Test debug_mode is False by default."""
        workflow = SimpleWorkflow()
        self.assertFalse(workflow.debug_mode)

    def test_debug_mode_enabled(self):
        """Test debug_mode can be enabled."""
        workflow = SimpleWorkflow(debug_mode=True)
        self.assertTrue(workflow.debug_mode)


class TestWorkflowWithAgents(unittest.TestCase):
    """Test cases for Workflow with Agent integration."""

    def test_workflow_with_mock_agent(self):
        """Test Workflow can work with mock agents."""
        from agentica.agent import Agent

        class AgentWorkflow(Workflow):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._agent = Agent(name="TestAgent")

            def run(self, message: str) -> RunResponse:
                return RunResponse(content=f"Agent: {self._agent.name}, Message: {message}")

        workflow = AgentWorkflow(name="AgentWorkflow")
        response = workflow.run("Hello")
        self.assertIn("TestAgent", response.content)


if __name__ == "__main__":
    unittest.main()
