# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Agent core class.
"""
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.agent import Agent
from agentica.memory import AgentMemory, Memory
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.run_response import RunResponse, RunEvent
from agentica.tools.base import Tool, Function


class TestAgentInitialization(unittest.TestCase):
    """Test cases for Agent initialization."""

    def test_default_initialization(self):
        """Test Agent with default parameters."""
        agent = Agent()
        self.assertIsNone(agent.model)
        self.assertIsNone(agent.name)
        self.assertIsNotNone(agent.agent_id)
        self.assertIsInstance(agent.memory, AgentMemory)
        self.assertEqual(agent.session_state, {})

    def test_initialization_with_name(self):
        """Test Agent with custom name."""
        agent = Agent(name="TestAgent")
        self.assertEqual(agent.name, "TestAgent")

    def test_initialization_with_custom_id(self):
        """Test Agent with custom agent_id."""
        agent = Agent(agent_id="custom-id-123")
        self.assertEqual(agent.agent_id, "custom-id-123")

    def test_initialization_with_user_id(self):
        """Test Agent with user_id."""
        agent = Agent(user_id="user-123")
        self.assertEqual(agent.user_id, "user-123")

    def test_initialization_with_session_id(self):
        """Test Agent with session_id."""
        agent = Agent(session_id="session-456")
        self.assertEqual(agent.session_id, "session-456")

    def test_initialization_with_instructions(self):
        """Test Agent with instructions."""
        instructions = ["Be helpful", "Be concise"]
        agent = Agent(instructions=instructions)
        self.assertEqual(agent.instructions, instructions)

    def test_initialization_with_description(self):
        """Test Agent with description."""
        agent = Agent(description="A test agent")
        self.assertEqual(agent.description, "A test agent")


class TestAgentTools(unittest.TestCase):
    """Test cases for Agent tool management."""

    def test_agent_with_no_tools(self):
        """Test Agent without tools."""
        agent = Agent()
        tools = agent.get_tools()
        self.assertEqual(tools, [])

    def test_agent_with_function_tool(self):
        """Test Agent with a function as tool."""
        def my_tool(x: int) -> int:
            """A simple tool that doubles the input."""
            return x * 2

        agent = Agent(tools=[my_tool])
        tools = agent.get_tools()
        self.assertEqual(len(tools), 1)

    def test_agent_with_tool_class(self):
        """Test Agent with Tool class."""
        tool = Tool(name="test_tool")

        def sample_func(x: str) -> str:
            """Sample function."""
            return x.upper()

        tool.register(sample_func)
        agent = Agent(tools=[tool])
        tools = agent.get_tools()
        self.assertGreaterEqual(len(tools), 1)


class TestAgentMemory(unittest.TestCase):
    """Test cases for Agent memory management."""

    def test_default_memory(self):
        """Test Agent with default memory."""
        agent = Agent()
        self.assertIsInstance(agent.memory, AgentMemory)
        self.assertEqual(len(agent.memory.messages), 0)

    def test_custom_memory(self):
        """Test Agent with custom memory."""
        memory = AgentMemory()
        memory.add_message(Message(role="user", content="Hello"))
        agent = Agent(memory=memory)
        self.assertEqual(len(agent.memory.messages), 1)

    def test_add_history_to_messages(self):
        """Test add_history_to_messages setting."""
        agent = Agent(add_history_to_messages=True)
        self.assertTrue(agent.add_history_to_messages)

    def test_num_history_responses(self):
        """Test num_history_responses setting."""
        agent = Agent(num_history_responses=5)
        self.assertEqual(agent.num_history_responses, 5)


class TestAgentRun(unittest.TestCase):
    """Test cases for Agent run method with mocked model."""

    def setUp(self):
        """Set up mock model for tests."""
        self.mock_model = Mock()
        self.mock_model.id = "mock-model"
        self.mock_model.name = "Mock Model"
        self.mock_model.provider = "mock"
        self.mock_model.tools = None
        self.mock_model.functions = {}
        self.mock_model.function_call_stack = None
        self.mock_model.run_tools = True

    @patch('agentica.agent.Agent.update_model')
    def test_run_returns_run_response(self, mock_update):
        """Test that run returns a RunResponse."""
        # Create mock response (ModelResponse is a dataclass without 'id' field)
        mock_response = ModelResponse(
            content="Hello, I'm an AI assistant."
        )
        self.mock_model.response = Mock(return_value=mock_response)

        agent = Agent(model=self.mock_model)
        # Mock the internal _run method to return a simple response
        with patch.object(agent, '_run') as mock_run:
            mock_run.return_value = iter([RunResponse(content="Test response")])
            response = agent.run("Hello")
            self.assertIsInstance(response, RunResponse)


class TestAgentSystemPrompt(unittest.TestCase):
    """Test cases for Agent system prompt generation."""

    def test_system_prompt_string(self):
        """Test Agent with string system_prompt."""
        agent = Agent(system_prompt="You are a helpful assistant.")
        self.assertEqual(agent.system_prompt, "You are a helpful assistant.")

    def test_system_prompt_callable(self):
        """Test Agent with callable system_prompt."""
        def get_prompt():
            return "Dynamic prompt"

        agent = Agent(system_prompt=get_prompt)
        self.assertTrue(callable(agent.system_prompt))

    def test_instructions_list(self):
        """Test Agent with list of instructions."""
        instructions = ["Be helpful", "Be concise", "Be accurate"]
        agent = Agent(instructions=instructions)
        self.assertEqual(len(agent.instructions), 3)


class TestAgentDeepCopy(unittest.TestCase):
    """Test cases for Agent deep copy functionality."""

    def test_deep_copy_preserves_name(self):
        """Test deep copy preserves agent name."""
        agent = Agent(name="Original")
        copied = agent.deep_copy()
        self.assertEqual(copied.name, "Original")

    def test_deep_copy_preserves_user_id(self):
        """Test deep copy preserves user_id."""
        agent = Agent(user_id="user-1")
        copied = agent.deep_copy()
        self.assertEqual(copied.user_id, "user-1")

    def test_deep_copy_with_name_update(self):
        """Test deep copy with name update."""
        agent = Agent(name="Original")
        copied = agent.deep_copy(update={"name": "Copied"})
        self.assertEqual(copied.name, "Copied")


class TestAgentTeam(unittest.TestCase):
    """Test cases for Agent team functionality."""

    def test_agent_with_team(self):
        """Test Agent with team members."""
        member1 = Agent(name="Member1", role="researcher")
        member2 = Agent(name="Member2", role="writer")
        leader = Agent(name="Leader", team=[member1, member2])

        self.assertEqual(len(leader.team), 2)
        self.assertEqual(leader.team[0].name, "Member1")
        self.assertEqual(leader.team[1].name, "Member2")

    def test_agent_role(self):
        """Test Agent role setting."""
        agent = Agent(role="assistant")
        self.assertEqual(agent.role, "assistant")


class TestAgentMultiRound(unittest.TestCase):
    """Test cases for Agent multi-round conversation settings."""

    def test_enable_multi_round(self):
        """Test enable_multi_round setting."""
        agent = Agent(enable_multi_round=True)
        self.assertTrue(agent.enable_multi_round)

    def test_max_rounds(self):
        """Test max_rounds setting."""
        agent = Agent(max_rounds=50)
        self.assertEqual(agent.max_rounds, 50)

    def test_max_tokens(self):
        """Test max_tokens setting."""
        agent = Agent(max_tokens=64000)
        self.assertEqual(agent.max_tokens, 64000)


class TestAgentStructuredOutput(unittest.TestCase):
    """Test cases for Agent structured output settings."""

    def test_structured_outputs_disabled(self):
        """Test structured_outputs disabled by default."""
        agent = Agent()
        self.assertFalse(agent.structured_outputs)

    def test_structured_outputs_enabled(self):
        """Test structured_outputs enabled."""
        agent = Agent(structured_outputs=True)
        self.assertTrue(agent.structured_outputs)


if __name__ == "__main__":
    unittest.main()
