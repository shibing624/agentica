# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Agent core class.
"""
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.agent import Agent
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig
from agentica.memory import WorkingMemory
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
        self.assertIsInstance(agent.working_memory, WorkingMemory)

    def test_initialization_with_name(self):
        """Test Agent with custom name."""
        agent = Agent(name="TestAgent")
        self.assertEqual(agent.name, "TestAgent")

    def test_initialization_with_custom_id(self):
        """Test Agent with custom agent_id."""
        agent = Agent(agent_id="custom-id-123")
        self.assertEqual(agent.agent_id, "custom-id-123")

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


class TestWorkingMemory(unittest.TestCase):
    """Test cases for Agent memory management."""

    def test_default_memory(self):
        """Test Agent with default memory."""
        agent = Agent()
        self.assertIsInstance(agent.working_memory, WorkingMemory)
        self.assertEqual(len(agent.working_memory.messages), 0)

    def test_add_history_to_messages(self):
        """Test add_history_to_messages setting."""
        agent = Agent(add_history_to_messages=True)
        self.assertTrue(agent.add_history_to_messages)

    def test_history_window(self):
        """Test history_window setting."""
        agent = Agent(history_window=5)
        self.assertEqual(agent.history_window, 5)


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
        agent = Agent(model=self.mock_model)
        # Mock the runner's run() to verify run_sync bridges correctly
        agent._runner.run = AsyncMock(return_value=RunResponse(content="Test response"))
        response = agent.run_sync("Hello")
        self.assertIsInstance(response, RunResponse)


class TestAgentSystemPrompt(unittest.TestCase):
    """Test cases for Agent system prompt generation."""

    def test_system_prompt_string(self):
        """Test Agent with string system_prompt via PromptConfig."""
        agent = Agent(prompt_config=PromptConfig(system_prompt="You are a helpful assistant."))
        self.assertEqual(agent.prompt_config.system_prompt, "You are a helpful assistant.")

    def test_system_prompt_callable(self):
        """Test Agent with callable system_prompt via PromptConfig."""
        def get_prompt():
            return "Dynamic prompt"

        agent = Agent(prompt_config=PromptConfig(system_prompt=get_prompt))
        self.assertTrue(callable(agent.prompt_config.system_prompt))

    def test_instructions_list(self):
        """Test Agent with list of instructions."""
        instructions = ["Be helpful", "Be concise", "Be accurate"]
        agent = Agent(instructions=instructions)
        self.assertEqual(len(agent.instructions), 3)


class TestAgentTeam(unittest.TestCase):
    """Test cases for Agent team functionality."""

    def test_agent_with_team(self):
        """Test Agent with team members."""
        member1 = Agent(name="Member1", prompt_config=PromptConfig(role="researcher"))
        member2 = Agent(name="Member2", prompt_config=PromptConfig(role="writer"))
        leader = Agent(name="Leader", team=[member1, member2])

        self.assertEqual(len(leader.team), 2)
        self.assertEqual(leader.team[0].name, "Member1")
        self.assertEqual(leader.team[1].name, "Member2")

    def test_agent_role(self):
        """Test Agent role setting via PromptConfig."""
        agent = Agent(prompt_config=PromptConfig(role="assistant"))
        self.assertEqual(agent.prompt_config.role, "assistant")


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


class TestAgentTimeout(unittest.TestCase):
    """Test cases for Agent timeout settings (now passed to run()/run_stream())."""

    def test_timeout_wrapper_methods_exist(self):
        """Test that timeout wrapper methods exist on Agent's runner."""
        agent = Agent()
        self.assertTrue(hasattr(agent._runner, '_run_with_timeout'))
        self.assertTrue(callable(agent._runner._run_with_timeout))
        self.assertTrue(hasattr(agent._runner, '_wrap_stream_with_timeout'))
        self.assertTrue(callable(agent._runner._wrap_stream_with_timeout))

    def test_run_timeout_triggers_timeout_response(self):
        """Test that timeout produces correct RunResponse event."""
        import asyncio

        agent = Agent()

        async def slow_consume(*args, **kwargs):
            await asyncio.sleep(1)  # Sleep longer than timeout
            return RunResponse(content="Should not reach here")

        with patch.object(agent._runner, "_consume_run", slow_consume):
            agent.model = Mock()
            agent.model.id = "test-model"
            agent.response_model = None
            agent.parse_response = True

            from agentica.run_config import RunConfig
            response = agent.run_sync("test", config=RunConfig(run_timeout=0.001))
            self.assertEqual(response.event, "RunTimeout")
            self.assertIn("timed out", response.content)

    def test_first_token_timeout_in_stream(self):
        """Test first token timeout in streaming mode."""
        import asyncio

        agent = Agent()

        async def slow_async_iterator():
            await asyncio.sleep(1)  # Sleep longer than first_token_timeout
            yield RunResponse(content="Should not reach here")

        async def get_first():
            async for item in agent._runner._wrap_stream_with_timeout(
                slow_async_iterator(), first_token_timeout=0.001
            ):
                return item
            return None

        result = asyncio.run(get_first())
        self.assertIsNotNone(result)
        self.assertEqual(result.event, "FirstTokenTimeout")
        self.assertIn("First token timed out", result.content)

    def test_run_timeout_in_stream(self):
        """Test run timeout in streaming mode."""
        import asyncio

        agent = Agent()

        async def slow_async_iterator():
            yield RunResponse(content="First")  # First token arrives quickly
            await asyncio.sleep(0.1)  # Then slow down beyond run_timeout
            yield RunResponse(content="Second")  # Should timeout before this

        async def collect():
            out = []
            async for item in agent._runner._wrap_stream_with_timeout(
                slow_async_iterator(), run_timeout=0.05
            ):
                out.append(item)
            return out

        results = asyncio.run(collect())
        # Should get first item, then timeout
        self.assertEqual(results[0].content, "First")
        self.assertEqual(results[-1].event, "RunTimeout")


if __name__ == "__main__":
    unittest.main()
