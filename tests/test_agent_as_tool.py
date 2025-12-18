# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for Agent.as_tool() functionality
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.tools.base import Function
from agentica.run_response import RunResponse


class AgentAsToolTest(unittest.TestCase):
    """Test cases for Agent.as_tool() method."""

    def test_as_tool_returns_function(self):
        """Test that as_tool() returns a Function object."""
        agent = Agent(
            name="Test Agent",
            instructions="You are a test agent",
        )
        tool = agent.as_tool()
        
        self.assertIsInstance(tool, Function)
    
    def test_as_tool_default_name_from_agent_name(self):
        """Test that tool name defaults to snake_case of agent name."""
        agent = Agent(
            name="Spanish Translator",
            instructions="Translate to Spanish",
        )
        tool = agent.as_tool()
        
        self.assertEqual(tool.name, "spanish_translator")
    
    def test_as_tool_custom_name(self):
        """Test that custom tool name is used when provided."""
        agent = Agent(
            name="Spanish Translator",
            instructions="Translate to Spanish",
        )
        tool = agent.as_tool(tool_name="translate_es")
        
        self.assertEqual(tool.name, "translate_es")
    
    def test_as_tool_default_description_from_agent_description(self):
        """Test that tool description defaults to agent description."""
        agent = Agent(
            name="Translator",
            description="A professional translator agent",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        
        self.assertIn("A professional translator agent", tool.description)
    
    def test_as_tool_default_description_from_agent_role(self):
        """Test that tool description falls back to agent role."""
        agent = Agent(
            name="Translator",
            role="Professional Spanish translator",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        
        self.assertIn("Professional Spanish translator", tool.description)
    
    def test_as_tool_custom_description(self):
        """Test that custom tool description is used when provided."""
        agent = Agent(
            name="Translator",
            instructions="Translate text",
        )
        tool = agent.as_tool(tool_description="Custom description for translation")
        
        self.assertIn("Custom description for translation", tool.description)
    
    def test_as_tool_name_fallback_to_agent_id(self):
        """Test that tool name falls back to agent_id when name is not set."""
        agent = Agent(
            instructions="Test agent",
        )
        tool = agent.as_tool()
        
        # Should start with "agent_" followed by first 8 chars of agent_id
        self.assertTrue(tool.name.startswith("agent_"))
        self.assertEqual(len(tool.name), 14)  # "agent_" + 8 chars
    
    def test_as_tool_has_entrypoint(self):
        """Test that the tool has an entrypoint function."""
        agent = Agent(
            name="Test Agent",
            instructions="Test",
        )
        tool = agent.as_tool()
        
        self.assertIsNotNone(tool.entrypoint)
        self.assertTrue(callable(tool.entrypoint))
    
    @patch.object(Agent, 'run')
    def test_as_tool_calls_agent_run(self, mock_run):
        """Test that calling the tool invokes agent.run()."""
        mock_response = RunResponse(content="Translated text")
        mock_run.return_value = mock_response
        
        agent = Agent(
            name="Translator",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        
        # Call the tool's entrypoint
        result = tool.entrypoint("Hello world")
        
        # Verify agent.run was called with the input
        mock_run.assert_called_once_with("Hello world", stream=False)
        self.assertEqual(result, "Translated text")
    
    @patch.object(Agent, 'run')
    def test_as_tool_handles_none_content(self, mock_run):
        """Test that tool handles None content from agent response."""
        mock_response = RunResponse(content=None)
        mock_run.return_value = mock_response
        
        agent = Agent(
            name="Translator",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        
        result = tool.entrypoint("Hello")
        
        self.assertEqual(result, "No response from agent.")
    
    @patch.object(Agent, 'run')
    def test_as_tool_custom_output_extractor(self, mock_run):
        """Test that custom output extractor is used when provided."""
        mock_response = RunResponse(content="Raw output", run_id="test-123")
        mock_run.return_value = mock_response
        
        def custom_extractor(response: RunResponse) -> str:
            return f"Extracted: {response.content} (run_id: {response.run_id})"
        
        agent = Agent(
            name="Translator",
            instructions="Translate text",
        )
        tool = agent.as_tool(custom_output_extractor=custom_extractor)
        
        result = tool.entrypoint("Hello")
        
        self.assertEqual(result, "Extracted: Raw output (run_id: test-123)")
    
    @patch.object(Agent, 'run')
    def test_as_tool_handles_dict_content(self, mock_run):
        """Test that tool handles dict content from agent response."""
        mock_response = RunResponse(content={"key": "value", "number": 42})
        mock_run.return_value = mock_response
        
        agent = Agent(
            name="Analyzer",
            instructions="Analyze text",
        )
        tool = agent.as_tool()
        
        result = tool.entrypoint("Analyze this")
        
        # Should be JSON serialized
        self.assertIn('"key"', result)
        self.assertIn('"value"', result)
        self.assertIn("42", result)


class AgentAsToolIntegrationTest(unittest.TestCase):
    """Integration tests for Agent as Tool pattern (requires API key)."""
    
    @unittest.skipIf(
        not os.environ.get("OPENAI_API_KEY"),
        "OPENAI_API_KEY not set, skipping integration test"
    )
    def test_orchestrator_with_agent_tools(self):
        """Test orchestrator agent using sub-agents as tools."""
        from agentica import OpenAIChat
        
        # Create a simple echo agent
        echo_agent = Agent(
            name="Echo Agent",
            model=OpenAIChat(id='gpt-4o-mini'),
            instructions="Simply repeat back the input text exactly as given.",
        )
        
        # Create orchestrator with echo agent as tool
        orchestrator = Agent(
            name="Orchestrator",
            model=OpenAIChat(id='gpt-4o-mini'),
            instructions="You have an echo tool. Use it when asked to echo something.",
            tools=[
                echo_agent.as_tool(
                    tool_name="echo",
                    tool_description="Echo back the input text",
                ),
            ],
        )
        
        # This test just verifies the setup works without errors
        # Actual API call would require credits
        self.assertIsNotNone(orchestrator.tools)
        self.assertEqual(len(orchestrator.tools), 1)


if __name__ == "__main__":
    unittest.main()
