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
            name="Chinese Translator",
            instructions="Translate to Chinese",
        )
        tool = agent.as_tool()
        
        self.assertEqual(tool.name, "chinese_translator")
    
    def test_as_tool_custom_name(self):
        """Test that custom tool name is used when provided."""
        agent = Agent(
            name="Chinese Translator",
            instructions="Translate to Chinese",
        )
        tool = agent.as_tool(tool_name="translate_zh")
        
        self.assertEqual(tool.name, "translate_zh")
    
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
            role="Professional Chinese translator",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        
        self.assertIn("Professional Chinese translator", tool.description)
    
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
        mock_response = RunResponse(content="翻译后的文本")
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
        self.assertEqual(result, "翻译后的文本")
    
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
        mock_response = RunResponse(content="原始输出", run_id="test-123")
        mock_run.return_value = mock_response
        
        def custom_extractor(response: RunResponse) -> str:
            return f"提取结果: {response.content} (run_id: {response.run_id})"
        
        agent = Agent(
            name="Translator",
            instructions="Translate text",
        )
        tool = agent.as_tool(custom_output_extractor=custom_extractor)
        
        result = tool.entrypoint("Hello")
        
        self.assertEqual(result, "提取结果: 原始输出 (run_id: test-123)")
    
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


class AgentAsToolMockedIntegrationTest(unittest.TestCase):
    """Integration tests for Agent as Tool pattern with mocked LLM."""
    
    @patch.object(Agent, 'run')
    def test_orchestrator_with_agent_tools(self, mock_run):
        """Test orchestrator agent using sub-agents as tools (mocked)."""
        # Mock the run method to return a predefined response
        mock_run.return_value = RunResponse(content="你好，今天你好吗？")
        
        # Create a simple translator agent
        translator_agent = Agent(
            name="Chinese Translator",
            instructions="将输入文本翻译成中文",
        )
        
        # Create orchestrator with translator agent as tool
        orchestrator = Agent(
            name="Orchestrator",
            instructions="你有一个翻译工具。当被要求翻译时使用它。",
            tools=[
                translator_agent.as_tool(
                    tool_name="translate_to_chinese",
                    tool_description="将文本翻译成中文",
                ),
            ],
        )
        
        # Verify the setup works without errors
        self.assertIsNotNone(orchestrator.tools)
        self.assertEqual(len(orchestrator.tools), 1)
        
        # Verify tool properties
        tool = orchestrator.tools[0]
        self.assertIsInstance(tool, Function)
        self.assertEqual(tool.name, "translate_to_chinese")
    
    @patch.object(Agent, 'run')
    def test_multiple_agent_tools(self, mock_run):
        """Test orchestrator with multiple agent tools (mocked)."""
        mock_run.return_value = RunResponse(content="处理结果")
        
        # Create multiple specialist agents
        translator = Agent(name="Translator", instructions="翻译文本")
        summarizer = Agent(name="Summarizer", instructions="总结文本")
        analyzer = Agent(name="Analyzer", instructions="分析文本")
        
        # Create orchestrator with multiple tools
        orchestrator = Agent(
            name="Orchestrator",
            instructions="协调多个专家Agent",
            tools=[
                translator.as_tool(tool_name="translate", tool_description="翻译"),
                summarizer.as_tool(tool_name="summarize", tool_description="总结"),
                analyzer.as_tool(tool_name="analyze", tool_description="分析"),
            ],
        )
        
        # Verify all tools are added
        self.assertEqual(len(orchestrator.tools), 3)
        
        # Verify tool names
        tool_names = [t.name for t in orchestrator.tools]
        self.assertIn("translate", tool_names)
        self.assertIn("summarize", tool_names)
        self.assertIn("analyze", tool_names)
    
    @patch.object(Agent, 'run')
    def test_agent_tool_execution_chain(self, mock_run):
        """Test chained execution of agent tools (mocked)."""
        # Setup mock to return different values on each call
        mock_run.side_effect = [
            RunResponse(content="AI是机器展示的智能。"),  # First call: summarize
            RunResponse(content="人工智能是机器展示的智能。"),  # Second call: translate
        ]
        
        summarizer = Agent(name="Summarizer", instructions="总结文本")
        translator = Agent(name="Translator", instructions="翻译文本")
        
        # Get tools
        summarize_tool = summarizer.as_tool(tool_name="summarize")
        translate_tool = translator.as_tool(tool_name="translate")
        
        # Execute chain: summarize then translate
        summary = summarize_tool.entrypoint("Long text about AI...")
        self.assertEqual(summary, "AI是机器展示的智能。")
        
        translation = translate_tool.entrypoint(summary)
        self.assertEqual(translation, "人工智能是机器展示的智能。")
        
        # Verify both agents were called
        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
