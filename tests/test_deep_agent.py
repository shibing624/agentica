# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for DeepAgent

This test module covers:
1. DeepAgent initialization and configuration
2. Human-in-the-loop with UserInputTool
3. Context overflow handling
4. Repetitive behavior detection
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import DeepAgent, UserInputTool


class TestDeepAgent(unittest.TestCase):
    """Test cases for DeepAgent."""

    def test_deep_agent_initialization(self):
        """Test basic DeepAgent initialization."""
        agent = DeepAgent(
            name="TestAgent",
            description="A test agent",
        )

        self.assertEqual(agent.name, "TestAgent")
        self.assertTrue(agent.prompt_config.enable_agentic_prompt)  # Default is True for DeepAgent

    def test_builtin_tools(self):
        """Test that builtin tools are correctly configured."""
        agent = DeepAgent(
            name="TestAgent",
            include_file_tools=True,
            include_execute=True,
            include_web_search=True,
            include_fetch_url=True,
            include_todos=True,
            include_task=True,
            include_skills=True,
            include_user_input=False,
        )

        tool_names = agent.get_builtin_tool_names()
        self.assertIn("ls", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("execute", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertIn("fetch_url", tool_names)
        self.assertIn("write_todos", tool_names)
        self.assertIn("task", tool_names)
        self.assertIn("list_skills", tool_names)
        self.assertNotIn("user_input", tool_names)

    def test_builtin_tools_selective(self):
        """Test selective builtin tools."""
        agent = DeepAgent(
            name="TestAgent",
            include_file_tools=False,
            include_execute=False,
            include_web_search=True,
            include_fetch_url=True,
            include_todos=False,
            include_task=False,
            include_skills=False,
        )

        tool_names = agent.get_builtin_tool_names()
        self.assertNotIn("ls", tool_names)
        self.assertNotIn("execute", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertIn("fetch_url", tool_names)
        self.assertNotIn("write_todos", tool_names)
        self.assertNotIn("task", tool_names)

    def test_user_input_tool(self):
        """Test DeepAgent with user input tool."""
        agent = DeepAgent(
            name="InteractiveAgent",
            include_user_input=True,
        )

        tool_names = agent.get_builtin_tool_names()
        self.assertIn("user_input", tool_names)
        self.assertIn("confirm", tool_names)

    def test_context_management_settings(self):
        """Test context management configuration."""
        agent = DeepAgent(
            name="TestAgent",
            context_soft_limit=50000,
            context_hard_limit=80000,
            enable_context_overflow_handling=True,
        )

        self.assertEqual(agent.context_soft_limit, 50000)
        self.assertEqual(agent.context_hard_limit, 80000)
        self.assertTrue(agent.enable_context_overflow_handling)

    def test_repetition_detection_settings(self):
        """Test repetition detection configuration."""
        agent = DeepAgent(
            name="TestAgent",
            enable_repetition_detection=True,
            max_same_tool_calls=5,
        )

        self.assertTrue(agent.enable_repetition_detection)
        self.assertEqual(agent.max_same_tool_calls, 5)

    def test_reflection_settings(self):
        """Test reflection configuration."""
        agent = DeepAgent(
            name="TestAgent",
            enable_step_reflection=True,
            reflection_frequency=5,
        )

        self.assertTrue(agent.enable_step_reflection)
        self.assertEqual(agent.reflection_frequency, 5)

    def test_repr(self):
        """Test string representation."""
        agent = DeepAgent(
            name="TestAgent",
        )

        repr_str = repr(agent)
        self.assertIn("DeepAgent", repr_str)
        self.assertIn("TestAgent", repr_str)

    def test_enable_agentic_prompt_default(self):
        """Test that DeepAgent enables agentic_prompt by default."""
        agent = DeepAgent(
            name="TestAgent",
        )
        self.assertTrue(agent.prompt_config.enable_agentic_prompt)


class TestUserInputTool(unittest.TestCase):
    """Test cases for UserInputTool."""

    def test_user_input_tool_initialization(self):
        """Test UserInputTool initialization."""
        tool = UserInputTool()

        self.assertEqual(tool.timeout, 300)

    def test_user_input_tool_custom_settings(self):
        """Test UserInputTool with custom settings."""
        tool = UserInputTool(
            timeout=60,
            default_on_timeout="no",
        )

        self.assertEqual(tool.timeout, 60)
        self.assertEqual(tool.default_on_timeout, "no")

    def test_user_input_tool_custom_callback(self):
        """Test UserInputTool with custom callback."""
        def custom_callback(prompt: str, options=None) -> str:
            return "custom_response"

        tool = UserInputTool(
            input_callback=custom_callback,
        )

        self.assertIsNotNone(tool.input_callback)


if __name__ == "__main__":
    unittest.main()
