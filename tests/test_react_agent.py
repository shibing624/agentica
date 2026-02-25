# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test cases for Agent with multi-round tool calling (ReAct-style)
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.agent.config import PromptConfig
from agentica.model.openai import OpenAIChat


class AgentMultiRoundTest(unittest.TestCase):
    """Test Agent basic behavior."""

    def test_agent_init(self):
        """Test Agent initialization."""
        agent = Agent(
            prompt_config=PromptConfig(add_datetime_to_instructions=True),
        )
        self.assertTrue(agent.prompt_config.add_datetime_to_instructions)

    def test_agent_with_instructions(self):
        """Test Agent with custom instructions."""
        custom_instructions = ["Always be helpful", "Use tools when needed"]
        agent = Agent(
            instructions=custom_instructions,
        )
        self.assertIsNotNone(agent.instructions)

    @patch.object(OpenAIChat, 'response')
    def test_agent_run(self, mock_response):
        """Test Agent run method with mocked model response."""
        # Mock the model response
        mock_model_response = MagicMock()
        mock_model_response.content = "Beijing is the capital of China."
        mock_model_response.parsed = None
        mock_model_response.audio = None
        mock_model_response.reasoning_content = None
        mock_model_response.created_at = None
        mock_response.return_value = mock_model_response

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        )
        
        # Run the agent
        response = agent.run_sync("What is the capital of China?")
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
