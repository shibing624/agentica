# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.react_agent import ReactAgent

pwd_path = Path(__file__).parent


class ReactAgentTest(unittest.TestCase):

    @patch('agentica.react_agent.OpenAI')
    def test_calc(self, mock_openai):
        """Test ReactAgent with mocked OpenAI API."""
        # Mock the OpenAI client and its chat.completions.create method
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Thought: 北京是中国的首都。\nAction: finish(北京是中国的首都，历史悠久，是政治、文化中心。)"
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        m = ReactAgent(model_name='gpt-4o-mini')
        r = m.run("一句话介绍北京")
        print(r)
        self.assertIsNotNone(r)
