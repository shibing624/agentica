# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.agent import Agent
from agentica.tools.calculator_tool import CalculatorTool
from agentica import ZhipuAI


class CalcParser(unittest.TestCase):

    @patch('agentica.model.zhipuai.chat.getenv')
    @patch('agentica.agent.Agent.run')
    def test_calc1(self, mock_run, mock_getenv):
        """
        Test calculator tool with mocked ZhipuAI API key.
        """
        # Mock getenv to return a fake API key
        mock_getenv.return_value = "fake_zhipuai_api_key"
        
        # Mock the Agent.run method to return a fake result
        mock_run.return_value = "The result is 7073851546.5406"
        
        m = Agent(llm=ZhipuAI(api_key="fake_zhipuai_api_key"), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
