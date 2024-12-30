# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
from pathlib import Path
import sys

sys.path.append('..')
from agentica import Agent
from agentica.tools.calculator_tool import CalculatorTool
from agentica import ZhipuAIChat
pwd_path = Path(__file__).parent


class CalcParser(unittest.TestCase):
    def setUp(self):
        llm = ZhipuAIChat()
        self.m = Agent(llm=llm, tools=[CalculatorTool()])

    def test_calc(self):
        """
        Test calculator tool.
        """
        r = self.m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
