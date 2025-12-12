# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.agent import Agent
from agentica.tools.calculator_tool import CalculatorTool
from agentica import WeatherTool
from agentica import Message, ZhipuAI

class CalcParser(unittest.TestCase):

    def test_calc1(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=ZhipuAI(), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
