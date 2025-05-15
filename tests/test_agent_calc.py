# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import unittest
import sys

sys.path.append('..')
from agentica.agent import Agent
from agentica.tools.calculator_tool import CalculatorTool
from agentica import WeatherTool
from agentica.model.zhipuai.chat import ZhipuAI
from agentica import Message, Yi, AzureOpenAIChat, DeepSeek, OpenAIChat, Moonshot, ZhipuAI, Doubao


class CalcParser(unittest.TestCase):

    def test_calc1(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=ZhipuAI(), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc1_plus(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=ZhipuAI(id='glm-4-plus'), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc2(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=Yi(), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc3(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=AzureOpenAIChat(id='gpt-4.1'), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc4(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=Doubao(id="ep-20250326193223-b2b4d"), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc5(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=OpenAIChat(id='gpt-4o'), tools=[CalculatorTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc6(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=OpenAIChat(id='gpt-4o-mini'), tools=[CalculatorTool(), WeatherTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
        r = m.run("查询北京天气，并用温度 乘以 * 212222.3 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc7(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=OpenAIChat(id='gpt-3.5-turbo'), tools=[CalculatorTool(), WeatherTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
        r = m.run("查询北京天气，并用温度 乘以 * 212222.3 =?")
        print(r)
        self.assertIsNotNone(r)

    def test_calc8(self):
        """
        Test calculator tool.
        """
        m = Agent(llm=Moonshot(), tools=[CalculatorTool(), WeatherTool()])
        r = m.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
        print(r)
        self.assertIsNotNone(r)
        r = m.run("查询北京天气，并用温度 乘以 * 212222.3 =?")
        print(r)
        self.assertIsNotNone(r)
