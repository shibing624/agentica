# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in tools overview - Demonstrates various built-in tools in Agentica

This example shows how to use common built-in tools:
1. WeatherTool - Get weather information
2. ShellTool - Execute shell commands
3. FileTool - File operations
4. CalculatorTool - Mathematical calculations
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica import WeatherTool, ShellTool, FileTool
from agentica.tools.calculator_tool import CalculatorTool


async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[
            WeatherTool(),
            ShellTool(),
            FileTool(),
            CalculatorTool(),
        ],
        add_datetime_to_instructions=True,
    )

    print("=" * 60)
    print("Example 1: Weather Tool")
    print("=" * 60)
    await agent.print_response("北京今天天气怎么样?")

    print("\n" + "=" * 60)
    print("Example 2: Shell Tool")
    print("=" * 60)
    await agent.print_response("列出当前目录下的文件")

    print("\n" + "=" * 60)
    print("Example 3: File Tool")
    print("=" * 60)
    await agent.print_response("读取当前目录下的README.md文件的前10行")

    print("\n" + "=" * 60)
    print("Example 4: Calculator Tool")
    print("=" * 60)
    await agent.print_response("计算 (123 + 456) * 789 / 2 的结果")

    print("\n" + "=" * 60)
    print("Example 5: Combining Multiple Tools")
    print("=" * 60)
    await agent.print_response("查询上海天气，然后计算如果温度乘以2会是多少度")


if __name__ == "__main__":
    asyncio.run(main())
