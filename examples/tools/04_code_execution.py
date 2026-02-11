# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Code execution demo - Demonstrates code tools

This example shows how to use code tools:
1. ShellTool - Execute shell commands
2. CodeTool - Code analysis and quality checking
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, ShellTool
from agentica.tools.code_tool import CodeTool


async def main():
    print("=" * 60)
    print("Example 1: Shell Command Execution")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[ShellTool()],
        instructions=[
            "You are an expert programmer.",
            "When asked to solve problems, write Python code and use shell to execute it.",
        ],
    )

    await agent.print_response("用python计算斐波那契数列的前20个数，直接用shell执行python -c命令")

    print("\n" + "=" * 60)
    print("Example 2: Code Analysis")
    print("=" * 60)

    agent2 = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[CodeTool()],
    )

    await agent2.print_response("分析当前目录下的 setup.py 文件结构")


if __name__ == "__main__":
    asyncio.run(main())
