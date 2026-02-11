# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in tools overview - Demonstrates various built-in tools in Agentica

This example shows how to use common built-in tools:
1. WeatherTool - Get weather information
2. ShellTool - Execute shell commands
3. JinaTool - Web content reading
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica import WeatherTool, ShellTool, JinaTool


async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[
            ShellTool(),
            JinaTool(),
        ],
        add_datetime_to_instructions=True,
    )

    print("\n" + "=" * 60)
    print("Example: Shell Tool")
    print("=" * 60)
    await agent.print_response("列出当前目录下的文件")

    print("\n" + "=" * 60)
    print("Example: Jina Tool (Web Reading)")
    print("=" * 60)
    await agent.print_response("读取 https://httpbin.org/get 的内容摘要")


if __name__ == "__main__":
    asyncio.run(main())
