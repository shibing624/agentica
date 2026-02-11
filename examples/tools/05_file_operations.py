# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: File operations demo - Demonstrates file manipulation tools

This example shows how to use file operation tools:
1. ShellTool - File listing and basic operations
2. PatchTool - File editing with diff patches
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, ShellTool
from agentica.tools.patch_tool import PatchTool


async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[ShellTool(), PatchTool()],
    )

    print("=" * 60)
    print("Example 1: List Files")
    print("=" * 60)
    await agent.print_response("列出当前目录下的所有Python文件")

    print("\n" + "=" * 60)
    print("Example 2: Read File")
    print("=" * 60)
    await agent.print_response("读取当前目录下README.md的前10行")

    print("\n" + "=" * 60)
    print("Example 3: Create and Edit File")
    print("=" * 60)
    await agent.print_response(
        "在outputs目录下创建一个test_hello.py文件，内容是打印Hello World，"
        "然后用patch工具添加一行打印当前时间的代码"
    )

    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    await agent.print_response("删除outputs/test_hello.py文件")


if __name__ == "__main__":
    asyncio.run(main())
