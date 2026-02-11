# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: File operations demo - Demonstrates file manipulation tools

This example shows how to use file operation tools:
1. FileTool - Basic file operations
2. EditTool - File editing
3. WorkspaceTool - Workspace navigation
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica import FileTool
from agentica.tools.edit_tool import EditTool
from agentica.tools.workspace_tool import WorkspaceTool


def main():
    # Create agent with file operation tools
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[FileTool(), EditTool(), WorkspaceTool()],
    )

    # Example 1: List files
    print("=" * 60)
    print("Example 1: List Files")
    print("=" * 60)
    agent.print_response_sync("列出当前目录下的所有Python文件")

    # Example 2: Read file
    print("\n" + "=" * 60)
    print("Example 2: Read File")
    print("=" * 60)
    agent.print_response_sync("读取当前目录下README.md的内容摘要")

    # Example 3: Create and write file
    print("\n" + "=" * 60)
    print("Example 3: Create File")
    print("=" * 60)
    agent.print_response_sync(
        "在outputs目录下创建一个test_hello.py文件，内容是打印Hello World"
    )

    # Example 4: Edit file
    print("\n" + "=" * 60)
    print("Example 4: Edit File")
    print("=" * 60)
    agent.print_response_sync(
        "修改outputs/test_hello.py，添加一行打印当前时间的代码"
    )

    # Example 5: Workspace navigation
    print("\n" + "=" * 60)
    print("Example 5: Workspace Navigation")
    print("=" * 60)
    agent.print_response_sync("查看当前工作目录的结构")

    # Cleanup
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    agent.print_response_sync("删除outputs/test_hello.py文件")


if __name__ == "__main__":
    main()
