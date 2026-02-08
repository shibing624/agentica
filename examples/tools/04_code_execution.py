# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Code execution demo - Demonstrates code execution tools

This example shows how to use code execution tools:
1. RunPythonCodeTool - Execute Python code
2. CodeTool - Code analysis and manipulation
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.tools.run_python_code_tool import RunPythonCodeTool
from agentica.tools.code_tool import CodeTool


def main():
    # Example 1: Python code execution
    print("=" * 60)
    print("Example 1: Python Code Execution")
    print("=" * 60)
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[RunPythonCodeTool(save_and_run=True, base_dir="outputs")],
        instructions=[
            "You are an expert Python programmer.",
            "When asked to solve problems, write and execute Python code.",
        ],
    )
    
    agent.print_response("计算斐波那契数列的前20个数")
    
    # Example 2: Data analysis
    print("\n" + "=" * 60)
    print("Example 2: Data Analysis")
    print("=" * 60)
    
    agent.print_response(
        "创建一个包含10个随机数的列表，计算它们的平均值、中位数和标准差"
    )
    
    # Example 3: Code with visualization (saves to file)
    print("\n" + "=" * 60)
    print("Example 3: Code Generation")
    print("=" * 60)
    
    agent2 = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[CodeTool()],
    )
    
    agent2.print_response("写一个冒泡排序的Python函数，并解释它的工作原理")


if __name__ == "__main__":
    main()
