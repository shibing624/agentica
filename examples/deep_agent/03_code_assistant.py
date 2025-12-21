# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent Code Assistant Demo

This example demonstrates DeepAgent as a coding assistant:
- Code execution with the 'execute' tool
- Code analysis and debugging
- Code generation and refactoring
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


def code_execution_demo():
    """Demo: Execute Python code."""
    print("=" * 60)
    print("Demo 1: Code Execution")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="CodeRunner",
        description="A code execution assistant",
        show_tool_calls=True,
        # debug_mode=True,
    )

    # Simple calculation
    response = agent.run(
        "Write and execute Python code to calculate the first 10 Fibonacci numbers"
    )
    print(f"\nResponse:\n{response.content}")


def code_analysis_demo():
    """Demo: Analyze and debug code."""
    print("\n" + "=" * 60)
    print("Demo 2: Code Analysis")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="CodeAnalyzer",
        description="A code analysis assistant",
        instructions=[
            "You are an expert code reviewer.",
            "Analyze code for bugs, performance issues, and best practices.",
        ],
        show_tool_calls=True,
        # debug_mode=True,
    )

    buggy_code = '''
def find_max(numbers):
    max_num = 0
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

# Test
print(find_max([-5, -2, -10]))  # Expected: -2, but returns 0
'''

    response = agent.run(
        f"Analyze this code and find the bug:\n```python\n{buggy_code}\n```"
    )
    print(f"\nResponse:\n{response.content}")


def code_generation_demo():
    """Demo: Generate code from requirements."""
    print("\n" + "=" * 60)
    print("Demo 3: Code Generation")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="CodeGenerator",
        description="A code generation assistant",
        instructions=[
            "You are an expert Python programmer.",
            "Generate clean, well-documented code.",
            "Include type hints and docstrings.",
            "IMPORTANT: Use Python syntax correctly - use `None` (not `null`), `True`/`False` (not `true`/`false`).",
            "Before executing code, verify the syntax is correct Python.",
        ],
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Write a Python function that implements binary search. "
        "Include type hints, docstring, and test cases. Then execute the tests. "
        "Remember: use None for null values in Python."
    )
    print(f"\nResponse:\n{response.content}")


def data_processing_demo():
    """Demo: Data processing with code execution."""
    print("\n" + "=" * 60)
    print("Demo 4: Data Processing")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DataProcessor",
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Write and execute Python code to:\n"
        "1. Generate a list of 100 random numbers between 1 and 1000\n"
        "2. Calculate mean, median, and standard deviation\n"
        "3. Find the top 5 largest numbers\n"
        "4. Print a summary of the results\n"
        "write the code to ./tmp/ directory."
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    code_execution_demo()
    code_analysis_demo()
    code_generation_demo()
    data_processing_demo()
