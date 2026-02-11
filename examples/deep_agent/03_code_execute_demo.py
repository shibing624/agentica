# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent Code Assistant Demo

This example demonstrates DeepAgent as a coding assistant:
- Code execution with the 'execute' tool
- Code analysis and debugging
- Code generation and refactoring
- Self-verification with lint/test commands
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


def code_execution_demo():
    """Demo: Execute Python code."""
    print("=" * 60)
    print("Demo 1: Code Execution")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        name="CodeRunner",
        description="A code execution assistant",
        debug_mode=True,
    )

    # Simple calculation
    response = agent.run_sync(
        "Write and execute Python code to calculate the first 10 Fibonacci numbers, "
        "and judge each num is even or odd. give me the result after execution."
    )
    print(f"\nResponse:\n{response}")


def code_analysis_demo():
    """Demo: Analyze and debug code."""
    print("\n" + "=" * 60)
    print("Demo 2: Code Analysis")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        name="CodeAnalyzer",
        description="A code analysis assistant",
        instructions=[
            "You are an expert code reviewer.",
            "Analyze code for bugs, performance issues, and best practices. last run it and fix the bug.",
        ],
        debug_mode=True,
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

    response = agent.run_sync(
        f"Analyze this code and find the bug:\n```python\n{buggy_code}\n``` --- last run it and verify the result."
    )
    print(f"\nResponse:\n{response}")


def code_generation_demo():
    """Demo: Generate code from requirements."""
    print("\n" + "=" * 60)
    print("Demo 3: Code Generation")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        name="CodeGenerator",
        description="A code generation assistant",
        instructions=[
            "You are an expert Python programmer.",
            "Generate clean, well-documented code.",
            "Include type hints and docstrings.",
            "IMPORTANT: Use Python syntax correctly - use `None` (not `null`), `True`/`False` (not `true`/`false`).",
            "Before executing code, verify the syntax is correct Python.",
        ],
        debug_mode=True,
    )

    response = agent.run_sync(
        "Write a Python function that implements binary search. "
        "Include type hints, docstring, and test cases. Then execute the tests. output the test results."
    )
    print(f"\nResponse:\n{response}")


def data_processing_demo():
    """Demo: Data processing with code execution."""
    print("\n" + "=" * 60)
    print("Demo 4: Data Processing")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        name="DataProcessor",
        instructions=[
            "You are a data processing expert.",
            "Use Python to manipulate and analyze data.",
            "Include error handling and validation. wirte the code to ./tmp/ directory.",
        ],
        debug_mode=True,
    )

    response = agent.run_sync(
        "Write and execute Python code to:\n"
        "1. Generate a list of 100 random numbers between 1 and 1000\n"
        "2. Calculate mean, median, and standard deviation\n"
        "3. Find the top 5 largest numbers\n"
        "4. Print a summary of the results\n"
    )
    print(f"\nResponse:\n{response}")


def self_verification_demo():
    """Demo: Write code to file and verify with lint/test."""
    print("\n" + "=" * 60)
    print("Demo 5: Self Verification - Write and Verify Code")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        debug_mode=True,
    )

    # Create output directory
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(demo_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)

    response = agent.run_sync(
        f"""Please complete the following tasks:

1. Write a Python module `calculator.py` in {output_dir}/ with these functions:
   - add(a, b) -> sum of two numbers
   - subtract(a, b) -> difference
   - multiply(a, b) -> product
   - divide(a, b) -> quotient (handle division by zero)
   All functions should have type hints and docstrings.

2. Write a test file `test_calculator.py` and run, last verify the code.
"""
    )
    print(f"\nResponse:\n{response}")


def lint_fix_demo():
    """Demo: Detect and fix lint issues in existing code."""
    print("\n" + "=" * 60)
    print("Demo 6: Lint Fix - Detect and Fix Code Issues")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(),
        debug_mode=True,
    )

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(demo_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with intentional lint issues
    bad_code = '''
import os
import sys
import json

def process_data(data,config):  # missing space after comma
    """Process data."""
    result=[]=
    for item in data:
        if item>0: >
            result.append(item*2)
    return result

x = 1
y = 2
z=3

'''

    bad_code_path = os.path.join(output_dir, "bad_code.py")
    with open(bad_code_path, "w") as f:
        f.write(bad_code)

    response = agent.run_sync(
        f"""I have a Python file at {bad_code_path} . fix，do Verification.
"""
    )
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    code_execution_demo()
    code_analysis_demo()
    # code_generation_demo()
    # data_processing_demo()
    # self_verification_demo()
    lint_fix_demo()
