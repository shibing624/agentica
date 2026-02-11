# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom tool demo - Demonstrates how to create function-based and class-based tools

This example shows:
1. Function-based tools (simple functions with docstrings)
2. Class-based tools (using Tool base class)
3. Combining custom tools with built-in tools
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, WeatherTool, ShellTool, Tool


# Simple function-based tools
def multiply(first_num: float, second_num: float) -> str:
    """Multiply two numbers together.
    
    Args:
        first_num: The first number to multiply
        second_num: The second number to multiply
        
    Returns:
        A string representation of the product
    """
    return str(first_num * second_num)


def exponentiate(base: float, exponent: float) -> str:
    """Exponentiate the base to the exponent power.
    
    Args:
        base: The base number
        exponent: The exponent to raise the base to
        
    Returns:
        A string representation of base raised to the power of exponent
    """
    if abs(exponent) > 1000:
        return f"Error: exponent {exponent} is too large, max allowed is 1000"
    return str(base ** exponent)


def get_text_length(text: str) -> str:
    """Get the length of the text.
    
    Args:
        text: The text string to measure
        
    Returns:
        A string representation of the text length
    """
    return str(len(text))


class AddTool(Tool):
    """A custom tool for adding two numbers."""
    
    def __init__(self):
        super().__init__(name="add_tool")
        self.register(self.run_add)

    def run_add(self, first_num: float, second_num: float) -> str:
        """Add two numbers together.
        
        Args:
            first_num: The first number to add
            second_num: The second number to add
            
        Returns:
            A string representation of the sum
        """
        return str(first_num + second_num)


async def main():
    agent = Agent(
        model=OpenAIChat(id='gpt-4o-mini'),
        # debug=True,
        tools=[
            multiply,
            AddTool(),
            exponentiate,
            get_text_length,
            WeatherTool(),
            ShellTool(),
        ],
    )
    
    print("=" * 60)
    print("Example 1: Multiplication")
    print("=" * 60)
    await agent.print_response("3乘以10000005是啥?")
    
    print("\n" + "=" * 60)
    print("Example 2: Complex calculation")
    print("=" * 60)
    await agent.print_response(
        "将3的五次方乘以(12和3的和). step by step to show the result. "
        "最后统计一下结果的字符长度。"
    )
    
    print("\n" + "=" * 60)
    print("Example 3: Weather + calculation")
    print("=" * 60)
    await agent.print_response("明天北京天气多少度？温度 乘以 2333 = ？")
    
    print("\n" + "=" * 60)
    print("Example 4: File operations")
    print("=" * 60)
    await agent.print_response("查询当前目录最大的py文件")


if __name__ == "__main__":
    asyncio.run(main())
