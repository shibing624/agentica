# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom tool demo, demonstrates how to create function-based and class-based tools
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, DeepAgent, OpenAIChat, WeatherTool, ShellTool, Tool, FileTool, ZhipuAI


# ============================================================================
# Simple function-based tools
# ============================================================================

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
    # Guard against extremely large exponents to prevent overflow
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


# ============================================================================
# Class-based tools
# ============================================================================

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


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main function to create and run the agent with custom tools."""
    # Create agent with custom and built-in tools
    agent = Agent(
        # model=OpenAIChat(id='gpt-4o'),
        model=ZhipuAI(),
        tools=[
            multiply,           # Function-based tool
            AddTool(),          # Class-based tool
            exponentiate,       # Function-based tool
            get_text_length,    # Function-based tool
            WeatherTool(),      # Built-in tool
            ShellTool(),        # Built-in tool
            FileTool()          # Built-in tool
        ],
        # debug_mode=True
    )
    
    # Example queries
    agent.print_response("3乘以10000005是啥?")
    agent.print_response(
        "将3的五次方乘以(12和3的和). step by step to show the result. "
        "最后统计一下结果的字符长度。"
    )
    agent.print_response("明天北京天气多少度？温度 乘以 2333 = ？")
    agent.print_response("查询当前目录最大的py文件")


if __name__ == "__main__":
    main()
