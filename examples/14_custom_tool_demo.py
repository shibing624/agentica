"""
Custom Tool Demo

This demo shows how to create custom tools for the Agent:
1. Simple function-based tools
2. Class-based tools inheriting from Tool
3. Using built-in tools alongside custom tools
"""

from agentica import Agent, OpenAIChat, WeatherTool, ShellTool, Tool, FileTool


# ============================================================================
# Simple function-based tools
# ============================================================================

def multiply(first_int: float, second_int: float) -> str:
    """Multiply two numbers together.
    
    Args:
        first_int: The first number to multiply
        second_int: The second number to multiply
        
    Returns:
        A string representation of the product
    """
    return str(first_int * second_int)


def exponentiate(base: float, exponent: float) -> str:
    """Exponentiate the base to the exponent power.
    
    Args:
        base: The base number
        exponent: The exponent to raise the base to
        
    Returns:
        A string representation of base raised to the power of exponent
    """
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

    def run_add(self, first_int: float, second_int: float) -> str:
        """Add two numbers together.
        
        Args:
            first_int: The first number to add
            second_int: The second number to add
            
        Returns:
            A string representation of the sum
        """
        return str(first_int + second_int)


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main function to create and run the agent with custom tools."""
    # Create agent with custom and built-in tools
    agent = Agent(
        model=OpenAIChat(id='gpt-4o'),
        tools=[
            multiply,           # Function-based tool
            AddTool(),          # Class-based tool
            exponentiate,       # Function-based tool
            get_text_length,    # Function-based tool
            WeatherTool(),      # Built-in tool
            ShellTool(),        # Built-in tool
            FileTool()          # Built-in tool
        ],
        debug_mode=True
    )
    
    # Example queries
    agent.print_response("3乘以10000005是啥?")
    agent.print_response(
        "将3的五次方乘以(12和3的和). step by step to show the result. "
        "最后统计一下结果的字符长度。"
    )
    agent.print_response("明天北京天气多少度？温度 乘以 2333 = ？")


if __name__ == "__main__":
    main()
