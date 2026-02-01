# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async tool demo - Demonstrates how to use async functions as tools

This example shows:
1. Async function-based tools
2. Async class-based tools
3. Mixing async and sync tools
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Tool


# ============================================================================
# Async function-based tools
# ============================================================================

async def async_fetch_data(url: str) -> str:
    """Simulate fetching data from a URL asynchronously.
    
    Args:
        url: The URL to fetch data from
        
    Returns:
        A simulated response from the URL
    """
    # Simulate async I/O operation
    await asyncio.sleep(0.1)
    return f"Fetched data from {url}: {{status: 200, data: 'example response'}}"


async def async_process_data(data: str, operation: str) -> str:
    """Process data asynchronously with a specified operation.
    
    Args:
        data: The data to process
        operation: The operation to perform (uppercase, lowercase, reverse)
        
    Returns:
        The processed data
    """
    await asyncio.sleep(0.05)
    if operation == "uppercase":
        return data.upper()
    elif operation == "lowercase":
        return data.lower()
    elif operation == "reverse":
        return data[::-1]
    else:
        return f"Unknown operation: {operation}"


async def async_calculate(a: float, b: float, op: str) -> str:
    """Perform async calculation.
    
    Args:
        a: First number
        b: Second number
        op: Operation (add, sub, mul, div)
        
    Returns:
        The result of the calculation
    """
    await asyncio.sleep(0.02)
    if op == "add":
        return str(a + b)
    elif op == "sub":
        return str(a - b)
    elif op == "mul":
        return str(a * b)
    elif op == "div":
        return str(a / b) if b != 0 else "Error: Division by zero"
    else:
        return f"Unknown operation: {op}"


# ============================================================================
# Sync function for comparison
# ============================================================================

def sync_multiply(x: float, y: float) -> str:
    """Multiply two numbers synchronously.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The product of x and y
    """
    return str(x * y)


# ============================================================================
# Async class-based tools
# ============================================================================

class AsyncDataTool(Tool):
    """A tool with async methods for data operations."""
    
    def __init__(self):
        super().__init__(name="async_data_tool")
        self.register(self.fetch_and_process)

    async def fetch_and_process(self, url: str, transform: str = "uppercase") -> str:
        """Fetch data from URL and transform it.
        
        Args:
            url: The URL to fetch from
            transform: Transformation to apply (uppercase, lowercase, reverse)
            
        Returns:
            The transformed data
        """
        # Simulate fetching
        await asyncio.sleep(0.1)
        data = f"Data from {url}"
        
        # Apply transformation
        if transform == "uppercase":
            return data.upper()
        elif transform == "lowercase":
            return data.lower()
        elif transform == "reverse":
            return data[::-1]
        return data


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main function to test async tools."""
    # Create agent with async and sync tools
    agent = Agent(
        model=OpenAIChat(id='gpt-4o-mini'),
        tools=[
            async_fetch_data,     # Async function tool
            async_process_data,   # Async function tool
            async_calculate,      # Async function tool
            sync_multiply,        # Sync function tool
            AsyncDataTool(),      # Async class-based tool
        ],
        show_tool_calls=True,
    )
    
    print("=" * 60)
    print("Example 1: Async fetch")
    print("=" * 60)
    agent.print_response("Fetch data from https://api.example.com/users")
    
    print("\n" + "=" * 60)
    print("Example 2: Async calculation")
    print("=" * 60)
    agent.print_response("Calculate 100 add 200 using async_calculate")
    
    print("\n" + "=" * 60)
    print("Example 3: Mix of async and sync")
    print("=" * 60)
    agent.print_response("Use sync_multiply to multiply 5 and 10, then use async_calculate to add 100 to the result")
    
    print("\n" + "=" * 60)
    print("Example 4: Async class-based tool")
    print("=" * 60)
    agent.print_response("Use AsyncDataTool to fetch and process data from https://example.com with uppercase transform")


async def async_main():
    """Async main function to test async tools with arun."""
    agent = Agent(
        model=OpenAIChat(id='gpt-4o-mini'),
        tools=[
            async_fetch_data,
            async_calculate,
            sync_multiply,
        ],
        show_tool_calls=True,
    )
    
    print("=" * 60)
    print("Example: Using arun with async tools")
    print("=" * 60)
    response = await agent.arun("Fetch data from https://api.test.com and then calculate 50 add 100")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()
    
    # Uncomment to test async execution
    # asyncio.run(async_main())
