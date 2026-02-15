# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent usage examples

DeepAgent = Agent + Built-in Tools
- Agent: Tools must be added manually
- DeepAgent: Automatically includes built-in tools

Built-in tools:
- File tools: ls, read_file, write_file, edit_file, glob, grep
- Execute: execute shell commands
- Web tools: web_search, fetch_url
- Task tools: write_todos, read_todos
- Subagent: task (launch subagents for parallel execution)
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


async def basic_example():
    """Basic example: Create DeepAgent with minimal config."""
    print("=" * 60)
    print("Basic Example: Minimal DeepAgent")
    print("=" * 60)

    # Minimal config - just model
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
    )

    print(f"Tools: {agent.get_builtin_tool_names()}")
    print(f"Total tools: {len(agent.tools)}")

    await agent.print_response_stream(
        "List all Python files in the current directory",
        show_tool_calls=True,
    )


async def workdir_example():
    """Example with work_dir for file operations."""
    print("\n" + "=" * 60)
    print("Example: With work_dir")
    print("=" * 60)

    # Set work_dir for file operations
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        work_dir="/path/to/project",  # File operations will use this as base
    )

    print(f"work_dir: {agent.work_dir}")
    print(f"Tools: {agent.get_builtin_tool_names()}")


async def disable_tools_example():
    """Example: Disable certain built-in tools for security."""
    print("\n" + "=" * 60)
    print("Example: Disable Tools for Security")
    print("=" * 60)

    # Disable dangerous tools for read-only scenarios
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="SafeExplorer",
        description="Read-only file explorer",
        include_execute=False,      # Disable shell execution
        include_web_search=False,    # Disable web access
        include_todos=False,         # Disable task management
        include_task=False,          # Disable subagent spawning
    )

    print(f"Tools: {agent.get_builtin_tool_names()}")


async def add_custom_tools_example():
    """Example: Add custom tools alongside built-in tools."""
    print("\n" + "=" * 60)
    print("Example: Add Custom Tools")
    print("=" * 60)

    def multiply(a: float, b: float) -> str:
        """Multiply two numbers."""
        return str(a * b)

    def weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 25°C"

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="Assistant",
        tools=[multiply, weather],  # Add custom tools
    )

    print(f"Built-in: {agent.get_builtin_tool_names()}")
    print(f"Custom: multiply, weather")

    await agent.print_response("What is 123 * 456? What's the weather in Tokyo?")


async def deep_research_example():
    """Example: Enable deep research mode."""
    print("\n" + "=" * 60)
    print("Example: Deep Research Mode")
    print("=" * 60)

    # Enable deep research prompt optimization
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        enable_deep_research=True,  # Optimized system prompt for research
        include_todos=False,
    )

    print(f"Deep research mode: {agent.enable_deep_research}")

    await agent.print_response(
        "Research the latest developments in AI agents in 2025",
        show_tool_calls=True,
    )


async def full_config_example():
    """Example: Full configuration with memory and workspace."""
    print("\n" + "=" * 60)
    print("Example: Full Configuration")
    print("=" * 60)

    from agentica import SqliteDb

    os.makedirs("tmp", exist_ok=True)
    db = SqliteDb(db_file="tmp/agent_sessions.db")

    agent = DeepAgent(
        # Core
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent",
        description="Full-featured DeepAgent",
        # DeepAgent specific
        work_dir=".",
        include_file_tools=True,
        include_execute=True,
        include_web_search=True,
        include_fetch_url=True,
        include_todos=True,
        include_task=True,
        include_skills=True,
        enable_deep_research=False,
        # Memory & History
        add_history_to_messages=True,
        history_window=4,
        # Tool config
        tool_call_limit=40,
        # Output
        markdown=True,
        debug=True,
    )

    print(f"Tools count: {len(agent.tools)}")
    print(f"Tools: {agent.get_builtin_tool_names()}")

    await agent.print_response("List current directory files")


async def parallel_subagent_example():
    """Example: Using subagents for parallel execution."""
    print("\n" + "=" * 60)
    print("Example: Parallel Subagents")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        include_todos=False,
    )

    print(f"Subagent tool available: {'task' in agent.get_builtin_tool_names()}")

    # The model can launch multiple subagents in parallel
    # Example prompt to trigger parallel subagents:
    await agent.print_response(
        """I need you to do three things in parallel:
        1. Use explore subagent to find all .py files in examples/
        2. Use research subagent to search for "LLM agents 2025"
        3. Use code subagent to create a simple hello.py file

        Launch all three at once using the task tool.""",
        show_tool_calls=True,
    )


if __name__ == '__main__':
    # Run basic example
    asyncio.run(basic_example())
    # asyncio.run(workdir_example())
    # asyncio.run(disable_tools_example())
    # asyncio.run(add_custom_tools_example())
    # asyncio.run(deep_research_example())
    # asyncio.run(full_config_example())
    # asyncio.run(parallel_subagent_example())
