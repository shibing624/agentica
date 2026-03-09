# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent with Built-in Tools usage examples

Agent + get_builtin_tools() provides:
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

from agentica import Agent, OpenAIChat
from agentica.tools.buildin_tools import get_builtin_tools
from agentica.agent.config import ToolConfig, PromptConfig


async def basic_example():
    """Basic example: Create Agent with built-in tools."""
    print("=" * 60)
    print("Basic Example: Agent with Built-in Tools")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=get_builtin_tools(),
        prompt_config=PromptConfig(enable_agentic_prompt=True),
    )

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

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=get_builtin_tools(),
        work_dir="/path/to/project",
    )

    print(f"work_dir: {agent.work_dir}")


async def selective_tools_example():
    """Example: Select specific built-in tools."""
    print("\n" + "=" * 60)
    print("Example: Selective Built-in Tools")
    print("=" * 60)

    # Only include file tools and web search (no execute, no todos, no task)
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="SafeExplorer",
        description="Read-only file explorer",
        tools=get_builtin_tools(
            include_execute=False,
            include_web_search=False,
            include_todos=False,
            include_task=False,
        ),
    )

    print(f"Tools: {[t.name for t in agent.tools]}")


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

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Assistant",
        tools=get_builtin_tools() + [multiply, weather],
    )

    print(f"Total tools: {len(agent.tools)}")

    await agent.print_response("What is 123 * 456? What's the weather in Tokyo?")


async def web_research_example():
    """Example: Web research with Agent."""
    print("\n" + "=" * 60)
    print("Example: Web Research")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=get_builtin_tools(include_todos=False),
    )

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

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="FullAgent",
        description="Full-featured Agent with built-in tools",
        work_dir=".",
        tools=get_builtin_tools(),
        add_history_to_messages=True,
        history_window=4,
        tool_config=ToolConfig(tool_call_limit=40),
        prompt_config=PromptConfig(markdown=True, enable_agentic_prompt=True),
        debug=True,
    )

    print(f"Tools count: {len(agent.tools)}")

    await agent.print_response("List current directory files")


async def parallel_subagent_example():
    """Example: Using subagents for parallel execution."""
    print("\n" + "=" * 60)
    print("Example: Parallel Subagents")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=get_builtin_tools(include_todos=False),
    )

    # The model can launch multiple subagents in parallel
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
    # asyncio.run(selective_tools_example())
    # asyncio.run(add_custom_tools_example())
    # asyncio.run(web_research_example())
    # asyncio.run(full_config_example())
    # asyncio.run(parallel_subagent_example())
