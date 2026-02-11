# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent basic usage examples

DeepAgent is an enhanced version of Agent that automatically includes built-in tools:
- File system tools: ls, read_file, write_file, edit_file, glob, grep
- Code execution tool: execute
- Web tools: web_search, fetch_url
- Task management tools: write_todos, read_todos
- Subagent tool: task

Difference from regular Agent:
- Agent: Tools must be added manually
- DeepAgent: Automatically includes built-in tools
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


async def basic_example():
    """Basic example: Create DeepAgent and execute simple tasks."""
    print("=" * 60)
    print("Basic Example: DeepAgent Basic Usage")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        add_datetime_to_instructions=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    await agent.print_response_stream(
        "List all Python files in the current directory and count them",
        show_tool_calls=True,
    )


async def custom_config_example():
    """Custom configuration example: Disable certain tools."""
    print("\n" + "=" * 60)
    print("Custom Config Example: Disable Code Execution Tool")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="SafeAssistant",
        description="A safe assistant without code execution",
        include_execute=False,
        include_web_search=False,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")


async def with_custom_tools_example():
    """Add custom tools alongside built-in tools."""
    print("\n" + "=" * 60)
    print("Custom Tools Example: Add Custom Function Tool")
    print("=" * 60)

    def multiply(a: float, b: float) -> str:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            The product as a string
        """
        return str(a * b)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="MathAssistant",
        description="A math assistant with custom tools",
        tools=[multiply],
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    await agent.print_response("Calculate (1111.2 * 22.1222) using the multiply tool")


async def complex_deep_agent():
    """DeepAgent with full configuration."""
    from agentica import SqliteDb

    print("\n" + "=" * 60)
    print("Complex DeepAgent Example")
    print("=" * 60)

    db_path = "tmp/agent_sessions.db"
    os.makedirs("tmp", exist_ok=True)
    db = SqliteDb(db_file=str(db_path))

    deep_agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent",
        db=db,
        load_workspace_context=True,
        load_workspace_memory=True,
        memory_days=7,
        add_history_to_messages=True,
        num_history_responses=4,
        tool_call_limit=40,
        add_datetime_to_instructions=True,
        auto_load_mcp=True,
        debug_mode=True,
    )
    print(f"DeepAgent builtin tools: {deep_agent.get_builtin_tool_names()}")
    print(f"DeepAgent: {deep_agent}")
    await deep_agent.print_response("List all Python files in the current directory and count them")


if __name__ == '__main__':
    asyncio.run(basic_example())
    asyncio.run(custom_config_example())
    asyncio.run(with_custom_tools_example())
    asyncio.run(complex_deep_agent())
