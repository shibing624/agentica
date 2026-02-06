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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, Agent, OpenAIChat


def basic_example():
    """Basic example: Create DeepAgent and execute simple tasks."""
    print("=" * 60)
    print("Basic Example: DeepAgent Basic Usage")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="CodingAssistant",
        description="A powerful coding assistant with file system access",
        add_datetime_to_instructions=True,
        # debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    response = agent.run("List all Python files in the current directory and count them")
    print(f"\nResponse:\n{response.content}")


def custom_config_example():
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
        # debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")


def with_custom_tools_example():
    """Add custom tools example."""
    print("\n" + "=" * 60)
    print("Custom Tools Example: Add Calculator Tool")
    print("=" * 60)

    from agentica.tools.calculator_tool import CalculatorTool

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="MathAssistant",
        description="A math assistant with calculator",
        tools=[CalculatorTool()],
        # debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    agent.print_response("Calculate (1111.2 * 22.1222) + (3333.3 / 4444.43)=?")


def complex_deep_agent():
    """DeepAgent."""
    from agentica import SqliteDb
    print("\n" + "=" * 60)
    print("Complex DeepAgent Example: Create a complex DeepAgent")
    print("=" * 60)
    db_path = "tmp/agent_sessions.db"
    db = SqliteDb(db_file=str(db_path))

    deep_agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent",
        db=db,
        # 工作空间配置（配置层）
        load_workspace_context=True,
        load_workspace_memory=True,
        memory_days=7,
        # 历史记录配置
        add_history_to_messages=True,
        num_history_responses=4,
        # 工具配置
        show_tool_calls=True,
        tool_call_limit=40,
        # 指令
        add_datetime_to_instructions=True,
        auto_load_mcp=True,
        # 调试
        debug_mode=True,
    )
    print(f"DeepAgent builtin tools: {deep_agent.get_builtin_tool_names()}")
    print(f"DeepAgent: {deep_agent}")
    deep_agent.print_response("List all Python files in the current directory and count them")
    while True:
        user_input = input("Enter your command (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        deep_agent.print_response(user_input)
        print("-" * 60)
        print("\n")



if __name__ == '__main__':
    basic_example()
    # custom_config_example()
    # with_custom_tools_example()
    complex_deep_agent()
