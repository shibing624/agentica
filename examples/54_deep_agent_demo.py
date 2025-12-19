# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent usage examples

DeepAgent is an enhanced version of Agent that automatically includes built-in tools:
- File system tools: ls, read_file, write_file, edit_file, glob, grep
- Code execution tool: execute
- Web tools: web_search, fetch_url
- Task management tools: write_todos, read_todos
- Subagent tool: task

Difference from regular Agent:
- Agent: Tools must be added manually
- DeepAgent: Automatically includes built-in tools, better suited for complex programming and research tasks
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica import DeepAgent, OpenAIChat


def basic_example():
    """Basic example: Create DeepAgent and execute simple tasks"""
    print("=" * 60)
    print("Basic Example: DeepAgent Basic Usage")
    print("=" * 60)

    # Create DeepAgent with all built-in tools
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="CodingAssistant",
        description="A powerful coding assistant with file system access",
        add_datetime_to_instructions=True,
        debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    # Execute task
    response = agent.run("List all Python files in the current directory and count them")
    print(f"\nResponse:\n{response.content}")


def custom_config_example():
    """Custom configuration example: Disable certain tools"""
    print("\n" + "=" * 60)
    print("Custom Config Example: Disable Code Execution Tool")
    print("=" * 60)

    # Create DeepAgent with code execution disabled (safer)
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="SafeAssistant",
        description="A safe assistant without code execution",
        include_execute=False,  # Disable code execution
        include_web_search=False,  # Disable web search
        debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")


def with_custom_tools_example():
    """Add custom tools example"""
    print("\n" + "=" * 60)
    print("Custom Tools Example: Add Calculator Tool")
    print("=" * 60)

    from agentica.tools.calculator_tool import CalculatorTool

    # Create DeepAgent with additional calculator tool
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="MathAssistant",
        description="A math assistant with calculator",
        tools=[CalculatorTool()],  # Add extra tool
        debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")

    # Execute math calculation task
    response = agent.run("Calculate the factorial of 10")
    print(f"\nResponse:\n{response.content}")


def file_operations_example():
    """File operations example"""
    print("\n" + "=" * 60)
    print("File Operations Example")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="FileManager",
        description="A file management assistant",
        work_dir=".",  # Set working directory
        debug_mode=True,
    )

    # Execute file operation tasks
    response = agent.run("""
    Please do the following:
    1. List all files in the current directory
    2. Find all Python files using glob
    3. Search for 'import' in Python files using grep
    """)
    print(f"\nResponse:\n{response.content}")


def todo_management_example():
    """Task management example"""
    print("\n" + "=" * 60)
    print("Task Management Example")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="TaskManager",
        description="A task management assistant",
        debug_mode=True,
    )

    # Execute task management
    response = agent.run("""
    I need to complete a project. Please create a todo list with the following tasks:
    1. Design the architecture (mark as in_progress)
    2. Implement the core features (pending)
    3. Write tests (pending)
    4. Deploy to production (pending)
    
    Then show me the current todo list.
    """)
    print(f"\nResponse:\n{response.content}")


def compare_agent_vs_deep_agent():
    """Compare Agent and DeepAgent"""
    print("\n" + "=" * 60)
    print("Comparison: Agent vs DeepAgent")
    print("=" * 60)

    from agentica import Agent

    # Regular Agent - no built-in tools
    normal_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="NormalAgent",
    )
    print(f"Normal Agent tools: {normal_agent.tools}")

    # DeepAgent - automatically includes built-in tools
    deep_agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="DeepAgent",
    )
    print(f"DeepAgent builtin tools: {deep_agent.get_builtin_tool_names()}")


if __name__ == '__main__':
    # Run basic example
    basic_example()

    # Run custom config example
    custom_config_example()

    # Run custom tools example
    with_custom_tools_example()

    # Run file operations example
    file_operations_example()

    # Run task management example
    todo_management_example()

    # Compare Agent and DeepAgent
    compare_agent_vs_deep_agent()
