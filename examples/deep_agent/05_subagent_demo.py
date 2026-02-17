# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Subagent Demo - Demonstrates using different subagent types

This example shows how to use the subagent system to:
1. Spawn explore subagents for codebase exploration (read-only)
2. Spawn general subagents for complex multi-step tasks
3. Spawn research subagents for web research
4. Register custom subagent types
5. Use parallel subagent execution for independent tasks

Built-in Subagent Types:
- explore: Read-only codebase explorer (fast, low context usage)
- general: Full capabilities for complex tasks
- research: Web search and document analysis
- code: Code generation and execution

Custom Subagent Types:
- Users can register custom subagent types via register_custom_subagent()
- Custom types are automatically available in the task tool prompt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat
from agentica.subagent import (
    SubagentRegistry,
    get_available_subagent_types,
    get_subagent_config,
    register_custom_subagent,
    get_custom_subagent_configs,
)


def show_available_subagent_types():
    """Display available subagent types and their capabilities."""
    print("=" * 60)
    print("Available Subagent Types")
    print("=" * 60)
    
    for st in get_available_subagent_types():
        custom_flag = " (custom)" if st.get("is_custom") else ""
        print(f"\n[{st['type']}] {st['name']}{custom_flag}")
        print(f"  {st['description'][:100]}...")
    print()


def demo_custom_subagent():
    """
    Demo: Registering and using a custom subagent type.
    
    This demonstrates how users can define their own subagent types
    without modifying the library code. Custom subagents are automatically
    included in the task tool's system prompt.
    """
    print("\n" + "=" * 60)
    print("Demo 1: Custom Subagent Registration")
    print("=" * 60)
    
    # Register a custom subagent for code review
    register_custom_subagent(
        name="code-reviewer",
        description="Reviews code for quality, bugs, security issues, and best practices",
        system_prompt="""You are an expert code reviewer. Your job is to:
1. Analyze code for potential bugs and issues
2. Check for security vulnerabilities
3. Suggest improvements for readability and maintainability
4. Verify adherence to best practices

Provide clear, actionable feedback with specific line references when possible.""",
        allowed_tools=["read_file", "ls", "glob", "grep"],
        tool_call_limit=10,
    )
    
    # Show that the custom subagent is now available
    print("\nAfter registration, available subagent types:")
    for st in get_available_subagent_types():
        custom_flag = " (custom)" if st.get("is_custom") else ""
        print(f"  - {st['type']}: {st['name']}{custom_flag}")
    
    # Show custom configs
    custom_configs = get_custom_subagent_configs()
    print(f"\nCustom subagent configs registered: {list(custom_configs.keys())}")


def demo_explore_subagent():
    """
    Demo: Using explore subagent for codebase exploration.
    
    The explore subagent is read-only and specialized for:
    - Finding files using glob patterns
    - Searching code with regex
    - Reading and analyzing source code
    """
    print("\n" + "=" * 60)
    print("Demo 2: Explore Subagent (Codebase Exploration)")
    print("=" * 60)
    
    # Get explore subagent config to show its capabilities
    config = get_subagent_config("explore")
    print(f"\nExplore Agent Config:")
    print(f"  - Allowed tools: {config.allowed_tools}")
    print(f"  - Tool call limit: {config.tool_call_limit}")
    print(f"  - Can spawn subagents: {config.can_spawn_subagents}")
    
    # Create DeepAgent with task tool
    agent = DeepAgent(
        model=OpenAIChat(),
        name="Explorer",
        include_task=True,
        work_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    
    # Ask the agent to explore the codebase
    query = """Use the task tool with subagent_type='explore' to find all Python files 
    in the agentica/tools directory that contain the word 'Tool' in their class names.
    Return a summary of what tools are available."""
    
    print(f"\nUser Query: {query}")
    print("\nAgent Response:")
    response = agent.run_sync(query)
    print(response.content if response else "No response")


def demo_parallel_subagents():
    """
    Demo: Using multiple subagents in parallel.
    
    This demonstrates launching multiple independent subagents
    to work on different aspects of a task simultaneously.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Parallel Subagent Execution")
    print("=" * 60)
    
    agent = DeepAgent(
        model=OpenAIChat(),
        name="Coordinator",
        include_task=True,
        work_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    
    # Ask the agent to use multiple subagents in parallel
    query = """I need to understand this project better. Please launch explore subagents 
    IN PARALLEL (multiple task tool calls in one response) to:
    
    1. Find and summarize the main Agent class structure
    2. List all available built-in tools
    3. Find how the memory system works
    
    After getting results from all subagents, synthesize the findings into a brief summary."""
    
    print(f"\nUser Query: {query}")
    print("\nAgent Response:")
    response = agent.run_sync(query)
    print(response.content if response else "No response")


def demo_research_subagent():
    """
    Demo: Using research subagent for web research.
    
    The research subagent is specialized for:
    - Web search
    - Fetching and analyzing web pages
    - Synthesizing research findings
    """
    print("\n" + "=" * 60)
    print("Demo 4: Research Subagent (Web Research)")
    print("=" * 60)
    
    # Get research subagent config
    config = get_subagent_config("research")
    print(f"\nResearch Agent Config:")
    print(f"  - Allowed tools: {config.allowed_tools}")
    print(f"  - Tool call limit: {config.tool_call_limit}")
    
    agent = DeepAgent(
        model=OpenAIChat(),
        name="Researcher",
        include_task=True,
        include_web_search=True,
        include_fetch_url=True,
    )
    
    query = """Use the task tool with subagent_type='research' to find information about 
    the latest developments in AI agent frameworks (like LangChain, AutoGPT, etc.).
    Return a brief summary of the top 3 frameworks and their key features."""
    
    print(f"\nUser Query: {query}")
    print("\nAgent Response:")
    response = agent.run_sync(query)
    print(response.content if response else "No response")


def demo_subagent_registry():
    """
    Demo: Using SubagentRegistry to track subagent runs.
    
    The registry keeps track of all subagent executions,
    allowing monitoring and management of background tasks.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Subagent Registry")
    print("=" * 60)
    
    registry = SubagentRegistry()
    
    # Get all runs (from previous demos if any)
    all_runs = list(registry._runs.values())
    print(f"\nTotal subagent runs tracked: {len(all_runs)}")
    
    if all_runs:
        print("\nRecent runs:")
        for run in all_runs[-5:]:
            status_icon = "[OK]" if run.status == "completed" else "[ERR]" if run.status == "error" else "[...]"
            print(f"  {status_icon} [{run.subagent_type.value}] {run.task_label}")
            print(f"       Status: {run.status}, Started: {run.started_at.strftime('%H:%M:%S')}")
    
    # Cleanup old completed runs
    cleaned = registry.cleanup_completed(max_age_seconds=3600)
    print(f"\nCleaned up {cleaned} old runs")


def main():
    """Run all demos."""
    # Show available subagent types first
    show_available_subagent_types()
    
    # Run demos (comment out ones you don't want to run)
    print("\n" + "=" * 60)
    print("Running Subagent Demos")
    print("=" * 60)
    
    # Demo 1: Custom subagent registration (no LLM call, just shows registration)
    # demo_custom_subagent()
    
    # Demo 2: Explore subagent
    demo_explore_subagent()
    
    # Demo 3: Parallel subagents
    demo_parallel_subagents()
    
    # Demo 4: Research subagent
    demo_research_subagent()
    
    # Demo 5: Subagent registry
    demo_subagent_registry()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
