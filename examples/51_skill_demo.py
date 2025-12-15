# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo - Async version

Agent Skill is a prompt-based skill system that extends agent capabilities.
Skills are not code-level extensions, but text instructions injected into
the system prompt, allowing the LLM to read and follow instructions.

This demo shows how to use the "Analyzing Agentica Library" skill to
answer questions about the agentica library.

Reference: https://claude.com/blog/skills
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, SkillTool
from agentica.tools.shell_tool import ShellTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool


async def main() -> None:
    """The main entry point for the skill agent example."""
    # Initialize SkillTool
    skill_tool = SkillTool()

    # Register the analyzing-py-lib skill
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/analyzing-py-lib"
    )
    skill_tool.register_skill(skill_dir)

    print("Registered skills:", skill_tool.list_skills())
    print("\n" + "=" * 60)

    # Print the skill prompt
    print("\nSkill Prompt (will be added to agent instructions):")
    print("-" * 60)
    skill_prompt = skill_tool.get_skill_prompt()
    if skill_prompt:
        print(skill_prompt)
    print("-" * 60)

    # Create agent with skills
    # SkillTool includes built-in file reading capabilities (read_file, list_files, list_skills)
    # Add ShellTool and RunPythonCodeTool for executing skill scripts
    agent = Agent(
        name="Agentica Expert",
        description="An AI assistant specialized in the Agentica library.",
        instructions=[
            "You are a helpful assistant that can answer questions about the Agentica library.",
            "Don't make any assumptions. All your knowledge about Agentica library must come from your equipped skills.",
            "When asked about Agentica, first check your available skills and use them to find accurate information.",
            skill_prompt,  # Add skill prompt to instructions
        ],
        # SkillTool includes read_file/list_files, combine with ShellTool and RunPythonCodeTool
        tools=[skill_tool, ShellTool(), RunPythonCodeTool()],
        show_tool_calls=True,
    )

    # Example questions
    print("\n" + "=" * 60)
    print("Demo Questions:")
    print("=" * 60)

    # Ask about available skills
    print("\nQuestion 1: What skills do you have?")
    response = await agent.arun("What skills do you have?")
    print(f"Response: {response.content}\n")

    # Ask a specific question about agentica
    print("\nQuestion 2: How to create a custom tool function?")
    response = await agent.arun("How to create a custom tool function for the agent in agentica?")
    print(f"Response: {response.content}\n")


def run_sync_demo():
    """Synchronous demo showing skill registration and prompt generation."""
    print("\n" + "=" * 60)
    print("Synchronous Demo - Testing view_agentica_module.py")
    print("=" * 60)

    # Test the view_agentica_module.py script
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/analyzing-py-lib"
    )
    view_module_path = os.path.join(skill_dir, "view_agentica_module.py")

    if os.path.exists(view_module_path):
        print(f"\nRunning: python {view_module_path} --module agentica")
        print("-" * 60)

        # Import and run the view function
        import importlib.util
        spec = importlib.util.spec_from_file_location("view_agentica_module", view_module_path)
        view_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(view_module)

        # View top-level modules
        result = view_module.view_agentica_library("agentica")
        print(result)

        print("\n" + "-" * 60)
        print("Running: python view_agentica_module.py --module agentica.Agent")
        print("-" * 60)

        # View Agent class
        result = view_module.view_agentica_library("agentica.Agent")
        print(result)


if __name__ == "__main__":
    # Run async demo
    asyncio.run(main())

    # Run sync demo to test the skill script
    run_sync_demo()
