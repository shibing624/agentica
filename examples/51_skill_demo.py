# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo - Using SkillTool

Agent Skill is a prompt-based skill system that extends agent capabilities.
Skills are not code-level extensions, but text instructions injected into
the system prompt, allowing the LLM to read and follow instructions.

This demo shows how to use the SkillTool to load and execute skills.

Usage:
    python 51_skill_demo.py

Reference: https://claude.com/blog/skills
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.tools.skill_tool import SkillTool
from agentica.tools.shell_tool import ShellTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool


async def main() -> None:
    """The main entry point for the skill agent example."""
    # Get the custom skill directory
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/analyzing-py-lib"
    )

    # Create SkillTool with custom skill directory
    # SkillTool auto-loads skills from standard directories:
    # - .claude/skills (project-level)
    # - .agentica/skills (project-level)
    # - ~/.claude/skills (user-level)
    # - ~/.agentica/skills (user-level)
    # Plus any custom directories you specify
    skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

    # Print available skills
    print("=" * 60)
    print("Available Skills:")
    print("=" * 60)
    print(skill_tool.list_skills())
    print()

    # Print the skill tool system prompt
    print("=" * 60)
    print("Skill Tool System Prompt (injected into agent):")
    print("=" * 60)
    system_prompt = skill_tool.get_system_prompt()
    if system_prompt:
        print(system_prompt[:1000] + "..." if len(system_prompt) > 1000 else system_prompt)
    print()

    # Create agent with SkillTool
    # The SkillTool's system prompt is automatically injected
    agent = Agent(
        name="Skill-Enabled Agent",
        description="An AI assistant with skill capabilities.",
        instructions=[
            "You are a helpful assistant that can use skills to answer questions.",
            "Use execute_skill(skill_name) to load a skill's instructions.",
            "Use list_skills() to see all available skills.",
            "Use list_skill_files(directory) to explore skill resources.",
            "Use read_skill_file(file_path) to read skill files.",
        ],
        tools=[skill_tool, ShellTool(), RunPythonCodeTool()],
        show_tool_calls=True,
    )

    # Example: Explore skill directory
    print("=" * 60)
    print("Exploring skill directory:")
    print("=" * 60)
    print(skill_tool.list_skill_files(skill_dir))
    print()

    # Example: Read SKILL.md
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    print("=" * 60)
    print("SKILL.md content preview:")
    print("=" * 60)
    content = skill_tool.read_skill_file(skill_md_path)
    print(content[:500] + "..." if len(content) > 500 else content)
    print()

    # Ask questions
    print("=" * 60)
    print("Demo Questions:")
    print("=" * 60)

    # Ask about available skills
    print("\nQuestion 1: What skills do you have?")
    response = await agent.arun("What skills do you have? Use list_skills() to check.")
    print(f"Response: {response.content}\n")

    # Ask a specific question
    print("\nQuestion 2: How to create a custom tool function?")
    response = await agent.arun(
        "How to create a custom tool function for the agent in agentica? "
        "First execute the 'Analyzing Agentica Library' skill if available."
    )
    print(f"Response: {response.content}\n")


def run_sync_demo():
    """Synchronous demo showing SkillTool usage."""
    print("\n" + "=" * 60)
    print("Synchronous Demo - SkillTool Basic Usage")
    print("=" * 60)

    # Create SkillTool
    skill_tool = SkillTool()

    # List all skills
    print("\nAll available skills:")
    print(skill_tool.list_skills())

    # Add a custom skill
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/analyzing-py-lib"
    )
    if os.path.exists(skill_dir):
        print(f"\nAdding custom skill from: {skill_dir}")
        skill = skill_tool.add_skill_dir(skill_dir)
        if skill:
            print(f"  Added: {skill.name}")

            # Execute the skill
            print(f"\nExecuting skill: {skill.name}")
            result = skill_tool.execute_skill(skill.name)
            print(result[:500] + "..." if len(result) > 500 else result)


if __name__ == "__main__":
    # Run sync demo first
    run_sync_demo()

    # Run async demo
    print("\n" + "=" * 60)
    print("Running Async Demo...")
    print("=" * 60)
    asyncio.run(main())
