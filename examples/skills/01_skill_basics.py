# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo - Using SkillTool

Agent Skill is a prompt-based skill system that extends agent capabilities.
Skills are text instructions injected into the system prompt, allowing the LLM
to read and follow instructions.

Reference: https://claude.com/blog/skills
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.tools.skill_tool import SkillTool
from agentica.tools.shell_tool import ShellTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool

pwd_path = os.path.dirname(os.path.abspath(__file__))


def run_sync_demo():
    """Synchronous demo showing SkillTool usage."""
    print("\n" + "=" * 60)
    print("Synchronous Demo - SkillTool Basic Usage")
    print("=" * 60)

    skill_tool = SkillTool()

    print("\nAll available skills:")
    print(skill_tool.list_skills())

    # Add a custom skill
    skill_dir = os.path.join(pwd_path, "../data/skill/analyzing-py-lib")
    if os.path.exists(skill_dir):
        print(f"\nAdding custom skill from: {skill_dir}")
        skill = skill_tool.add_skill_dir(skill_dir)
        if skill:
            print(f"  Added: {skill.name}")
            # Show skill info instead of execute_skill (which is now injected via system prompt)
            result = skill_tool.get_skill_info(skill.name)
            print(result[:500] + "..." if len(result) > 500 else result)


async def main() -> None:
    """The main entry point for the skill agent example."""
    # Get the custom skill directory
    skill_dir = os.path.join(pwd_path, "../data/skill/analyzing-py-lib")

    # Create SkillTool with custom skill directory
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
    # Note: Skill prompts are automatically injected into the system prompt via get_system_prompt()
    agent = Agent(
        name="Skill-Enabled Agent",
        description="An AI assistant with skill capabilities.",
        instructions=[
            "You are a helpful assistant that can use skills to answer questions.",
            "Skill instructions are automatically loaded into the system prompt.",
            "Use list_skills() to see all available skills.",
            "Use get_skill_info(skill_name) to get details about a specific skill.",
        ],
        tools=[skill_tool, ShellTool(), RunPythonCodeTool()],
        show_tool_calls=True,
    )

    # Ask questions
    print("=" * 60)
    print("Demo Questions:")
    print("=" * 60)

    print("\nQuestion 1: What skills do you have?")
    response = await agent.arun("What skills do you have? Use list_skills() to check.")
    print(f"Response: {response.content}\n")


if __name__ == "__main__":
    run_sync_demo()
    
    print("\n" + "=" * 60)
    print("Running Async Demo...")
    print("=" * 60)
    asyncio.run(main())
