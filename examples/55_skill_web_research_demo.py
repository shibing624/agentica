# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo: web-research with SkillTool

This demo shows how to use the SkillTool with DeepAgent for web research tasks.
The skill provides instructions for conducting web research.

Usage:
    python 55_skill_web_research_demo.py
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import DeepAgent, OpenAIChat
from agentica.tools.skill_tool import SkillTool


async def main() -> None:
    """The main entry point for the skill agent example."""
    # Get the web-research skill directory
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/web-research"
    )

    # Create SkillTool with custom skill directory
    # If use standard directories, cp -rf data/skill/* ~/.agentica/skill/
    # SkillTool auto-loads skills from standard directories plus custom ones
    skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

    # Print available skills
    print("=" * 60)
    print("Available Skills:")
    print("=" * 60)
    print(skill_tool.list_skills())
    print()

    # Explore skill directory using skill-specific tool
    print("=" * 60)
    print("Skill directory contents:")
    print("=" * 60)
    print(skill_tool.list_skill_files(skill_dir))
    print()

    # Read the SKILL.md file using skill-specific tool
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    print("=" * 60)
    print("SKILL.md content preview:")
    print("=" * 60)
    content = skill_tool.read_skill_file(skill_md_path)
    print(content[:800] + "..." if len(content) > 800 else content)
    print()

    # Create DeepAgent with SkillTool
    # DeepAgent has built-in tools for file operations, web search, etc.
    # SkillTool adds skill execution capabilities
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent-web-research",
        instructions=[
            "You are a research assistant with skill capabilities.",
            "Use execute_skill(skill_name) to load a skill's instructions.",
            "Use list_skills() to see all available skills.",
            "Use the web-research skill to conduct thorough research on topics.",
        ],
        tools=[skill_tool],  # Add SkillTool to DeepAgent
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        # debug_mode=True,
    )

    print(f"Agent tools: {agent.tools}")
    print()

    # First, let's execute the web-research skill
    print("\nExecuting web-research skill...")
    result = skill_tool.execute_skill("web-research")
    print(result)

    # Research question
    question = "帮我调研各种可再生能源的环境影响，并写出一份详尽的中文调研报告。"
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    # Run the agent
    response = await agent.arun(question)
    print(f"\nResponse: {response.content}")


def run_sync_demo():
    """Synchronous demo showing SkillTool with web-research skill."""
    print("\n" + "=" * 60)
    print("Synchronous Demo - SkillTool with web-research")
    print("=" * 60)

    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/web-research"
    )

    # Create SkillTool
    skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

    # Get skill info
    print("\nSkill info:")
    result = skill_tool.get_skill_info("web-research")
    print(result)


if __name__ == "__main__":
    # Run sync demo first
    run_sync_demo()

    # Run async demo
    print("\n" + "=" * 60)
    print("Running Async Demo...")
    print("=" * 60)
    asyncio.run(main())
