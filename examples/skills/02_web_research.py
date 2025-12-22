# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo: web-research with SkillTool

This demo shows how to use the SkillTool with DeepAgent for web research tasks.
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat
from agentica.tools.skill_tool import SkillTool

pwd_path = os.path.dirname(os.path.abspath(__file__))


def run_sync_demo():
    """Synchronous demo showing SkillTool with web-research skill."""
    print("\n" + "=" * 60)
    print("Synchronous Demo - SkillTool with web-research")
    print("=" * 60)

    skill_dir = os.path.join(pwd_path,  "../data/skill/web-research")
    skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

    print("\nSkill info:")
    result = skill_tool.get_skill_info("web-research")
    print(result)

async def main() -> None:
    """The main entry point for the skill agent example."""
    # Get the web-research skill directory
    skill_dir = os.path.join(pwd_path, "../data/skill/web-research")

    # Create SkillTool with custom skill directory
    skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

    # Print available skills
    print("=" * 60)
    print("Available Skills:")
    print("=" * 60)
    print(skill_tool.list_skills())
    print()

    # Print skill info
    print("=" * 60)
    print("Skill Info:")
    print("=" * 60)
    print(skill_tool.get_skill_info("web-research"))
    print()

    # Create DeepAgent with SkillTool
    # Note: Skill prompts are automatically injected into the system prompt via get_system_prompt()
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent-web-research",
        instructions=[
            "You are a research assistant with skill capabilities.",
            "Skill instructions are automatically loaded into the system prompt.",
            "Use the web-research skill instructions to conduct thorough research on topics.",
        ],
        tools=[skill_tool],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
    )

    print(f"Agent tools: {agent.tools}")
    print()

    # Show system prompt (skill instructions are injected)
    print("\nSkill instructions are now injected into the system prompt automatically.")
    prompt = skill_tool.get_system_prompt()
    if prompt:
        print(f"System prompt preview: {prompt[:500]}...")

    # Research question
    question = "帮我调研可再生能源的环境影响，写出简短的中文调研报告。"
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    response = await agent.arun(question)
    print(f"\nResponse: {response.content}")


if __name__ == "__main__":
    run_sync_demo()
    
    print("\n" + "=" * 60)
    print("Running Async Demo...")
    print("=" * 60)
    asyncio.run(main())
