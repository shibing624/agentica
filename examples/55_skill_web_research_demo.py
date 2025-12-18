# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo: web-research
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import DeepAgent, SkillTool, OpenAIChat


async def main() -> None:
    """The main entry point for the skill agent example."""
    # Initialize SkillTool
    skill_tool = SkillTool()

    # Register the analyzing-py-lib skill
    skill_dir = os.path.join(
        os.path.dirname(__file__),
        "data/skill/web-research"
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
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepAgent-web-research",
        instructions=[
            "可以用web research深度分析任何主题。",
            skill_prompt,  # Add skill prompt to instructions
        ],
        tools=[skill_tool],
        show_tool_calls=True,
        # enable_multi_round=True,
        # debug_mode=True,
    )
    question = "用web-research深度分析英伟达的股价走势"
    print(f"Question: {question}\n")
    response = await agent.arun(question)
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
