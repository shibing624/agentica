# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Agent Demo — interactive demo of DeepAgent.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agentica import DeepAgent, RunConfig, pprint_run_response
from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig


async def main():
    """Run DeepAgent with sample queries to demonstrate full capabilities."""
    print("=" * 60)
    print("Deep Agent Demo")
    print("=" * 60)

    # Turn on the experience → SKILL.md upgrade pipeline. After a correction
    # has been repeated ~3 times across runs (repeat_count >= min_repeat_count)
    # AND the workspace has at least one tool_recovery event, the experience
    # is compiled into a SKILL.md under
    # ``~/.agentica/workspace/users/<user>/generated_skills/<slug>/``
    # and auto-installed into the skill registry for the next run.
    experience_config = ExperienceConfig(
        capture_tool_errors=True,
        capture_user_corrections=True,
        capture_success_patterns=False,
        skill_upgrade=SkillUpgradeConfig(
            mode="shadow",  # auto-install to generated_skills/
            min_repeat_count=3,
            min_tier="hot",
            min_success_applications=1,
        ),
    )
    agent = DeepAgent(
        debug=True,
        experience_config=experience_config,
    )
    print(f"Model: {agent.model.id}")
    print(f"Tools: {len(agent.get_tools() or [])} loaded")
    while True:
        query = input("Enter your query: ")
        if query.lower() in ["exit", "quit", "bye"]:
            break
        await agent.print_response_stream(query)

    print("Goodbye!")
if __name__ == "__main__":
    asyncio.run(main())
