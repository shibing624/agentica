# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills system usage example

This example shows how to:
1. Load skills from directories
2. Match trigger commands
3. Get skill prompts for injection
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI
from agentica.skills import (
    SkillLoader,
    get_skill_registry,
    load_skills,
    get_available_skills,
)


def main():
    print("=== Loading Skills ===")

    # Load all skills from standard directories
    registry = load_skills()

    # List available skills
    skills = get_available_skills()
    print(f"\nFound {len(skills)} skills:")
    for skill in skills:
        trigger_info = f" (trigger: {skill.trigger})" if skill.trigger else ""
        print(f"  - {skill.name}: {skill.description}{trigger_info}")

    # Get skills summary for system prompt
    print("\n=== Skills Summary ===")
    summary = registry.get_skills_summary()
    print(summary)

    # Match trigger commands
    print("\n=== Trigger Matching ===")
    test_inputs = [
        "/commit fix bug",
        "/github pr list",
        "regular message",
    ]

    for text in test_inputs:
        matched = registry.match_trigger(text)
        if matched:
            print(f"'{text}' -> matches skill: {matched.name}")
        else:
            print(f"'{text}' -> no match")

    # List all registered triggers
    print("\n=== Registered Triggers ===")
    triggers = registry.list_triggers()
    for trigger, skill_name in triggers.items():
        print(f"  {trigger} -> {skill_name}")

    # Get skill instruction for agent
    print("\n=== Skill Instructions ===")
    instructions = registry.get_skill_instruction()
    print(instructions[:500] + "..." if len(instructions) > 500 else instructions)

    # Example: Create agent with skills
    print("\n=== Creating Agent with Skills ===")
    agent = Agent(
        model=ZhipuAI(model="glm-4-flash"),
        instructions=f"You are a helpful assistant.\n\n{registry.get_skills_summary()}",
    )

    # Run with skill context (manually inject skill prompt when triggered)
    user_input = "/github pr view 123"
    matched_skill = registry.match_trigger(user_input)

    if matched_skill:
        print(f"Detected skill: {matched_skill.name}")
        # Get skill prompt and inject it
        skill_prompt = matched_skill.get_prompt()
        print(f"Skill prompt preview: {skill_prompt[:200]}...")


if __name__ == "__main__":
    main()
