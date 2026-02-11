# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills with Agent Integration Demo

This example shows how to:
1. Match trigger commands and inject skill prompts into Agent
2. Use Agent.add_instruction() for dynamic prompt injection
3. Handle skill-based conversations
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI
from agentica.tools.shell_tool import ShellTool
from agentica.skills import (
    SkillLoader,
    load_skills,
    get_skill_registry,
)


def demo_skill_injection():
    """Demo: Inject skill prompt when trigger is detected."""
    print("=" * 60)
    print("Demo 1: Skill Prompt Injection")
    print("=" * 60)

    # Load all skills
    load_skills()
    registry = get_skill_registry()
    loader = SkillLoader()

    # Create a basic agent
    agent = Agent(
        model=ZhipuAI(model="glm-4-flash"),
        instructions="You are a helpful coding assistant.",
        tools=[ShellTool()],
    )

    # Simulate user input with trigger
    user_input = "/commit fix the login validation bug"

    # Check if input matches a trigger
    matched_skill = registry.match_trigger(user_input)

    if matched_skill:
        print(f"\nDetected skill trigger: {matched_skill.name}")

        # Get the skill prompt
        skill_prompt = matched_skill.get_prompt()
        print(f"Skill prompt loaded ({len(skill_prompt)} chars)")

        # Inject skill prompt into agent using add_instruction()
        agent.add_instruction(skill_prompt)

        # Extract the actual message (remove trigger prefix)
        # e.g., "/commit fix bug" -> "fix bug"
        trigger = matched_skill.trigger
        if trigger and user_input.startswith(trigger):
            actual_message = user_input[len(trigger):].strip()
        else:
            actual_message = user_input

        print(f"\nProcessing: '{actual_message}'")
        # In real usage, you would call: agent.run_sync(actual_message)

    else:
        print(f"\nNo skill trigger detected in: '{user_input}'")
        # Process as normal message


def demo_skill_based_agent():
    """Demo: Create an agent that automatically handles skill triggers."""
    print("\n" + "=" * 60)
    print("Demo 2: Skill-Based Agent")
    print("=" * 60)

    # Load skills
    load_skills()
    registry = get_skill_registry()

    # Get skills summary for base prompt
    skills_summary = registry.get_skills_summary()

    # Create agent with skills awareness
    agent = Agent(
        model=ZhipuAI(model="glm-4-flash"),
        instructions=[
            "You are a helpful coding assistant with skill capabilities.",
            "When users use trigger commands (like /commit or /github), "
            "the corresponding skill instructions will be loaded.",
            "",
            "Available skills:",
            skills_summary,
        ],
        tools=[ShellTool()],
    )

    def process_with_skills(user_message: str):
        """Process a message, handling skill triggers."""
        matched = registry.match_trigger(user_message)

        if matched:
            print(f"[Skill activated: {matched.name}]")

            # Inject skill prompt
            skill_prompt = matched.get_prompt()
            agent.add_instruction(skill_prompt)

            # Extract actual message
            trigger = matched.trigger
            if trigger and user_message.startswith(trigger):
                actual_message = user_message[len(trigger):].strip()
            else:
                actual_message = user_message
        else:
            actual_message = user_message

        # Process the message
        print(f"Message: {actual_message}")
        # response = agent.run_sync(actual_message)
        # return response

    # Test with different inputs
    test_messages = [
        "/commit add user authentication feature",
        "/github pr list --state open",
        "What is the current time?",
    ]

    print("\nProcessing test messages:")
    print("-" * 40)
    for msg in test_messages:
        print(f"\nInput: {msg}")
        process_with_skills(msg)


def demo_custom_skill_with_agent():
    """Demo: Use custom skill with actual agent execution."""
    print("\n" + "=" * 60)
    print("Demo 3: Custom Skill with Agent Execution")
    print("=" * 60)

    from agentica.skills import register_skill

    # Register custom skill
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    custom_skill_dir = os.path.join(pwd_path, "../data/skills/python-lib-analyzer")

    if os.path.exists(custom_skill_dir):
        skill = register_skill(custom_skill_dir, location="project")
        if skill:
            print(f"Loaded custom skill: {skill.name}")

            # Create agent with custom skill prompt
            agent = Agent(
                model=ZhipuAI(model="glm-4.7-flash"),
                instructions=[
                    "You are a Python library expert.",
                    "",
                    "# Skill Instructions",
                    skill.get_prompt(),
                ],
                tools=[ShellTool()],
            )
            response = agent.run_sync("What are the main modules in numpy ?")
            print(response.content)
    else:
        print(f"Custom skill directory not found: {custom_skill_dir}")


def demo_list_triggers():
    """Demo: List all available triggers for user reference."""
    print("\n" + "=" * 60)
    print("Demo 4: Available Triggers")
    print("=" * 60)

    load_skills()
    registry = get_skill_registry()

    print("\nAvailable slash commands:")
    triggers = registry.list_triggers()

    if triggers:
        for trigger, skill_name in sorted(triggers.items()):
            skill = registry.get(skill_name)
            if skill:
                desc = skill.description[:50] + "..." if len(skill.description) > 50 else skill.description
                print(f"  {trigger:15} - {desc}")
    else:
        print("  No triggers available")

    print("\nUsage examples:")
    print("  /commit fix authentication bug")
    print("  /github pr create --title 'New feature'")
    print("  /github issue list --state open")


if __name__ == "__main__":
    demo_skill_injection()
    demo_skill_based_agent()
    demo_custom_skill_with_agent()
    demo_list_triggers()
