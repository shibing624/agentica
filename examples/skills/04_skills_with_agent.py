# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills with Agent Integration Demo

This example shows how to:
1. Match trigger commands and inject skill prompts into Agent
2. Use SkillTool(custom_skill_dirs=...) to load SKILL.md folders at runtime
3. Use Agent.add_instruction() for dynamic prompt injection
4. Handle skill-based conversations
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAIChat
from agentica.tools.skill_tool import SkillTool
from agentica.tools.shell_tool import ShellTool
from agentica.skills import (
    SkillLoader,
    load_skills,
    get_skill_registry,
    reset_skill_registry,
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
        model=ZhipuAIChat(model="glm-4-flash"),
        instructions="You are a helpful coding assistant.",
        tools=[ShellTool()],
    )

    # Simulate user input with trigger
    user_input = "/commit fix the login validation bug"

    # Check if input matches a trigger
    matched_skill = registry.match_trigger(user_input)

    if matched_skill:
        print(f"\nDetected skill trigger: {matched_skill.name}")

        # Extract the arguments that followed the trigger
        # e.g., "/commit fix bug" -> "fix bug"
        trigger = matched_skill.trigger
        if trigger and user_input.startswith(trigger):
            arguments = user_input[len(trigger):].strip()
        else:
            arguments = user_input

        # render_invocation() combines the skill body with the arguments and
        # states that the arguments are the workflow's target, not a separate
        # instruction. Sending the bare arguments instead lets the model read
        # something like "git status的代码" as a task of its own.
        message = matched_skill.render_invocation(arguments)
        print(f"Skill prompt loaded ({len(message)} chars)")

        print(f"\nProcessing arguments: '{arguments}'")
        # In real usage, you would call: agent.run_sync(message)

    else:
        print(f"\nNo skill trigger detected in: '{user_input}'")
        # Process as normal message


def demo_skill_tool_custom_dirs():
    """Demo: Load a folder of SKILL.md packages through SkillTool."""
    print("\n" + "=" * 60)
    print("Demo 2: SkillTool custom_skill_dirs")
    print("=" * 60)

    reset_skill_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_root = Path(tmpdir) / "my-skills"
        skill_dir = skills_root / "python-style-guide"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
name: python-style-guide
description: Apply a concise Python coding style guide.
when-to-use: python, style, code review
---

Prefer small functions, explicit names, and direct control flow.

## Gotchas
- Avoid broad try/except blocks that hide failures.
- Keep imports at module top unless a lazy import is necessary.

## Minimal Example
```python
def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")
```
""",
            encoding="utf-8",
        )

        # Point at the parent folder. SkillTool discovers any child directory
        # that contains SKILL.md, matching the common "drop in a skill folder"
        # workflow used by coding agents.
        skill_tool = SkillTool(custom_skill_dirs=[str(skills_root)])
        skill_tool.initialize()

        agent = Agent(
            model=ZhipuAIChat(model="glm-4-flash"),
            instructions="You are a Python coding assistant.",
            tools=[skill_tool, ShellTool()],
        )

        print(f"Agent created: {agent.name or 'unnamed'}")
        print("\nRuntime skills:")
        print(skill_tool.list_skills())
        print("\nLoaded skill instructions:")
        print(skill_tool.get_skill_info("python-style-guide"))
    reset_skill_registry()


def demo_skill_based_agent():
    """Demo: Create an agent that automatically handles skill triggers."""
    print("\n" + "=" * 60)
    print("Demo 3: Skill-Based Agent")
    print("=" * 60)

    # Load skills
    load_skills()
    registry = get_skill_registry()

    # Get skills summary for base prompt
    skills_summary = registry.get_skills_summary()

    # Create agent with skills awareness
    agent = Agent(
        model=ZhipuAIChat(model="glm-4-flash"),
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

            trigger = matched.trigger
            if trigger and user_message.startswith(trigger):
                arguments = user_message[len(trigger):].strip()
            else:
                arguments = user_message
            actual_message = matched.render_invocation(arguments)
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
    print("Demo 4: Custom Skill with Agent Execution")
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
                model=ZhipuAIChat(model="glm-4.7-flash"),
                instructions=[
                    "You are a Python library expert.",
                    "",
                    "# Skill Instructions",
                    skill.get_prompt(),
                ],
                tools=[ShellTool()],
            )
            query = "What are the main modules in numpy ?"
            if os.environ.get("RUN_SKILL_AGENT_LLM") == "1":
                response = agent.run_sync(query)
                print(response.content)
            else:
                print(f"Ready to run with injected skill. Query: {query}")
                print("Set RUN_SKILL_AGENT_LLM=1 to call the model.")
    else:
        print(f"Custom skill directory not found: {custom_skill_dir}")


def demo_list_triggers():
    """Demo: List all available triggers for user reference."""
    print("\n" + "=" * 60)
    print("Demo 5: Available Triggers")
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
    demo_skill_tool_custom_dirs()
    demo_skill_based_agent()
    demo_custom_skill_with_agent()
    demo_list_triggers()
