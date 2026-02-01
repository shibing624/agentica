# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills system usage with Workspace example

This example shows how to:
1. Load skills from directories including workspace skills
2. Match trigger commands and inject prompts
3. Integrate skills with workspace-based agents
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI, Workspace
from agentica.skills import (
    SkillLoader,
    get_skill_registry,
    load_skills,
    get_available_skills,
)


def demo_workspace_with_skills():
    """Demo: Create workspace-based agent with skills."""
    print("=" * 60)
    print("Demo: Workspace + Skills Integration")
    print("=" * 60)

    # Load all skills from standard directories
    registry = load_skills()

    # List available skills
    skills = get_available_skills()
    print(f"\nFound {len(skills)} skills:")
    for skill in skills:
        trigger_info = f" (trigger: {skill.trigger})" if skill.trigger else ""
        print(f"  - {skill.name}: {skill.description[:40]}...{trigger_info}")

    # Create a workspace-based agent
    # The workspace provides: AGENT.md, PERSONA.md, TOOLS.md, USER.md, MEMORY.md
    print("\n" + "-" * 40)
    print("Creating Agent from Workspace")
    print("-" * 40)

    # Use default workspace path from config (~/.agentica/workspace)
    workspace = Workspace()
    workspace.initialize()  # Create if not exists

    # Get workspace context
    context = workspace.get_context_prompt()
    if context:
        print(f"Workspace context loaded ({len(context)} chars)")

    # Get skills summary
    skills_summary = registry.get_skills_summary()

    # Create agent with workspace context + skills
    agent = Agent(
        model=ZhipuAI(model="glm-4-flash"),
        name="Workspace-Skills-Agent",
        instructions=[
            "You are a helpful assistant with workspace and skill capabilities.",
            "",
            "# Workspace Context",
            context if context else "No workspace context available.",
            "",
            "# Available Skills",
            skills_summary,
        ],
    )

    print(f"Agent created: {agent.name}")

    # Demo: Match trigger and inject skill
    print("\n" + "-" * 40)
    print("Trigger Matching Demo")
    print("-" * 40)

    user_inputs = [
        "/commit add new feature",
        "/github pr list",
        "What is the weather today?",
    ]

    for user_input in user_inputs:
        matched = registry.match_trigger(user_input)
        if matched:
            print(f"'{user_input}' -> {matched.name} skill activated")

            # Inject skill prompt dynamically
            agent.add_instruction(f"\n# {matched.name} Skill\n{matched.get_prompt()}")

            # Extract actual message
            trigger = matched.trigger
            if trigger and user_input.startswith(trigger):
                actual_msg = user_input[len(trigger):].strip()
                print(f"  Actual message: '{actual_msg}'")
        else:
            print(f"'{user_input}' -> no skill trigger")


def demo_workspace_skills_directory():
    """Demo: Skills from workspace skills directory."""
    print("\n" + "=" * 60)
    print("Demo: Workspace Skills Directory")
    print("=" * 60)

    workspace = Workspace()
    skills_dir = workspace.get_skills_dir()

    print(f"\nWorkspace skills directory: {skills_dir}")
    print(f"Exists: {skills_dir.exists()}")

    if skills_dir.exists():
        # List skills in workspace
        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
        if skill_dirs:
            print(f"\nFound {len(skill_dirs)} skill directories:")
            for sd in skill_dirs:
                skill_md = sd / "SKILL.md"
                status = "✓" if skill_md.exists() else "✗"
                print(f"  {status} {sd.name}/")
        else:
            print("No skills in workspace directory yet.")
    else:
        print("Workspace skills directory not created yet.")

    # Show how to add a skill to workspace
    print("\nTo add a skill to workspace:")
    print(f"  1. Create directory: {skills_dir}/my-skill/")
    print(f"  2. Create SKILL.md with YAML frontmatter")
    print(f"  3. Skills will be auto-loaded on next load_skills()")


if __name__ == "__main__":
    demo_workspace_with_skills()
    demo_workspace_skills_directory()
