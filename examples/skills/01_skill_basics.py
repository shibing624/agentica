# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Demo - Basic Usage

Agent Skill is a prompt-based skill system that extends agent capabilities.
Skills are text instructions defined in SKILL.md files that can be injected
into agent prompts, allowing the LLM to follow specialized instructions.

Skills support:
- Trigger commands (e.g., /commit, /github)
- Automatic discovery from multiple directories
- Built-in skills (github, commit)

Reference: https://github.com/shibing624/agentica
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.skills import (
    SkillLoader,
    load_skills,
    get_available_skills,
    get_skill_registry,
)


def demo_skill_discovery():
    """Demo: Discover and list all available skills."""
    print("=" * 60)
    print("Demo 1: Skill Discovery")
    print("=" * 60)

    # Load all skills from standard directories
    # Search paths (in priority order):
    # 1. .claude/skills, .agentica/skills (project-level)
    # 2. ~/.agentica/skill (user-level from config)
    # 3. ~/.claude/skills, ~/.agentica/skills (user-level)
    # 4. agentica/skills/builtin/ (built-in skills)
    registry = load_skills()

    # List all available skills
    skills = get_available_skills()
    print(f"\nFound {len(skills)} skills:")
    for skill in skills:
        trigger = f" (trigger: {skill.trigger})" if skill.trigger else ""
        location = f" [{skill.location}]" if skill.location else ""
        print(f"  - {skill.name}: {skill.description[:50]}...{trigger}{location}")

    # Get skills summary for system prompt injection
    print("\n" + "-" * 40)
    print("Skills Summary (for system prompt):")
    print("-" * 40)
    summary = registry.get_skills_summary()
    print(summary[:500] + "..." if len(summary) > 500 else summary)


def demo_trigger_matching():
    """Demo: Match user input to skill triggers."""
    print("\n" + "=" * 60)
    print("Demo 2: Trigger Matching")
    print("=" * 60)

    # Ensure skills are loaded
    registry = get_skill_registry()
    if len(registry) == 0:
        load_skills()

    # Test trigger matching
    test_inputs = [
        "/commit fix login bug",          # Should match 'commit' skill
        "/github pr list",                # Should match 'github' skill
        "/github issue create",           # Should match 'github' skill
        "regular message without trigger", # Should not match
        "/unknown command",               # Should not match (no such trigger)
    ]

    print("\nTesting trigger matches:")
    for text in test_inputs:
        matched = registry.match_trigger(text)
        if matched:
            print(f"  '{text}' -> matched: {matched.name}")
        else:
            print(f"  '{text}' -> no match")

    # List all registered triggers
    print("\n" + "-" * 40)
    print("Registered Triggers:")
    print("-" * 40)
    triggers = registry.list_triggers()
    if triggers:
        for trigger, skill_name in triggers.items():
            print(f"  {trigger} -> {skill_name}")
    else:
        print("  No triggers registered")


def demo_skill_loader_api():
    """Demo: Using SkillLoader convenience methods."""
    print("\n" + "=" * 60)
    print("Demo 3: SkillLoader API")
    print("=" * 60)

    loader = SkillLoader()

    # Ensure skills are loaded
    loader.load_all()

    # Get a skill by name
    commit_skill = loader.get_skill("commit")
    if commit_skill:
        print(f"\nSkill: {commit_skill.name}")
        print(f"  Description: {commit_skill.description[:60]}...")
        print(f"  Trigger: {commit_skill.trigger}")
        print(f"  Requires: {commit_skill.requires}")
        print(f"  Location: {commit_skill.location}")

        # Get skill prompt (for injection into agent)
        prompt = loader.get_skill_prompt("commit")
        if prompt:
            print(f"\n  Prompt preview:")
            print(f"  {prompt[:300]}...")

    # Match trigger using loader
    user_input = "/commit refactor auth module"
    matched = loader.match_trigger(user_input)
    if matched:
        print(f"\nMatched '{user_input}' to skill: {matched.name}")


def demo_skill_files():
    """Demo: List and read skill files."""
    print("\n" + "=" * 60)
    print("Demo 4: Skill File Operations")
    print("=" * 60)

    from agentica.skills import list_skill_files, read_skill_file
    from pathlib import Path

    # List files in built-in skills directory
    builtin_dir = Path(__file__).parent.parent.parent / "agentica/skills/builtin"
    if builtin_dir.exists():
        print(f"\nBuilt-in skills directory: {builtin_dir}")
        result = list_skill_files(str(builtin_dir))
        print(result)

        # Read a skill file
        commit_skill_md = builtin_dir / "commit" / "SKILL.md"
        if commit_skill_md.exists():
            print(f"\n" + "-" * 40)
            print(f"Content of {commit_skill_md.name}:")
            print("-" * 40)
            content = read_skill_file(str(commit_skill_md))
            print(content[:500] + "..." if len(content) > 500 else content)


def demo_custom_skill():
    """Demo: Register a custom skill from a directory."""
    print("\n" + "=" * 60)
    print("Demo 5: Custom Skill Registration")
    print("=" * 60)

    from agentica.skills import register_skill

    # Custom skill directory (example data)
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    custom_skill_dir = os.path.join(pwd_path, "../data/skill/python-lib-analyzer")

    if os.path.exists(custom_skill_dir):
        print(f"\nRegistering custom skill from: {custom_skill_dir}")
        skill = register_skill(custom_skill_dir, location="project")
        if skill:
            print(f"  Registered: {skill.name}")
            print(f"  Trigger: {skill.trigger}")
            print(f"  Description: {skill.description[:60]}...")
        else:
            print("  Skill already registered or failed to load")
    else:
        print(f"  Custom skill directory not found: {custom_skill_dir}")


if __name__ == "__main__":
    demo_skill_discovery()
    demo_trigger_matching()
    demo_skill_loader_api()
    demo_skill_files()
    demo_custom_skill()
