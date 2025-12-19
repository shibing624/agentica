# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills module for Agentica.

Agent Skill is an approach proposed by Anthropic to improve agent capabilities on specific tasks.
Skills are not code-level extensions, but text instructions injected into the system prompt,
allowing the LLM to read and follow the instructions to complete tasks.

This module provides:
- Skill: Data class representing a loaded skill
- SkillRegistry: Registry for managing loaded skills
- SkillLoader: Discovers and loads skills from standard directories

Search paths (in priority order):
1. .claude/skills (project-level)
2. .agentica/skills (project-level)
3. ~/.claude/skills (user-level)
4. ~/.agentica/skills (user-level)

Usage:
    from agentica.skills import load_skills, get_available_skills, register_skill

    # Load all skills from standard directories
    registry = load_skills()

    # Get all available skills
    skills = get_available_skills()

    # Register a single skill
    skill = register_skill("./my-skills/web-research")

Reference: https://claude.com/blog/skills
"""

from agentica.skills.skill import Skill
from agentica.skills.skill_registry import (
    SkillRegistry,
    get_skill_registry,
    reset_skill_registry,
)
from agentica.skills.skill_loader import (
    SkillLoader,
    load_skills,
    get_available_skills,
    register_skill,
    register_skills,
    list_skill_files,
    read_skill_file,
)

__all__ = [
    # Skill class
    "Skill",
    # Registry
    "SkillRegistry",
    "get_skill_registry",
    "reset_skill_registry",
    # Loader
    "SkillLoader",
    "load_skills",
    "get_available_skills",
    "register_skill",
    "register_skills",
    # File operations
    "list_skill_files",
    "read_skill_file",
]
