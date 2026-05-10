# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Prompt loaders for the experience-to-skill upgrade lifecycle.
"""
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8").strip() + "\n\n"


def get_skill_spawn_prompt() -> str:
    """Get the prompt for deciding whether to spawn a generated skill."""
    return _load_prompt("skill_spawn.md")


def get_skill_judge_prompt() -> str:
    """Get the prompt for judging shadow skill runtime performance."""
    return _load_prompt("skill_judge.md")


def get_skill_maintenance_prompt() -> str:
    """Get the prompt for repairing or discarding a failing generated skill."""
    return _load_prompt("skill_maintenance.md")
