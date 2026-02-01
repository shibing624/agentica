# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SOUL module - Behavioral guidelines and professional objectivity

This module provides prompts for agent behavior:
1. Professional objectivity - prioritize accuracy over validation
2. Tone and style - concise, direct communication
3. No time estimates - focus on what, not how long
4. Genuine helpfulness - skip pleasantries, just help

Based on OpenCode's anthropic.txt and OpenClaw's SOUL.md
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


# Load prompts from MD files
SOUL_PROMPT = _load_prompt("soul.md")
SOUL_PROMPT_COMPACT = _load_prompt("soul_compact.md")


def get_soul_prompt(compact: bool = False) -> str:
    """Get the soul prompt.

    Args:
        compact: If True, return the compact version

    Returns:
        The appropriate soul prompt string
    """
    return SOUL_PROMPT_COMPACT if compact else SOUL_PROMPT
