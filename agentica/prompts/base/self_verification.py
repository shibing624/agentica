# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SELF_VERIFICATION module - Code validation guidance

This module provides prompts for self-verification after code changes.
Key principles:
1. Run lint/typecheck/test commands after completing code changes
2. Discover validation commands from project configuration
3. Fix errors and re-run until passing
4. Use shell tool to execute validation commands

Inspired by OpenCode's anthropic.txt and gemini.txt verification guidance.
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
SELF_VERIFICATION_PROMPT = _load_prompt("self_verification.md")
SELF_VERIFICATION_PROMPT_COMPACT = _load_prompt("self_verification_compact.md")


def get_self_verification_prompt(compact: bool = False) -> str:
    """Get the self verification prompt.

    Args:
        compact: If True, return the compact version

    Returns:
        The appropriate self verification prompt string
    """
    return SELF_VERIFICATION_PROMPT_COMPACT if compact else SELF_VERIFICATION_PROMPT
