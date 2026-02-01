# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Claude-specific prompt optimizations

Optimizations for Claude models (Opus, Sonnet, Haiku):
1. Thorough thinking process
2. Autonomous tool calling
3. Concise, idiomatic code generation
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


CLAUDE_SPECIFIC_PROMPT = _load_prompt("anthropic.md")


def get_claude_prompt() -> str:
    """Get the Claude-specific prompt.

    Returns:
        The Claude-specific prompt string
    """
    return CLAUDE_SPECIFIC_PROMPT
