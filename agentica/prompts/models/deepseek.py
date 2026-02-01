# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepSeek-specific prompt optimizations

Optimizations for DeepSeek models:
1. Extended thinking for complex reasoning
2. Code generation best practices
3. Tool calling guidelines
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


DEEPSEEK_SPECIFIC_PROMPT = _load_prompt("deepseek.md")


def get_deepseek_prompt() -> str:
    """Get the DeepSeek-specific prompt.

    Returns:
        The DeepSeek-specific prompt string
    """
    return DEEPSEEK_SPECIFIC_PROMPT
