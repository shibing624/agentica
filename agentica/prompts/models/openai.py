# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OpenAI GPT-specific prompt optimizations

Optimizations for GPT models (GPT-4o, GPT-4, o1, o3):
1. Structured output formatting
2. Step-by-step reasoning
3. Function calling best practices
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


OPENAI_SPECIFIC_PROMPT = _load_prompt("openai.md")


def get_openai_prompt() -> str:
    """Get the OpenAI-specific prompt.

    Returns:
        The OpenAI-specific prompt string
    """
    return OPENAI_SPECIFIC_PROMPT
