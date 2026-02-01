# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Zhipu GLM-specific prompt optimizations

Optimizations for Zhipu GLM models.
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


ZHIPU_SPECIFIC_PROMPT = _load_prompt("zhipu.md")


def get_zhipu_prompt() -> str:
    """Get the Zhipu-specific prompt.

    Returns:
        The Zhipu-specific prompt string
    """
    return ZHIPU_SPECIFIC_PROMPT
