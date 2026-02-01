# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Default prompt for unrecognized models

Generic prompt that works well across different model providers.
Used as fallback when model-specific optimizations are not available.
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


DEFAULT_PROMPT = _load_prompt("default.md")


def get_default_prompt() -> str:
    """Get the default prompt.

    Returns:
        The default prompt string
    """
    return DEFAULT_PROMPT
