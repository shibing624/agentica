# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared utility for loading prompt markdown files.
"""
from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def load_prompt(filename: str, **kwargs) -> str:
    """Load prompt content from MD file in the md/ directory.

    Args:
        filename: Name of the .md file (e.g., "heartbeat.md")
        **kwargs: Optional format variables to substitute in the template

    Returns:
        Stripped content of the file, or empty string if not found.
    """
    filepath = _BASE_DIR / filename
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8").strip()
        if kwargs:
            content = content.format(**kwargs)
        return content
    return ""
