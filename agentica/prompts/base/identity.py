# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IDENTITY module - Agent identity definitions

This module provides prompts for different agent identities:
1. CLI agent - Interactive command line assistant
2. API agent - Programmatic API assistant
3. Default - General purpose assistant

Based on OpenClaw's IDENTITY.md
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
IDENTITY_CLI_PROMPT = _load_prompt("identity_cli.md")
IDENTITY_API_PROMPT = _load_prompt("identity_api.md")
IDENTITY_DEFAULT_PROMPT = _load_prompt("identity_default.md")


def get_identity_prompt(identity_type: str = "default") -> str:
    """Get the identity prompt based on type.

    Args:
        identity_type: Type of identity ("cli", "api", "default")

    Returns:
        The appropriate identity prompt string
    """
    prompts = {
        "cli": IDENTITY_CLI_PROMPT,
        "api": IDENTITY_API_PROMPT,
        "default": IDENTITY_DEFAULT_PROMPT,
    }
    return prompts.get(identity_type, IDENTITY_DEFAULT_PROMPT)
