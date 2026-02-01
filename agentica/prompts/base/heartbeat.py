# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: HEARTBEAT module - Forced iteration mechanism

This is the CORE module for improving task completion rates.
It instructs the model to:
1. Keep iterating until the problem is completely solved
2. Never end prematurely without verification
3. Follow a self-driven workflow
4. Verify changes before marking complete

Inspired by OpenCode's beast.txt and anthropic.txt
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
HEARTBEAT_PROMPT = _load_prompt("heartbeat.md")
HEARTBEAT_PROMPT_COMPACT = _load_prompt("heartbeat_compact.md")


def get_heartbeat_prompt(compact: bool = False) -> str:
    """Get the heartbeat prompt.

    Args:
        compact: If True, return the compact version for context-sensitive situations

    Returns:
        The appropriate heartbeat prompt string
    """
    return HEARTBEAT_PROMPT_COMPACT if compact else HEARTBEAT_PROMPT


def get_iteration_reminder(step: int) -> str:
    """Generate an iteration reminder message for multi-round execution.

    Args:
        step: Current step number

    Returns:
        Reminder message to inject during execution
    """
    return f"""
Step {step} checkpoint:
- Have you fully solved the problem?
- Are there any remaining tasks in the task list?
- Did you verify your changes?

If not complete, continue working. Do NOT end your turn prematurely.
"""
