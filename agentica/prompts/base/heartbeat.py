# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: HEARTBEAT module - Iteration control mechanism

Core module for task completion. Instructs the model to:
1. Keep iterating until the problem is completely solved
2. Never end prematurely without verification
3. Degrade strategy after persistent failures
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file
HEARTBEAT_PROMPT = _load_prompt("heartbeat.md")


def get_heartbeat_prompt() -> str:
    """Get the heartbeat prompt.

    Returns:
        The heartbeat prompt string
    """
    return HEARTBEAT_PROMPT


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
