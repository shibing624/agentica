# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Agent specific prompts

Prompts for DeepAgent functionality:
- step_reflection: Reflection prompt between steps
- force_answer: Force answer when context limit reached
- repetitive_behavior: Warning for repetitive tool calls
- force_strategy_change: Force strategy change after persistent repetition
- iteration_checkpoint: Checkpoint during iteration
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt


# Load prompts from MD files
STEP_REFLECTION_PROMPT = _load_prompt("step_reflection.md")
FORCE_ANSWER_PROMPT = _load_prompt("force_answer.md")
REPETITIVE_BEHAVIOR_PROMPT = _load_prompt("repetitive_behavior.md")
FORCE_STRATEGY_CHANGE_PROMPT = _load_prompt("force_strategy_change.md")
ITERATION_CHECKPOINT_PROMPT = _load_prompt("iteration_checkpoint.md")


def get_step_reflection_prompt() -> str:
    """Get the step reflection prompt."""
    return STEP_REFLECTION_PROMPT


def get_force_answer_prompt() -> str:
    """Get the force answer prompt."""
    return FORCE_ANSWER_PROMPT


def get_repetitive_behavior_prompt(tool_name: str, count: int) -> str:
    """Get the repetitive behavior warning prompt."""
    return REPETITIVE_BEHAVIOR_PROMPT.format(tool_name=tool_name, count=count)


def get_force_strategy_change_prompt(tool_name: str, count: int) -> str:
    """Get the force strategy change prompt for persistent repetition."""
    return FORCE_STRATEGY_CHANGE_PROMPT.format(tool_name=tool_name, count=count)


def get_iteration_checkpoint_prompt(step: int) -> str:
    """Get the iteration checkpoint prompt."""
    return ITERATION_CHECKPOINT_PROMPT.format(step=step)


__all__ = [
    "STEP_REFLECTION_PROMPT",
    "FORCE_ANSWER_PROMPT",
    "REPETITIVE_BEHAVIOR_PROMPT",
    "FORCE_STRATEGY_CHANGE_PROMPT",
    "ITERATION_CHECKPOINT_PROMPT",
    "get_step_reflection_prompt",
    "get_force_answer_prompt",
    "get_repetitive_behavior_prompt",
    "get_force_strategy_change_prompt",
    "get_iteration_checkpoint_prompt",
]
