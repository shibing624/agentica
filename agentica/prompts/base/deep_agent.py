# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Agent specific prompts

Prompts for DeepAgent functionality:
- deep_research: System prompt for deep research mode
- step_reflection: Reflection prompt between steps
- force_answer: Force answer when context limit reached
- repetitive_behavior: Warning for repetitive tool calls
- iteration_checkpoint: Checkpoint during iteration
- must_continue: Reminder to continue until complete
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt from markdown file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


# Load prompts from MD files
DEEP_RESEARCH_PROMPT = _load_prompt("deep_research.md")
STEP_REFLECTION_PROMPT = _load_prompt("step_reflection.md")
FORCE_ANSWER_PROMPT = _load_prompt("force_answer.md")
REPETITIVE_BEHAVIOR_PROMPT = _load_prompt("repetitive_behavior.md")
ITERATION_CHECKPOINT_PROMPT = _load_prompt("iteration_checkpoint.md")
MUST_CONTINUE_PROMPT = _load_prompt("must_continue.md")


def get_deep_research_prompt(current_date: str = "") -> str:
    """Get the deep research system prompt with current date."""
    if current_date:
        return DEEP_RESEARCH_PROMPT.replace("{current_date}", current_date)
    return DEEP_RESEARCH_PROMPT


def get_step_reflection_prompt() -> str:
    """Get the step reflection prompt."""
    return STEP_REFLECTION_PROMPT


def get_force_answer_prompt() -> str:
    """Get the force answer prompt."""
    return FORCE_ANSWER_PROMPT


def get_repetitive_behavior_prompt(tool_name: str, count: int) -> str:
    """Get the repetitive behavior warning prompt."""
    return REPETITIVE_BEHAVIOR_PROMPT.format(tool_name=tool_name, count=count)


def get_iteration_checkpoint_prompt(step: int) -> str:
    """Get the iteration checkpoint prompt."""
    return ITERATION_CHECKPOINT_PROMPT.format(step=step)


def get_must_continue_prompt() -> str:
    """Get the must continue reminder prompt."""
    return MUST_CONTINUE_PROMPT


__all__ = [
    "DEEP_RESEARCH_PROMPT",
    "STEP_REFLECTION_PROMPT",
    "FORCE_ANSWER_PROMPT",
    "REPETITIVE_BEHAVIOR_PROMPT",
    "ITERATION_CHECKPOINT_PROMPT",
    "MUST_CONTINUE_PROMPT",
    "get_deep_research_prompt",
    "get_step_reflection_prompt",
    "get_force_answer_prompt",
    "get_repetitive_behavior_prompt",
    "get_iteration_checkpoint_prompt",
    "get_must_continue_prompt",
]
