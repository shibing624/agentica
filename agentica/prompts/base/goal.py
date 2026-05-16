# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: GOAL prompts — judge system message and continuation template.

Two static templates used by the standing-goal loop in ``agentica.goals``:

- ``GOAL_JUDGE_SYSTEM_PROMPT`` — system message for the LLM-as-judge that
  decides ``done`` / ``continue`` after each agent turn. No placeholders.
- ``GOAL_CONTINUATION_PROMPT_TEMPLATE`` — user-role prompt fed back into
  the agent for the next turn. Placeholders: ``{objective}``,
  ``{subgoals_block}``.

The dynamic per-turn judge USER prompt (which interleaves tools,
subgoals, evidence rules) is assembled in ``agentica.goals`` because it's
a conditional builder, not a static template.
"""

from agentica.prompts.base.utils import load_prompt

GOAL_JUDGE_SYSTEM_PROMPT = load_prompt("goal_judge.md")
GOAL_CONTINUATION_PROMPT_TEMPLATE = load_prompt("goal_continuation.md")


def get_goal_judge_system_prompt() -> str:
    return GOAL_JUDGE_SYSTEM_PROMPT


def render_goal_continuation_prompt(objective: str, subgoals_block: str = "") -> str:
    """Fill the continuation template. ``subgoals_block`` may be empty."""
    return GOAL_CONTINUATION_PROMPT_TEMPLATE.format(
        objective=objective,
        subgoals_block=subgoals_block,
    )
