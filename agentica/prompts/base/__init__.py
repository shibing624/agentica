# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base prompt modules

Core prompt components:
- heartbeat: Forced iteration mechanism (must iterate until solved)
- tools: Tool usage strategy (parallel, file ops, execution)
- soul: Professional objectivity and tone guidelines
- self_verification: Code validation after changes (lint/test/typecheck)
- deep_agent: DeepAgent specific prompts (reflection, repetition, etc.)
"""

from agentica.prompts.base.heartbeat import (
    HEARTBEAT_PROMPT,
    get_heartbeat_prompt,
    get_iteration_reminder,
)
from agentica.prompts.base.tools import (
    TOOLS_PRIORITY_PROMPT,
    get_tools_prompt,
)
from agentica.prompts.base.soul import (
    SOUL_PROMPT,
    get_soul_prompt,
)
from agentica.prompts.base.self_verification import (
    SELF_VERIFICATION_PROMPT,
    get_self_verification_prompt,
)
from agentica.prompts.base.deep_agent import (
    STEP_REFLECTION_PROMPT,
    FORCE_ANSWER_PROMPT,
    REPETITIVE_BEHAVIOR_PROMPT,
    FORCE_STRATEGY_CHANGE_PROMPT,
    ITERATION_CHECKPOINT_PROMPT,
    get_step_reflection_prompt,
    get_force_answer_prompt,
    get_repetitive_behavior_prompt,
    get_force_strategy_change_prompt,
    get_iteration_checkpoint_prompt,
)

__all__ = [
    "HEARTBEAT_PROMPT",
    "get_heartbeat_prompt",
    "get_iteration_reminder",
    "TOOLS_PRIORITY_PROMPT",
    "get_tools_prompt",
    "SOUL_PROMPT",
    "get_soul_prompt",
    # Self Verification prompts
    "SELF_VERIFICATION_PROMPT",
    "get_self_verification_prompt",
    # Deep Agent prompts
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
