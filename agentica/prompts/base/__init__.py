# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Base prompt modules

Core prompt components:
- heartbeat: Forced iteration mechanism (must iterate until solved)
- task_management: Task tracking with TodoWrite
- tools: Tool usage priority and parallel strategy
- soul: Professional objectivity and tone guidelines
- self_verification: Code validation after changes (lint/test/typecheck)
- deep_agent: DeepAgent specific prompts (research, reflection, etc.)
"""

from agentica.prompts.base.heartbeat import (
    HEARTBEAT_PROMPT,
    get_heartbeat_prompt,
    get_iteration_reminder,
)
from agentica.prompts.base.task_management import (
    TASK_MANAGEMENT_PROMPT,
    get_task_management_prompt,
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
    DEEP_RESEARCH_PROMPT,
    STEP_REFLECTION_PROMPT,
    FORCE_ANSWER_PROMPT,
    REPETITIVE_BEHAVIOR_PROMPT,
    ITERATION_CHECKPOINT_PROMPT,
    MUST_CONTINUE_PROMPT,
    get_deep_research_prompt,
    get_step_reflection_prompt,
    get_force_answer_prompt,
    get_repetitive_behavior_prompt,
    get_iteration_checkpoint_prompt,
    get_must_continue_prompt,
)

__all__ = [
    "HEARTBEAT_PROMPT",
    "get_heartbeat_prompt",
    "get_iteration_reminder",
    "TASK_MANAGEMENT_PROMPT",
    "get_task_management_prompt",
    "TOOLS_PRIORITY_PROMPT",
    "get_tools_prompt",
    "SOUL_PROMPT",
    "get_soul_prompt",
    # Self Verification prompts
    "SELF_VERIFICATION_PROMPT",
    "get_self_verification_prompt",
    # Deep Agent prompts
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
