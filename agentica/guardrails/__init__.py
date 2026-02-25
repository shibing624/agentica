# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Guardrails package for Agentica.

Unified guardrail abstraction with three layers:
- core.py: Base exception, output, guard class, execution engine
- agent.py: Agent-level Input/Output guardrails (inherits core)
- tool.py: Tool-level Input/Output guardrails (inherits core)
"""

# Core abstractions
from agentica.guardrails.core import (
    GuardrailTriggered,
    GuardrailOutput,
    BaseGuardrail,
    run_guardrails_seq,
)

# Agent-level guardrails (from agent.py, backward-compatible with old base.py)
from agentica.guardrails.agent import (
    GuardrailTripwireTriggered,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    InputGuardrailResult,
    OutputGuardrailResult,
    run_input_guardrails,
    run_output_guardrails,
    input_guardrail,
    output_guardrail,
)

# Tool-level guardrails
from agentica.guardrails.tool import (
    ToolGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolGuardrailFunctionOutput,
    ToolContext,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolInputGuardrailResult,
    ToolOutputGuardrailResult,
    run_tool_input_guardrails,
    run_tool_output_guardrails,
    tool_input_guardrail,
    tool_output_guardrail,
)

__all__ = [
    # Core
    "GuardrailTriggered",
    "GuardrailOutput",
    "BaseGuardrail",
    "run_guardrails_seq",
    # Agent-level
    "GuardrailTripwireTriggered",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    "InputGuardrail",
    "OutputGuardrail",
    "GuardrailFunctionOutput",
    "InputGuardrailResult",
    "OutputGuardrailResult",
    "run_input_guardrails",
    "run_output_guardrails",
    "input_guardrail",
    "output_guardrail",
    # Tool-level
    "ToolGuardrailTripwireTriggered",
    "ToolInputGuardrailTripwireTriggered",
    "ToolOutputGuardrailTripwireTriggered",
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    "ToolGuardrailFunctionOutput",
    "ToolContext",
    "ToolInputGuardrailData",
    "ToolOutputGuardrailData",
    "ToolInputGuardrailResult",
    "ToolOutputGuardrailResult",
    "run_tool_input_guardrails",
    "run_tool_output_guardrails",
    "tool_input_guardrail",
    "tool_output_guardrail",
]
