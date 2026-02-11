# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Guardrails package for Agentica.

Guardrails are validation mechanisms that can be applied at different stages:
- Agent level: Input/Output guardrails for the entire agent
- Tool level: Input/Output guardrails for individual tool calls

Example:
    >>> from agentica.guardrails import (
    ...     InputGuardrail, OutputGuardrail,
    ...     ToolInputGuardrail, ToolOutputGuardrail,
    ... )
    >>>
    >>> # Agent-level guardrail
    >>> @input_guardrail
    ... def check_input(context, agent, input_message):
    ...     return GuardrailFunctionOutput(output_info={"checked": True})
    >>>
    >>> # Tool-level guardrail
    >>> @tool_input_guardrail
    ... def validate_tool_input(context, agent, tool_input):
    ...     return ToolGuardrailFunctionOutput(output_info={"validated": True})
"""

# Agent-level guardrails (from base.py)
from agentica.guardrails.base import (
    # Exceptions
    GuardrailTripwireTriggered,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    # Data classes
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
    InputGuardrailResult,
    OutputGuardrailResult,
    # Functions
    run_input_guardrails,
    run_output_guardrails,
    # Decorators
    input_guardrail,
    output_guardrail,
)

# Tool-level guardrails (from tool.py)
from agentica.guardrails.tool import (
    # Exceptions
    ToolGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
    # Data classes
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolGuardrailFunctionOutput,
    ToolContext,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolInputGuardrailResult,
    ToolOutputGuardrailResult,
    # Functions
    run_tool_input_guardrails,
    run_tool_output_guardrails,
    # Decorators
    tool_input_guardrail,
    tool_output_guardrail,
)

__all__ = [
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
