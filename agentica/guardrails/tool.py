# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tool-level Guardrails for tool input/output validation.

Architecture:
  Tool Call → ToolInputGuardrail.run() → [allow/reject/exception] → Tool Exec → ToolOutputGuardrail.run() → Return
"""

import inspect
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Union,
    Awaitable,
    overload,
)
from typing_extensions import TypedDict, TypeVar

from agentica.utils.log import logger
from agentica.guardrails.core import (
    GuardrailTriggered,
    BaseGuardrail,
)

if TYPE_CHECKING:
    from agentica.agent import Agent

TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)


# =============================================================================
# Exceptions
# =============================================================================

class ToolGuardrailTripwireTriggered(GuardrailTriggered):
    """Base exception for tool guardrail tripwire triggers."""
    pass


class ToolInputGuardrailTripwireTriggered(ToolGuardrailTripwireTriggered):
    """Exception raised when a tool input guardrail's tripwire is triggered."""
    pass


class ToolOutputGuardrailTripwireTriggered(ToolGuardrailTripwireTriggered):
    """Exception raised when a tool output guardrail's tripwire is triggered."""
    pass


# =============================================================================
# Behavior Types
# =============================================================================

class AllowBehavior(TypedDict):
    type: Literal["allow"]


class RejectContentBehavior(TypedDict):
    type: Literal["reject_content"]
    message: str


class RaiseExceptionBehavior(TypedDict):
    type: Literal["raise_exception"]


# =============================================================================
# Tool Guardrail Output
# =============================================================================

@dataclass
class ToolGuardrailFunctionOutput:
    """Output of a tool guardrail function with three-way behavior."""

    output_info: Any = None
    behavior: Union[AllowBehavior, RejectContentBehavior, RaiseExceptionBehavior] = field(
        default_factory=lambda: AllowBehavior(type="allow")
    )

    @classmethod
    def allow(cls, output_info: Any = None) -> "ToolGuardrailFunctionOutput":
        return cls(output_info=output_info, behavior=AllowBehavior(type="allow"))

    @classmethod
    def reject_content(cls, message: str, output_info: Any = None) -> "ToolGuardrailFunctionOutput":
        return cls(output_info=output_info, behavior=RejectContentBehavior(type="reject_content", message=message))

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> "ToolGuardrailFunctionOutput":
        return cls(output_info=output_info, behavior=RaiseExceptionBehavior(type="raise_exception"))

    def is_allow(self) -> bool:
        return self.behavior.get("type") == "allow"

    def is_reject_content(self) -> bool:
        return self.behavior.get("type") == "reject_content"

    def is_raise_exception(self) -> bool:
        return self.behavior.get("type") == "raise_exception"

    def get_reject_message(self) -> Optional[str]:
        if self.is_reject_content():
            return self.behavior.get("message")
        return None


# =============================================================================
# Tool Context & Data
# =============================================================================

@dataclass
class ToolContext:
    """Context information for tool guardrails."""
    tool_name: str
    tool_arguments: Optional[str] = None
    tool_call_id: Optional[str] = None
    agent: Optional["Agent[Any]"] = None


@dataclass
class ToolInputGuardrailData:
    """Input data passed to a tool input guardrail function."""
    context: ToolContext
    agent: "Agent[Any]"


@dataclass
class ToolOutputGuardrailData:
    """Output data passed to a tool output guardrail function."""
    context: ToolContext
    agent: "Agent[Any]"
    output: Any


# =============================================================================
# Results
# =============================================================================

@dataclass
class ToolInputGuardrailResult:
    guardrail: "ToolInputGuardrail[Any]"
    output: ToolGuardrailFunctionOutput


@dataclass
class ToolOutputGuardrailResult:
    guardrail: "ToolOutputGuardrail[Any]"
    output: ToolGuardrailFunctionOutput


# =============================================================================
# Tool Input Guardrail
# =============================================================================

ToolInputGuardrailFunc = Callable[
    [ToolInputGuardrailData],
    Union[ToolGuardrailFunctionOutput, Awaitable[ToolGuardrailFunctionOutput]],
]


@dataclass
class ToolInputGuardrail(BaseGuardrail, Generic[TContext_co]):
    """Guardrail that runs before a tool is invoked."""

    guardrail_function: ToolInputGuardrailFunc = None  # type: ignore
    name: Optional[str] = None

    async def run(self, data: ToolInputGuardrailData) -> ToolInputGuardrailResult:
        logger.debug(f"Running tool input guardrail: {self.get_name()}")
        result = await self._invoke(data)
        return ToolInputGuardrailResult(guardrail=self, output=result)


# =============================================================================
# Tool Output Guardrail
# =============================================================================

ToolOutputGuardrailFunc = Callable[
    [ToolOutputGuardrailData],
    Union[ToolGuardrailFunctionOutput, Awaitable[ToolGuardrailFunctionOutput]],
]


@dataclass
class ToolOutputGuardrail(BaseGuardrail, Generic[TContext_co]):
    """Guardrail that runs after a tool is invoked."""

    guardrail_function: ToolOutputGuardrailFunc = None  # type: ignore
    name: Optional[str] = None

    async def run(self, data: ToolOutputGuardrailData) -> ToolOutputGuardrailResult:
        logger.debug(f"Running tool output guardrail: {self.get_name()}")
        result = await self._invoke(data)
        return ToolOutputGuardrailResult(guardrail=self, output=result)


# =============================================================================
# Decorators
# =============================================================================

_ToolInputFuncSync = Callable[[ToolInputGuardrailData], ToolGuardrailFunctionOutput]
_ToolInputFuncAsync = Callable[[ToolInputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_input_guardrail(func: _ToolInputFuncSync) -> ToolInputGuardrail[Any]: ...
@overload
def tool_input_guardrail(func: _ToolInputFuncAsync) -> ToolInputGuardrail[Any]: ...
@overload
def tool_input_guardrail(*, name: Optional[str] = None) -> Callable[
    [Union[_ToolInputFuncSync, _ToolInputFuncAsync]], ToolInputGuardrail[Any]
]: ...


def tool_input_guardrail(func=None, *, name: Optional[str] = None):
    """Decorator to create a ToolInputGuardrail from a function."""
    def decorator(f):
        return ToolInputGuardrail(guardrail_function=f, name=name or f.__name__)
    if func is not None:
        return decorator(func)
    return decorator


_ToolOutputFuncSync = Callable[[ToolOutputGuardrailData], ToolGuardrailFunctionOutput]
_ToolOutputFuncAsync = Callable[[ToolOutputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_output_guardrail(func: _ToolOutputFuncSync) -> ToolOutputGuardrail[Any]: ...
@overload
def tool_output_guardrail(func: _ToolOutputFuncAsync) -> ToolOutputGuardrail[Any]: ...
@overload
def tool_output_guardrail(*, name: Optional[str] = None) -> Callable[
    [Union[_ToolOutputFuncSync, _ToolOutputFuncAsync]], ToolOutputGuardrail[Any]
]: ...


def tool_output_guardrail(func=None, *, name: Optional[str] = None):
    """Decorator to create a ToolOutputGuardrail from a function."""
    def decorator(f):
        return ToolOutputGuardrail(guardrail_function=f, name=name or f.__name__)
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Execution Functions
# =============================================================================

async def run_tool_input_guardrails(
    data: ToolInputGuardrailData,
    guardrails: List[ToolInputGuardrail[Any]],
) -> ToolGuardrailFunctionOutput:
    """Run all tool input guardrails. Returns first non-allow result."""
    for guardrail in guardrails:
        result = await guardrail.run(data)
        if result.output.is_raise_exception():
            logger.warning(f"Tool input guardrail '{guardrail.get_name()}' triggered exception")
            raise ToolInputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(), output=result.output,
            )
        if result.output.is_reject_content():
            logger.info(f"Tool input guardrail '{guardrail.get_name()}' rejected content")
            return result.output
    return ToolGuardrailFunctionOutput.allow()


async def run_tool_output_guardrails(
    data: ToolOutputGuardrailData,
    guardrails: List[ToolOutputGuardrail[Any]],
) -> ToolGuardrailFunctionOutput:
    """Run all tool output guardrails. Returns first non-allow result."""
    for guardrail in guardrails:
        result = await guardrail.run(data)
        if result.output.is_raise_exception():
            logger.warning(f"Tool output guardrail '{guardrail.get_name()}' triggered exception")
            raise ToolOutputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(), output=result.output,
            )
        if result.output.is_reject_content():
            logger.info(f"Tool output guardrail '{guardrail.get_name()}' rejected content")
            return result.output
    return ToolGuardrailFunctionOutput.allow()


__all__ = [
    "ToolGuardrailTripwireTriggered",
    "ToolInputGuardrailTripwireTriggered",
    "ToolOutputGuardrailTripwireTriggered",
    "AllowBehavior",
    "RejectContentBehavior",
    "RaiseExceptionBehavior",
    "ToolGuardrailFunctionOutput",
    "ToolContext",
    "ToolInputGuardrailData",
    "ToolOutputGuardrailData",
    "ToolInputGuardrailResult",
    "ToolOutputGuardrailResult",
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    "tool_input_guardrail",
    "tool_output_guardrail",
    "run_tool_input_guardrails",
    "run_tool_output_guardrails",
]
