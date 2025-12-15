# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tool Guardrails for validating tool inputs and outputs.

Tool Guardrails are checks that run before/after tool execution to:
- Validate tool input arguments
- Check tool outputs for sensitive data
- Block dangerous operations
- Implement security policies

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      Tool Execution Flow                     │
├─────────────────────────────────────────────────────────────┤
│  1. Tool Call Received                                       │
│         ↓                                                    │
│  2. ToolInputGuardrail.run() ──→ Check if input is valid     │
│         ↓                                                    │
│     ┌─ allow ──────────→ Continue execution                  │
│     ├─ reject_content ─→ Return message, skip tool execution │
│     └─ raise_exception → Raise exception, halt execution     │
│         ↓                                                    │
│  3. Tool Function Execution                                  │
│         ↓                                                    │
│  4. ToolOutputGuardrail.run() ──→ Check if output is valid   │
│         ↓                                                    │
│     ┌─ allow ──────────→ Return original result              │
│     ├─ reject_content ─→ Replace result with message         │
│     └─ raise_exception → Raise exception, halt execution     │
│         ↓                                                    │
│  5. Return Result to LLM                                     │
└─────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

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

if TYPE_CHECKING:
    from agentica.agent import Agent

TContext = TypeVar("TContext", bound=Any, default=Any)
TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)


# =============================================================================
# Exceptions
# =============================================================================

class ToolGuardrailTripwireTriggered(Exception):
    """Base exception for tool guardrail tripwire triggers."""

    def __init__(self, guardrail_name: str, output: "ToolGuardrailFunctionOutput"):
        self.guardrail_name = guardrail_name
        self.output = output
        super().__init__(f"Tool guardrail '{guardrail_name}' tripwire triggered")


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
    """Allows normal tool execution to continue."""
    type: Literal["allow"]


class RejectContentBehavior(TypedDict):
    """Rejects the tool call/output but continues execution with a message."""
    type: Literal["reject_content"]
    message: str


class RaiseExceptionBehavior(TypedDict):
    """Raises an exception to halt execution."""
    type: Literal["raise_exception"]


# =============================================================================
# Tool Guardrail Output
# =============================================================================

@dataclass
class ToolGuardrailFunctionOutput:
    """The output of a tool guardrail function.

    Attributes:
        output_info: Optional data about checks performed.
        behavior: Defines how the system should respond:
            - allow: Continue normal tool execution (default)
            - reject_content: Reject but continue with a message to the model
            - raise_exception: Halt execution by raising an exception
    """

    output_info: Any = None
    """Optional data about checks performed."""

    behavior: Union[AllowBehavior, RejectContentBehavior, RaiseExceptionBehavior] = field(
        default_factory=lambda: AllowBehavior(type="allow")
    )
    """Defines how the system should respond when this guardrail result is processed."""

    @classmethod
    def allow(cls, output_info: Any = None) -> "ToolGuardrailFunctionOutput":
        """Create a guardrail output that allows tool execution to continue.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to allow execution.
        """
        return cls(output_info=output_info, behavior=AllowBehavior(type="allow"))

    @classmethod
    def reject_content(
        cls, message: str, output_info: Any = None
    ) -> "ToolGuardrailFunctionOutput":
        """Create a guardrail output that rejects the tool call/output.

        The tool execution is skipped/result is replaced, but agent continues.

        Args:
            message: Message to send to the model instead of the tool result.
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to reject the content.
        """
        return cls(
            output_info=output_info,
            behavior=RejectContentBehavior(type="reject_content", message=message),
        )

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> "ToolGuardrailFunctionOutput":
        """Create a guardrail output that raises an exception to halt execution.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to raise an exception.
        """
        return cls(
            output_info=output_info,
            behavior=RaiseExceptionBehavior(type="raise_exception"),
        )

    def is_allow(self) -> bool:
        """Check if behavior is 'allow'."""
        return self.behavior.get("type") == "allow"

    def is_reject_content(self) -> bool:
        """Check if behavior is 'reject_content'."""
        return self.behavior.get("type") == "reject_content"

    def is_raise_exception(self) -> bool:
        """Check if behavior is 'raise_exception'."""
        return self.behavior.get("type") == "raise_exception"

    def get_reject_message(self) -> Optional[str]:
        """Get the rejection message if behavior is 'reject_content'."""
        if self.is_reject_content():
            return self.behavior.get("message")
        return None


# =============================================================================
# Tool Guardrail Data
# =============================================================================

@dataclass
class ToolContext:
    """Context information for tool guardrails."""

    tool_name: str
    """The name of the tool being called."""

    tool_arguments: Optional[str] = None
    """The JSON string of tool arguments."""

    tool_call_id: Optional[str] = None
    """The ID of the tool call."""

    agent: Optional["Agent[Any]"] = None
    """The agent executing the tool."""


@dataclass
class ToolInputGuardrailData:
    """Input data passed to a tool input guardrail function."""

    context: ToolContext
    """The tool context containing information about the current tool execution."""

    agent: "Agent[Any]"
    """The agent that is executing the tool."""


@dataclass
class ToolOutputGuardrailData:
    """Input data passed to a tool output guardrail function."""

    context: ToolContext
    """The tool context containing information about the current tool execution."""

    agent: "Agent[Any]"
    """The agent that is executing the tool."""

    output: Any
    """The output produced by the tool function."""


# =============================================================================
# Tool Guardrail Results
# =============================================================================

@dataclass
class ToolInputGuardrailResult:
    """The result of a tool input guardrail run."""

    guardrail: "ToolInputGuardrail[Any]"
    """The guardrail that was run."""

    output: ToolGuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class ToolOutputGuardrailResult:
    """The result of a tool output guardrail run."""

    guardrail: "ToolOutputGuardrail[Any]"
    """The guardrail that was run."""

    output: ToolGuardrailFunctionOutput
    """The output of the guardrail function."""


# =============================================================================
# Tool Input Guardrail
# =============================================================================

ToolInputGuardrailFunc = Callable[
    [ToolInputGuardrailData],
    Union[ToolGuardrailFunctionOutput, Awaitable[ToolGuardrailFunctionOutput]],
]


@dataclass
class ToolInputGuardrail(Generic[TContext_co]):
    """A guardrail that runs before a tool is invoked.

    Use this to validate tool input arguments before execution.

    Example:
        ```python
        @tool_input_guardrail
        def check_file_path(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            import json
            args = json.loads(data.context.tool_arguments or "{}")
            if "/etc/" in str(args):
                return ToolGuardrailFunctionOutput.reject_content(
                    message="Access to system directories is forbidden"
                )
            return ToolGuardrailFunctionOutput.allow()
        ```
    """

    guardrail_function: ToolInputGuardrailFunc
    """The function that implements the guardrail logic."""

    name: Optional[str] = None
    """Optional name for the guardrail. If not provided, uses the function name."""

    def get_name(self) -> str:
        """Get the name of the guardrail."""
        return self.name or getattr(
            self.guardrail_function, "__name__", "unknown_guardrail"
        )

    async def run(self, data: ToolInputGuardrailData) -> ToolInputGuardrailResult:
        """Run the guardrail on the tool input.

        Args:
            data: The input data to check.

        Returns:
            ToolInputGuardrailResult with the guardrail output.

        Raises:
            ValueError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise ValueError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        logger.debug(f"Running tool input guardrail: {self.get_name()}")
        result = self.guardrail_function(data)

        if inspect.isawaitable(result):
            result = await result

        return ToolInputGuardrailResult(guardrail=self, output=result)


# =============================================================================
# Tool Output Guardrail
# =============================================================================

ToolOutputGuardrailFunc = Callable[
    [ToolOutputGuardrailData],
    Union[ToolGuardrailFunctionOutput, Awaitable[ToolGuardrailFunctionOutput]],
]


@dataclass
class ToolOutputGuardrail(Generic[TContext_co]):
    """A guardrail that runs after a tool is invoked.

    Use this to validate or filter tool outputs.

    Example:
        ```python
        @tool_output_guardrail
        def sanitize_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
            if "password" in str(data.output).lower():
                return ToolGuardrailFunctionOutput.reject_content(
                    message="[REDACTED - sensitive data removed]"
                )
            return ToolGuardrailFunctionOutput.allow()
        ```
    """

    guardrail_function: ToolOutputGuardrailFunc
    """The function that implements the guardrail logic."""

    name: Optional[str] = None
    """Optional name for the guardrail. If not provided, uses the function name."""

    def get_name(self) -> str:
        """Get the name of the guardrail."""
        return self.name or getattr(
            self.guardrail_function, "__name__", "unknown_guardrail"
        )

    async def run(self, data: ToolOutputGuardrailData) -> ToolOutputGuardrailResult:
        """Run the guardrail on the tool output.

        Args:
            data: The output data to check.

        Returns:
            ToolOutputGuardrailResult with the guardrail output.

        Raises:
            ValueError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise ValueError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        logger.debug(f"Running tool output guardrail: {self.get_name()}")
        result = self.guardrail_function(data)

        if inspect.isawaitable(result):
            result = await result

        return ToolOutputGuardrailResult(guardrail=self, output=result)


# =============================================================================
# Decorators
# =============================================================================

_ToolInputFuncSync = Callable[[ToolInputGuardrailData], ToolGuardrailFunctionOutput]
_ToolInputFuncAsync = Callable[
    [ToolInputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]
]


@overload
def tool_input_guardrail(
    func: _ToolInputFuncSync,
) -> ToolInputGuardrail[Any]: ...


@overload
def tool_input_guardrail(
    func: _ToolInputFuncAsync,
) -> ToolInputGuardrail[Any]: ...


@overload
def tool_input_guardrail(
    *, name: Optional[str] = None
) -> Callable[
    [Union[_ToolInputFuncSync, _ToolInputFuncAsync]], ToolInputGuardrail[Any]
]: ...


def tool_input_guardrail(
    func: Optional[Union[_ToolInputFuncSync, _ToolInputFuncAsync]] = None,
    *,
    name: Optional[str] = None,
) -> Union[
    ToolInputGuardrail[Any],
    Callable[
        [Union[_ToolInputFuncSync, _ToolInputFuncAsync]], ToolInputGuardrail[Any]
    ],
]:
    """Decorator to create a ToolInputGuardrail from a function.

    Can be used directly or with keyword arguments:

        @tool_input_guardrail
        def my_guardrail(data: ToolInputGuardrailData): ...

        @tool_input_guardrail(name="my_guardrail")
        async def my_async_guardrail(data: ToolInputGuardrailData): ...

    Args:
        func: The guardrail function to wrap.
        name: Optional name for the guardrail.

    Returns:
        A ToolInputGuardrail instance.
    """

    def decorator(
        f: Union[_ToolInputFuncSync, _ToolInputFuncAsync],
    ) -> ToolInputGuardrail[Any]:
        return ToolInputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator


_ToolOutputFuncSync = Callable[[ToolOutputGuardrailData], ToolGuardrailFunctionOutput]
_ToolOutputFuncAsync = Callable[
    [ToolOutputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]
]


@overload
def tool_output_guardrail(
    func: _ToolOutputFuncSync,
) -> ToolOutputGuardrail[Any]: ...


@overload
def tool_output_guardrail(
    func: _ToolOutputFuncAsync,
) -> ToolOutputGuardrail[Any]: ...


@overload
def tool_output_guardrail(
    *, name: Optional[str] = None
) -> Callable[
    [Union[_ToolOutputFuncSync, _ToolOutputFuncAsync]], ToolOutputGuardrail[Any]
]: ...


def tool_output_guardrail(
    func: Optional[Union[_ToolOutputFuncSync, _ToolOutputFuncAsync]] = None,
    *,
    name: Optional[str] = None,
) -> Union[
    ToolOutputGuardrail[Any],
    Callable[
        [Union[_ToolOutputFuncSync, _ToolOutputFuncAsync]], ToolOutputGuardrail[Any]
    ],
]:
    """Decorator to create a ToolOutputGuardrail from a function.

    Can be used directly or with keyword arguments:

        @tool_output_guardrail
        def my_guardrail(data: ToolOutputGuardrailData): ...

        @tool_output_guardrail(name="my_guardrail")
        async def my_async_guardrail(data: ToolOutputGuardrailData): ...

    Args:
        func: The guardrail function to wrap.
        name: Optional name for the guardrail.

    Returns:
        A ToolOutputGuardrail instance.
    """

    def decorator(
        f: Union[_ToolOutputFuncSync, _ToolOutputFuncAsync],
    ) -> ToolOutputGuardrail[Any]:
        return ToolOutputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Utility Functions
# =============================================================================

async def run_tool_input_guardrails(
    data: ToolInputGuardrailData,
    guardrails: List[ToolInputGuardrail[Any]],
) -> ToolGuardrailFunctionOutput:
    """Run all tool input guardrails and return the first non-allow result.

    Args:
        data: The input data to check.
        guardrails: List of input guardrails to run.

    Returns:
        ToolGuardrailFunctionOutput - allow if all pass, otherwise first rejection.

    Raises:
        ToolInputGuardrailTripwireTriggered: If any guardrail raises an exception.
    """
    for guardrail in guardrails:
        result = await guardrail.run(data)

        if result.output.is_raise_exception():
            logger.warning(
                f"Tool input guardrail '{guardrail.get_name()}' triggered exception"
            )
            raise ToolInputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(),
                output=result.output,
            )

        if result.output.is_reject_content():
            logger.info(
                f"Tool input guardrail '{guardrail.get_name()}' rejected content"
            )
            return result.output

    return ToolGuardrailFunctionOutput.allow()


async def run_tool_output_guardrails(
    data: ToolOutputGuardrailData,
    guardrails: List[ToolOutputGuardrail[Any]],
) -> ToolGuardrailFunctionOutput:
    """Run all tool output guardrails and return the first non-allow result.

    Args:
        data: The output data to check.
        guardrails: List of output guardrails to run.

    Returns:
        ToolGuardrailFunctionOutput - allow if all pass, otherwise first rejection.

    Raises:
        ToolOutputGuardrailTripwireTriggered: If any guardrail raises an exception.
    """
    for guardrail in guardrails:
        result = await guardrail.run(data)

        if result.output.is_raise_exception():
            logger.warning(
                f"Tool output guardrail '{guardrail.get_name()}' triggered exception"
            )
            raise ToolOutputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(),
                output=result.output,
            )

        if result.output.is_reject_content():
            logger.info(
                f"Tool output guardrail '{guardrail.get_name()}' rejected content"
            )
            return result.output

    return ToolGuardrailFunctionOutput.allow()


__all__ = [
    # Exceptions
    "ToolGuardrailTripwireTriggered",
    "ToolInputGuardrailTripwireTriggered",
    "ToolOutputGuardrailTripwireTriggered",
    # Behavior types
    "AllowBehavior",
    "RejectContentBehavior",
    "RaiseExceptionBehavior",
    # Output types
    "ToolGuardrailFunctionOutput",
    # Context and data
    "ToolContext",
    "ToolInputGuardrailData",
    "ToolOutputGuardrailData",
    # Results
    "ToolInputGuardrailResult",
    "ToolOutputGuardrailResult",
    # Guardrail classes
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    # Decorators
    "tool_input_guardrail",
    "tool_output_guardrail",
    # Utility functions
    "run_tool_input_guardrails",
    "run_tool_output_guardrails",
]
