# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Guardrails for Agent input and output validation.

Guardrails are checks that run on agent inputs/outputs to validate, filter, or block content.
They can be used to:
- Check if input messages are off-topic or inappropriate
- Validate agent outputs before returning to user
- Block sensitive or dangerous content
- Implement content moderation policies

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution Flow                      │
├─────────────────────────────────────────────────────────────┤
│  1. User Input Received                                      │
│         ↓                                                    │
│  2. InputGuardrail.run() ──→ Check if input is valid         │
│         ↓                                                    │
│     ┌─ tripwire_triggered=False → Continue execution         │
│     └─ tripwire_triggered=True  → Raise exception, halt      │
│         ↓                                                    │
│  3. Agent Processing (LLM call, tool calls, etc.)            │
│         ↓                                                    │
│  4. OutputGuardrail.run() ──→ Check if output is valid       │
│         ↓                                                    │
│     ┌─ tripwire_triggered=False → Return output              │
│     └─ tripwire_triggered=True  → Raise exception, halt      │
│         ↓                                                    │
│  5. Return Result to User                                    │
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
    Optional,
    Union,
    Awaitable,
    overload,
)
from typing_extensions import TypeVar

from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent

TContext = TypeVar("TContext", bound=Any, default=Any)
TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)


# =============================================================================
# Exceptions
# =============================================================================

class GuardrailTripwireTriggered(Exception):
    """Base exception for guardrail tripwire triggers."""

    def __init__(self, guardrail_name: str, output: "GuardrailFunctionOutput"):
        self.guardrail_name = guardrail_name
        self.output = output
        super().__init__(f"Guardrail '{guardrail_name}' tripwire triggered")


class InputGuardrailTripwireTriggered(GuardrailTripwireTriggered):
    """Exception raised when an input guardrail's tripwire is triggered."""
    pass


class OutputGuardrailTripwireTriggered(GuardrailTripwireTriggered):
    """Exception raised when an output guardrail's tripwire is triggered."""
    pass


# =============================================================================
# Guardrail Output
# =============================================================================

@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function.

    Attributes:
        output_info: Optional information about the guardrail's output.
            For example, the guardrail could include information about
            the checks it performed and granular results.
        tripwire_triggered: Whether the tripwire was triggered.
            If triggered, the agent's execution will be halted.
    """

    output_info: Any = None
    """Optional information about the guardrail's output."""

    tripwire_triggered: bool = False
    """Whether the tripwire was triggered. If True, execution will be halted."""

    @classmethod
    def allow(cls, output_info: Any = None) -> "GuardrailFunctionOutput":
        """Create a guardrail output that allows execution to continue.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            GuardrailFunctionOutput configured to allow execution.
        """
        return cls(output_info=output_info, tripwire_triggered=False)

    @classmethod
    def block(cls, output_info: Any = None) -> "GuardrailFunctionOutput":
        """Create a guardrail output that blocks execution.

        Args:
            output_info: Optional data about why execution was blocked.

        Returns:
            GuardrailFunctionOutput configured to block execution.
        """
        return cls(output_info=output_info, tripwire_triggered=True)


# =============================================================================
# Guardrail Results
# =============================================================================

@dataclass
class InputGuardrailResult:
    """The result of an input guardrail run."""

    guardrail: "InputGuardrail[Any]"
    """The guardrail that was run."""

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class OutputGuardrailResult:
    """The result of an output guardrail run."""

    guardrail: "OutputGuardrail[Any]"
    """The guardrail that was run."""

    agent_output: Any
    """The output of the agent that was checked by the guardrail."""

    agent: "Agent[Any]"
    """The agent that was checked by the guardrail."""

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


# =============================================================================
# Input Guardrail
# =============================================================================

# Type aliases for guardrail functions
InputGuardrailFunc = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]],
    Union[GuardrailFunctionOutput, Awaitable[GuardrailFunctionOutput]],
]


@dataclass
class InputGuardrail(Generic[TContext]):
    """Input guardrails are checks that run before or in parallel with the agent.

    They can be used to:
    - Check if input messages are off-topic
    - Take over control of the agent's execution if an unexpected input is detected
    - Validate input format or content

    You can use the `@input_guardrail` decorator to turn a function into an
    `InputGuardrail`, or create an `InputGuardrail` manually.

    If `result.tripwire_triggered` is `True`, the agent's execution will
    immediately stop, and an `InputGuardrailTripwireTriggered` exception
    will be raised.

    Example:
        ```python
        @input_guardrail
        async def check_topic(ctx, agent, input_data):
            # Check if input is on-topic
            if "off-topic" in str(input_data):
                return GuardrailFunctionOutput(
                    output_info={"reason": "off-topic content"},
                    tripwire_triggered=True
                )
            return GuardrailFunctionOutput.allow()

        agent = Agent(
            name="MyAgent",
            input_guardrails=[check_topic],
        )
        ```
    """

    guardrail_function: InputGuardrailFunc
    """A function that receives the context, agent, and input, and returns a
    GuardrailFunctionOutput. The result marks whether the tripwire was triggered.
    """

    name: Optional[str] = None
    """The name of the guardrail, used for tracing. If not provided, uses the
    guardrail function's name.
    """

    run_in_parallel: bool = True
    """Whether the guardrail runs concurrently with the agent (True, default)
    or before the agent starts (False).
    """

    def get_name(self) -> str:
        """Get the name of the guardrail."""
        if self.name:
            return self.name
        return getattr(self.guardrail_function, "__name__", "unknown_guardrail")

    async def run(
        self,
        agent: "Agent[Any]",
        input_data: Union[str, List[Any]],
        context: Optional[Any] = None,
    ) -> InputGuardrailResult:
        """Run the guardrail on the input.

        Args:
            agent: The agent being guarded.
            input_data: The input to check.
            context: Optional context for the guardrail.

        Returns:
            InputGuardrailResult with the guardrail output.

        Raises:
            ValueError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise ValueError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        logger.debug(f"Running input guardrail: {self.get_name()}")
        output = self.guardrail_function(context, agent, input_data)

        if inspect.isawaitable(output):
            output = await output

        return InputGuardrailResult(guardrail=self, output=output)


# =============================================================================
# Output Guardrail
# =============================================================================

OutputGuardrailFunc = Callable[
    [Any, "Agent[Any]", Any],
    Union[GuardrailFunctionOutput, Awaitable[GuardrailFunctionOutput]],
]


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Output guardrails are checks that run on the final output of an agent.

    They can be used to:
    - Check if the output passes certain validation criteria
    - Filter or modify sensitive content
    - Ensure output format compliance

    You can use the `@output_guardrail` decorator to turn a function into an
    `OutputGuardrail`, or create an `OutputGuardrail` manually.

    If `result.tripwire_triggered` is `True`, an `OutputGuardrailTripwireTriggered`
    exception will be raised.

    Example:
        ```python
        @output_guardrail
        async def check_sensitive_data(ctx, agent, output):
            if "password" in str(output).lower():
                return GuardrailFunctionOutput(
                    output_info={"reason": "sensitive data detected"},
                    tripwire_triggered=True
                )
            return GuardrailFunctionOutput.allow()

        agent = Agent(
            name="MyAgent",
            output_guardrails=[check_sensitive_data],
        )
        ```
    """

    guardrail_function: OutputGuardrailFunc
    """A function that receives the context, agent, and output, and returns a
    GuardrailFunctionOutput. The result marks whether the tripwire was triggered.
    """

    name: Optional[str] = None
    """The name of the guardrail, used for tracing. If not provided, uses the
    guardrail function's name.
    """

    def get_name(self) -> str:
        """Get the name of the guardrail."""
        if self.name:
            return self.name
        return getattr(self.guardrail_function, "__name__", "unknown_guardrail")

    async def run(
        self,
        agent: "Agent[Any]",
        agent_output: Any,
        context: Optional[Any] = None,
    ) -> OutputGuardrailResult:
        """Run the guardrail on the output.

        Args:
            agent: The agent being guarded.
            agent_output: The output to check.
            context: Optional context for the guardrail.

        Returns:
            OutputGuardrailResult with the guardrail output.

        Raises:
            ValueError: If guardrail_function is not callable.
        """
        if not callable(self.guardrail_function):
            raise ValueError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        logger.debug(f"Running output guardrail: {self.get_name()}")
        output = self.guardrail_function(context, agent, agent_output)

        if inspect.isawaitable(output):
            output = await output

        return OutputGuardrailResult(
            guardrail=self,
            agent=agent,
            agent_output=agent_output,
            output=output,
        )


# =============================================================================
# Decorators
# =============================================================================

# Type definitions for decorator overloads
_InputGuardrailFuncSync = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]],
    GuardrailFunctionOutput,
]
_InputGuardrailFuncAsync = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def input_guardrail(
    func: _InputGuardrailFuncSync,
) -> InputGuardrail[Any]: ...


@overload
def input_guardrail(
    func: _InputGuardrailFuncAsync,
) -> InputGuardrail[Any]: ...


@overload
def input_guardrail(
    *,
    name: Optional[str] = None,
    run_in_parallel: bool = True,
) -> Callable[
    [Union[_InputGuardrailFuncSync, _InputGuardrailFuncAsync]],
    InputGuardrail[Any],
]: ...


def input_guardrail(
    func: Optional[Union[_InputGuardrailFuncSync, _InputGuardrailFuncAsync]] = None,
    *,
    name: Optional[str] = None,
    run_in_parallel: bool = True,
) -> Union[
    InputGuardrail[Any],
    Callable[
        [Union[_InputGuardrailFuncSync, _InputGuardrailFuncAsync]],
        InputGuardrail[Any],
    ],
]:
    """Decorator that transforms a function into an InputGuardrail.

    Can be used directly (no parentheses) or with keyword args:

        @input_guardrail
        def my_sync_guardrail(ctx, agent, input_data): ...

        @input_guardrail(name="my_guardrail", run_in_parallel=False)
        async def my_async_guardrail(ctx, agent, input_data): ...

    Args:
        func: The guardrail function to wrap.
        name: Optional name for the guardrail. If not provided, uses function's name.
        run_in_parallel: Whether to run concurrently with agent (True) or before (False).

    Returns:
        An InputGuardrail instance.
    """

    def decorator(
        f: Union[_InputGuardrailFuncSync, _InputGuardrailFuncAsync],
    ) -> InputGuardrail[Any]:
        return InputGuardrail(
            guardrail_function=f,
            name=name if name else f.__name__,
            run_in_parallel=run_in_parallel,
        )

    if func is not None:
        return decorator(func)

    return decorator


_OutputGuardrailFuncSync = Callable[
    [Any, "Agent[Any]", Any],
    GuardrailFunctionOutput,
]
_OutputGuardrailFuncAsync = Callable[
    [Any, "Agent[Any]", Any],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def output_guardrail(
    func: _OutputGuardrailFuncSync,
) -> OutputGuardrail[Any]: ...


@overload
def output_guardrail(
    func: _OutputGuardrailFuncAsync,
) -> OutputGuardrail[Any]: ...


@overload
def output_guardrail(
    *,
    name: Optional[str] = None,
) -> Callable[
    [Union[_OutputGuardrailFuncSync, _OutputGuardrailFuncAsync]],
    OutputGuardrail[Any],
]: ...


def output_guardrail(
    func: Optional[Union[_OutputGuardrailFuncSync, _OutputGuardrailFuncAsync]] = None,
    *,
    name: Optional[str] = None,
) -> Union[
    OutputGuardrail[Any],
    Callable[
        [Union[_OutputGuardrailFuncSync, _OutputGuardrailFuncAsync]],
        OutputGuardrail[Any],
    ],
]:
    """Decorator that transforms a function into an OutputGuardrail.

    Can be used directly (no parentheses) or with keyword args:

        @output_guardrail
        def my_sync_guardrail(ctx, agent, output): ...

        @output_guardrail(name="my_guardrail")
        async def my_async_guardrail(ctx, agent, output): ...

    Args:
        func: The guardrail function to wrap.
        name: Optional name for the guardrail. If not provided, uses function's name.

    Returns:
        An OutputGuardrail instance.
    """

    def decorator(
        f: Union[_OutputGuardrailFuncSync, _OutputGuardrailFuncAsync],
    ) -> OutputGuardrail[Any]:
        return OutputGuardrail(
            guardrail_function=f,
            name=name if name else f.__name__,
        )

    if func is not None:
        return decorator(func)

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================

async def run_input_guardrails(
    agent: "Agent[Any]",
    input_data: Union[str, List[Any]],
    guardrails: List[InputGuardrail[Any]],
    context: Optional[Any] = None,
) -> List[InputGuardrailResult]:
    """Run all input guardrails and collect results.

    Args:
        agent: The agent being guarded.
        input_data: The input to check.
        guardrails: List of input guardrails to run.
        context: Optional context for guardrails.

    Returns:
        List of InputGuardrailResult.

    Raises:
        InputGuardrailTripwireTriggered: If any guardrail's tripwire is triggered.
    """
    results = []
    for guardrail in guardrails:
        result = await guardrail.run(agent, input_data, context)
        results.append(result)

        if result.output.tripwire_triggered:
            logger.warning(
                f"Input guardrail '{guardrail.get_name()}' tripwire triggered"
            )
            raise InputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(),
                output=result.output,
            )

    return results


async def run_output_guardrails(
    agent: "Agent[Any]",
    agent_output: Any,
    guardrails: List[OutputGuardrail[Any]],
    context: Optional[Any] = None,
) -> List[OutputGuardrailResult]:
    """Run all output guardrails and collect results.

    Args:
        agent: The agent being guarded.
        agent_output: The output to check.
        guardrails: List of output guardrails to run.
        context: Optional context for guardrails.

    Returns:
        List of OutputGuardrailResult.

    Raises:
        OutputGuardrailTripwireTriggered: If any guardrail's tripwire is triggered.
    """
    results = []
    for guardrail in guardrails:
        result = await guardrail.run(agent, agent_output, context)
        results.append(result)

        if result.output.tripwire_triggered:
            logger.warning(
                f"Output guardrail '{guardrail.get_name()}' tripwire triggered"
            )
            raise OutputGuardrailTripwireTriggered(
                guardrail_name=guardrail.get_name(),
                output=result.output,
            )

    return results


__all__ = [
    # Exceptions
    "GuardrailTripwireTriggered",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    # Output types
    "GuardrailFunctionOutput",
    # Results
    "InputGuardrailResult",
    "OutputGuardrailResult",
    # Guardrail classes
    "InputGuardrail",
    "OutputGuardrail",
    # Decorators
    "input_guardrail",
    "output_guardrail",
    # Utility functions
    "run_input_guardrails",
    "run_output_guardrails",
]
