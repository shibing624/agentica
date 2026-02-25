# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent-level Guardrails for input/output validation.

Architecture:
  User Input → InputGuardrail.run() → [allow/block] → Agent → OutputGuardrail.run() → [allow/block] → Return
"""

from dataclasses import dataclass
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
from agentica.guardrails.core import (
    GuardrailTriggered,
    GuardrailOutput,
    BaseGuardrail,
    run_guardrails_seq,
)

if TYPE_CHECKING:
    from agentica.agent import Agent

TContext = TypeVar("TContext", bound=Any, default=Any)

# Backward-compatible alias
GuardrailFunctionOutput = GuardrailOutput
GuardrailTripwireTriggered = GuardrailTriggered


# =============================================================================
# Exceptions
# =============================================================================

class InputGuardrailTripwireTriggered(GuardrailTriggered):
    """Exception raised when an input guardrail's tripwire is triggered."""
    pass


class OutputGuardrailTripwireTriggered(GuardrailTriggered):
    """Exception raised when an output guardrail's tripwire is triggered."""
    pass


# =============================================================================
# Results
# =============================================================================

@dataclass
class InputGuardrailResult:
    """The result of an input guardrail run."""
    guardrail: "InputGuardrail[Any]"
    output: GuardrailOutput


@dataclass
class OutputGuardrailResult:
    """The result of an output guardrail run."""
    guardrail: "OutputGuardrail[Any]"
    agent_output: Any
    agent: "Agent[Any]"
    output: GuardrailOutput


# =============================================================================
# Input Guardrail
# =============================================================================

InputGuardrailFunc = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]],
    Union[GuardrailOutput, Awaitable[GuardrailOutput]],
]


@dataclass
class InputGuardrail(BaseGuardrail, Generic[TContext]):
    """Input guardrail that validates agent input before/during processing.

    Example:
        @input_guardrail
        def check_topic(ctx, agent, input_data):
            if "off-topic" in str(input_data):
                return GuardrailFunctionOutput.block()
            return GuardrailFunctionOutput.allow()
    """

    guardrail_function: InputGuardrailFunc = None  # type: ignore
    name: Optional[str] = None
    run_in_parallel: bool = True

    async def run(
        self,
        agent: "Agent[Any]",
        input_data: Union[str, List[Any]],
        context: Optional[Any] = None,
    ) -> InputGuardrailResult:
        """Run the guardrail on the input."""
        logger.debug(f"Running input guardrail: {self.get_name()}")
        output = await self._invoke(context, agent, input_data)
        return InputGuardrailResult(guardrail=self, output=output)


# =============================================================================
# Output Guardrail
# =============================================================================

OutputGuardrailFunc = Callable[
    [Any, "Agent[Any]", Any],
    Union[GuardrailOutput, Awaitable[GuardrailOutput]],
]


@dataclass
class OutputGuardrail(BaseGuardrail, Generic[TContext]):
    """Output guardrail that validates agent output before returning.

    Example:
        @output_guardrail
        def check_sensitive(ctx, agent, output):
            if "password" in str(output).lower():
                return GuardrailFunctionOutput.block()
            return GuardrailFunctionOutput.allow()
    """

    guardrail_function: OutputGuardrailFunc = None  # type: ignore
    name: Optional[str] = None

    async def run(
        self,
        agent: "Agent[Any]",
        agent_output: Any,
        context: Optional[Any] = None,
    ) -> OutputGuardrailResult:
        """Run the guardrail on the output."""
        logger.debug(f"Running output guardrail: {self.get_name()}")
        output = await self._invoke(context, agent, agent_output)
        return OutputGuardrailResult(
            guardrail=self, agent=agent, agent_output=agent_output, output=output,
        )


# =============================================================================
# Decorators
# =============================================================================

_InputGuardrailFuncSync = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]], GuardrailOutput,
]
_InputGuardrailFuncAsync = Callable[
    [Any, "Agent[Any]", Union[str, List[Any]]], Awaitable[GuardrailOutput],
]


@overload
def input_guardrail(func: _InputGuardrailFuncSync) -> InputGuardrail[Any]: ...
@overload
def input_guardrail(func: _InputGuardrailFuncAsync) -> InputGuardrail[Any]: ...
@overload
def input_guardrail(*, name: Optional[str] = None, run_in_parallel: bool = True) -> Callable[
    [Union[_InputGuardrailFuncSync, _InputGuardrailFuncAsync]], InputGuardrail[Any],
]: ...


def input_guardrail(
    func=None, *, name: Optional[str] = None, run_in_parallel: bool = True,
):
    """Decorator to create an InputGuardrail from a function.

    Usage:
        @input_guardrail
        def my_guardrail(ctx, agent, input_data): ...

        @input_guardrail(name="custom", run_in_parallel=False)
        async def my_guardrail(ctx, agent, input_data): ...
    """
    def decorator(f):
        return InputGuardrail(
            guardrail_function=f,
            name=name if name else f.__name__,
            run_in_parallel=run_in_parallel,
        )
    if func is not None:
        return decorator(func)
    return decorator


_OutputGuardrailFuncSync = Callable[[Any, "Agent[Any]", Any], GuardrailOutput]
_OutputGuardrailFuncAsync = Callable[[Any, "Agent[Any]", Any], Awaitable[GuardrailOutput]]


@overload
def output_guardrail(func: _OutputGuardrailFuncSync) -> OutputGuardrail[Any]: ...
@overload
def output_guardrail(func: _OutputGuardrailFuncAsync) -> OutputGuardrail[Any]: ...
@overload
def output_guardrail(*, name: Optional[str] = None) -> Callable[
    [Union[_OutputGuardrailFuncSync, _OutputGuardrailFuncAsync]], OutputGuardrail[Any],
]: ...


def output_guardrail(func=None, *, name: Optional[str] = None):
    """Decorator to create an OutputGuardrail from a function.

    Usage:
        @output_guardrail
        def my_guardrail(ctx, agent, output): ...

        @output_guardrail(name="custom")
        async def my_guardrail(ctx, agent, output): ...
    """
    def decorator(f):
        return OutputGuardrail(
            guardrail_function=f,
            name=name if name else f.__name__,
        )
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Execution Functions
# =============================================================================

async def run_input_guardrails(
    agent: "Agent[Any]",
    input_data: Union[str, List[Any]],
    guardrails: List[InputGuardrail[Any]],
    context: Optional[Any] = None,
) -> List[InputGuardrailResult]:
    """Run all input guardrails sequentially.

    Raises InputGuardrailTripwireTriggered if any guardrail blocks.
    """
    async def _run_one(guard):
        result = await guard.run(agent, input_data, context)
        return result, result.output.tripwire_triggered, guard.get_name(), result.output

    return await run_guardrails_seq(guardrails, _run_one, InputGuardrailTripwireTriggered)


async def run_output_guardrails(
    agent: "Agent[Any]",
    agent_output: Any,
    guardrails: List[OutputGuardrail[Any]],
    context: Optional[Any] = None,
) -> List[OutputGuardrailResult]:
    """Run all output guardrails sequentially.

    Raises OutputGuardrailTripwireTriggered if any guardrail blocks.
    """
    async def _run_one(guard):
        result = await guard.run(agent, agent_output, context)
        return result, result.output.tripwire_triggered, guard.get_name(), result.output

    return await run_guardrails_seq(guardrails, _run_one, OutputGuardrailTripwireTriggered)


__all__ = [
    "GuardrailTripwireTriggered",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    "GuardrailFunctionOutput",
    "InputGuardrailResult",
    "OutputGuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "input_guardrail",
    "output_guardrail",
    "run_input_guardrails",
    "run_output_guardrails",
]
