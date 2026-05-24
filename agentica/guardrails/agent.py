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
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Union,
    overload,
)

from agentica.model.message import Message
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
                return GuardrailOutput.block()
            return GuardrailOutput.allow()
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
                return GuardrailOutput.block()
            return GuardrailOutput.allow()
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


# =============================================================================
# Input normalization
# =============================================================================

def _media_markers(audio: Any, images: Any, videos: Any) -> List[str]:
    """Return short string tags describing attached media payloads.

    The actual bytes/URLs are NOT included to keep the guardrail input
    compact and avoid blowing up large multimodal payloads inside the
    serialised representation; the marker tells a guardrail "this turn
    has N images / has audio" so a policy can decide accordingly.
    """
    markers: List[str] = []
    if audio is not None:
        markers.append("audio:1")
    if images:
        try:
            markers.append(f"images:{len(images)}")
        except TypeError:
            markers.append("images:?")
    if videos:
        try:
            markers.append(f"videos:{len(videos)}")
        except TypeError:
            markers.append("videos:?")
    return markers


def _coerce_message_to_dict(item: Any) -> Optional[dict]:
    """Convert a Message / dict / str into a plain dict for guardrail inspection."""
    if isinstance(item, Message):
        d: dict = {"role": item.role or "user", "content": item.content}
        if item.images:
            d["media"] = (d.get("media") or []) + [f"image:{i}" for i in range(len(item.images))]
        if item.audio is not None:
            d["media"] = (d.get("media") or []) + ["audio"]
        if item.videos:
            d["media"] = (d.get("media") or []) + [f"video:{i}" for i in range(len(item.videos))]
        return d
    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if item is None:
        return None
    return {"role": "user", "content": str(item)}


def normalize_input_for_guardrails(
    *,
    message: Any = None,
    audio: Any = None,
    images: Any = None,
    videos: Any = None,
    messages: Optional[Sequence[Any]] = None,
) -> List[dict]:
    """Build the full inbound payload visible to input guardrails.

    A single ``message`` argument is *not* enough: callers may also pass
    a full conversation via ``messages=[...]`` plus multimodal attachments
    (``audio``/``images``/``videos``). All of these reach the model, so
    guardrails MUST be able to inspect all of them — otherwise a malicious
    image or earlier turn slips past.

    Returns a list of ``{"role", "content", "media"?}`` dicts in the order
    they will be sent to the model. ``str(result)`` keeps backward-compat
    with simple ``"<keyword>" in str(input_data)`` guardrail patterns
    while exposing multimodal markers like ``"images:2"``.
    """
    items: List[dict] = []
    if messages:
        for m in messages:
            d = _coerce_message_to_dict(m)
            if d is not None:
                items.append(d)
    if message is not None:
        d = _coerce_message_to_dict(message)
        if d is not None:
            items.append(d)
    media = _media_markers(audio, images, videos)
    if media:
        # Attach media markers to the last user-turn item so a guardrail
        # iterating a single dict can still see them; if there is no item
        # (caller only sent multimodal), synthesize an empty user turn.
        if not items:
            items.append({"role": "user", "content": ""})
        last = items[-1]
        last["media"] = (last.get("media") or []) + media
    return items


__all__ = [
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    "InputGuardrailResult",
    "OutputGuardrailResult",
    "InputGuardrail",
    "OutputGuardrail",
    "input_guardrail",
    "output_guardrail",
    "run_input_guardrails",
    "run_output_guardrails",
    "normalize_input_for_guardrails",
]
