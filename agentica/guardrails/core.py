# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Guardrail unified abstraction layer.

Provides base exception, output types, guard base class, and execution engine.
agent.py and tool.py inherit/compose from this module for concrete logic.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Union,
    Awaitable,
)
from typing_extensions import TypeVar

from agentica.utils.log import logger

TContext = TypeVar("TContext", bound=Any, default=Any)


# =============================================================================
# Unified Exception Hierarchy
# =============================================================================

class GuardrailTriggered(Exception):
    """Base exception for all guardrail triggers (agent-level and tool-level)."""

    def __init__(self, guardrail_name: str, output: Any):
        self.guardrail_name = guardrail_name
        self.output = output
        super().__init__(f"Guardrail '{guardrail_name}' triggered")


# =============================================================================
# Unified Guardrail Output
# =============================================================================

@dataclass
class GuardrailOutput:
    """Unified guardrail function output.

    Attributes:
        output_info: Optional information about the guardrail's checks.
        tripwire_triggered: Whether the tripwire was triggered (block).
    """

    output_info: Any = None
    tripwire_triggered: bool = False

    @classmethod
    def allow(cls, output_info: Any = None) -> "GuardrailOutput":
        """Create output that allows execution to continue."""
        return cls(output_info=output_info, tripwire_triggered=False)

    @classmethod
    def block(cls, output_info: Any = None) -> "GuardrailOutput":
        """Create output that blocks execution."""
        return cls(output_info=output_info, tripwire_triggered=True)


# =============================================================================
# Base Guardrail Class
# =============================================================================

@dataclass
class BaseGuardrail(Generic[TContext]):
    """Base guardrail class with common logic.

    Subclasses implement specific `run()` signatures for agent-level or tool-level.
    """

    guardrail_function: Callable
    """The function that implements the guardrail logic."""

    name: Optional[str] = None
    """Optional name for the guardrail."""

    def get_name(self) -> str:
        """Get the name of the guardrail."""
        if self.name:
            return self.name
        return getattr(self.guardrail_function, "__name__", "unknown_guardrail")

    async def _invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the guardrail function, handling both sync and async."""
        if not callable(self.guardrail_function):
            raise ValueError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )
        result = self.guardrail_function(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


# =============================================================================
# Unified Execution Engine
# =============================================================================

def _batches(
    guardrails: List[Any],
    parallel_when: Optional[Callable[[Any], bool]],
) -> List[List[Any]]:
    """Group consecutive parallel-safe guardrails; everything else stands alone."""
    if parallel_when is None:
        return [[guard] for guard in guardrails]
    batches: List[List[Any]] = []
    for guard in guardrails:
        if batches and parallel_when(guard) and parallel_when(batches[-1][-1]):
            batches[-1].append(guard)
        else:
            batches.append([guard])
    return batches


async def run_guardrails(
    guardrails: List[Any],
    run_one: Callable,
    exception_class: type = GuardrailTriggered,
    parallel_when: Optional[Callable[[Any], bool]] = None,
) -> List[Any]:
    """Guardrail execution engine.

    Guardrails run in declaration order. ``parallel_when(guard)`` marks one as
    safe to overlap with its neighbours, and consecutive marked guardrails are
    awaited together — a policy built from three moderation calls should cost
    one round trip, not three. Order is preserved instead of hoisting all the
    parallel ones to the front, because opting out is precisely how a caller
    puts a cheap filter ahead of an expensive one.

    Within a batch every guardrail finishes before a tripwire is raised, and
    the one raised is the first in declaration order: otherwise which guardrail
    blocked a request would depend on which call happened to return first.

    Args:
        guardrails: List of guardrail instances.
        run_one: Async callable taking a guardrail, returning
            ``(result, triggered, name, output)``.
        exception_class: Exception class to raise on trigger.
        parallel_when: Optional predicate; omit to run everything serially.

    Returns:
        List of results, in declaration order.
    """
    results = []
    for batch in _batches(guardrails, parallel_when):
        if len(batch) == 1:
            outcomes: List[Any] = [await run_one(batch[0])]
        else:
            outcomes = await asyncio.gather(
                *(run_one(guard) for guard in batch), return_exceptions=True,
            )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            result, triggered, name, output = outcome
            results.append(result)
            if triggered:
                logger.warning(f"Guardrail '{name}' triggered")
                raise exception_class(guardrail_name=name, output=output)
    return results


__all__ = [
    "GuardrailTriggered",
    "GuardrailOutput",
    "BaseGuardrail",
    "run_guardrails",
]
