# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Guardrail unified abstraction layer.

Provides base exception, output types, guard base class, and execution engine.
agent.py and tool.py inherit/compose from this module for concrete logic.
"""

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

async def run_guardrails_seq(
    guardrails: List[Any],
    run_one: Callable,
    exception_class: type = GuardrailTriggered,
) -> List[Any]:
    """Sequential guardrail execution engine.

    Args:
        guardrails: List of guardrail instances.
        run_one: Async callable that takes a guardrail and returns (result, triggered: bool, name: str).
        exception_class: Exception class to raise on trigger.

    Returns:
        List of results.
    """
    results = []
    for guard in guardrails:
        result, triggered, name, output = await run_one(guard)
        results.append(result)
        if triggered:
            logger.warning(f"Guardrail '{name}' triggered")
            raise exception_class(guardrail_name=name, output=output)
    return results


__all__ = [
    "GuardrailTriggered",
    "GuardrailOutput",
    "BaseGuardrail",
    "run_guardrails_seq",
]
