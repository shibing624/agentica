# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner shared result types: LoopBreak, ModelCallResult, ToolHandlingResult
"""

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple

from agentica.model.base import Model


class LoopBreak(NamedTuple):
    """Structured result of a Runner safety check that aborts the agentic loop.

    ``reason`` is a stable machine code (RunBreakReason value); ``message`` is a
    human-readable detail. Both are surfaced on ``RunResponse`` so downstream
    never has to parse internal error text out of ``content``.
    """

    reason: str
    message: str


@dataclass
class ModelCallResult:
    """A model response plus the concrete provider that produced it."""

    response: Any
    used_model: Model
    used_fallback: bool = False

    def __getattr__(self, name: str) -> Any:
        # Transparent delegation to the wrapped response (ModelResponse or the
        # stream iterator) for ergonomic access at call sites/tests. Guard the
        # backing field so a half-built instance (deepcopy/pickle) can't recurse
        # forever via getattr(self.response, ...) before `response` is set.
        if name == "response":
            raise AttributeError(name)
        return getattr(self.response, name)


@dataclass
class ToolHandlingResult:
    """Runner-owned tool execution summary for the current LLM turn."""

    had_tool_calls: bool
    tool_results: List[Dict[str, Any]]


