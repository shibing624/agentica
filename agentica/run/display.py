# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: UI-agnostic classification for RunResponse streaming events.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agentica.run.response import RunEvent, RunResponse


class RunDisplayEventKind(str, Enum):
    """Stable display categories consumed by CLI and gateway layers."""

    CONTENT_DELTA = "content_delta"
    FINAL_CONTENT = "final_content"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    METADATA_SKIP = "metadata_skip"
    TELEMETRY_ONLY = "telemetry_only"


@dataclass(frozen=True)
class RunDisplayEvent:
    kind: RunDisplayEventKind
    response: RunResponse
    run_event: Optional[RunEvent] = None


_METADATA_SKIP_EVENTS = {
    RunEvent.run_started,
    RunEvent.run_completed,
    RunEvent.updating_memory,
    RunEvent.reasoning_started,
    RunEvent.reasoning_completed,
    RunEvent.workflow_started,
    RunEvent.workflow_completed,
    RunEvent.subagent_spawned,
    RunEvent.subagent_completed,
}

_TELEMETRY_ONLY_EVENTS = {
    RunEvent.run_failed,
    RunEvent.run_cancelled,
}

_LEGACY_METADATA_EVENT_VALUES = {
    "MultiRoundTurn",
    "MultiRoundToolCall",
    "MultiRoundToolResult",
    "MultiRoundCompleted",
}


def _coerce_run_event(event: str) -> Optional[RunEvent]:
    try:
        return RunEvent(event)
    except ValueError:
        return None


def classify_run_response(response: RunResponse, is_final: bool = False) -> RunDisplayEvent:
    """Classify a RunResponse for display without importing UI frameworks."""
    run_event = _coerce_run_event(response.event)

    if run_event == RunEvent.tool_call_started:
        return RunDisplayEvent(RunDisplayEventKind.TOOL_STARTED, response, run_event)
    if run_event in (RunEvent.tool_call_completed, RunEvent.tool_call_failed):
        return RunDisplayEvent(RunDisplayEventKind.TOOL_COMPLETED, response, run_event)
    if run_event == RunEvent.run_response:
        if response.content is not None or response.reasoning_content is not None:
            kind = RunDisplayEventKind.FINAL_CONTENT if is_final else RunDisplayEventKind.CONTENT_DELTA
            return RunDisplayEvent(kind, response, run_event)
        return RunDisplayEvent(RunDisplayEventKind.METADATA_SKIP, response, run_event)
    if run_event in _METADATA_SKIP_EVENTS or response.event in _LEGACY_METADATA_EVENT_VALUES:
        return RunDisplayEvent(RunDisplayEventKind.METADATA_SKIP, response, run_event)
    if run_event in _TELEMETRY_ONLY_EVENTS:
        return RunDisplayEvent(RunDisplayEventKind.TELEMETRY_ONLY, response, run_event)
    if response.content is not None or response.reasoning_content is not None:
        kind = RunDisplayEventKind.FINAL_CONTENT if is_final else RunDisplayEventKind.CONTENT_DELTA
        return RunDisplayEvent(kind, response, run_event)
    return RunDisplayEvent(RunDisplayEventKind.METADATA_SKIP, response, run_event)
