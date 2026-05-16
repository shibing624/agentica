# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Internal RunEvent surface (Phase 3 of arch_v5.md).

The legacy `RunEvent` enum in `run_response.py` was streaming-content-centric
(RunStarted / RunResponse / ToolCallStarted ...). Phase 3 introduces a *flat,
typed* event record that hooks, tracing exporters and the gateway can all
consume without re-parsing free-text content.

`RunEventType` is the union of legacy stream events plus the new lifecycle
events the Runner now emits explicitly (`run.failed`, `run.cancelled`,
`tool.failed`, `subagent.spawned`, ...). Keep additions backward-compatible:
existing CLI display reads `event` strings, so we ALSO keep the old names in
`RunEvent`.

This module is intentionally dependency-free so it can be imported by any
layer (Runner / Workspace / Gateway / tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional


class RunEventType(str, Enum):
    """Canonical, dot-namespaced event types emitted by the Runner.

    Legacy `RunEvent.*` values from run_response.py remain valid for the
    streaming-content path; this enum is for the *internal* lifecycle bus
    consumed by hooks, langfuse exporter, learning_report and gateway.

    Scope discipline (YAGNI): every value listed here MUST have a real
    `_emit_event` call somewhere in the codebase. Adding a new value without
    wiring its emission turns this enum into a stale interface promise that
    external hook consumers will start depending on. Add the emit site first,
    then add the enum value in the same change.
    """

    # Run lifecycle (Phase 3 — wired in Runner._run_impl)
    run_started = "run.started"
    run_completed = "run.completed"
    run_failed = "run.failed"
    run_cancelled = "run.cancelled"

    # Standing goal loop (emitted from agentica.goals.GoalManager via
    # an optional event_callback wired by the CLI / SDK consumer).
    goal_set = "goal.set"
    goal_continuing = "goal.continuing"
    goal_completed = "goal.completed"
    goal_paused = "goal.paused"


@dataclass
class RunEventRecord:
    """A single event emitted from a Runner execution.

    Hooks receive `RunEventRecord` instances via `agent._event_callback`
    (see `Runner._emit_event`). Existing dict-based callback consumers
    continue to work because `to_dict()` produces the same flat shape.
    """

    run_id: str
    event_type: RunEventType
    timestamp: float = field(default_factory=time)
    agent_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "timestamp": self.timestamp,
            **self.payload,
        }
