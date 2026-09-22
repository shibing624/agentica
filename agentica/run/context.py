# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: RunContext / TaskAnchor / RunStatus

Public Python SDK surface (re-exported from `agentica.__init__`), but
intentionally NOT exposed over HTTP. Phase 0 of arch_v5.md is "SDK-first":
hooks, tracing exporters, tests and downstream Python code may freely import
and read these types; the gateway only sees `/api/chat` payloads.

Phase 0 of arch_v5.md: SDK-first run lifecycle. These types let the Runner,
Session, hooks and tracing layers agree on a single run identity, status set
and original task description -- without exposing a /api/runs endpoint.

Why a TaskAnchor:
    Long, multi-turn agent runs lose the original goal because:
      - `agent.run_input` is the *latest* user turn, not the original goal
      - compression may drop or summarize the first user message
      - retrieval (memory / experience) keeps re-querying with the latest
        turn instead of the original task description

    `TaskAnchor` pins the run-local original goal so prompts.py, the
    compression pipeline and workspace retrieval can all reference the
    *same* anchor for the lifetime of the run.

This module deliberately has zero runtime dependency on Agent / Runner so it
can be safely imported from anywhere (including tests and gateway layers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4


class RunStatus(str, Enum):
    """Lifecycle states for a single Runner execution.

    Mirrors arch_v5.md §"Internal Run Lifecycle". Keep this enum minimal --
    new states require the Runner / hooks contract to be updated as well.
    """

    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class RunSource(str, Enum):
    """Where a run originates. Used for provenance + report grouping.

    Entry points set this explicitly through RunConfig.source. The Runner only
    overrides it for subagent children because parent_run_id is authoritative.
    """

    sdk = "sdk"              # Direct agent.run() / runner.run()
    cli = "cli"              # Agentica interactive CLI
    gateway = "gateway"      # Gateway HTTP/SSE entry points
    cron = "cron"            # Scheduled cron task execution
    workflow = "workflow"    # Deterministic workflow execution
    subagent = "subagent"    # Spawned via subagent.spawn() (parent_run_id is set)


@dataclass
class TaskAnchor:
    """Run-local original goal anchor.

    Captured ONCE at run start from the user's first message and frozen for
    the lifetime of the run. Compression and retrieval read from here so the
    primary intent never drifts.

    Fields are intentionally simple strings / lists so the anchor can be
    serialized into a single system-prompt block without round-tripping
    through pydantic.
    """

    # Original user request, verbatim. Never mutated after run start.
    goal: str = ""

    # Same as `goal` by default, but exposed separately because retrieval
    # layers may want a normalized / shorter query string while prompts may
    # want the verbatim goal. Keep them split so we can iterate independently.
    source_query: str = ""

    # Acceptance criteria parsed (or hinted) from the original message. Free
    # text -- the LLM populates this on its own; we just store it.
    acceptance_criteria: List[str] = field(default_factory=list)

    # User-imposed constraints ("don't touch X", "must use Y").
    constraints: List[str] = field(default_factory=list)

    # Confirmed facts that should survive compression (URLs, IDs, file paths
    # the user explicitly asked us to operate on).
    confirmed_facts: List[str] = field(default_factory=list)

    # Optional one-line "what to do next" hint. May be updated mid-run by the
    # planning layer; everything else above is frozen.
    next_step_hint: Optional[str] = None

    # Where this anchor came from. Gates system-prompt rendering:
    #   "message" — auto-built from `agent.run(message)`'s first message;
    #               never rendered into the system prompt, so a transcript /
    #               replay / seed message can't leak into every turn.
    #               Still used as the retrieval query (`_get_anchor_query`).
    #   "goal"    — built from an explicit goal entry point: `run_goal()`,
    #               CLI `/goal`, or a `session_log.load_goal()` active goal.
    #               Rendered as the `## Original Task` block for long-task
    #               drift defense.
    source: Literal["message", "goal"] = "message"

    @classmethod
    def from_message(cls, message: Any) -> "TaskAnchor":
        """Build an anchor from the very first user-supplied message.

        Accepts strings, dicts (with `content`), or `Message` instances.
        Anything we can't extract is silently dropped (anchor stays empty).
        """
        text = ""
        if isinstance(message, str):
            text = message
        elif isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                text = content
        else:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = content

        text = (text or "").strip()
        return cls(goal=text, source_query=text)

    def to_prompt_block(self) -> str:
        """Render anchor as a stable system-prompt block.

        Used by prompts.py to inject the original goal at the top of the
        message list. Only renders for explicit goal-sourced anchors —
        message-sourced anchors (the default for plain `agent.run(msg)`)
        always return "" so a transcript / replay / seed message can never
        leak into the system prompt every turn. Callers that need long-task
        drift defense should go through `Agent.run_goal()` or set
        `source="goal"` explicitly.
        """
        if self.source != "goal":
            return ""
        if not self.goal and not (
            self.acceptance_criteria
            or self.constraints
            or self.confirmed_facts
            or self.next_step_hint
        ):
            return ""

        lines = ["<original_task>"]
        if self.goal:
            lines.append(f"GOAL: {self.goal}")
        if self.acceptance_criteria:
            lines.append("ACCEPTANCE CRITERIA:")
            for item in self.acceptance_criteria:
                lines.append(f"  - {item}")
        if self.constraints:
            lines.append("CONSTRAINTS:")
            for item in self.constraints:
                lines.append(f"  - {item}")
        if self.confirmed_facts:
            lines.append("CONFIRMED FACTS:")
            for item in self.confirmed_facts:
                lines.append(f"  - {item}")
        if self.next_step_hint:
            lines.append(f"NEXT STEP HINT: {self.next_step_hint}")
        # Standing-goal completion contract. This block is re-rendered every
        # turn (including turn 1), so it is the earliest place to tell the
        # model HOW to finish: prove completion with the verify_completion
        # tool rather than just narrating "done". The run_goal loop only ends
        # when verify_completion passes (or a budget cap is hit).
        lines.append(
            "COMPLETION: Do not stop until done. When you believe the goal is "
            "met, call the `verify_completion` tool to PROVE it — for code use "
            "mode=\"test\" with a verify_command like `pytest ...` (exit 0 == "
            "done); for other deliverables use mode=\"criteria\". Only a passing "
            "verify_completion ends the task; if it fails, fix the gap and "
            "verify again. If blocked, call update_goal(status=\"paused\")."
        )
        lines.append("</original_task>")
        return "\n".join(lines)


@dataclass
class RunContext:
    """Single source of truth for a run's identity and lifecycle.

    Created by the Runner at run start, attached to `Agent.run_context`,
    and read by hooks / tracing / session archives. Fields here are
    deliberately additive -- existing code that only reads `agent.run_id`
    keeps working.
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None  # set when run is a subagent / workflow child
    agent_id: Optional[str] = None

    source: RunSource = RunSource.sdk
    status: RunStatus = RunStatus.created

    started_at: float = field(default_factory=lambda: time())
    ended_at: Optional[float] = None

    trace_id: Optional[str] = None
    error: Optional[str] = None

    task_anchor: TaskAnchor = field(default_factory=TaskAnchor)

    # Free-form metadata. Used by exporters (Langfuse) and reports.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_running(self) -> None:
        self.status = RunStatus.running

    def mark_completed(self) -> None:
        self.status = RunStatus.completed
        self.ended_at = time()

    def mark_failed(self, error: str) -> None:
        self.status = RunStatus.failed
        self.error = error
        self.ended_at = time()

    def mark_cancelled(self, reason: str = "user_cancelled") -> None:
        self.status = RunStatus.cancelled
        self.error = reason
        self.ended_at = time()

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "agent_id": self.agent_id,
            "source": self.source.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "trace_id": self.trace_id,
            "error": self.error,
            "task_anchor": {
                "goal": self.task_anchor.goal,
                "source_query": self.task_anchor.source_query,
                "acceptance_criteria": list(self.task_anchor.acceptance_criteria),
                "constraints": list(self.task_anchor.constraints),
                "confirmed_facts": list(self.task_anchor.confirmed_facts),
                "next_step_hint": self.task_anchor.next_step_hint,
                "source": self.task_anchor.source,
            },
            "metadata": dict(self.metadata),
        }
