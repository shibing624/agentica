# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LearningReport — first-class observability of self-evolution.

§"Experience & Self-Evolution Governance" requires every run that
touches the experience / skill subsystem to emit a structured report so the
operator can answer "did this run actually learn something, and if not why?"
without grepping logs.

A LearningReport captures, per run:
  - tool_errors_captured:    raw error events written
  - corrections_persisted:   user corrections promoted to experience cards
  - cards_written:           new / updated experience cards
  - candidates_promoted:     candidate cards reaching tier promotion
  - skill_candidate:         which experience cluster proposed a skill
  - skill_state_change:      shadow → auto / rollback decisions
  - upgrade_decision:        "spawned" / "promoted" / "rejected" / "no_action"
  - skip_reason:             when no learning happened, the *reason* (e.g.
                             "no_evidence", "below_threshold", "duplicate")

Reports are persisted as markdown under
`workspace/users/{user_id}/reports/learning/{run_id}.md` and a flat JSONL
sidecar lives at `learning.jsonl` for cheap aggregation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional

from agentica.utils.log import logger


class LearningStatus(str, Enum):
    """Outcome of the learning subsystem for one run."""

    no_action = "no_action"           # nothing happened (and that's fine)
    learned = "learned"               # at least one card / skill change persisted
    skipped = "skipped"               # something was *attempted* but gated out
    error = "error"                   # learning subsystem itself raised


@dataclass
class LearningReport:
    """Structured outcome of one run's learning pipeline.

    Field set is intentionally flat so the report can be rendered as both
    markdown (for humans) and JSON (for aggregation) without nesting.
    """

    run_id: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: float = field(default_factory=time)
    status: LearningStatus = LearningStatus.no_action

    # Counters
    tool_errors_captured: int = 0
    corrections_persisted: int = 0
    cards_written: int = 0
    candidates_promoted: int = 0
    memory_entries_written: int = 0
    memory_candidates_written: int = 0

    # Skill subsystem
    skill_candidate: Optional[str] = None       # name of skill proposed (if any)
    skill_state_change: Optional[str] = None    # "spawned"/"promoted"/"rolled_back"
    upgrade_decision: str = "no_action"

    # Why nothing happened, when status != learned
    skip_reason: Optional[str] = None

    # Free-form notes (one-liners). Avoid storing big payloads here.
    notes: List[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        if note:
            self.notes.append(note)

    def mark_learned(self) -> None:
        self.status = LearningStatus.learned
        self.skip_reason = None

    def mark_skipped(self, reason: str) -> None:
        self.status = LearningStatus.skipped
        self.skip_reason = reason

    def mark_error(self, reason: str) -> None:
        self.status = LearningStatus.error
        self.skip_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_markdown(self) -> str:
        """Render as a human-readable markdown report."""
        lines: List[str] = []
        lines.append(f"# Learning Report — {self.run_id}")
        lines.append("")
        lines.append(f"- **Status:** `{self.status.value}`")
        if self.session_id:
            lines.append(f"- **Session:** `{self.session_id}`")
        if self.agent_id:
            lines.append(f"- **Agent:** `{self.agent_id}`")
        lines.append(f"- **Created at:** `{self.created_at}`")
        lines.append("")
        lines.append("## Counters")
        lines.append(f"- tool_errors_captured: {self.tool_errors_captured}")
        lines.append(f"- corrections_persisted: {self.corrections_persisted}")
        lines.append(f"- cards_written: {self.cards_written}")
        lines.append(f"- candidates_promoted: {self.candidates_promoted}")
        lines.append(f"- memory_entries_written: {self.memory_entries_written}")
        lines.append(f"- memory_candidates_written: {self.memory_candidates_written}")
        lines.append("")
        lines.append("## Skill subsystem")
        lines.append(f"- skill_candidate: {self.skill_candidate or '-'}")
        lines.append(f"- skill_state_change: {self.skill_state_change or '-'}")
        lines.append(f"- upgrade_decision: {self.upgrade_decision}")
        if self.skip_reason:
            lines.append("")
            lines.append(f"## Skip reason\n{self.skip_reason}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines).rstrip() + "\n"


def write_learning_report(workspace: Any, report: LearningReport) -> Optional[str]:
    """Persist the report under the workspace's reports/learning folder.

    Returns the markdown file path, or None if persistence failed. Errors are
    logged at WARNING level (with full path + reason) so missing reports are
    immediately visible to the operator -- learning observability must never
    fail silently. Learning failures still don't raise: an unhealthy reports
    folder must not crash the user's run.
    """
    if workspace is None:
        return None

    try:
        target_dir = workspace.get_user_learning_reports_dir()
    except OSError as e:
        logger.warning(
            f"learning report skipped (reports dir unreachable): {e}"
        )
        return None

    # Defensive: callers may stub workspace in unit tests. Without a real
    # Path we can't write anything, and proceeding would cascade the stub
    # value into json.dumps below. Skip silently in that case.
    if not isinstance(target_dir, Path):
        return None

    md_path = target_dir / f"{report.run_id}.md"
    jsonl_path = target_dir / "learning.jsonl"

    try:
        md_path.write_text(report.to_markdown(), encoding="utf-8")
    except OSError as e:
        logger.warning(f"learning report md write failed at {md_path}: {e}")
        return None

    try:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
    except OSError as e:
        # md already on disk -- jsonl is the bonus aggregation index. Warn but
        # still return the md path so callers can confirm primary persistence.
        logger.warning(f"learning report jsonl append failed at {jsonl_path}: {e}")

    return str(md_path)
