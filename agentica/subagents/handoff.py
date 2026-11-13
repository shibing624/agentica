# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Structured handoff context for parent → child agent task delegation.

Used by ``Subagent.spawn()`` and ``Swarm`` autonomous mode to bundle the
artefacts a child needs (parent identity, condensed instructions, workspace
summary, recent history, free-form extra context) into one Markdown block
injected into the child's prompt.

Replaces ad-hoc string concatenation that was scattered across spawn paths
and made it impossible to override what the child sees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent.base import Agent


# Hard cap on each text section's length so a giant parent prompt cannot
# blow up the child's context window. Chosen as ~600 chars (≈150 tokens).
_HANDOFF_SECTION_CAP_CHARS = 600


def _truncate(text: str, cap: int = _HANDOFF_SECTION_CAP_CHARS) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 3] + "..."


@dataclass
class HandoffContext:
    """Structured context handed off from a parent agent to a child task."""

    parent_name: str
    task: str
    parent_instructions: Optional[str] = None
    parent_workspace_summary: Optional[str] = None
    parent_history_excerpt: Optional[str] = None
    extra_context: Optional[str] = None

    def render(self) -> str:
        """Render as a Markdown block injectable into the child prompt."""
        parts: List[str] = [f"## Handoff from {self.parent_name}"]
        parts.append(f"### Task\n{self.task}")
        if self.parent_instructions:
            parts.append(f"### Parent Instructions\n{_truncate(self.parent_instructions)}")
        if self.parent_workspace_summary:
            parts.append(f"### Workspace Summary\n{_truncate(self.parent_workspace_summary)}")
        if self.parent_history_excerpt:
            parts.append(f"### Recent History\n{_truncate(self.parent_history_excerpt)}")
        if self.extra_context:
            parts.append(f"### Additional Context\n{_truncate(self.extra_context)}")
        return "\n\n".join(parts)


HandoffMapper = Callable[["Agent", str, Optional[str]], HandoffContext]


def _stringify_instructions(instructions: Any) -> Optional[str]:
    """Best-effort conversion of Agent.instructions into a static string.

    Callable instructions (dynamic per-run) are intentionally excluded —
    they require an active run context that the handoff layer does not own.
    """
    if instructions is None:
        return None
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, (list, tuple)):
        return "\n".join(str(x) for x in instructions if x)
    if callable(instructions):
        return None
    return str(instructions)


def _summarize_workspace(parent: "Agent") -> Optional[str]:
    """Extract a frozen-context snapshot of the parent's workspace.

    Tries, in order:
      1. ``parent.workspace.get_frozen_context()`` — explicit workspace.
      2. ``parent.working_memory.summary.summary`` — periodic compaction
         summary that the parent already paid LLM cost to produce.
    """
    workspace = parent.workspace
    if workspace is not None:
        try:
            ctx = workspace.get_frozen_context()
        except Exception as exc:
            logger.debug(f"handoff: workspace.get_frozen_context skipped ({exc})")
            ctx = None
        if ctx:
            return ctx if isinstance(ctx, str) else str(ctx)

    wm = parent.working_memory
    if wm is not None and getattr(wm, "summary", None) is not None:
        summary_text = getattr(wm.summary, "summary", None)
        if isinstance(summary_text, str) and summary_text.strip():
            return summary_text.strip()
    return None


def _recent_history_excerpt(parent: "Agent") -> Optional[str]:
    """Pull the last assistant turn out of parent's working memory."""
    wm = parent.working_memory
    if wm is None or not getattr(wm, "runs", None):
        return None
    last_run = wm.runs[-1]
    msgs = getattr(last_run, "messages", None) or []
    if not msgs:
        return None
    tail = msgs[-2:]
    lines: List[str] = []
    for m in tail:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        lines.append(f"{role}: {str(content)[:300]}")
    return "\n".join(lines) if lines else None


def default_handoff_mapper(
    parent_agent: "Agent",
    task: str,
    extra_context: Optional[str] = None,
) -> HandoffContext:
    """Build a default :class:`HandoffContext` from a parent agent + task."""
    return HandoffContext(
        parent_name=parent_agent.name or "ParentAgent",
        task=task,
        parent_instructions=_stringify_instructions(parent_agent.instructions),
        parent_workspace_summary=_summarize_workspace(parent_agent),
        parent_history_excerpt=_recent_history_excerpt(parent_agent),
        extra_context=extra_context,
    )


__all__ = [
    "HandoffContext",
    "HandoffMapper",
    "default_handoff_mapper",
]
