# -*- coding: utf-8 -*-
"""High-level shell-hook notices that do not belong to one Runner run."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentica.shell_hooks.egress import hook_egress_dispatch


def emit_session_started(
    agent: Any,
    *,
    source: str,
    profile: Optional[str] = None,
) -> None:
    """Announce the logical CLI session now occupying this process."""
    payload: Dict[str, Any] = {"source": source}
    if agent.model is not None:
        payload["model"] = agent.model.id
    if profile:
        payload["profile"] = profile
    payload["permission_mode"] = agent.tool_config.permission_mode
    if agent.session_log is not None:
        payload["transcript_path"] = str(agent.session_log.path)
    hook_egress_dispatch(
        "session.started",
        payload,
        session_id=agent.session_id,
        work_dir=agent.work_dir,
        agent=agent,
    )


def emit_session_ended(agent: Any, *, reason: str) -> None:
    """Announce that a logical CLI session is no longer active."""
    hook_egress_dispatch(
        "session.ended",
        {"reason": reason},
        session_id=agent.session_id,
        work_dir=agent.work_dir,
        agent=agent,
    )


def emit_request_resolved(
    agent: Any,
    *,
    request_id: str,
    event: str,
    decided_by: str,
    decision: Optional[str] = None,
) -> None:
    """Dismiss a request after the terminal, a hook, or cancellation wins."""
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "event": event,
        "decided_by": decided_by,
    }
    if decision:
        payload["decision"] = decision
    hook_egress_dispatch(
        "needs.resolved",
        payload,
        session_id=agent.session_id,
        work_dir=agent.work_dir,
        agent=agent,
    )
