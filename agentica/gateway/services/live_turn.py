# -*- coding: utf-8 -*-
"""In-process registry for web chat runs.

The HTTP connection is a subscriber, not the owner of the Agent run. A
refresh or dropped SSE unsubscribes; only an explicit cancel stops the
agent and waits for the session lock.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_service import AgentService

_CANCEL_WAIT_S = 30.0
_CANCEL_FORCE_S = 2.0
# Finished runs stay long enough for a refresh to replay, then go away.
RETAIN_AFTER_DONE_S = 600.0

# starting | running | cancelling | completed | cancelled | failed
_TERMINAL = frozenset({"completed", "cancelled", "failed"})


class LiveTurn:
    """One background run plus a seq-numbered event buffer for reconnect."""

    def __init__(self, session_id: str, owner: str, kind: str = "chat"):
        self.run_id = uuid.uuid4().hex[:16]
        self.session_id = session_id
        self.owner = owner
        self.kind = kind
        self.status = "starting"
        self.seq = 0
        self.events: list[dict] = []
        self.task: Optional[asyncio.Task] = None
        self.done = False
        self.finished_at = 0.0
        self._subs: list[asyncio.Queue] = []

    def public(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "kind": self.kind,
            "seq": self.seq,
        }

    def publish(self, item: Dict[str, Any]) -> None:
        if self.status == "starting":
            self.status = "running"
        self.seq += 1
        wrapped = {"seq": self.seq, "event": item.get("event"), "data": item.get("data")}
        self.events.append(wrapped)
        for q in list(self._subs):
            q.put_nowait(wrapped)

    def finish(self, status: str) -> None:
        if self.done:
            return
        self.status = status
        self.done = True
        self.finished_at = time.monotonic()
        for q in list(self._subs):
            q.put_nowait(None)

    def subscribe(self, after: int = 0) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for ev in self.events:
            if ev["seq"] > after:
                q.put_nowait(ev)
        if self.done:
            q.put_nowait(None)
        else:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            pass


_by_run: Dict[str, LiveTurn] = {}
_by_session: Dict[str, str] = {}


def gc() -> None:
    """Drop finished runs whose reconnect window has expired."""
    now = time.monotonic()
    for rid, turn in list(_by_run.items()):
        if turn.done and turn.finished_at and now - turn.finished_at >= RETAIN_AFTER_DONE_S:
            _drop_run(rid)


def get_run(run_id: str) -> Optional[LiveTurn]:
    gc()
    return _by_run.get(run_id)


def get(session_id: str) -> Optional[LiveTurn]:
    gc()
    rid = _by_session.get(session_id)
    return _by_run.get(rid) if rid else None


def active(session_id: str) -> Optional[LiveTurn]:
    turn = get(session_id)
    if turn is None or turn.done:
        return None
    return turn


def start(session_id: str, owner: str, kind: str = "chat") -> LiveTurn:
    gc()
    turn = LiveTurn(session_id, owner, kind)
    _by_run[turn.run_id] = turn
    _by_session[session_id] = turn.run_id
    return turn


def _drop_run(run_id: str) -> None:
    turn = _by_run.pop(run_id, None)
    if turn is None:
        return
    if _by_session.get(turn.session_id) == run_id:
        _by_session.pop(turn.session_id, None)


def drop(session_id: str) -> None:
    rid = _by_session.pop(session_id, None)
    if rid:
        _by_run.pop(rid, None)
    for r, t in list(_by_run.items()):
        if t.session_id == session_id:
            _by_run.pop(r, None)


def reset() -> None:
    """Test helper: drop every live turn."""
    _by_run.clear()
    _by_session.clear()


def owned(turn: LiveTurn, owner: str) -> bool:
    return turn.owner == owner


async def cancel_and_wait(
    svc: "AgentService",
    *,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    owner: Optional[str] = None,
    timeout: float = _CANCEL_WAIT_S,
) -> Dict[str, Any]:
    """Cancel the agent and wait until this session can take a new run.

    Idempotent: a run that already finished returns its terminal status
    (``completed`` / ``cancelled`` / ``failed``) and does not error.
    ``cancel_session`` is a no-op until the Agent is cached, so a Stop
    during ``_get_agent`` also cancels the wrapper task.
    """
    turn = get_run(run_id) if run_id else get(session_id or "")
    if turn is None:
        return {"status": "completed", "cancelled": False}
    if owner is not None and not owned(turn, owner):
        raise PermissionError("not the run owner")
    if turn.done or turn.status in _TERMINAL:
        return {"status": turn.status, "cancelled": False, "run_id": turn.run_id}

    turn.status = "cancelling"
    svc.cancel_session(turn.session_id)
    if (
        turn.task is not None
        and not turn.task.done()
        and not svc.has_cached_session(turn.session_id)
    ):
        turn.task.cancel()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lock_held = svc.is_session_active(turn.session_id)
        if not lock_held and turn.done:
            return {
                "status": turn.status,
                "cancelled": turn.status == "cancelled",
                "run_id": turn.run_id,
            }
        await asyncio.sleep(0.05)

    if turn.task is not None and not turn.task.done():
        turn.task.cancel()
        try:
            await asyncio.wait_for(turn.task, timeout=_CANCEL_FORCE_S)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    if not turn.done:
        turn.finish("cancelled")
    return {
        "status": turn.status,
        "cancelled": not svc.is_session_active(turn.session_id),
        "run_id": turn.run_id,
    }
