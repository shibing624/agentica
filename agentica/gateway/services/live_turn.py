# -*- coding: utf-8 -*-
"""In-process registry for web chat runs.

The HTTP connection is a subscriber, not the owner of the Agent run. A
refresh or dropped SSE unsubscribes; only an explicit cancel stops the
agent and waits for the session lock.

Session indexes are ``(owner, session_id)``. The client supplies
``session_id``, so a shared registry keyed on that alone lets one account
conflict with — or drop — another's run.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_service import AgentService

_CANCEL_WAIT_S = 30.0
_CANCEL_FORCE_S = 2.0
# Finished runs stay long enough for a refresh to replay, then go away.
RETAIN_AFTER_DONE_S = 600.0

# Replay buffer caps (run-in-progress has no TTL). Consecutive content /
# thinking deltas are merged so a long /goal does not store one row per token.
MAX_BUFFER_EVENTS = 4000
MAX_BUFFER_CHARS = 1_000_000
# Slow subscribers are disconnected rather than growing an unbounded queue.
SUB_QUEUE_MAX = 256
_COALESCE_EVENTS = frozenset({"content", "thinking"})

# starting | running | cancelling | completed | cancelled | failed
_TERMINAL = frozenset({"completed", "cancelled", "failed"})

_SessionKey = Tuple[str, str]


def _event_chars(ev: dict) -> int:
    data = ev.get("data")
    if isinstance(data, str):
        return len(data)
    if isinstance(data, dict):
        return sum(len(str(v)) for v in data.values())
    return len(str(data or ""))


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
        self._buffer_chars = 0

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
        last = self.events[-1] if self.events else None
        if (
            last is not None
            and last.get("event") in _COALESCE_EVENTS
            and wrapped["event"] == last["event"]
            and isinstance(wrapped.get("data"), str)
            and isinstance(last.get("data"), str)
        ):
            self._buffer_chars -= _event_chars(last)
            last["data"] += wrapped["data"]
            last["seq"] = wrapped["seq"]
            self._buffer_chars += _event_chars(last)
        else:
            self.events.append(wrapped)
            self._buffer_chars += _event_chars(wrapped)
        self._trim_buffer()
        for q in list(self._subs):
            try:
                q.put_nowait(wrapped)
            except asyncio.QueueFull:
                self.unsubscribe(q)
                dropped = getattr(q, "_dropped", None)
                if dropped is not None:
                    dropped.set()

    def _trim_buffer(self) -> None:
        while len(self.events) > 1 and (
            len(self.events) > MAX_BUFFER_EVENTS
            or self._buffer_chars > MAX_BUFFER_CHARS
        ):
            dropped = self.events.pop(0)
            self._buffer_chars -= _event_chars(dropped)

    def replay(self, after: int = 0) -> list[dict]:
        return [ev for ev in self.events if ev["seq"] > after]

    def finish(self, status: str) -> None:
        if self.done:
            return
        self.status = status
        self.done = True
        self.finished_at = time.monotonic()
        for q in list(self._subs):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                dropped = getattr(q, "_dropped", None)
                if dropped is not None:
                    dropped.set()

    def subscribe(self, after: int = 0) -> asyncio.Queue:
        """Attach a live subscriber. Replay is ``replay(after)``, not this queue.

        Prefilling the queue with the whole buffer would either overflow a
        bounded queue or re-copy megabytes per tab refresh.
        """
        del after  # replay is separate; keep the signature the routes already use
        q: asyncio.Queue = asyncio.Queue(maxsize=SUB_QUEUE_MAX)
        q._dropped = asyncio.Event()
        if not self.done:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            pass


_by_run: Dict[str, LiveTurn] = {}
_by_session: Dict[_SessionKey, str] = {}


def _skey(session_id: str, owner: str) -> _SessionKey:
    return (owner, session_id)


def gc() -> None:
    """Drop finished runs whose reconnect window has expired."""
    now = time.monotonic()
    for rid, turn in list(_by_run.items()):
        if turn.done and turn.finished_at and now - turn.finished_at >= RETAIN_AFTER_DONE_S:
            _drop_run(rid)


def get_run(run_id: str) -> Optional[LiveTurn]:
    gc()
    return _by_run.get(run_id)


def get(session_id: str, owner: str) -> Optional[LiveTurn]:
    gc()
    rid = _by_session.get(_skey(session_id, owner))
    return _by_run.get(rid) if rid else None


def active(session_id: str, owner: str) -> Optional[LiveTurn]:
    turn = get(session_id, owner)
    if turn is None or turn.done:
        return None
    return turn


def iter_owner(owner: str) -> list[LiveTurn]:
    """Live (and recently finished) turns for one account."""
    gc()
    return [t for t in _by_run.values() if t.owner == owner]


def start(session_id: str, owner: str, kind: str = "chat") -> LiveTurn:
    gc()
    key = _skey(session_id, owner)
    old_rid = _by_session.get(key)
    if old_rid:
        old = _by_run.get(old_rid)
        if old is not None and old.done:
            _drop_run(old_rid)
    turn = LiveTurn(session_id, owner, kind)
    _by_run[turn.run_id] = turn
    _by_session[key] = turn.run_id
    return turn


def _drop_run(run_id: str) -> None:
    turn = _by_run.pop(run_id, None)
    if turn is None:
        return
    key = _skey(turn.session_id, turn.owner)
    if _by_session.get(key) == run_id:
        _by_session.pop(key, None)


def drop(session_id: str, owner: str) -> None:
    """Remove this owner's runs for ``session_id``. Other accounts are untouched."""
    rid = _by_session.pop(_skey(session_id, owner), None)
    if rid:
        _by_run.pop(rid, None)
    for r, t in list(_by_run.items()):
        if t.session_id == session_id and t.owner == owner:
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
    if run_id:
        turn = get_run(run_id)
    elif session_id and owner is not None:
        turn = get(session_id, owner)
    else:
        turn = None
    if turn is None:
        return {"status": "completed", "cancelled": False}
    if owner is not None and not owned(turn, owner):
        raise PermissionError("not the run owner")
    if turn.done or turn.status in _TERMINAL:
        return {"status": turn.status, "cancelled": False, "run_id": turn.run_id}

    turn.status = "cancelling"
    svc.cancel_session(turn.session_id, owner=turn.owner)
    if (
        turn.task is not None
        and not turn.task.done()
        and not svc.has_cached_session(turn.session_id, owner=turn.owner)
    ):
        turn.task.cancel()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lock_held = svc.is_session_active(turn.session_id, owner=turn.owner)
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
        "cancelled": not svc.is_session_active(turn.session_id, owner=turn.owner),
        "run_id": turn.run_id,
    }
