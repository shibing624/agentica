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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentica.agent.approvals import ApprovalDecision, ApprovalRegistry, PendingApproval
from agentica.session.rows import split_image_markers

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
QuestionAnswer = str | dict[str, str]


@dataclass
class PendingQuestion:
    """One ``ask_user_question`` call waiting on this turn for an answer."""

    question_id: str
    prompt: str
    options: List[str] = field(default_factory=list)
    # The model asked for one-or-more answers; the card offers checkboxes
    # instead of one-tap buttons.
    multi_select: bool = False
    questions: list[dict] | None = None
    can_engage: bool = False
    engaged: bool = False


def _event_chars(ev: dict) -> int:
    data = ev.get("data")
    if isinstance(data, str):
        return len(data)
    if isinstance(data, dict):
        return sum(len(str(v)) for v in data.values())
    return len(str(data or ""))


class LiveTurn:
    """One background run plus a seq-numbered event buffer for reconnect."""

    def __init__(self, session_id: str, owner: str, kind: str = "chat", user_message: str = ""):
        self.run_id = uuid.uuid4().hex[:16]
        # A scheduled run: ``{"id", "name"}`` of the job that delivered it, and
        # the message it delivered (logged as automated, not as user_message).
        self.cron_job: Optional[Dict[str, str]] = None
        self.automated_message = ""
        self.session_id = session_id
        self.owner = owner
        self.kind = kind
        self.user_message = user_message
        self.status = "starting"
        self.seq = 0
        self.events: list[dict] = []
        self.task: Optional[asyncio.Task] = None
        self.user_logged: asyncio.Event = asyncio.Event()
        self.done = False
        self.started_wall = time.time()
        self.finished_at = 0.0
        self._subs: list[asyncio.Queue] = []
        self._buffer_chars = 0
        self.approvals = ApprovalRegistry()
        # ask_user_question: open questions and the futures their answers land in.
        self._questions: Dict[str, Tuple[PendingQuestion, asyncio.Future[QuestionAnswer | None]]] = {}
        # Set by the surface that started the turn when it can show the
        # question itself too (IM sends a /c/<token> card). Awaited on the loop.
        self.question_notifier: Optional[Callable[[PendingQuestion], Awaitable[None]]] = None
        # Called synchronously when a question closes (answered anywhere, or
        # timed out), so that surface takes its own copy down at once.
        self.question_closed: Optional[Callable[[str, str], None]] = None
        # Same for an approval card resolved on Web, IM, or by a mode switch.
        self.approval_closed: Optional[Callable[[str, str], None]] = None

    def resolve_approval(self, tool_call_id: str, decision: ApprovalDecision, by: str = "") -> bool:
        """First decision wins; every surface showing the card hears it closed."""
        ok = (
            self.approvals.resolve_by_mode(tool_call_id, decision)
            if by == "mode"
            else self.approvals.decide(tool_call_id, decision)
        )
        if not ok:
            return False
        self.publish({
            "event": "approval_resolved",
            "data": {"tool_call_id": tool_call_id, "decision": decision, "by": by},
        })
        if self.approval_closed is not None:
            self.approval_closed(tool_call_id, by)
        return True

    def expire_approval(self, tool_call_id: str, waited_seconds: float) -> bool:
        """Take down an approval card nobody answered, without running the call.

        Not a refusal: the registry records it as expired, so the model is told
        nobody answered and no deny-similar grant is written. A decision that
        landed first wins (``expire`` then returns False and nothing is sent).
        """
        if not self.approvals.expire(tool_call_id, waited_seconds):
            return False
        self.publish({
            "event": "approval_resolved",
            "data": {"tool_call_id": tool_call_id, "decision": "deny", "by": "timeout"},
        })
        if self.approval_closed is not None:
            self.approval_closed(tool_call_id, "timeout")
        return True

    @property
    def open_questions(self) -> List[PendingQuestion]:
        return [q for q, _ in self._questions.values()]

    def open_question(
        self,
        prompt: str = "",
        options: List[str] | None = None,
        multi_select: bool = False,
        *,
        questions: list[dict] | None = None,
    ) -> Tuple[PendingQuestion, asyncio.Future[QuestionAnswer | None]]:
        """Register a question and publish it. Must run on the event loop."""
        question = PendingQuestion(
            question_id=f"q-{uuid.uuid4().hex[:10]}",
            prompt=prompt,
            options=list(options or []),
            multi_select=multi_select,
            questions=questions,
            can_engage=self.question_notifier is None,
        )
        future = asyncio.get_running_loop().create_future()
        self._questions[question.question_id] = (question, future)
        self.publish(question_event(question))
        return question, future

    def engage_question(self, question_id: str) -> bool:
        """The user started answering in Web/Desktop; keep the card open."""
        entry = self._questions.get(question_id)
        if entry is None or not entry[0].can_engage:
            return False
        question = entry[0]
        if not question.engaged:
            question.engaged = True
            self.publish(question_event(question))
        return True

    async def wait_for_question(
        self, question: PendingQuestion, future: asyncio.Future[QuestionAnswer | None], timeout: float,
    ) -> QuestionAnswer | None:
        try:
            if timeout and not question.engaged:
                await asyncio.wait((future,), timeout=timeout)
                if not future.done() and not question.engaged:
                    self.close_question(question.question_id, by="timeout")
            return await future
        finally:
            self.close_question(question.question_id)

    def answer_question(self, question_id: str, answer: QuestionAnswer, by: str) -> bool:
        """First answer wins; every surface showing the question hears it closed."""
        entry = self._questions.get(question_id)
        if entry is None or entry[1].done():
            return False
        question, future = entry
        if question.questions is None:
            if not isinstance(answer, str):
                raise ValueError("A single question requires a text answer")
        elif not isinstance(answer, dict) or answer.keys() != {item["id"] for item in question.questions}:
            raise ValueError("Provide one answer for each question ID")
        del self._questions[question_id]
        future.set_result(answer)
        self.publish({"event": "question_resolved", "data": {"question_id": question_id, "answer": answer, "by": by}})
        if self.question_closed is not None:
            self.question_closed(question_id, by)
        return True

    def close_question(self, question_id: str, by: str = "") -> None:
        """Release an unanswered question and close its cards everywhere."""
        entry = self._questions.pop(question_id, None)
        if entry is None:
            return
        if not entry[1].done():
            entry[1].set_result(None)
        self.publish({"event": "question_resolved", "data": {"question_id": question_id, "answer": "", "by": by}})
        if self.question_closed is not None:
            self.question_closed(question_id, by)

    def release_questions(self) -> None:
        """Unblock every open question and take its cards down.

        Stop and a turn that ends while a question is open both come through
        here. Setting the future is what lets the tool return; the resolved
        event and ``question_closed`` are what take the Web card and the IM
        card down. Each question is removed before publishing its close event.
        """
        for qid in list(self._questions):
            self.close_question(qid)

    def progress_summary(self) -> Dict[str, Any]:
        """Current live step summary for IM /status and /ps."""
        tool_calls = 0
        open_calls: list[Tuple[str, str, dict]] = []
        last_note = ""
        for ev in self.events:
            kind = ev.get("event")
            data = ev.get("data")
            if kind == "tool_call" and isinstance(data, dict):
                tool_calls += 1
                tc_id = str(data.get("tool_call_id") or "")
                t_name = str(data.get("name") or "tool")
                t_args = data.get("args") if isinstance(data.get("args"), dict) else {}
                open_calls.append((tc_id, t_name, t_args))
            elif kind == "tool_result" and isinstance(data, dict):
                tc_id = str(data.get("tool_call_id") or "")
                if tc_id:
                    open_calls = [c for c in open_calls if c[0] != tc_id]
                elif open_calls:
                    open_calls.pop()
            elif kind in ("thinking", "content") and isinstance(data, str) and data.strip():
                lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
                if lines:
                    last_note = lines[-1][:140]

        active_tool = ""
        active_detail = ""
        if open_calls:
            _, active_tool, args = open_calls[-1]
            for key in ("description", "file_path", "path", "pattern", "command", "url", "task"):
                val = args.get(key)
                if isinstance(val, str) and val.strip():
                    active_detail = " ".join(val.strip().splitlines()[0].split())[:140]
                    break
        elif last_note:
            active_detail = last_note

        pending = self.approvals.list()
        return {
            "elapsed_s": max(0.0, time.time() - self.started_wall),
            "tool_calls": tool_calls,
            "active_tool": active_tool,
            "active_detail": active_detail,
            "pending_approval": pending[0].name if pending else "",
        }

    def public(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "kind": self.kind,
            "seq": self.seq,
            "first_seq": (self.events[0].get("cuts") or [(self.events[0]["seq"], 0)])[0][0] if self.events else 1,
            "started_at_ms": int(self.started_wall * 1000),
        }
        if self.user_message:
            text, images = split_image_markers(self.user_message)
            out["user_message"] = text
            if images:
                out["user_images"] = images
        if self.cron_job:
            out["cron_job"] = self.cron_job
        if self.automated_message:
            out["automated_message"] = self.automated_message
        return out

    def seq_after_tool(self, tool_call_id: str) -> int:
        """Seq of ``tool_result`` for ``tool_call_id`` in the replay buffer, or 0."""
        if not tool_call_id:
            return 0
        for ev in reversed(self.events):
            if ev.get("event") == "tool_result" and isinstance(ev.get("data"), dict):
                if ev["data"].get("tool_call_id") == tool_call_id:
                    return int(ev["seq"])
        return 0

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
            # Adjacent text deltas share one buffered event so a long reply
            # does not exhaust MAX_BUFFER_EVENTS. ``cuts`` remembers where each
            # merged delta started, so a client that already holds a prefix of
            # this event is replayed only what it has not seen.
            self._buffer_chars -= _event_chars(last)
            cuts = last.setdefault("cuts", [(last["seq"], 0)])
            cuts.append((wrapped["seq"], len(last["data"])))
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
        """Events with ``seq > after``, as the client would have received them.

        A coalesced event whose merged deltas straddle ``after`` is cut: only
        the text of the deltas past ``after`` is sent, under the event's seq.
        """
        out: list[dict] = []
        for ev in self.events:
            if ev["seq"] <= after:
                continue
            cuts = ev.get("cuts")
            if not cuts or after < cuts[0][0]:
                out.append({"seq": ev["seq"], "event": ev["event"], "data": ev["data"]})
                continue
            start = next(at for seq, at in cuts if seq > after)
            out.append({"seq": ev["seq"], "event": ev["event"], "data": ev["data"][start:]})
        return out

    def finish(self, status: str) -> None:
        if self.done:
            return
        self.approvals.deny_all()
        self.release_questions()
        self.status = status
        self.done = True
        self.finished_at = time.monotonic()
        bump_pulse(self.owner)
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

    def republish_pending_approvals(self) -> None:
        """Re-push still-pending cards (approvals and questions) so a refresh/reconnect sees them."""
        for pending in self.approvals.list():
            self.publish(approval_event(pending))
        for question in self.open_questions:
            self.publish(question_event(question))


_by_run: Dict[str, LiveTurn] = {}
_by_session: Dict[_SessionKey, str] = {}
_pulse_ver: Dict[str, int] = {}


def bump_pulse(owner: str) -> int:
    """Increment the per-owner activity counter so open Web/Desktop tabs know to sync."""
    v = _pulse_ver.get(owner, 0) + 1
    _pulse_ver[owner] = v
    return v


def pulse(owner: str) -> int:
    """This process's change counter for ``owner``: moves when a run starts or ends."""
    return _pulse_ver.get(owner, 0)


def _skey(session_id: str, owner: str) -> _SessionKey:
    return (owner, session_id)


def approval_event(pending: PendingApproval) -> Dict[str, Any]:
    """SSE envelope for one parked tool call (no ``seq``; ``publish`` adds it)."""
    return {
        "event": "approval_request",
        "data": {
            "tool_call_id": pending.tool_call_id,
            "name": pending.name,
            "args": pending.arguments,
            "question": pending.question,
            "preview": pending.preview,
            "similar_label": pending.similar_label,
            "options": list(pending.options),
        },
    }


def question_event(question: PendingQuestion) -> Dict[str, Any]:
    """SSE envelope for one open ``ask_user_question``."""
    data: Dict[str, Any] = {
        "question_id": question.question_id,
        "can_engage": question.can_engage,
        "engaged": question.engaged,
    }
    if question.questions is not None:
        data["questions"] = question.questions
    else:
        data.update(prompt=question.prompt, options=list(question.options), multi_select=question.multi_select)
    return {
        "event": "question_request",
        "data": data,
    }


def gc() -> None:
    """Drop finished runs whose reconnect window has expired.

    A turn with pending approvals or questions is never dropped: the card is
    still waiting, even if ``done`` was set somehow without ``deny_all``.
    """
    now = time.monotonic()
    for rid, turn in list(_by_run.items()):
        if turn.approvals.size or turn.open_questions:
            continue
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


def start(session_id: str, owner: str, kind: str = "chat", user_message: str = "") -> LiveTurn:
    gc()
    key = _skey(session_id, owner)
    old_rid = _by_session.get(key)
    if old_rid:
        old = _by_run.get(old_rid)
        if old is not None and old.done:
            _drop_run(old_rid)
    turn = LiveTurn(session_id, owner, kind, user_message=user_message)
    _by_run[turn.run_id] = turn
    _by_session[key] = turn.run_id
    bump_pulse(owner)
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
    _pulse_ver.clear()


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
    turn.approvals.deny_all()
    # Same as deny_all: a question parks the tool thread, so Stop cannot
    # reach the runner's cancel check until that wait is released.
    turn.release_questions()
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
