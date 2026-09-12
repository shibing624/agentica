# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The notify sink — a one-way observation channel plus an optional
two-way decision channel to a local desktop app.

Talks HTTP over a Unix domain socket. Two paths, and which one blocks is decided
by the *path*, never by a body field — a malformed body must not be able to hang
a run:

    POST /event   fire and forget    2s     non-blocking run lifecycle
    POST /await   waits for a human 55s    approval / question

**The degradation ladder is the whole point.** Any failure at any stage falls
back to the terminal prompt and never returns "allow":

    1. socket reachable            -> normal round trip
    2. connect fails               -> fall back *immediately*, do not wait out
                                      the timeout (the desktop app is not
                                      running; that user must see no difference)
    3. connected but no answer     -> timeout, then fall back
    4. unparseable / unknown body  -> "no decision", fall back. Never guess,
                                      never default to allow.

Non-blocking delivery runs on a dedicated daemon thread with a bounded queue
rather than an asyncio task: the CLI runs each turn through its own
``asyncio.run()`` (see ``utils/async_utils.run_sync``), so a task bound to that
loop dies when the turn ends. A thread outlives all of them, and a synchronous
client means no loop at all to get this wrong.

Failures are swallowed on purpose. This is an observation channel: **its faults
must never become the agent's faults**, matching ``Runner._emit_event``'s
contract for the in-process callback.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from typing import Any, Callable, Dict, Optional

import httpx

from agentica.notify.config import (
    DELIVERY_TIMEOUT_SECONDS,
    QUEUE_MAXSIZE,
    NotifyConfig,
    load_notify_config,
)
from agentica.utils.log import logger

SOURCE = "agentica"
ENVELOPE_VERSION = 1

#: Titles the desktop app can show as-is (<= 40 chars per the contract).
_EVENT_TITLES = {
    "run.started": "run started",
    "run.completed": "run completed",
    "run.failed": "run failed",
    "run.cancelled": "run cancelled",
    "needs.approval": "waiting for approval",
    "needs.input": "waiting for your answer",
}

#: Decisions the contract allows back. Anything else is treated as "no
#: decision" rather than being coerced — guessing here would approve a command.
_ALLOWED_DECISIONS = frozenset({"allow", "deny", "allow_prefix", "deny_prefix"})


def _tty_name() -> Optional[str]:
    """The controlling terminal name, best effort. Used only to jump back."""
    try:
        return os.ttyname(sys.stdin.fileno())
    except Exception:
        return None


class NotifySink:
    """Durable sink instance. Built once per process; see ``install_sink``.

    Not thread-safe by design requirement but is safe in practice: the queue is
    a ``queue.Queue`` and every shared field is either set at construction or a
    plain counter read under the GIL.
    """

    def __init__(
        self,
        config: NotifyConfig,
        *,
        transport_factory: Optional[Callable[[str], Any]] = None,
    ):
        self._cfg = config
        # Injectable so tests can point at an in-process fake server instead of
        # mocking HTTP (mocking it would skip the timeout behaviour we care about).
        self._transport_factory = transport_factory or (
            lambda path: httpx.HTTPTransport(uds=path)
        )
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._dropped = 0
        self._sent = 0
        if config.enabled:
            self._start_worker()

    # ------------------------------------------------------------------ setup

    @property
    def config(self) -> NotifyConfig:
        return self._cfg

    def _start_worker(self) -> None:
        # A thread, not an asyncio task, and deliberately so: the CLI drives
        # each turn through its own ``asyncio.run()``, so a task created from
        # one turn's loop is dead by the next turn. This thread spans the whole
        # session. See the module docstring.
        self._worker = threading.Thread(
            target=self._drain_queue, name="agentica-notify-sink", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------- non-blocking send

    def emit_event(
        self,
        event: str,
        *,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        """Queue one lifecycle event. Never raises, never blocks."""
        if not self._cfg.enabled or not self._cfg.event_enabled(event):
            return
        try:
            envelope = self._envelope(
                event, session_id=session_id, payload=payload, work_dir=work_dir
            )
        except Exception as exc:
            logger.debug(f"notify sink: could not build {event} envelope: {exc}")
            return
        self._enqueue(envelope)

    def _enqueue(self, envelope: dict) -> None:
        """Put without blocking; drop the oldest when the desktop app lags."""
        try:
            self._queue.put_nowait(envelope)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()  # oldest
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 50 == 0:
                logger.debug(
                    f"notify sink: queue full, dropped {self._dropped} event(s) so far"
                )
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            pass

    def _drain_queue(self) -> None:
        """Worker loop: deliver queued events, one short-lived client at a time."""
        while not self._stop.is_set():
            try:
                envelope = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # raise_transport=True: the worker has to be able to tell a
                # delivery that happened from one that did not, or a dead
                # desktop app would look like a healthy stream of events.
                self._post(
                    "/event",
                    envelope,
                    timeout=DELIVERY_TIMEOUT_SECONDS,
                    wait=False,
                    raise_transport=True,
                )
                self._sent += 1
            except Exception as exc:
                # Observation only: a dead desktop app is debug noise, not a fault.
                logger.debug(f"notify sink: /event delivery failed: {exc}")

    # ------------------------------------------------------------ two-way send

    def await_decision(
        self,
        event: str,
        *,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Block for a desktop decision. ``None`` means "fall back to terminal".

        Returns ``{"decision": ...}`` or ``{"answer": ...}`` only when the
        desktop app gave a usable answer. Every other outcome — disabled,
        not permitted, connect failure, timeout, HTTP error, unparseable body —
        returns None so the caller falls through to the terminal prompt.
        """
        if not self._cfg.enabled:
            return None
        # The switch gates *deciding*, not *knowing*. With it off the app may
        # still be told a decision is pending; it just cannot make one.
        if not self._cfg.approve_from_desktop:
            return None
        try:
            envelope = self._envelope(
                event, session_id=session_id, payload=payload, work_dir=work_dir
            )
        except Exception as exc:
            logger.debug(f"notify sink: could not build {event} envelope: {exc}")
            return None
        try:
            response = self._post(
                "/await",
                envelope,
                timeout=self._cfg.timeout_seconds,
                wait=True,
                raise_transport=True,
            )
        except httpx.TimeoutException:
            logger.debug(
                f"notify sink: {event} not answered within "
                f"{self._cfg.timeout_seconds:g}s; falling back to the terminal"
            )
            return None
        except httpx.ConnectError:
            # Level 2 of the ladder: the desktop app is not running. Fall back
            # now — making the user wait out a timeout for an app that is off
            # is how a feature like this gets switched off for good.
            logger.debug(f"notify sink: no desktop app at {self._cfg.resolved_socket}")
            return None
        except Exception as exc:
            logger.debug(f"notify sink: {event} transport failed: {exc}")
            return None
        if response is None:
            return None
        return self._parse_decision(response, event)

    @staticmethod
    def _parse_decision(response: httpx.Response, event: str) -> Optional[Dict[str, Any]]:
        """Validate a decision body. Anything unrecognized means "no decision"."""
        if response.status_code != 200:
            logger.debug(
                f"notify sink: {event} answered HTTP {response.status_code}; "
                f"falling back to the terminal"
            )
            return None
        try:
            body = response.json()
        except Exception:
            logger.debug(f"notify sink: {event} response was not JSON; falling back")
            return None
        if not isinstance(body, dict):
            return None
        decision = body.get("decision")
        if isinstance(decision, str) and decision in _ALLOWED_DECISIONS:
            return {"decision": decision}
        answer = body.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"answer": answer}
        # Includes {"reject": true} and anything malformed: no decision is a
        # valid outcome, and it must not be confused with approval.
        logger.debug(f"notify sink: {event} returned no usable decision; falling back")
        return None

    # --------------------------------------------------------------- transport

    def _post(
        self,
        path: str,
        envelope: dict,
        *,
        timeout: float,
        wait: bool,
        raise_transport: bool = False,
    ) -> Optional[httpx.Response]:
        """One request over a fresh short-lived client.

        A client per request keeps this usable from the worker thread and from
        whatever thread is parked on a decision, without shared-state questions
        about a pooled connection the desktop app may have closed.
        """
        headers = {"Content-Type": "application/json"}
        token = self._cfg.resolved_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        transport = self._transport_factory(self._cfg.resolved_socket)
        try:
            with httpx.Client(
                transport=transport, base_url="http://localhost", timeout=timeout
            ) as client:
                response = client.post(path, json=envelope, headers=headers)
        except Exception:
            if raise_transport:
                raise
            return None
        if not wait:
            # The reply to /event is not part of the contract; reading it only
            # confirms the app accepted the POST.
            return None
        return response

    # ---------------------------------------------------------------- envelope

    def _envelope(
        self,
        event: str,
        *,
        session_id: Optional[str],
        payload: Optional[Dict[str, Any]],
        work_dir: Optional[str],
    ) -> dict:
        import time as _time

        transport: Dict[str, Any] = {"ppid": os.getppid()}
        cwd = work_dir or os.getcwd()
        if cwd:
            transport["cwd"] = str(cwd)
        tty = _tty_name()
        if tty:
            transport["tty"] = tty
        body = dict(payload or {})
        body.setdefault("title", _EVENT_TITLES.get(event, event)[:40])
        return {
            "v": ENVELOPE_VERSION,
            "source": SOURCE,
            "session_key": session_id or "",
            "event": event,
            "ts": _time.time(),
            "transport": transport,
            "payload": body,
        }

    # ----------------------------------------------------------------- testing

    def stop(self, timeout: float = 1.0) -> None:
        """Stop the worker. For teardown and tests."""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None


# ---------------------------------------------------------------- process-wide

_sink: Optional[NotifySink] = None
_lock = threading.Lock()
#: Install-time decision: ``False`` means "this process never wires a sink".
_installed = False

#: Optional caller-supplied "nothing more is queued" probe. The CLI sets it to
#: look at its pending-input queue, which the sink cannot see. When unset, or
#: when it raises, completion is reported — suppress only on a confident False.
_idle_provider: Optional[Callable[[], bool]] = None



def set_idle_provider(provider: Optional[Callable[[], bool]]) -> None:
    """Register a probe for "is there more work queued for this session?".

    Segregated from the sink because only the host knows: the CLI's
    ``pending_queue`` lives in the interactive app, not on the agent. Called at
    install time; a process with no queue simply never sets it.
    """
    global _idle_provider
    _idle_provider = provider


def _nothing_more_queued() -> bool:
    """True only when the host is confident nothing else is waiting.

    False on absent provider or any exception: this gates reporting, so an
    unanswerable question must not silence a real completion.
    """
    provider = _idle_provider
    if provider is None:
        return True
    try:
        return bool(provider())
    except Exception as exc:
        logger.debug(f"notify sink: idle probe failed: {exc}")
        return True


def install_sink(
    config: Optional[NotifyConfig] = None,
    *,
    transport_factory: Optional[Callable[[str], Any]] = None,
) -> Optional[NotifySink]:
    """Build the process sink, or return None when it is disabled.

    Called once per process (the CLI installs it at startup). ``enabled: false``
    means *nothing* is wired — no queue, no worker, no callbacks — decided here
    once rather than re-read every turn.
    """
    global _sink, _installed
    cfg = config if config is not None else load_notify_config()
    with _lock:
        _installed = True
        if not cfg.enabled:
            _sink = None
            return None
        # A local socket is still an attack surface: without a shared secret,
        # any process on this machine could forge a "needs.approval" and collect
        # an "allow". Established once, here, because both sides read the file.
        from agentica.notify.token import ensure_token

        ensure_token(cfg)
        _sink = NotifySink(cfg, transport_factory=transport_factory)
        return _sink


def get_sink() -> Optional[NotifySink]:
    """The installed sink, or None when this process has none."""
    return _sink


def reset_sink_for_tests() -> None:
    """Drop any installed sink so a test starts from a clean process state."""
    global _sink, _installed
    with _lock:
        if _sink is not None:
            _sink.stop()
        _sink = None
        _installed = False


def notify_sink_dispatch(
    record: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
) -> None:
    """Hand one ``RunEventRecord`` to the sink. Never raises, never blocks.

    Called from ``Runner._emit_event`` alongside (not instead of) the in-process
    callback, so a broken sink and a broken callback cannot take each other down.

    ``goal.*`` events deliberately do not come through here: they are emitted by
    ``GoalManager`` on its own callback, and the desktop app has no use for the
    goal loop. Wiring them is a future decision, not an oversight.

    But "no goal events" is not the same as "goal does not affect the protocol":
    a standing goal turns one user request into N runs, so ``run.completed`` is
    held back while a goal is still active (see ``_goal_still_running``). The
    agent is passed for that one check.
    """
    sink = _sink
    if sink is None:
        return
    try:
        event = getattr(record, "event_type", None)
        name = getattr(event, "value", None) or str(event)
        if name == "run.completed":
            # Deferred while a goal is driving this agent: the run that just
            # ended is one lap of several, so "you can come back now" is not
            # true yet. The signal is two-sided on purpose — see
            # ``goal_finished`` for why one point in time cannot answer this.
            if _goal_is_driving(agent) or not _nothing_more_queued():
                # Remember on the agent itself, not in a process registry: this
                # is per-session state, it dies with the agent, and Agent is not
                # hashable so a set/dict keyed by it would raise outright.
                _mark_deferred(agent)
                return
            _emit_completion(sink, record, session_id=session_id, work_dir=work_dir)
            return
        sink.emit_event(
            name,
            session_id=session_id,
            work_dir=work_dir,
            payload=_run_event_payload(record),
        )
    except Exception as exc:
        logger.debug(f"notify sink: dispatch failed: {exc}")


def _emit_completion(sink: NotifySink, record: Any, *, session_id, work_dir) -> None:
    """Report one completed run, once."""
    payload = _run_event_payload(record)
    payload.setdefault("title", "run completed")
    sink.emit_event("run.completed", session_id=session_id, work_dir=work_dir, payload=payload)


def goal_finished(agent: Any, *, session_id: Optional[str] = None,
                  work_dir: Optional[str] = None) -> None:
    """A goal drove this agent and has now stopped. Report the held completion.

    Called by the CLI's goal hook on its exit path, which is the one place that
    knows whether another lap is coming.

    Only fires when a completion was actually held back: a session with no goal,
    or one whose goal never ran a lap, reports nothing extra.
    """
    sink = _sink
    if sink is None or agent is None:
        return
    try:
        if not getattr(agent, _DEFERRED_FLAG, False):
            return
        setattr(agent, _DEFERRED_FLAG, False)
        _emit_completion(sink, None, session_id=session_id, work_dir=work_dir)
    except Exception as exc:
        logger.debug(f"notify sink: could not report the deferred completion: {exc}")


#: Set on an agent whose ``run.completed`` was held back mid-goal. On the agent
#: rather than in a module-level registry because ``Agent`` is unhashable (so a
#: set would raise) and because this is per-session state that should die with
#: the agent.
_DEFERRED_FLAG = "_notify_completion_deferred"


def _mark_deferred(agent: Any) -> None:
    if agent is None:
        return
    try:
        setattr(agent, _DEFERRED_FLAG, True)
    except Exception as exc:
        # An agent that refuses attributes just means no deferred release; the
        # completion stays suppressed for that session, which is the safe side.
        logger.debug(f"notify sink: could not mark a deferred completion: {exc}")


def goal_is_driving(agent: Any) -> bool:
    """Is a standing goal still driving this session? (public wrapper)"""
    return _goal_is_driving(agent)


def _goal_is_driving(agent: Any) -> bool:
    """Is a standing goal driving this session, as of right now?

    ``run.completed`` is read by the desktop app as "you can come back now",
    which is only true when the work is actually over. A standing goal turns one
    request into N runs (``loop.py`` emits ``run.completed`` at the end of every
    one, then the CLI's goal hook queues the next continuation), so without this
    check a 5-lap goal reports "done" five times. The same repeat happens when
    several user messages are queued: each is its own run.

    Read from the session log rather than ``agent.goal_manager``, and that is
    deliberate: the CLI keeps its own ``GoalManager`` (``state.goal_manager``)
    and a second instance on the agent caches the log once, lazily, on first
    touch. Measured: set a goal through the CLI's manager and the agent's copy
    still reports ``is_active() == False`` forever, because it read the log
    before the goal existed. Asking the log is always current and cannot go
    stale that way. Returns False on any doubt — this gates reporting, so
    "not sure" must stay quiet rather than suppress a real completion.
    """
    try:
        session_log = getattr(agent, "_session_log", None)
        if session_log is None:
            return False
        from agentica.goals import GoalManager

        state = GoalManager(session_log).load()
        return state is not None and state.status == "active"
    except Exception as exc:
        logger.debug(f"notify sink: could not read goal state: {exc}")
        return False


def _run_event_payload(record: Any) -> Dict[str, Any]:
    """The metadata slice of a run event, per the contract.

    Only what the desktop app needs: no prompts, no tool output, no file
    contents. The discipline is the same as the budget docs' "aggregates, not
    raw streams" — this channel stays narrow even though it is a local socket.
    """
    raw = getattr(record, "payload", None)
    payload: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("agent_name", "duration_seconds", "had_response", "reason", "error"):
            if key in raw and raw[key] is not None:
                payload[key] = raw[key]
    return payload
