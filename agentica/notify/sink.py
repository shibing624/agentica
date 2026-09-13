# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The notify sink — report run state to a local desktop app.

One path, fire and forget, over HTTP on a Unix domain socket:

    POST /event   fire and forget    2s    run lifecycle notices

**The sink is observe-only.** It used to have a second path, ``POST /await``,
that blocked for the user's answer to an approval or a question. Replies no
longer travel through this channel: the user's own hook command takes them
(``agentica/shell_hooks``), so there is no HTTP request here that waits on a
person. Removing that half is why ``needs.*`` no longer appears in this module.

**The degradation ladder is the whole point.** Delivery is observation: a
missing desktop app, a refused socket, a non-2xx response or an unusable body
are all "the notice was not shown", and none of them may affect a run. This is
also why nothing here reads a reply body any more — there is nothing to read.

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
#: The payload discipline lives in ``wire`` because the hook egress puts the
#: same strings on its own wire. Aliased to the old private name so the call
#: sites below read unchanged.
from agentica.notify.wire import clip_text as _clip_text
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


def _fan_out_event(
    name: str,
    payload: Dict[str, Any],
    *,
    session_id: Optional[str],
    work_dir: Optional[str],
    agent: Any = None,
) -> None:
    """Deliver one event to every installed external egress.

    Both are observation channels: neither may break the other, and neither may
    break the run. The hook egress is imported lazily because it imports this
    package's ``wire`` helpers — a module-level import would be a cycle.
    """
    sink = _sink
    if sink is not None:
        try:
            sink.emit_event(
                name, session_id=session_id, work_dir=work_dir, payload=payload
            )
        except Exception as exc:
            logger.debug(f"notify sink: could not emit {name}: {exc}")
    try:
        from agentica.shell_hooks.egress import hook_egress_dispatch

        hook_egress_dispatch(
            name, payload, session_id=session_id, work_dir=work_dir, agent=agent
        )
    except Exception as exc:
        logger.debug(f"shell hooks: could not send {name}: {exc}")


def notify_sink_dispatch(
    record: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
) -> None:
    """Hand one ``RunEventRecord`` to every installed external egress.

    Called from ``Runner._emit_event`` alongside (not instead of) the in-process
    callback, so a broken egress and a broken callback cannot take each other
    down.

    This is also the shared home of the ``run.completed`` deferral below: both
    egresses must agree on whether a run is really over, so that decision is
    taken once here rather than once per transport.

    ``goal.*`` events deliberately do not come through here: they are emitted by
    ``GoalManager`` on its own callback, and an external consumer has no use for
    the goal loop — one request becoming N runs is an agentica implementation
    detail, not part of the contract.
    """
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
                # The answer is taken here rather than at release time: by then
                # several laps may have run, and the reply worth showing is the
                # last one, not whatever the agent happens to hold later.
                _mark_deferred(agent, _completion_payload(agent, record))
                return
            _emit_completion(
                _completion_payload(agent, record),
                session_id=session_id,
                work_dir=work_dir,
                agent=agent,
            )
            return
        _fan_out_event(
            name,
            _run_event_payload(record),
            session_id=session_id,
            work_dir=work_dir,
            agent=agent,
        )
    except Exception as exc:
        logger.debug(f"notify sink: dispatch failed: {exc}")


def _emit_completion(
    source: Any,
    *,
    session_id: Optional[str],
    work_dir: Optional[str],
    agent: Any = None,
) -> None:
    """Report one completed run, once, to every egress.

    ``source`` is either a ``RunEventRecord`` (the run reporting itself) or an
    already-extracted payload dict (a completion released later). Both produce
    the same event shape — a consumer must not have to know which path it came
    from.
    """
    payload = dict(source) if isinstance(source, dict) else _run_event_payload(source)
    payload.setdefault("title", "run completed")
    _fan_out_event(
        "run.completed",
        payload,
        session_id=session_id,
        work_dir=work_dir,
        agent=agent,
    )


def goal_finished(agent: Any, *, session_id: Optional[str] = None,
                  work_dir: Optional[str] = None) -> None:
    """A goal drove this agent and has now stopped. Report the held completion.

    Called from every place that knows no further lap is coming: the CLI's goal
    hook (each of its exit paths), ``Agent.run_goal`` (the SDK / Gateway driver),
    and the interactive loop's failure path.

    Only fires when a completion was actually held back: a session with no goal,
    or one whose goal never ran a lap, reports nothing extra.
    """
    if agent is None:
        return
    try:
        if not getattr(agent, _DEFERRED_FLAG, False):
            return
        setattr(agent, _DEFERRED_FLAG, False)
        # The held payload, not an empty one: a completion that carries no
        # duration or agent name would be a different shape from every other
        # completion on this wire, and the consumer has no way to know why.
        held = getattr(agent, _DEFERRED_PAYLOAD, None) or {}
        setattr(agent, _DEFERRED_PAYLOAD, None)
        _emit_completion(
            held, session_id=session_id, work_dir=work_dir, agent=agent
        )
    except Exception as exc:
        logger.debug(f"notify sink: could not report the deferred completion: {exc}")


#: Set on an agent whose ``run.completed`` was held back mid-goal. On the agent
#: rather than in a module-level registry because ``Agent`` is unhashable (so a
#: set would raise) and because this is per-session state that should die with
#: the agent.
_DEFERRED_FLAG = "_notify_completion_deferred"
#: Metadata of the held completion(s), so the released event has the same shape
#: as a normal one. See ``_accumulate_held`` for what "the held metadata" means
#: when several laps were held.
_DEFERRED_PAYLOAD = "_notify_completion_held"


def _mark_deferred(agent: Any, payload: Optional[Dict[str, Any]] = None) -> None:
    if agent is None:
        return
    try:
        setattr(agent, _DEFERRED_FLAG, True)
        setattr(agent, _DEFERRED_PAYLOAD, _accumulate_held(
            getattr(agent, _DEFERRED_PAYLOAD, None), payload or {}
        ))
    except Exception as exc:
        # An agent that refuses attributes just means no deferred release; the
        # completion stays suppressed for that session, which is the safe side.
        logger.debug(f"notify sink: could not mark a deferred completion: {exc}")


def _accumulate_held(held: Optional[Dict[str, Any]],
                     lap: Dict[str, Any]) -> Dict[str, Any]:
    """Fold one held lap's metadata into the completion that will be released.

    ``duration_seconds`` sums, deliberately: the released event ends a *goal*,
    and what the reader wants from it is "how long was it busy while I was away",
    which is every lap rather than whichever one happened to finish last. The
    single-run path keeps meaning "this run", so the two agree on the field's
    reading — total busy time for the thing that just ended.

    ``had_response`` is sticky: if any lap produced a response, the goal did.
    ``agent_name`` takes the first non-empty value; it does not vary per lap.
    """
    out = dict(held or {})
    if lap.get("agent_name") and not out.get("agent_name"):
        out["agent_name"] = lap["agent_name"]
    if lap.get("had_response"):
        out["had_response"] = True
    duration = lap.get("duration_seconds")
    if isinstance(duration, (int, float)):
        out["duration_seconds"] = round(
            float(out.get("duration_seconds") or 0) + float(duration), 2
        )
    for key in ("reason", "error"):
        if lap.get(key) and not out.get(key):
            out[key] = lap[key]
    # The answer and its time describe the lap that just ended, so a later lap
    # replaces them. The released event ends a *goal*, and what the reader wants
    # there is the last thing it said while they were away, not the first.
    if lap.get("answer"):
        out["answer"] = lap["answer"]
    if lap.get("answered_at"):
        out["answered_at"] = lap["answered_at"]
    return out


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

    Only what the desktop app needs: no tool output, no file contents. Text the
    user already saw on their own screen (the turn's prompt, the answer) is
    included, clipped hard by ``_clip_text`` — the bubble shows the gist, and
    the terminal remains the place to read the whole thing.

    ``source_query`` is the run's anchor text, which is the user's message for a
    normal turn but the *goal objective* in a goal-driven session. It is sent as
    ``prompt``; a desktop showing it in a bubble is showing what started the
    work, which is right in both readings.
    """
    raw = getattr(record, "payload", None)
    payload: Dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("agent_name", "duration_seconds", "had_response", "reason", "error"):
            if key in raw and raw[key] is not None:
                payload[key] = raw[key]
        prompt = _clip_text(raw.get("prompt") or raw.get("source_query"))
        if prompt:
            payload["prompt"] = prompt
    return payload


def _completion_payload(agent: Any, source: Any) -> Dict[str, Any]:
    """The payload for ``run.completed``, including the answer if there is one.

    The answer is read from the live agent rather than carried on the run event:
    the run-event bus has other consumers (telemetry, hooks) that have no use
    for a copy of the reply, and this channel is meant to stay narrow.

    ``answered_at`` is the run's own timestamp, not the envelope's ``ts``. Those
    differ whenever a completion was held back for a goal: the envelope is
    stamped when the event is *sent* (at release, after the goal stopped), while
    this is when the reply was actually produced. A UI showing "when did it
    answer" needs the latter.
    """
    payload = source if isinstance(source, dict) else _run_event_payload(source)
    payload = dict(payload)
    answer = _clip_text(getattr(getattr(agent, "run_response", None), "content", None))
    if answer:
        payload["answer"] = answer
    if not isinstance(source, dict):
        stamp = getattr(source, "timestamp", None)
        if isinstance(stamp, (int, float)):
            payload["answered_at"] = stamp
    return payload
