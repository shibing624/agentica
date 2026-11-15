# -*- coding: utf-8 -*-
"""The CLI's approval offer to a hook command: decisions, races, provenance.

Only the process side is faked. ``ApprovalRegistry`` parks and decides for real,
``_offer_approval_to_hook`` runs its real daemon-thread poll loop, and the fake
``HookRequest`` delivers scripted replies — so the race under test is the
production one between the terminal and the hook's poll thread.

No process, pipe or marker file is involved: spawning consumers, writing the
payload to stdin and parsing stdout is development-time acceptance for
``agentica.shell_hooks`` and is covered there. What is asserted here is the
decision logic layered on top of it:

* a hook decision resolves the parked wait as the hook's own answer;
* an empty / absent reply is *no decision*, never a deny, so the approval stays
  parked and the terminal can still answer;
* the terminal winning kills the losing hook before its answer can be applied;
* a reply correlates by ``tool_call_id`` — without one, or with another, it is
  discarded;
* the wake-up is announced with ``decided_by="hook"`` and the prompt remnant is
  reprinted.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, List

import pytest

from agentica.agent.approvals import ApprovalRegistry, PendingApproval
from agentica.cli.approvals import (
    _apply_hook_decision,
    _offer_approval_to_hook,
    complete_approval,
)
from agentica.cli.interactive.session_state import _InputRequest
from agentica.shell_hooks.egress import reset_hook_egress_for_tests

#: How often the polling helpers re-check a condition. Bounded well under the
#: 100ms a test is allowed to sleep, and only used where the poll thread owns
#: the event being awaited.
_POLL_SECONDS = 0.005


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Wait for a condition driven by the hook's poll thread.

    ``timeout`` only bounds a broken test; a passing one returns as soon as the
    condition holds. Checking instead of sleeping prevents a missing kill() or
    a missing announcement from passing silently.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(_POLL_SECONDS)
    return predicate()


class _FakeHookRequest:
    """A ``HookRequest`` answered from a script instead of from a process.

    The real interface is deliberately small: ``still_waiting`` says whether a
    reply could still arrive, ``wait_for_reply`` returns one reply or ``None``,
    and ``kill`` stops the wait. ``None`` means "no decision yet", never "deny",
    which is the contract these tests must preserve.
    """

    def __init__(
        self,
        replies: Any = (),
        *,
        event: str = "",
        payload: Any = None,
        request_id: str = "request-1",
    ) -> None:
        self.event = event
        self.payload = payload or {}
        self.request_id = request_id
        self._replies = list(replies or ())
        self.wait_count = 0
        self.kill_count = 0
        self.killed = threading.Event()
        #: Set once the poll thread has entered ``wait_for_reply``.
        self.first_wait = threading.Event()
        self._released = threading.Event()

    @property
    def still_waiting(self) -> bool:
        return not self.killed.is_set()

    def wait_for_reply(self, timeout: Any = None):
        self.wait_count += 1
        self.first_wait.set()
        if self._replies:
            return self._replies.pop(0)
        # Nothing scripted: behave like a consumer that has not answered yet.
        # A 10ms bounded wait keeps this stand-in from spinning without
        # inventing a cap on the caller (the real class deliberately has none).
        self._released.wait(0.01)
        return None

    def kill(self) -> None:
        # Count only the transition: callers call kill() in a ``finally`` after
        # teardown may already have killed the request.
        if not self.killed.is_set():
            self.kill_count += 1
        self.killed.set()
        self._released.set()


class _FakeHooks:
    """The monkeypatched ``start_hook_request``: records calls, scripts replies."""

    def __init__(self) -> None:
        self.payloads: List[dict] = []
        self.requests: List[_FakeHookRequest] = []
        #: One list of replies per started request. An exhausted script — or an
        #: empty one — models a consumer that never answers.
        self.script: List[List[dict]] = []

    def __call__(self, event: str, payload: dict):
        self.payloads.append(payload)
        replies = self.script.pop(0) if self.script else []
        request = _FakeHookRequest(
            replies,
            event=event,
            payload=payload,
            request_id=str(payload.get("request_id") or ""),
        )
        self.requests.append(request)
        return request

    def kill_all(self) -> None:
        for request in self.requests:
            request.kill()


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No real egress and no console output: both channels belong to the test."""
    reset_hook_egress_for_tests()
    monkeypatch.setattr(
        "agentica.cli.approvals._print_approval_record", lambda *args, **kwargs: None
    )
    yield
    reset_hook_egress_for_tests()


@pytest.fixture
def hooks(monkeypatch):
    """Replace only the process factory; everything above it stays production."""
    fake = _FakeHooks()
    monkeypatch.setattr("agentica.cli.approvals.start_hook_request", fake)
    try:
        yield fake
    finally:
        # A failed assertion must not leave a poll thread spinning on a fake.
        fake.kill_all()


@pytest.fixture
def loop():
    """A real asyncio loop: the registry future belongs to it, as in the CLI."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


class _Agent:
    session_id = "sess-1"
    work_dir = "/w"
    task_anchor = None

    class _ToolConfig:
        permission_mode = "read-only"

    tool_config = _ToolConfig()


class _State:
    def __init__(self, registry, agent=None):
        self.current_agent = agent or _Agent()
        self.approval_registry = registry
        self.input_request = None


def _pending(tool_call_id: str = "call_1") -> PendingApproval:
    return PendingApproval(
        tool_call_id=tool_call_id,
        name="execute",
        arguments={"command": "rm -rf build"},
        question="run it?",
        preview="rm -rf build",
        options=("allow", "deny"),
    )


def _request(pending: PendingApproval) -> _InputRequest:
    return _InputRequest(
        prompt="approve?",
        kind="approval",
        approval_id=pending.tool_call_id,
        approval_pending=pending,
        hook_request_id="request-1",
    )


async def _call(registry, tool_call_id, decision):
    return registry.decide(tool_call_id, decision)


def _decide_on_loop(loop, registry, tool_call_id, decision):
    """Answer the way the TUI does.

    The key handler runs on the prompt_toolkit thread, which *is* the asyncio
    loop thread, and calls ``registry.decide`` directly. Deciding from a foreign
    thread without ``call_soon_threadsafe`` would not wake the loop's future —
    which is why the hook path above routes its answer through the loop.
    """
    return asyncio.run_coroutine_threadsafe(
        _call(registry, tool_call_id, decision), loop
    ).result(timeout=5)


def _park(loop, registry, pending):
    """Register the wait the way the runner does: inside the loop.

    Returns a ``concurrent.futures.Future`` rather than the asyncio one: an
    ``asyncio.Future.result()`` takes no timeout and is not safe to wait on from
    this thread, so the test thread waits on the loop's own hand-off instead.
    """
    async def _register():
        return registry.wait(pending)

    waiter = asyncio.run_coroutine_threadsafe(_register(), loop).result(timeout=5)

    async def _await():
        return await waiter

    return asyncio.run_coroutine_threadsafe(_await(), loop)


def test_the_hook_answer_resolves_the_parked_approval(loop, hooks):
    hooks.script = [[{"decision": "allow"}]]
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    req = _request(pending)

    _offer_approval_to_hook(pending, _State(registry), loop, req)

    assert future.result(timeout=5) == "allow"
    # The answer is applied as the hook's own, not the user's.
    assert req.resolved_by == "hook"
    assert hooks.requests[0].payload["request_id"] == "request-1"
    assert _wait_until(lambda: hooks.requests[0].kill_count == 1)


def test_a_deny_is_applied_as_a_deny(loop, hooks):
    hooks.script = [[{"decision": "deny"}]]
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    req = _request(pending)

    _offer_approval_to_hook(pending, _State(registry), loop, req)

    assert future.result(timeout=5) == "deny"
    assert req.resolved_by == "hook"


def test_the_terminal_answering_first_kills_the_loser(loop, hooks, monkeypatch):
    """The hook would answer later; the user typed y now.

    The user's answer must stand, and the losing hook must be killed before it
    can answer — killed, not awaited.
    """
    emitted = []
    monkeypatch.setattr(
        "agentica.cli.approvals.emit_request_resolved",
        lambda agent, **payload: emitted.append(payload),
    )
    hooks.script = [[]]  # the hook has not answered yet
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    request = hooks.requests[0]
    assert request.first_wait.wait(2), "the hook poll never started"

    # The user answers in the terminal: this is what the TUI's key handler does.
    assert _decide_on_loop(loop, registry, "call_1", "allow") is True
    assert future.result(timeout=5) == "allow"

    assert _wait_until(lambda: request.kill_count == 1), "the losing hook was not killed"
    assert emitted == [], "a hook that lost the race must not announce a decision"


def test_a_hook_with_no_decision_leaves_the_parked_approval_alone(loop, hooks):
    hooks.script = [[]]  # the hook never answers
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    request = hooks.requests[0]
    assert _wait_until(lambda: request.wait_count >= 2), "the hook poll never ran"
    assert not future.done(), "an empty reply must not resolve the approval"
    assert registry.is_parked("call_1")

    # The terminal can still answer, which is the whole fallback contract.
    assert _decide_on_loop(loop, registry, "call_1", "allow") is True
    assert future.result(timeout=5) == "allow"
    assert _wait_until(lambda: request.kill_count == 1)


def test_no_hook_installed_is_a_no_op(loop):
    """The real ``start_hook_request`` finds nothing wired: no process, no thread."""
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()

    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    assert registry.is_parked("call_1")
    assert not future.done()
    assert _decide_on_loop(loop, registry, "call_1", "deny") is True
    assert future.result(timeout=5) == "deny"


def test_a_request_without_a_correlation_id_is_killed_at_once(loop, hooks):
    """Without a tool_call_id an answer could not be applied, so nothing waits."""
    pending = _pending(tool_call_id="")

    _offer_approval_to_hook(pending, _State(ApprovalRegistry()), loop, _request(pending))

    request = hooks.requests[0]
    assert request.kill_count == 1
    assert request.wait_count == 0  # no poll thread was started
    assert "tool_call_id" not in hooks.payloads[0]


def test_a_hook_answer_for_another_call_is_ignored(loop, hooks, monkeypatch):
    """The reply correlates by tool_call_id: another id must not decide call_1."""
    emitted = []
    monkeypatch.setattr(
        "agentica.cli.approvals.emit_request_resolved",
        lambda agent, **payload: emitted.append(payload),
    )
    hooks.script = [[{"decision": "deny"}]]
    registry = ApprovalRegistry()
    parked = _pending()
    future = _park(loop, registry, parked)
    # The card and the answering request both belong to call_1; the offer is
    # made for a different call, which is what a mis-correlated reply looks like.
    req = _request(parked)
    other = _pending(tool_call_id="call_2")

    _offer_approval_to_hook(other, _State(registry), loop, req)

    request = hooks.requests[0]
    assert _wait_until(lambda: request.kill_count == 1)
    assert not future.done()
    assert registry.is_parked("call_1")
    assert emitted == []

    # The parked call is still decided only by the terminal.
    assert _decide_on_loop(loop, registry, "call_1", "allow") is True
    assert future.result(timeout=5) == "allow"


def test_a_late_hook_decision_cannot_emit_a_conflicting_resolution(monkeypatch):
    """Once the terminal's answer is in the slot, the hook's is second and mute."""
    registry = ApprovalRegistry()
    state = _State(registry)
    pending = _pending()
    req = _request(pending)
    state.input_request = req
    assert req.submit("allow", source="terminal")
    emitted = []
    monkeypatch.setattr(
        "agentica.cli.approvals.emit_request_resolved",
        lambda *args, **kwargs: emitted.append(kwargs),
    )

    _apply_hook_decision(
        state,
        "call_1",
        "deny",
        "request-1",
        req,
    )

    assert req.result.get_nowait() == "allow"
    assert emitted == []


def test_the_payload_carries_the_anchor_and_the_offered_options(loop, hooks):
    hooks.script = [[{"decision": "deny"}]]
    agent = _Agent()
    agent.task_anchor = type("A", (), {"source_query": "the user's message"})()
    state = _State(ApprovalRegistry(), agent)
    pending = _pending()
    future = _park(loop, state.approval_registry, pending)

    _offer_approval_to_hook(pending, state, loop, _request(pending))

    assert future.result(timeout=5) == "deny"
    doc = hooks.payloads[0]
    assert doc["hook_event_name"] == "needs.approval"
    assert doc["prompt"] == "the user's message"
    assert doc["options"] == ["allow", "deny"]
    assert doc["tool_call_id"] == "call_1"
    assert doc["session_id"] == "sess-1"
    assert doc["request_id"] == "request-1"
    assert "arguments" not in doc


def test_the_hook_answer_reprints_the_approval_record(loop, hooks, monkeypatch):
    """The card is layout, not scrollback. Hiding it without a remnant would
    drop the command that was just approved."""
    printed = []
    monkeypatch.setattr(
        "agentica.cli.approvals._print_approval_record",
        lambda pending, decision, **kwargs: printed.append(
            (pending.tool_call_id, decision)
        ),
    )
    hooks.script = [[{"decision": "allow"}]]
    pending = _pending()
    registry = ApprovalRegistry()
    state = _State(registry)
    state.input_request = _request(pending)
    future = _park(loop, registry, pending)

    _offer_approval_to_hook(pending, state, loop, state.input_request)

    assert future.result(timeout=5) == "allow"
    assert _wait_until(lambda: printed == [("call_1", "allow")])
    # The hook's answer closes the prompt slot exactly like a keypress does.
    assert _wait_until(lambda: state.input_request is None)


def test_a_hook_decision_is_announced_as_decided_by_hook(loop, hooks, monkeypatch):
    """The wake-up says who decided: an automated answer, not a typed one."""
    emitted = []
    monkeypatch.setattr(
        "agentica.cli.approvals.emit_request_resolved",
        lambda agent, **payload: emitted.append(payload),
    )
    hooks.script = [[{"decision": "deny"}]]
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()

    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    assert future.result(timeout=5) == "deny"
    assert _wait_until(
        lambda: emitted
        == [
            {
                "request_id": "request-1",
                "event": "needs.approval",
                "decided_by": "hook",
                "decision": "deny",
            }
        ]
    )


def test_terminal_resolution_is_announced_once(monkeypatch):
    pending = _pending()
    state = _State(ApprovalRegistry())
    state.input_request = _request(pending)
    seen = []
    monkeypatch.setattr(
        "agentica.cli.approvals.emit_request_resolved",
        lambda agent, **payload: seen.append(payload),
    )
    monkeypatch.setattr("agentica.cli.approvals._print_approval_record", lambda *_: None)

    assert complete_approval(state, "allow") is True
    assert complete_approval(state, "allow") is False
    assert seen == [
        {
            "request_id": "request-1",
            "event": "needs.approval",
            "decided_by": "terminal",
            "decision": "allow",
        }
    ]
