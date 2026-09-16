# -*- coding: utf-8 -*-
"""The CLI's approval offer to a hook command: the answer is applied to the
registry as the user's own, and the terminal winning kills the loser.

A real asyncio loop and a real ``ApprovalRegistry``: the thing under test is the
race between two threads, so a synchronous stand-in for the loop would test the
stand-in instead.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest

from agentica.agent.approvals import ApprovalRegistry, PendingApproval
from agentica.cli.approvals import (
    _apply_hook_decision,
    _offer_approval_to_hook,
    complete_approval,
)
from agentica.cli.interactive.session_state import _InputRequest
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class _Agent:
    session_id = "sess-1"
    work_dir = "/w"
    task_anchor = None
    run_context = None
    model = None
    session_log = None

    class _ToolConfig:
        permission_mode = "ask"

    tool_config = _ToolConfig()


class _State:
    def __init__(self, registry):
        self.current_agent = _Agent()
        self.approval_registry = registry
        self.input_request = None


def _pending():
    return PendingApproval(
        tool_call_id="call_1",
        name="execute",
        arguments={"command": "rm -rf build"},
        question="run it?",
        preview="rm -rf build",
        options=("allow", "deny"),
    )


def _script(tmp_path, body):
    path = tmp_path / "hook.py"
    path.write_text(
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "request_id=payload['request_id']\n"
        + body,
        encoding="utf-8",
    )
    return [sys.executable, str(path)]


def _config(command):
    return ShellHooksConfig(
        enabled=True,
        consumers=[
            HookConsumer(
                name="desktop",
                command=command,
                events={"needs.resolved": False},
            )
        ],
    )


def _request(pending):
    return _InputRequest(
        prompt="approve?",
        kind="approval",
        approval_id=pending.tool_call_id,
        approval_pending=pending,
        hook_request_id="request-1",
    )


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


async def _call(registry, tool_call_id, decision):
    return registry.decide(tool_call_id, decision)


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


def test_the_hook_answer_resolves_the_parked_approval(tmp_path, loop):
    install_hook_egress(
        _config(
            _script(
                tmp_path,
                "print(json.dumps({'request_id':request_id,'decision':'allow'}))",
            )
        )
    )
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))
    assert future.result(timeout=10) == "allow"


def test_a_deny_is_applied_as_a_deny(tmp_path, loop):
    install_hook_egress(
        _config(
            _script(
                tmp_path,
                "print(json.dumps({'request_id':request_id,'decision':'deny'}))",
            )
        )
    )
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))
    assert future.result(timeout=10) == "deny"


def test_the_terminal_answering_first_kills_the_loser(tmp_path, loop):
    """The hook would answer in 2s; the user typed y now. The user's answer must
    stand, and the hook must be killed before it can answer — not awaited."""
    marker = tmp_path / "hook_answered"
    install_hook_egress(
        _config(
            _script(
                tmp_path,
                "import time,pathlib;"
                "time.sleep(2);"
                f"pathlib.Path({str(marker)!r}).write_text('answered');"
                "print(json.dumps({'request_id':request_id,'decision':'deny'}))",
            )
        )
    )
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    # The user answers in the terminal: this is what the TUI's key handler does.
    assert _decide_on_loop(loop, registry, "call_1", "allow") is True
    assert future.result(timeout=5) == "allow"

    time.sleep(3.0)
    assert not marker.exists(), "the hook was not killed after the terminal answered"


def test_a_hook_with_no_decision_leaves_the_parked_approval_alone(tmp_path, loop):
    install_hook_egress(
        _config(_script(tmp_path, "pass"))
    )
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))

    time.sleep(1.0)
    assert not future.done(), "an empty reply must not resolve the approval"
    # The terminal can still answer, which is the whole fallback contract.
    assert _decide_on_loop(loop, registry, "call_1", "allow") is True
    assert future.result(timeout=5) == "allow"


def test_no_hook_is_a_no_op(tmp_path, loop):
    registry = ApprovalRegistry()
    future = _park(loop, registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, _State(registry), loop, _request(pending))
    time.sleep(0.4)
    assert not future.done()
    assert _decide_on_loop(loop, registry, "call_1", "deny") is True
    assert future.result(timeout=5) == "deny"


def test_a_late_hook_decision_cannot_emit_a_conflicting_resolution(monkeypatch):
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


def test_the_payload_carries_the_anchor_and_the_offered_options(tmp_path, loop):
    out = tmp_path / "seen.json"
    install_hook_egress(
        _config(
            _script(
                tmp_path,
                "import pathlib;"
                f"pathlib.Path({str(out)!r}).write_text(json.dumps(payload));"
                "print(json.dumps({'request_id':request_id,'decision':'deny'}))",
            )
        )
    )
    agent = _Agent()
    agent.task_anchor = type("A", (), {"source_query": "the user's message"})()
    state = _State(ApprovalRegistry())
    state.current_agent = agent
    future = _park(loop, state.approval_registry, _pending())
    pending = _pending()
    _offer_approval_to_hook(pending, state, loop, _request(pending))
    assert future.result(timeout=10) == "deny"

    import json

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["hook_event_name"] == "needs.approval"
    assert doc["prompt"] == "the user's message"
    assert doc["options"] == ["allow", "deny"]
    assert doc["tool_call_id"] == "call_1"
    assert "arguments" not in doc


def test_the_hook_answer_reprints_the_approval_record(tmp_path, loop):
    """The card is layout, not scrollback. Hiding it without a remnant would
    drop the command that was just approved."""
    printed = []
    done = threading.Event()

    def _capture(pending, decision):
        printed.append((pending.tool_call_id, decision))
        done.set()

    import agentica.cli.approvals as approvals_mod

    original = approvals_mod._print_approval_record
    approvals_mod._print_approval_record = _capture
    try:
        install_hook_egress(
            _config(
                _script(
                    tmp_path,
                    "print(json.dumps({'request_id':request_id,'decision':'allow'}))",
                )
            )
        )
        pending = _pending()
        registry = ApprovalRegistry()
        state = _State(registry)
        state.input_request = _InputRequest(
            prompt="ok",
            kind="approval",
            approval_id="call_1",
            approval_pending=pending,
            hook_request_id="request-1",
        )
        future = _park(loop, registry, pending)
        _offer_approval_to_hook(pending, state, loop, state.input_request)
        assert future.result(timeout=10) == "allow"
        assert done.wait(5)
        assert printed == [("call_1", "allow")]
        assert state.input_request is None
    finally:
        approvals_mod._print_approval_record = original


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
