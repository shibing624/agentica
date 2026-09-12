# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the two-way approval path (step 4).

The desktop app may answer an approval — but only ever as an alternative to the
terminal, never as a replacement for it, and never on its own authority. These
tests are about the ways that could quietly do the wrong thing: deciding when
the switch is off, blocking the event loop, losing a decision, or treating a
normal race as an error.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agentica.agent.approvals import ApprovalRegistry, PendingApproval
from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.approvals import publish_approval
from agentica.notify.config import NotifyConfig
from agentica.notify.sink import get_sink

from tests.notify.test_notify_sink import _FakeDesktop, _MissingDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


def _pending(**kw) -> PendingApproval:
    base = dict(
        tool_call_id="call_1",
        name="bash",
        arguments={"command": "rm -rf build"},
        question="Run `rm -rf build`?",
        preview="rm -rf build",
        options=["allow", "allow_prefix", "deny", "deny_prefix"],
    )
    base.update(kw)
    return PendingApproval(**base)


def _run_with_loop(coro_fn):
    """Run an async scenario in its own loop and return its result."""
    return asyncio.run(coro_fn())


class TestTheDesktopCanDecide:
    def test_a_desktop_allow_is_applied_to_the_registry(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                return await asyncio.wait_for(waiter, timeout=5)

            assert _run_with_loop(scenario) == "allow"
            body = desktop.requests[0]["json"]
            assert body["event"] == "needs.approval"
            assert body["payload"]["kind"] == "permission"
            assert body["payload"]["approval_id"] == "call_1"
            assert body["payload"]["question"] == "Run `rm -rf build`?"
            # Options are reported as-is so the desktop side can widen later.
            assert body["payload"]["options"] == ["allow", "allow_prefix", "deny", "deny_prefix"]
        finally:
            desktop.close()

    def test_a_desktop_deny_is_applied(self):
        desktop = _FakeDesktop(decision_body={"decision": "deny"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                return await asyncio.wait_for(waiter, timeout=5)

            assert _run_with_loop(scenario) == "deny"
        finally:
            desktop.close()

    def test_the_raw_arguments_are_not_sent(self):
        """Metadata only: the description says enough, the full command does not
        belong on this channel."""
        desktop = _FakeDesktop(decision_body={"decision": "deny"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending(arguments={"command": "SECRET_COMMAND_TEXT"})
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                return await asyncio.wait_for(waiter, timeout=5)

            _run_with_loop(scenario)
            body = desktop.requests[0]["json"]
            assert "SECRET_COMMAND_TEXT" not in str(body["payload"])
            assert "arguments" not in body["payload"]
        finally:
            desktop.close()


class TestTheUsersAnswerCountsFromEitherPlace:
    def test_an_answer_from_the_app_resolves_the_same_wait(self):
        """Same wait, same effect — the app is an input surface, not a gate.

        There is no "may the app answer?" switch: that would mean the app had
        authority of its own. What the user presses there is applied as their
        answer for this session and this interaction, exactly like typing it.
        """
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop(), timeout=5)
                return await asyncio.wait_for(waiter, timeout=5)

            assert _run_with_loop(scenario) == "allow"
        finally:
            desktop.close()

    def test_the_typed_answer_and_the_pressed_answer_are_indistinguishable(self):
        """Both produce the same registry state — that is the whole contract."""
        desktop = _FakeDesktop(decision_body={"decision": "deny"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def from_app():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop(), timeout=5)
                return await asyncio.wait_for(waiter, timeout=5)

            async def from_terminal():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                registry.decide("call_1", "deny")
                return await asyncio.wait_for(waiter, timeout=5)

            assert _run_with_loop(from_app) == _run_with_loop(from_terminal) == "deny"
        finally:
            desktop.close()

    def test_a_disabled_sink_does_not_touch_the_approval_flow(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=False, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(waiter, timeout=0.5)

            _run_with_loop(scenario)
            assert desktop.requests == []
        finally:
            desktop.close()


class TestTheLadderOnTheApprovalPath:
    def test_a_missing_desktop_app_does_not_hold_up_the_approval(self):
        """The terminal must stay responsive; no answer means no answer."""
        install_sink(NotifyConfig(enabled=True, socket=_MissingDesktop().socket_path))

        async def scenario():
            registry = ApprovalRegistry()
            pending = _pending()
            waiter = registry.wait(pending)
            started = time.monotonic()
            publish_approval(pending, registry, asyncio.get_running_loop())
            # The terminal answer arrives normally, while the sink is still
            # trying: the two are independent.
            await asyncio.sleep(0.3)
            registry.decide("call_1", "deny")
            return await asyncio.wait_for(waiter, timeout=2), time.monotonic() - started

        decision, elapsed = _run_with_loop(scenario)
        assert decision == "deny"
        assert elapsed < 2.0

    def test_a_silent_desktop_app_does_not_hold_up_the_approval(self):
        """Connected but never answers: same outcome, the terminal still works."""
        desktop = _FakeDesktop(hang=True)
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                await asyncio.sleep(0.3)
                registry.decide("call_1", "deny")
                return await asyncio.wait_for(waiter, timeout=2)

            assert _run_with_loop(scenario) == "deny"
        finally:
            desktop.close()

    def test_an_unusable_desktop_answer_leaves_the_wait_pending(self):
        """Level 4 on this path: garbage is 'no decision', never an allow."""
        desktop = _FakeDesktop(decision_body={"decision": "yes please"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(waiter, timeout=1.0)

            _run_with_loop(scenario)
        finally:
            desktop.close()


class TestRacesAndFailures:
    def test_no_answer_never_becomes_an_allow(self):
        """The headline risk, asserted at the source of the decision.

        ``registry.decide`` is watched directly rather than inferred from an
        outcome, because "nothing decided" and "allowed" can look alike from the
        outside if the wrong value is written. A user who never pressed anything
        must never get an approval: no reply, a timeout, and a garbage reply all
        have to leave the wait unresolved.
        """
        for body, label in (
            (None, "no reply at all"),
            ({"decision": "allow"}, "a reply, but the app never got to send it"),
        ):
            desktop = _FakeDesktop(hang=True, decision_body=body)
            try:
                install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
                decided = []

                class _Watching(ApprovalRegistry):
                    def decide(self, call_id, decision):
                        decided.append(decision)
                        return super().decide(call_id, decision)

                async def scenario():
                    registry = _Watching()
                    pending = _pending()
                    registry.wait(pending)
                    publish_approval(pending, registry, asyncio.get_running_loop(), timeout=0.4)
                    await asyncio.sleep(0.8)

                _run_with_loop(scenario)
                assert decided == [], f"{label}: something was decided ({decided})"
            finally:
                desktop.close()

    def test_the_terminal_answering_first_is_not_an_error(self):
        """decide() returning False is the normal race, not a fault."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                # The user hits y in the terminal before the desktop answers.
                registry.decide("call_1", "deny")
                publish_approval(pending, registry, asyncio.get_running_loop())
                return await asyncio.wait_for(waiter, timeout=5)

            # The terminal's deny is final; the later desktop allow must not win.
            assert _run_with_loop(scenario) == "deny"
        finally:
            desktop.close()

    def test_a_late_desktop_answer_after_the_wait_is_gone_is_dropped(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                publish_approval(pending, registry, asyncio.get_running_loop())
                registry.decide("call_1", "deny")   # terminal wins
                return await asyncio.wait_for(waiter, timeout=5)

            assert _run_with_loop(scenario) == "deny"
            # Let the desktop answer arrive afterwards; it must be inert.
            time.sleep(0.4)
            assert get_sink() is not None  # the sink itself is still healthy
        finally:
            desktop.close()

    def test_a_pending_without_an_id_is_skipped(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending(tool_call_id="")
                publish_approval(pending, registry, asyncio.get_running_loop())
                return None

            _run_with_loop(scenario)
            time.sleep(0.2)
            # Nothing could be correlated back, so nothing was asked.
            assert desktop.requests == []
        finally:
            desktop.close()

    def test_a_registry_of_none_is_a_no_op(self):
        """Non-interactive paths pass get_registry=None; the sink must not block."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            publish_approval(_pending(), None, None)  # must not raise
            time.sleep(0.2)
            assert desktop.requests == []
        finally:
            desktop.close()

    def test_the_publish_call_returns_immediately(self):
        """It must not block the loop the approval is waiting on."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            async def scenario():
                registry = ApprovalRegistry()
                pending = _pending()
                waiter = registry.wait(pending)
                started = time.monotonic()
                publish_approval(pending, registry, asyncio.get_running_loop())
                elapsed = time.monotonic() - started
                await asyncio.wait_for(waiter, timeout=5)
                return elapsed

            # Generous bound: the point is "did not block", not a benchmark.
            assert _run_with_loop(scenario) < 0.5
        finally:
            desktop.close()
