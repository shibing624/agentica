# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: End-to-end ``run.completed`` semantics through the real goal loop.

The unit tests pin the sink's rule; these drive the actual CLI goal hook
(``_maybe_continue_goal``), because that is where the rule is answerable. A bug
here is silent in the worst way: the sink keeps working, tests stay green, and
the desktop app simply never learns the work is over — so the user never comes
back to look.

Two shapes of the same mistake have now been made, in opposite directions:

  1. reporting per lap  -> N "done" for one request;
  2. judging a lap by "is a goal active" -> the last lap is suppressed too, and
     because the goal stops *after* that lap, and a stopped goal never queues
     another lap, no completion is ever reported at all.

Both are covered below, plus the ordering: the release has to happen after the
evaluation that decides there is no next lap, not before it.
"""

from __future__ import annotations

import pathlib
import tempfile
import time

import pytest

from agentica.agent import Agent
from agentica.cli.interactive import goal_hook as cli_goal_hook
from agentica.cli.interactive.session_state import SessionState
from agentica.cli.commands.context import PendingQueue
from agentica.goals import CONTINUATION_PROMPT_PREFIX, GoalDecision, GoalManager
from agentica.model.openai import OpenAIChat
from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.notify.sink import notify_sink_dispatch, set_idle_provider
from agentica.run_events import RunEventRecord, RunEventType

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean(tmp_path):
    reset_sink_for_tests()
    set_idle_provider(None)
    yield
    reset_sink_for_tests()
    set_idle_provider(None)


def _record() -> RunEventRecord:
    return RunEventRecord(
        run_id="r", event_type=RunEventType.run_completed, agent_id="a", payload={}
    )


def _agent(tmp_path, session_id="s"):
    return Agent(
        name="P",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
        session_id=session_id,
        session_base_dir=str(tmp_path),
    )


def _events(desktop):
    return [r["json"]["event"] for r in desktop.requests if r["json"]]


def _decision(status: str, should_continue: bool, prompt=None) -> GoalDecision:
    return GoalDecision(
        status=status,
        should_continue=should_continue,
        continuation_prompt=prompt,
        verdict="continue" if should_continue else "done",
        reason="",
        message="",
    )


#: A continuation as the goal loop really writes it. A bare string would be
#: read as a real user message — the hook treats anything not matching this as
#: the user speaking, which outranks the next lap and releases immediately.
_CONTINUATION = f"{CONTINUATION_PROMPT_PREFIX}\nGoal: ship it"


def _lap(agent, decision, state, pending_queue, monkeypatch):
    """One lap exactly as the CLI runs it: emit, then evaluate, then decide."""
    # loop.py emits this inside the run — the goal is necessarily still active.
    notify_sink_dispatch(_record(), agent=agent)
    # app.py calls the hook after the run returns.
    # A real turn produces text; with an empty one the hook returns before it
    # ever consults the decision, which is a different path (covered separately
    # by test_an_empty_turn_releases).
    monkeypatch.setattr(
        type(state.goal_manager), "extract_turn_signals",
        staticmethod(lambda rr, baseline: ("did some work", 10, baseline + 10, [])),
    )
    # Patch the async verdict itself and leave the real _run_async_safe bridge in
    # place: patching the bridge would strand the coroutine it is meant to await.
    async def _verdict(*_a, **_k):
        return decision

    monkeypatch.setattr(type(state.goal_manager), "evaluate_after_turn", _verdict)
    monkeypatch.setattr(cli_goal_hook, "_cprint", lambda *a, **k: None)
    cli_goal_hook._maybe_continue_goal(state, pending_queue, {})
    return decision


class TestARealGoalReportsExactlyOnce:
    def test_a_three_lap_goal_reports_one_completion(self, tmp_path, monkeypatch):
        """The regression: N laps, exactly one 'come back' — on the last one."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-3")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None  # extract_turn_signals tolerates None
            state = SessionState(current_agent=agent, goal_manager=mgr)
            queue = PendingQueue()

            # Laps 1 and 2 continue; lap 3 is the one that finishes the goal.
            _lap(agent, _decision("active", True, _CONTINUATION), state, queue, monkeypatch)
            _lap(agent, _decision("active", True, _CONTINUATION), state, queue, monkeypatch)
            _lap(agent, _decision("complete", False), state, queue, monkeypatch)

            time.sleep(0.4)
            got = _events(desktop)
            assert got.count("run.completed") == 1, f"expected exactly one, got {got}"
        finally:
            desktop.close()

    def test_two_laps_also_report_once(self, tmp_path, monkeypatch):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-2")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            _lap(agent, _decision("complete", False), state, PendingQueue(), monkeypatch)

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()

    def test_the_lap_inside_a_goal_reports_nothing(self, tmp_path, monkeypatch):
        """Only the last lap: an intermediate one must stay quiet."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-inner")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            time.sleep(0.3)

            assert _events(desktop) == []
        finally:
            desktop.close()


class TestEveryStoppingPointReleases:
    """The goal hook can return early in several ways; each one means "no more
    laps", so each one has to release the held completion. Enumerating the
    branches of ``goals.py`` instead would miss one."""

    @pytest.mark.parametrize("status", ["complete", "paused", "budget_limited"])
    def test_each_terminal_status_releases(self, tmp_path, monkeypatch, status):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, f"goal-{status}")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            # The manager's own verdict is what ends it.
            _lap(agent, _decision(status, False), state, PendingQueue(), monkeypatch)

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1, status
        finally:
            desktop.close()

    def test_a_cancelled_turn_releases(self, tmp_path, monkeypatch):
        """Ctrl+C pauses the goal, which also means no next lap."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-cancel")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            agent._cancelled = True
            cli_goal_hook._maybe_continue_goal(state, PendingQueue(), {})

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()

    def test_a_real_user_message_takes_priority_and_releases(self, tmp_path, monkeypatch):
        """A queued user message outranks the next lap, so the goal stops
        driving — and the deferred completion is true from that point."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-user")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            queue = PendingQueue()
            queue.put("actually, do this instead")
            cli_goal_hook._maybe_continue_goal(state, queue, {})

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()

    def test_an_empty_turn_releases(self, tmp_path, monkeypatch):
        """Nothing to judge means no continuation either."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-empty")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)
            empty = _decision("active", True, _CONTINUATION)

            import asyncio

            class _EmptySignals:
                pass

            monkeypatch.setattr(
                type(mgr), "extract_turn_signals",
                staticmethod(lambda rr, baseline: ("   ", 0, baseline, [])),
            )
            cli_goal_hook._maybe_continue_goal(state, PendingQueue(), {})

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()

    def test_an_evaluator_failure_releases(self, tmp_path, monkeypatch):
        """The goal cannot continue if evaluating it blew up."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-boom")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            _lap(agent, _decision("active", True, _CONTINUATION), state, PendingQueue(), monkeypatch)

            async def boom(*_a, **_k):
                raise RuntimeError("judge unavailable")

            monkeypatch.setattr(type(mgr), "evaluate_after_turn", boom)
            monkeypatch.setattr(cli_goal_hook, "_cprint", lambda *a, **k: None)
            cli_goal_hook._maybe_continue_goal(state, PendingQueue(), {})

            time.sleep(0.4)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()


class TestOrderingAndNonInterference:
    def test_releasing_happens_after_an_active_goal_would_have_queued(self, tmp_path, monkeypatch):
        """While laps keep coming, nothing is released — otherwise the display
        would say 'done' in the middle of the work."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "goal-mid")
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)
            queue = PendingQueue()

            for _ in range(5):
                _lap(agent, _decision("active", True, _CONTINUATION), state, queue, monkeypatch)
            time.sleep(0.3)

            assert _events(desktop) == []
            assert queue.peek_all(), "the continuation must still be queued"
        finally:
            desktop.close()

    def test_a_goal_that_never_started_is_untouched(self, tmp_path, monkeypatch):
        """No deferred completion -> the release must not invent one."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "no-goal")
            mgr = GoalManager(agent._session_log)  # nothing set
            agent.run_response = None
            state = SessionState(current_agent=agent, goal_manager=mgr)

            cli_goal_hook._maybe_continue_goal(state, PendingQueue(), {})
            time.sleep(0.3)

            assert _events(desktop) == []
        finally:
            desktop.close()

    def test_a_plain_run_is_unaffected(self, tmp_path):
        """No goal at all: one run, one completion, no bookkeeping."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, "plain")
            notify_sink_dispatch(_record(), agent=agent)
            time.sleep(0.3)
            assert _events(desktop).count("run.completed") == 1
        finally:
            desktop.close()
