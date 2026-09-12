# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: What ``run.completed`` means on the notify wire.

A generic agent status display reads it as "you can come back now". That is only
true when the work is actually over, and two things in agentica make one apparent
request into several runs:

  * a standing goal — the CLI's goal hook queues a continuation after each run,
    so a 5-lap goal ends 5 runs;
  * several queued user messages — each is promoted to its own run.

Both would report "done" N times. The desktop app has no way to tell an
intermediate run from the last one (its own debounce cannot help: the gap between
two runs includes a judge LLM call, so the interval is unbounded), so the sink
has to be the one that stays quiet.

These tests are about a failure that is silent by nature: nothing errors, the
display just starts lying, and the user stops trusting it.
"""

from __future__ import annotations

import json
import time

import pytest

from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.notify.sink import notify_sink_dispatch
from agentica.notify.sink import set_idle_provider
from agentica.run_events import RunEventRecord, RunEventType

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    set_idle_provider(None)
    yield
    reset_sink_for_tests()
    set_idle_provider(None)


def _agent(tmp_path, session_id="sess-1"):
    return Agent(
        name="Probe",
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
        session_id=session_id,
        session_base_dir=str(tmp_path),
    )


def _record(event: RunEventType = RunEventType.run_completed) -> RunEventRecord:
    return RunEventRecord(
        run_id="run-1",
        event_type=event,
        agent_id="a",
        payload={"duration_seconds": 1.0, "had_response": True},
    )


def _events(desktop):
    return [r["json"]["event"] for r in desktop.requests if r["json"]]


class TestAGoalDoesNotReportDonePerLap:
    def test_no_completed_reaches_the_sink_while_a_goal_is_active(self, tmp_path):
        """The regression test for the reported bug: three laps, zero 'done'."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path)

            from agentica.goals import GoalManager
            GoalManager(agent._session_log).set("跑通全部测试")

            # Three laps of a standing goal.
            for _ in range(3):
                notify_sink_dispatch(_record(), session_id="sess-1", agent=agent)
            time.sleep(0.3)

            assert _events(desktop) == []
        finally:
            desktop.close()

    def test_started_still_goes_out_for_every_lap(self, tmp_path):
        """Only 'done' is held back: the display must still know work is
        happening, otherwise it looks frozen instead of busy."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path)

            from agentica.goals import GoalManager
            GoalManager(agent._session_log).set("目标")

            for _ in range(3):
                notify_sink_dispatch(
                    _record(RunEventType.run_started), session_id="sess-1", agent=agent,
                )
            time.sleep(0.3)

            assert _events(desktop) == ["run.started"] * 3
        finally:
            desktop.close()

    def test_failed_and_cancelled_are_never_suppressed(self, tmp_path):
        """A goal that stops because it broke must say so — suppressing those
        would leave the display stuck on 'working' forever."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path)

            from agentica.goals import GoalManager
            GoalManager(agent._session_log).set("目标")

            notify_sink_dispatch(_record(RunEventType.run_failed), agent=agent)
            notify_sink_dispatch(_record(RunEventType.run_cancelled), agent=agent)
            time.sleep(0.3)

            assert sorted(_events(desktop)) == ["run.cancelled", "run.failed"]
        finally:
            desktop.close()

    def test_the_completion_goes_out_once_the_goal_is_done(self, tmp_path):
        """The whole point: the real last run must still report done."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path)

            from agentica.goals import GoalManager
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            notify_sink_dispatch(_record(), agent=agent)   # suppressed
            # The model calls verify_completion and the goal closes.
            mgr.complete(reason="verified") if hasattr(mgr, "complete") else mgr.clear()
            notify_sink_dispatch(_record(), agent=agent)   # must get through
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_a_paused_goal_does_not_suppress_completion(self, tmp_path):
        """Paused is not active: the current run really did finish."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path)

            from agentica.goals import GoalManager
            mgr = GoalManager(agent._session_log)
            mgr.set("目标")
            mgr.pause(reason="user-interrupted")

            notify_sink_dispatch(_record(), agent=agent)
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()


class TestItReadsTheSignalThatIsActuallyCurrent:
    """The trap this check nearly fell into.

    The CLI owns ``state.goal_manager``; the agent has a *second* manager that
    loads the log once, lazily, on first touch. Ask the agent's copy and you can
    get a stale "no goal" forever — which would make the suppression above look
    correct while doing nothing.
    """

    def test_the_agents_own_manager_can_be_stale(self, tmp_path):
        """Pins the hazard, so the fix is not 'simplified' back into it."""
        agent = _agent(tmp_path, session_id="stale-1")
        agent.enable_goal_tool()            # agent.goal_manager is now cached
        assert agent.goal_manager.is_active() is False

        from agentica.goals import GoalManager
        GoalManager(agent._session_log).set("设定目标")

        # The authoritative manager sees it...
        assert GoalManager(agent._session_log).is_active() is True
        # ...while the agent's cached copy still does not.
        assert agent.goal_manager.is_active() is False

    def test_the_sink_still_sees_the_active_goal(self, tmp_path):
        """Because it reads the log, not the agent's cached manager."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _agent(tmp_path, session_id="stale-2")
            agent.enable_goal_tool()

            from agentica.goals import GoalManager
            GoalManager(agent._session_log).set("设定目标")

            notify_sink_dispatch(_record(), agent=agent)
            time.sleep(0.3)

            assert _events(desktop) == [], "the stale-cache hazard must not resurface"
        finally:
            desktop.close()

class TestQueuedInputAlsoDefersDone:
    """The same repeat, without a goal involved.

    Several messages typed in a row are each promoted to their own run, so the
    first one finishing is not "you can come back now". The sink cannot see the
    CLI's queue; the CLI registers a probe.
    """

    def test_a_queued_message_defers_completion(self, tmp_path):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            queued = ["second message"]
            set_idle_provider(lambda: not queued)

            notify_sink_dispatch(_record(), agent=_agent(tmp_path))
            time.sleep(0.3)

            assert _events(desktop) == []
        finally:
            desktop.close()

    def test_the_last_run_reports_done_once_the_queue_drains(self, tmp_path):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            queued = ["second message"]
            set_idle_provider(lambda: not queued)

            notify_sink_dispatch(_record(), agent=_agent(tmp_path))  # deferred
            queued.pop()                                             # nothing left
            notify_sink_dispatch(_record(), agent=_agent(tmp_path))  # reported
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_the_probe_only_gates_completion(self, tmp_path):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            set_idle_provider(lambda: False)  # never idle

            notify_sink_dispatch(_record(RunEventType.run_started), agent=_agent(tmp_path))
            notify_sink_dispatch(_record(RunEventType.run_failed), agent=_agent(tmp_path))
            time.sleep(0.3)

            assert sorted(_events(desktop)) == ["run.failed", "run.started"]
        finally:
            desktop.close()

    def test_an_absent_probe_reports_normally(self, tmp_path):
        """The SDK and the non-interactive paths have no queue to ask about."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            set_idle_provider(None)
            notify_sink_dispatch(_record(), agent=_agent(tmp_path))
            time.sleep(0.3)
            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_a_raising_probe_reports_normally(self, tmp_path):
        """Must degrade toward reporting: a display that never says 'done'
        because a probe broke is worse than one extra 'done'."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            def boom():
                raise RuntimeError("queue unreadable")

            set_idle_provider(boom)
            notify_sink_dispatch(_record(), agent=_agent(tmp_path))
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_a_wrong_probe_shape_is_treated_as_idle_not_as_an_error(self, tmp_path):
        """Regression guard: len(PendingQueue) raises TypeError, and the sink
        reads a raising probe as idle — so a probe written against a queue that
        has no __len__ would disable this check with no visible symptom."""
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        assert not hasattr(q, "__len__"), (
            "PendingQueue gained __len__; the CLI probe uses peek_all(), so "
            "update this test and the comment in app.py"
        )
        q.put("something")
        set_idle_provider(lambda: not q.peek_all())

        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            notify_sink_dispatch(_record(), agent=_agent(tmp_path))
            time.sleep(0.3)
            assert _events(desktop) == []
        finally:
            desktop.close()


class TestDegradesToReporting:
    """When in doubt, report. Suppressing a real completion is worse than an
    extra one: a display that never says 'done' is useless."""

    def test_an_agent_without_a_session_log_reports_normally(self):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = Agent(name="Bare", model=OpenAIChat(id="gpt-4o-mini", api_key="fake"))
            assert agent._session_log is None

            notify_sink_dispatch(_record(), agent=agent)
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_a_broken_goal_read_reports_normally(self, tmp_path):
        """A raising log read must not swallow the completion."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            class _BrokenLog:
                def load_goal(self):
                    raise RuntimeError("log unreadable")

            class _Agent:
                _session_log = _BrokenLog()

            notify_sink_dispatch(_record(), agent=_Agent())
            time.sleep(0.3)

            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()

    def test_no_agent_passed_reports_normally(self):
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            notify_sink_dispatch(_record())
            time.sleep(0.3)
            assert _events(desktop) == ["run.completed"]
        finally:
            desktop.close()
