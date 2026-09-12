# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The sink wired into a run — the step-3 milestone.

Proves the whole chain the desktop app cares about: an agent runs, and the
notify sink hears about it. Uses a real Unix-socket server, because the point
here is that the envelope actually leaves the process.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.run_events import RunEventType
from agentica.runner import Runner

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


def _drain(desktop, expected: int, timeout: float = 5.0) -> None:
    """Wait for ``expected`` requests, so a slow worker cannot flake the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(desktop.requests) >= expected:
            return
        time.sleep(0.02)


class TestRunLifecycleReachesTheDesktop:
    def test_emit_event_dispatches_to_the_sink(self):
        """The wiring itself: ``Runner._emit_event`` must side-mount the sink."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            agent = Agent(
                name="Probe",
                model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
                session_id="sess-42",
                work_dir="/tmp/proj",
            )
            runner = Runner(agent)

            class _Ctx:
                run_id = "run-1"
                agent_id = "agent-1"
                parent_run_id = None

            # _emit_event no-ops without a run context, which is the documented
            # boundary (these four events belong to a run).
            agent.run_context = None
            runner._emit_event(RunEventType.run_started)
            time.sleep(0.2)
            assert desktop.requests == [], "no run context must mean no event"

            agent.run_context = _Ctx()
            runner._emit_event(
                RunEventType.run_started, {"agent_name": "Probe", "source_query": "hi"}
            )
            _drain(desktop, 1)

            assert len(desktop.requests) == 1
            body = desktop.requests[0]["json"]
            assert body["event"] == "run.started"
            assert body["session_key"] == "sess-42"
            assert body["transport"]["cwd"] == "/tmp/proj"
            assert body["payload"]["agent_name"] == "Probe"
        finally:
            desktop.close()

    def test_the_agent_callback_still_fires_alongside_the_sink(self):
        """Side-mounted, not replacing: both consumers get the event."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            agent = Agent(name="Probe", model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
                          session_id="sess-1")
            seen = []
            agent._event_callback = seen.append
            runner = Runner(agent)

            class _Ctx:
                run_id = "run-1"
                agent_id = "a"
                parent_run_id = None

            agent.run_context = _Ctx()
            runner._emit_event(RunEventType.run_completed, {"duration_seconds": 2.0})
            _drain(desktop, 1)

            assert len(seen) == 1, "the in-process callback must still fire"
            assert len(desktop.requests) == 1, "and the sink must get it too"
        finally:
            desktop.close()

    def test_a_broken_callback_does_not_stop_the_sink(self):
        """Their independence is the reason for side-mounting."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))

            def boom(_payload):
                raise RuntimeError("display callback is broken")

            agent = Agent(name="Probe", model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
                          session_id="sess-1")
            agent._event_callback = boom
            runner = Runner(agent)

            class _Ctx:
                run_id = "run-1"
                agent_id = "a"
                parent_run_id = None

            agent.run_context = _Ctx()
            runner._emit_event(RunEventType.run_failed, {"error": "x"})
            _drain(desktop, 1)

            assert len(desktop.requests) == 1
            assert desktop.requests[0]["json"]["event"] == "run.failed"
        finally:
            desktop.close()

    def test_a_disabled_sink_stays_out_of_the_way(self):
        desktop = _FakeDesktop()
        try:
            assert install_sink(NotifyConfig(enabled=False, socket=desktop.socket_path)) is None

            agent = Agent(name="Probe", model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
                          session_id="sess-1")
            runner = Runner(agent)

            class _Ctx:
                run_id = "run-1"
                agent_id = "a"
                parent_run_id = None

            agent.run_context = _Ctx()
            runner._emit_event(RunEventType.run_started)
            time.sleep(0.3)

            assert desktop.requests == []
        finally:
            desktop.close()

    def test_a_broken_sink_never_breaks_the_emit(self):
        """An observation channel must not become the agent's problem."""
        def boom(_path):
            raise RuntimeError("sink is broken")

        from agentica.notify.sink import NotifySink
        sink = NotifySink(NotifyConfig(enabled=True, socket="/tmp/x.sock"),
                          transport_factory=boom)
        import agentica.notify.sink as sink_mod
        sink_mod._sink = sink

        agent = Agent(name="Probe", model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
                      session_id="sess-1")
        runner = Runner(agent)

        class _Ctx:
            run_id = "run-1"
            agent_id = "a"
            parent_run_id = None

        agent.run_context = _Ctx()
        runner._emit_event(RunEventType.run_started)   # must not raise
        time.sleep(0.3)
        sink.stop()
