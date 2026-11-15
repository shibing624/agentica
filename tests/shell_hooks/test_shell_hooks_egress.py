# -*- coding: utf-8 -*-
"""The lifecycle fan-out: one event reaches every subscribed consumer, and a
broken consumer cannot affect the sink or the run.

The egress is a decision layer — which consumers are subscribed, what the
payload says, which events are dropped, and what shutdown does to a straggler.
Every test here scripts those decisions with in-process fakes
(``_fakes.FakeHookProcess``); real process spawning, signalling and reaping are
development-time validation, not per-``pytest`` acceptance.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agentica.run.events import RunEventRecord, RunEventType
from agentica.shell_hooks import egress as egress_mod
from agentica.shell_hooks import prompt as prompt_mod
from agentica.shell_hooks import requests as requests_mod
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import (
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)
from tests.shell_hooks._fakes import FakeHookProcess, install_fakes


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


class _SinkSpy:
    def __init__(self):
        self.events = []

    def emit_event(self, event, *, session_id=None, payload=None, work_dir=None):
        self.events.append((event, payload))


def _config(*labels, events=None, enabled=True):
    """One subscribed consumer per label; the label is scripted, never executed."""
    return ShellHooksConfig(
        enabled=enabled,
        consumers=[
            HookConsumer(name=label, command=[label], events=events or {})
            for label in labels
        ],
    )


class TestInstall:
    def test_disabled_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=False)) is None
        assert get_hook_egress() is None

    def test_enabled_without_a_command_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=True)) is None

    def test_enabled_with_a_command_is_wired(self):
        cfg = install_hook_egress(_config("desktop"))
        assert cfg is not None
        assert get_hook_egress() is not None

    def test_disabled_forks_nothing_even_when_events_are_dispatched(self, monkeypatch):
        """Verified by construction count, not by reading the config.

        Reading the config would only prove the config says "off"; what must
        hold is that no consumer is started, on either the notice path or the
        ``needs.*`` path, even when the dispatch points are called as they are
        in a real run.
        """
        egress_hooks = install_fakes(monkeypatch, egress_mod)
        request_hooks = install_fakes(monkeypatch, requests_mod)
        install_hook_egress(ShellHooksConfig(enabled=False))
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        assert requests_mod.start_hook_request("needs.approval", {}) is None
        assert requests_mod.start_hook_request("needs.input", {}) is None
        assert egress_hooks.made == []
        assert request_hooks.made == []


class TestDispatch:
    def test_an_event_reaches_the_consumer_with_its_identity(self, monkeypatch):
        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        agent = SimpleNamespace(run_context=SimpleNamespace(run_id="run-1"))
        hook_egress_dispatch(
            "run.started",
            {"agent_name": "Agent", "prompt": "do the thing"},
            session_id="sess-1",
            work_dir="/w",
            agent=agent,
        )
        assert len(hooks.made) == 1
        doc = hooks.made[0].payload
        assert doc["hook_event_name"] == "run.started"
        assert doc["session_id"] == "sess-1"
        assert doc["run_id"] == "run-1"
        assert doc["prompt"] == "do the thing"
        assert doc["cwd"] == "/w"
        assert doc["agent_name"] == "Agent"
        # Process identity always rides along, so a consumer can address an
        # answer back to this process rather than to a guessed one.
        assert doc["transport"]["ppid"] > 0
        assert doc["transport"]["cwd"] == "/w"

    def test_a_switched_off_event_is_not_sent(self, monkeypatch):
        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop", events={"run.started": False}))
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        assert [doc["hook_event_name"] for doc in hooks.payloads] == ["run.completed"]

    def test_no_egress_is_a_no_op(self, monkeypatch):
        hooks = install_fakes(monkeypatch, egress_mod)
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise
        assert hooks.made == []

    def test_a_command_that_cannot_start_is_swallowed(self, monkeypatch):
        install_fakes(monkeypatch, egress_mod, dead={"starts": False})
        install_hook_egress(_config("dead"))
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise
        assert egress_mod._live_notices == set()

    def test_a_consumer_constructor_that_raises_is_swallowed(self, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("consumer construction failed")

        monkeypatch.setattr(egress_mod, "HookProcess", explode)
        install_hook_egress(_config("desktop"))
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_hanging_command_does_not_block_the_caller(self, monkeypatch):
        hooks = install_fakes(monkeypatch, egress_mod, hang={"hang": True})
        install_hook_egress(_config("hang"))
        started = time.monotonic()
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert time.monotonic() - started < 0.5
        # Fire-and-forget: the outstanding consumer is tracked for shutdown.
        assert hooks.made[0] in egress_mod._live_notices

    def test_a_hanging_notice_consumer_is_killed_after_the_notice_timeout(
        self, monkeypatch
    ):
        """A notice must not leak a consumer for the life of the process: the
        cleanup path waits a bounded time, then kills the group."""
        hooks = install_fakes(monkeypatch, egress_mod, hang={"hang": True})
        monkeypatch.setattr(egress_mod, "NOTICE_PROCESS_TIMEOUT_SECONDS", 0.02)
        install_hook_egress(_config("hang"))
        hook_egress_dispatch("run.started", {}, session_id="s")
        process = hooks.made[0]
        deadline = time.monotonic() + 2.0
        while not process.killed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert process.killed is True
        while process in egress_mod._live_notices and time.monotonic() < deadline:
            time.sleep(0.005)
        assert egress_mod._live_notices == set()

    def test_each_subscribed_consumer_receives_the_event(self, monkeypatch):
        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("first", "second"))
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert [doc["hook_event_name"] for doc in hooks.payloads] == [
            "run.started",
            "run.started",
        ]

    def test_atexit_shutdown_is_registered_once(self, monkeypatch):
        registered = []
        monkeypatch.setattr(egress_mod, "_atexit_registered", False)
        monkeypatch.setattr(
            egress_mod.atexit, "register", lambda fn, **kw: registered.append(fn)
        )
        install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        assert len(registered) == 1

    def test_exit_cleanup_kills_without_starting_a_thread(self, monkeypatch):
        """At interpreter exit no thread can start (Python raises ``can't create
        new thread at interpreter shutdown``), so the atexit path must kill the
        consumer without spawning the zombie reaper."""
        process = FakeHookProcess(
            ["fake-hook"], {"hook_event_name": "run.started"}, hang=True
        )
        assert process.start()
        egress_mod._live_notices.add(process)

        def refuse(self):
            raise RuntimeError("can't create new thread at interpreter shutdown")

        with monkeypatch.context() as m:
            m.setattr(threading.Thread, "start", refuse)
            egress_mod._shutdown_notice_processes(at_exit=True)

        # Killed, not merely abandoned: the exit status is -9.
        assert process.killed is True
        assert process.returncode == -9
        assert egress_mod._live_notices == set()

    def test_sdk_process_never_reads_the_user_config(self, monkeypatch):
        """An SDK process that did not install the egress must not pick up
        the CLI user's ``settings.hooks`` — not on dispatch, not on prompt
        context, not on ``needs.*``."""
        from agentica.shell_hooks.prompt import collect_prompt_context

        egress_hooks = install_fakes(monkeypatch, egress_mod)
        prompt_hooks = install_fakes(monkeypatch, prompt_mod)
        request_hooks = install_fakes(monkeypatch, requests_mod)
        loads = []

        def load():
            loads.append(True)
            return _config(
                "desktop",
                events={
                    "run.started": True,
                    "UserPromptSubmit": True,
                    "needs.approval": True,
                },
            )

        monkeypatch.setattr(egress_mod, "load_shell_hooks_config", load)
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert collect_prompt_context("hello", session_id="s") is None
        assert requests_mod.start_hook_request("needs.approval", {"request_id": "r"}) is None
        assert loads == []
        assert egress_hooks.made == []
        assert prompt_hooks.made == []
        assert request_hooks.made == []


class TestTheSinkStillWorks:
    def test_both_egresses_receive_one_event(self, monkeypatch):
        """Fan-out, not replacement: the sink half is untouched."""
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        spy = _SinkSpy()
        monkeypatch.setattr(sink_mod, "_sink", spy)
        record = RunEventRecord(
            run_id="r1", event_type=RunEventType.run_started, payload={"agent_name": "A"}
        )
        sink_mod.notify_sink_dispatch(record, session_id="s1")
        assert [e for e, _ in spy.events] == ["run.started"]
        assert [doc["hook_event_name"] for doc in hooks.payloads] == ["run.started"]

    def test_the_hook_is_reached_with_no_sink_installed(self, monkeypatch):
        """The early return that used to guard the sink would swallow this.

        A user with hooks configured and no notify sink must still get events —
        that combination is the whole point of a second egress.
        """
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        monkeypatch.setattr(sink_mod, "_sink", None)
        record = RunEventRecord(run_id="r1", event_type=RunEventType.run_started)
        sink_mod.notify_sink_dispatch(record, session_id="s1")
        assert [doc["hook_event_name"] for doc in hooks.payloads] == ["run.started"]

    def test_a_deferred_completion_reaches_the_hook_on_release(self, monkeypatch):
        """The goal deferral is shared, so the hook sees the same one release the
        sink does — not one event per lap."""
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        monkeypatch.setattr(sink_mod, "_sink", None)
        monkeypatch.setattr(sink_mod, "_goal_is_driving", lambda agent: True)

        class _Agent:
            run_response = None
            _session_log = None
            run_context = None

        agent = _Agent()
        record = RunEventRecord(
            run_id="r1",
            event_type=RunEventType.run_completed,
            payload={"duration_seconds": 1.5},
        )
        sink_mod.notify_sink_dispatch(record, session_id="s1", agent=agent)
        assert hooks.made == [], "the completion must be held while a goal drives"

        monkeypatch.setattr(sink_mod, "_goal_is_driving", lambda agent: False)
        sink_mod.goal_finished(agent, session_id="s1")
        assert len(hooks.made) == 1
        doc = hooks.made[0].payload
        assert doc["hook_event_name"] == "run.completed"
        # The held payload is the shape a consumer already knows.
        assert doc["duration_seconds"] == 1.5


class TestSubagentRunsStayOffTheWire:
    """A subagent is not a session.

    Children are spawned with no ``session_id`` of their own, so their events
    would be keyed on the process-wide fallback: every consumer would grow one
    phantom session that lights up whenever any child touches a tool, never gets
    ``session.started`` / ``session.ended``, and merges all concurrent children
    into one row. ``tool.*`` is what makes it obvious, because children are
    tool-heavy.
    """

    def test_a_subagent_tool_event_reaches_neither_egress(self, monkeypatch):
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        spy = _SinkSpy()
        monkeypatch.setattr(sink_mod, "_sink", spy)
        record = RunEventRecord(
            run_id="child-run",
            event_type=RunEventType.tool_started,
            parent_run_id="parent-run",
            payload={"tool_name": "read_file", "tool_call_id": "c1"},
        )

        sink_mod.notify_sink_dispatch(record, session_id=None)

        assert spy.events == []
        assert hooks.made == []

    def test_a_subagent_completion_cannot_be_mistaken_for_the_users_turn(
        self, monkeypatch
    ):
        """The one that would actually mislead: 'done' for work still running.

        The deferral inputs are pinned rather than left to ambient state: with
        `agent=None`, or with an idle provider left behind by another test, the
        completion path bails on its own and this would pass whether or not the
        guard exists.
        """
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        monkeypatch.setattr(sink_mod, "_sink", None)
        monkeypatch.setattr(sink_mod, "_goal_is_driving", lambda agent: False)
        monkeypatch.setattr(sink_mod, "_nothing_more_queued", lambda: True)

        class _Agent:
            run_response = None
            _session_log = None
            run_context = None

        record = RunEventRecord(
            run_id="child-run",
            event_type=RunEventType.run_completed,
            parent_run_id="parent-run",
        )

        sink_mod.notify_sink_dispatch(record, session_id=None, agent=_Agent())

        assert hooks.made == []

    def test_the_parents_own_events_still_go_out(self, monkeypatch):
        """The guard keys on lineage, so it must not silence the top-level run."""
        import agentica.notify.sink as sink_mod

        hooks = install_fakes(monkeypatch, egress_mod)
        install_hook_egress(_config("desktop"))
        monkeypatch.setattr(sink_mod, "_sink", None)
        record = RunEventRecord(
            run_id="parent-run",
            event_type=RunEventType.tool_started,
            parent_run_id=None,
            payload={"tool_name": "read_file", "tool_call_id": "c1"},
        )

        sink_mod.notify_sink_dispatch(record, session_id="s1")

        assert [doc["hook_event_name"] for doc in hooks.payloads] == ["tool.started"]
