# -*- coding: utf-8 -*-
"""The lifecycle fan-out: one event reaches every installed egress, and a broken
consumer cannot affect the sink or the run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from agentica.run.events import RunEventRecord, RunEventType
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import (
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)


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


def _recorder(tmp_path, name="hook"):
    """A hook command that appends each payload to a file."""
    out = tmp_path / f"{name}.jsonl"
    script = tmp_path / f"{name}.py"
    script.write_text(
        "import json,sys\n"
        f"open({str(out)!r},'a').write(json.dumps(json.load(sys.stdin))+'\\n')\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)], out


def _wait_for(path, count, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(lines) >= count:
                return [json.loads(line) for line in lines]
        time.sleep(0.05)
    raise AssertionError(f"{path} never reached {count} payloads")


def _assert_nothing_arrives(path, settle=2.0):
    """Prove an absence, which a bare ``not path.exists()`` cannot.

    The hook runs in a spawned process, so straight after dispatch the file is
    missing whether it was suppressed or merely slow — an immediate assertion
    passes for the wrong reason and hides a regression.
    """
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if path.exists():
            raise AssertionError(f"{path} received {path.read_text(encoding='utf-8')!r}")
        time.sleep(0.05)


def _config(command, *, events=None, enabled=True):
    consumers = []
    if command:
        consumers.append(
            HookConsumer(name="desktop", command=command, events=events or {})
        )
    return ShellHooksConfig(enabled=enabled, consumers=consumers)


class TestInstall:
    def test_disabled_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=False)) is None
        assert get_hook_egress() is None

    def test_enabled_without_a_command_wires_nothing(self):
        assert install_hook_egress(_config([])) is None

    def test_enabled_with_a_command_is_wired(self):
        cfg = install_hook_egress(_config(["/bin/true"]))
        assert cfg is not None
        assert get_hook_egress() is not None

    def test_disabled_forks_nothing_even_when_events_are_dispatched(self, monkeypatch):
        """Verified by process count, not by reading the config.

        Reading the config would only prove the config says "off"; what must hold
        is that no ``Popen`` happens, on either path, even when the dispatch
        points are called as they are in a real run.
        """
        import subprocess

        from agentica.shell_hooks.requests import start_hook_request

        calls = []
        real_popen = subprocess.Popen

        def spy(*args, **kwargs):
            calls.append(args)
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", spy)
        install_hook_egress(_config([]))
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        assert start_hook_request("needs.approval", {}) is None
        assert start_hook_request("needs.input", {}) is None
        assert calls == []


class TestDispatch:
    def test_an_event_reaches_the_command(self, tmp_path):
        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
        hook_egress_dispatch(
            "run.started",
            {"agent_name": "Agent", "prompt": "do the thing"},
            session_id="sess-1",
            work_dir="/w",
        )
        doc = _wait_for(out, 1)[0]
        assert doc["hook_event_name"] == "run.started"
        assert doc["session_id"] == "sess-1"
        assert doc["prompt"] == "do the thing"
        assert doc["cwd"] == "/w"
        assert doc["agent_name"] == "Agent"

    def test_a_switched_off_event_is_not_sent(self, tmp_path):
        command, out = _recorder(tmp_path)
        install_hook_egress(
            _config(command, events={"run.started": False})
        )
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        docs = _wait_for(out, 1)
        assert [d["hook_event_name"] for d in docs] == ["run.completed"]

    def test_no_egress_is_a_no_op(self):
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_broken_command_is_swallowed(self):
        install_hook_egress(_config(["/nonexistent/xyz"]))
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_hanging_command_does_not_block_the_caller(self, tmp_path):
        install_hook_egress(
            _config([sys.executable, "-c", "import time; time.sleep(30)"])
        )
        started = time.monotonic()
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert time.monotonic() - started < 1.0

    def test_a_hanging_notice_consumer_is_killed_and_reaped(
        self, tmp_path, monkeypatch
    ):
        import agentica.shell_hooks.egress as egress_mod

        pid_file = tmp_path / "pid"
        command = [
            sys.executable,
            "-c",
            (
                "import json,os,pathlib,sys,time;"
                "json.load(sys.stdin);"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
                "time.sleep(30)"
            ),
        ]
        monkeypatch.setattr(egress_mod, "NOTICE_PROCESS_TIMEOUT_SECONDS", 0.05)
        install_hook_egress(_config(command))
        hook_egress_dispatch("run.started", {}, session_id="s")
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int(pid_file.read_text(encoding="utf-8"))
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("notice consumer was not killed and reaped")

    def test_process_exit_kills_outstanding_notice_consumers(self, tmp_path):
        """Daemon cleanup dies with the parent; start_new_session children
        would otherwise outlive a CLI / --query process that has already
        exited. atexit must kill the group."""
        pid_file = tmp_path / "pid"
        child = tmp_path / "exiting_parent.py"
        hang = tmp_path / "hang.py"
        hang.write_text(
            "import json, os, pathlib, sys, time\n"
            "json.load(sys.stdin)\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        child.write_text(
            "from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig\n"
            "from agentica.shell_hooks.egress import hook_egress_dispatch, install_hook_egress\n"
            "import time\n"
            "from pathlib import Path\n"
            f"pid_file = Path({str(pid_file)!r})\n"
            "install_hook_egress(ShellHooksConfig(\n"
            "    enabled=True,\n"
            "    consumers=[HookConsumer(\n"
            f"        name='hang', command=[{sys.executable!r}, {str(hang)!r}]\n"
            "    )],\n"
            "))\n"
            "hook_egress_dispatch('run.started', {}, session_id='s')\n"
            "deadline = time.monotonic() + 5\n"
            "while time.monotonic() < deadline and not pid_file.exists():\n"
            "    time.sleep(0.01)\n"
            "raise SystemExit(0)\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(child)],
            cwd=str(tmp_path),
            env={**os.environ, "AGENTICA_HOOKS_ENABLED": "1"},
            timeout=15,
        )
        assert completed.returncode == 0
        pid = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("notice consumer outlived the parent process")

    def test_each_subscribed_consumer_receives_the_event(self, tmp_path):
        first_command, first_out = _recorder(tmp_path, "first")
        second_command, second_out = _recorder(tmp_path, "second")
        install_hook_egress(
            ShellHooksConfig(
                enabled=True,
                consumers=[
                    HookConsumer(name="first", command=first_command),
                    HookConsumer(name="second", command=second_command),
                ],
            )
        )
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert _wait_for(first_out, 1)[0]["hook_event_name"] == "run.started"
        assert _wait_for(second_out, 1)[0]["hook_event_name"] == "run.started"

    def test_dispatch_lazily_installs_for_noninteractive_runs(self, tmp_path, monkeypatch):
        import agentica.shell_hooks.egress as egress_mod

        command, out = _recorder(tmp_path, "lazy")
        monkeypatch.setattr(
            egress_mod,
            "load_shell_hooks_config",
            lambda: _config(command),
        )
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert _wait_for(out, 1)[0]["hook_event_name"] == "run.started"

    def test_concurrent_lazy_install_reads_config_once(self, monkeypatch):
        import agentica.shell_hooks.egress as egress_mod

        calls = []

        def load():
            calls.append(True)
            time.sleep(0.05)
            return _config(["/bin/true"])

        monkeypatch.setattr(egress_mod, "load_shell_hooks_config", load)
        barrier = threading.Barrier(8)
        threads = [
            threading.Thread(
                target=lambda: (
                    barrier.wait(),
                    egress_mod.ensure_hook_egress_installed(),
                )
            )
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert calls == [True]


class TestTheSinkStillWorks:
    def test_both_egresses_receive_one_event(self, tmp_path, monkeypatch):
        """Fan-out, not replacement: the sink half is untouched."""
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
        spy = _SinkSpy()
        monkeypatch.setattr(sink_mod, "_sink", spy)
        record = RunEventRecord(
            run_id="r1", event_type=RunEventType.run_started, payload={"agent_name": "A"}
        )
        sink_mod.notify_sink_dispatch(record, session_id="s1")
        assert [e for e, _ in spy.events] == ["run.started"]
        assert _wait_for(out, 1)[0]["hook_event_name"] == "run.started"

    def test_the_hook_is_reached_with_no_sink_installed(self, tmp_path, monkeypatch):
        """The early return that used to guard the sink would swallow this.

        A user with hooks configured and no notify sink must still get events —
        that combination is the whole point of a second egress.
        """
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
        monkeypatch.setattr(sink_mod, "_sink", None)
        record = RunEventRecord(run_id="r1", event_type=RunEventType.run_started)
        sink_mod.notify_sink_dispatch(record, session_id="s1")
        assert _wait_for(out, 1)[0]["hook_event_name"] == "run.started"

    def test_a_deferred_completion_reaches_the_hook_on_release(self, tmp_path, monkeypatch):
        """The goal deferral is shared, so the hook sees the same one release the
        sink does — not one event per lap."""
        import agentica.notify.sink as sink_mod
        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
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
        assert not out.exists(), "the completion must be held while a goal drives"

        monkeypatch.setattr(sink_mod, "_goal_is_driving", lambda agent: False)
        sink_mod.goal_finished(agent, session_id="s1")
        doc = _wait_for(out, 1)[0]
        assert doc["hook_event_name"] == "run.completed"


class TestSubagentRunsStayOffTheWire:
    """A subagent is not a session.

    Children are spawned with no ``session_id`` of their own, so their events
    would be keyed on the process-wide fallback: every consumer would grow one
    phantom session that lights up whenever any child touches a tool, never gets
    ``session.started`` / ``session.ended``, and merges all concurrent children
    into one row. ``tool.*`` is what makes it obvious, because children are
    tool-heavy.
    """

    def test_a_subagent_tool_event_reaches_neither_egress(self, tmp_path, monkeypatch):
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
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
        _assert_nothing_arrives(out)

    def test_a_subagent_completion_cannot_be_mistaken_for_the_users_turn(
        self, tmp_path, monkeypatch
    ):
        """The one that would actually mislead: 'done' for work still running.

        The deferral inputs are pinned rather than left to ambient state: with
        `agent=None`, or with an idle provider left behind by another test, the
        completion path bails on its own and this would pass whether or not the
        guard exists.
        """
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
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

        _assert_nothing_arrives(out)

    def test_the_parents_own_events_still_go_out(self, tmp_path, monkeypatch):
        """The guard keys on lineage, so it must not silence the top-level run."""
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(_config(command))
        monkeypatch.setattr(sink_mod, "_sink", None)
        record = RunEventRecord(
            run_id="parent-run",
            event_type=RunEventType.tool_started,
            parent_run_id=None,
            payload={"tool_name": "read_file", "tool_call_id": "c1"},
        )

        sink_mod.notify_sink_dispatch(record, session_id="s1")

        assert _wait_for(out, 1)[0]["hook_event_name"] == "tool.started"
