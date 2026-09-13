# -*- coding: utf-8 -*-
"""The lifecycle fan-out: one event reaches every installed egress, and a broken
consumer cannot affect the sink or the run."""

from __future__ import annotations

import json
import sys
import time

import pytest

from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import (
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)
from agentica.run_events import RunEventRecord, RunEventType


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


def _recorder(tmp_path):
    """A hook command that appends each payload to a file."""
    out = tmp_path / "seen.jsonl"
    script = tmp_path / "hook.py"
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
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) >= count:
                return [json.loads(l) for l in lines]
        time.sleep(0.05)
    raise AssertionError(f"{path} never reached {count} payloads")


class TestInstall:
    def test_disabled_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=False)) is None
        assert get_hook_egress() is None

    def test_enabled_without_a_command_wires_nothing(self):
        assert install_hook_egress(ShellHooksConfig(enabled=True, command=[])) is None

    def test_enabled_with_a_command_is_wired(self):
        cfg = install_hook_egress(ShellHooksConfig(enabled=True, command=["/bin/true"]))
        assert cfg is not None
        assert get_hook_egress() is not None


class TestDispatch:
    def test_an_event_reaches_the_command(self, tmp_path):
        command, out = _recorder(tmp_path)
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
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
            ShellHooksConfig(enabled=True, command=command, events={"run.started": False})
        )
        hook_egress_dispatch("run.started", {}, session_id="s")
        hook_egress_dispatch("run.completed", {}, session_id="s")
        docs = _wait_for(out, 1)
        assert [d["hook_event_name"] for d in docs] == ["run.completed"]

    def test_no_egress_is_a_no_op(self):
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_broken_command_is_swallowed(self):
        install_hook_egress(ShellHooksConfig(enabled=True, command=["/nonexistent/xyz"]))
        hook_egress_dispatch("run.started", {}, session_id="s")  # must not raise

    def test_a_hanging_command_does_not_block_the_caller(self, tmp_path):
        install_hook_egress(
            ShellHooksConfig(
                enabled=True,
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
            )
        )
        started = time.monotonic()
        hook_egress_dispatch("run.started", {}, session_id="s")
        assert time.monotonic() - started < 1.0


class TestTheSinkStillWorks:
    def test_both_egresses_receive_one_event(self, tmp_path, monkeypatch):
        """Fan-out, not replacement: the sink half is untouched."""
        import agentica.notify.sink as sink_mod

        command, out = _recorder(tmp_path)
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
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
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
        monkeypatch.setattr(sink_mod, "_sink", None)
        record = RunEventRecord(run_id="r1", event_type=RunEventType.run_started)
        sink_mod.notify_sink_dispatch(record, session_id="s1")
        assert _wait_for(out, 1)[0]["hook_event_name"] == "run.started"

    def test_a_deferred_completion_reaches_the_hook_on_release(self, tmp_path, monkeypatch):
        """The goal deferral is shared, so the hook sees the same one release the
        sink does — not one event per lap."""
        import agentica.notify.sink as sink_mod
        from agentica.notify import set_idle_provider

        command, out = _recorder(tmp_path)
        install_hook_egress(ShellHooksConfig(enabled=True, command=command))
        monkeypatch.setattr(sink_mod, "_sink", None)
        monkeypatch.setattr(sink_mod, "_goal_is_driving", lambda agent: True)

        class _Agent:
            run_response = None
            _session_log = None

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
