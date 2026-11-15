# -*- coding: utf-8 -*-
"""`agentica watch` decisions, with probes stubbed.

The contract is exit codes and streamed progress, and the loop implementing it
is pure bookkeeping. The previous suite launched real shell jobs and waited
seconds per scenario; the probe runner is the seam that matters.
"""
import pytest

from agentica.tools import job_watch
from agentica.tools.job_watch import (
    LOST_CONTACT_EXIT,
    RESULT_UNKNOWN_EXIT,
    new_lines,
    watch,
)


def _probes(monkeypatch, queues):
    """Answer each probe from a per-command queue; the last reply repeats."""
    calls = []

    def fake_run(command, timeout):
        calls.append(command)
        queue = queues.get(command) or [(0, "")]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(job_watch, "_run", fake_run)
    monkeypatch.setattr(job_watch.time, "sleep", lambda seconds: None)
    return calls


def _spec(**overrides):
    spec = {"done": "done", "progress": "progress", "result": "result", "interval": 0.001}
    spec.update(overrides)
    return spec


def test_ends_with_the_observed_job_exit_code_and_streams_progress(monkeypatch, capsys):
    calls = _probes(monkeypatch, {
        "done": [(1, ""), (0, "")],
        "progress": [(0, "step 1\nstep 2\n")],
        "result": [(3, "boom\n")],
    })

    assert watch(_spec()) == 3

    out = capsys.readouterr().out
    assert out.count("step 1") == 1 and out.count("step 2") == 1
    assert "[result]" in out and "boom" in out
    assert "[job finished: exit 3]" in out
    assert calls.count("done") == 2


def test_overlapping_progress_tails_print_each_line_once(monkeypatch, capsys):
    _probes(monkeypatch, {
        "done": [(1, ""), (1, ""), (0, "")],
        "progress": [(0, "a\nb\n"), (0, "b\nc\n")],
        "result": [(0, "")],
    })

    assert watch(_spec()) == 0
    out = capsys.readouterr().out
    assert out.splitlines().count("b") == 1
    assert out.index("a") < out.index("b") < out.index("c")


def test_repeated_probe_failures_end_as_lost_contact_without_running_the_result(monkeypatch, capsys):
    calls = _probes(monkeypatch, {"done": [(255, "ssh: connect failed")]})

    assert watch(_spec(max_failures=2)) == LOST_CONTACT_EXIT
    out = capsys.readouterr().out
    assert "lost contact" in out
    assert "result" not in calls


def test_a_result_probe_timeout_is_unknown_not_success(monkeypatch, capsys):
    _probes(monkeypatch, {"done": [(0, "")], "result": [(None, "partial\n")]})

    assert watch(_spec()) == RESULT_UNKNOWN_EXIT
    out = capsys.readouterr().out
    assert "result probe timed out" in out
    assert "partial" in out


def test_without_a_result_command_the_watch_ends_zero(monkeypatch, capsys):
    _probes(monkeypatch, {"done": [(0, "")]})
    assert watch(_spec(result="")) == 0
    assert "[job finished]" in capsys.readouterr().out


def test_the_default_poll_interval_is_thirty_seconds(monkeypatch):
    sleeps = []
    _probes(monkeypatch, {"done": [(1, ""), (0, "")]})
    monkeypatch.setattr(job_watch.time, "sleep", sleeps.append)

    watch({"done": "done", "result": "", "interval": 0})

    assert sleeps == [30.0]


# ---------------------------------------------------------------- new_lines


def test_new_lines_strips_the_previous_tail():
    assert new_lines(["a", "b"], ["b", "c"]) == ["c"]
    assert new_lines(["a"], ["x", "y"]) == ["x", "y"]
    assert new_lines([], ["x"]) == ["x"]
    assert new_lines(["a"], ["a"]) == []
