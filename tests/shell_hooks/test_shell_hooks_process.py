# -*- coding: utf-8 -*-
"""Spawning the user's command: real processes, because the failure modes are
process failure modes (a non-zero exit, an empty stdout, a child that outlives
its parent)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

from agentica.shell_hooks.process import HookProcess


def _py(body: str):
    return [sys.executable, "-c", body]


class TestTheDocumentArrivesOnStdin:
    def test_the_hook_sees_the_payload(self):
        reader = _py(
            "import json,sys;"
            "doc=json.load(sys.stdin);"
            "print(json.dumps({'answer': doc['hook_event_name']}))"
        )
        proc = HookProcess(reader, {"hook_event_name": "needs.input"})
        assert proc.start() is True
        assert proc.wait(timeout=10) is True
        assert proc.stdout == '{"answer": "needs.input"}\n'

    def test_a_complete_document_does_not_wait_for_process_exit(self):
        proc = HookProcess(
            _py(
                "import json,sys,time;"
                "print(json.dumps({'answer':'ready'}));sys.stdout.flush();"
                "time.sleep(5)"
            ),
            {},
        )
        assert proc.start() is True
        started = time.monotonic()
        assert proc.wait(timeout=1) is True
        assert time.monotonic() - started < 1
        assert proc.stdout == '{"answer": "ready"}\n'
        proc.kill()


class TestFailuresAreNotDecisions:
    def test_a_non_zero_exit_still_returns_whatever_was_printed(self):
        proc = HookProcess(_py("import sys; print(''); sys.exit(3)"), {})
        assert proc.start() is True
        assert proc.wait(timeout=10) is True
        assert proc.stdout is not None  # the caller decides; see parse_reply

    def test_a_missing_command_does_not_raise(self):
        proc = HookProcess(["/nonexistent/notifier-xyz"], {})
        assert proc.start() is False

    def test_an_empty_command_does_not_raise(self):
        assert HookProcess([], {}).start() is False

    def test_a_hook_that_never_reads_stdin_does_not_hang_the_spawn(self):
        """A consumer that exits without reading must produce a broken pipe, not
        a wedged run."""
        proc = HookProcess(_py("raise SystemExit(0)"), {"k": "v" * 40_000_000})
        assert proc.start() in (True, False)
        proc.kill()


class TestKilling:
    def test_kill_is_idempotent_and_leaves_nothing_running(self):
        proc = HookProcess(_py("import time; time.sleep(60)"), {})
        assert proc.start() is True
        assert proc.returncode is None
        proc.kill()
        proc.kill()
        time.sleep(0.3)
        # returncode is not None once the child has been reaped. Checking
        # os.kill(pid, 0) here would be wrong: signal 0 succeeds for a zombie,
        # so it cannot tell "dead" from "dead and not yet collected".
        assert proc.returncode is not None

    def test_kill_takes_the_whole_group(self):
        """A hook that spawned its own child must not leave it behind."""
        child = _py("import time; time.sleep(60)")
        body = (
            "import subprocess,sys,time;"
            f"subprocess.Popen({child!r});"
            "time.sleep(60)"
        )
        proc = HookProcess(_py(body), {})
        assert proc.start() is True
        time.sleep(0.8)  # let the grandchild exist
        proc.kill()
        time.sleep(0.3)
        assert _group_empty(proc.pid)

    def test_kill_group_after_the_original_hook_has_exited(self):
        """A grandchild may retain stdout after the session leader exits."""
        child = _py("import time; time.sleep(60)")
        body = f"import subprocess;subprocess.Popen({child!r})"
        proc = HookProcess(_py(body), {})
        assert proc.start() is True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and proc.returncode is None:
            time.sleep(0.02)
        assert proc.returncode == 0
        assert not _group_empty(proc.pid)
        proc.kill()
        time.sleep(0.3)
        assert _group_empty(proc.pid)

    def test_kill_leaves_no_zombie(self):
        """A killed child whose status is never collected stays ``<defunct>``.

        The acceptance for this feature is "no zombie after the terminal
        answered first", and ``os.kill(pid, 0)`` cannot see the difference, so
        this reads the process table.
        """
        proc = HookProcess(_py("import time; time.sleep(60)"), {})
        assert proc.start() is True
        pid = proc.pid
        proc.kill()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if _is_gone_from_ps(pid):
                return
            time.sleep(0.1)
        state = _ps_line(pid)
        raise AssertionError(f"process {pid} is still in the table: {state!r}")


def _group_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except OSError:
        return True
    return False


def _ps_line(pid) -> str:
    out = subprocess.run(
        ["ps", "-eo", "pid=,stat=,command="], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        parts = line.split(None, 2)
        if parts and parts[0] == str(pid):
            return line.strip()
    return ""


def _is_gone_from_ps(pid) -> bool:
    line = _ps_line(pid)
    if not line:
        return True
    # ``Z`` is a zombie: killed, but its status was never collected.
    return False
