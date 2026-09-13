# -*- coding: utf-8 -*-
"""Spawning the user's command: real processes, because the failure modes are
process failure modes (a non-zero exit, an empty stdout, a child that outlives
its parent)."""

from __future__ import annotations

import os
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


def _group_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except OSError:
        return True
    return False
