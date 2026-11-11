# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: `agentica watch`: track a job agentica did not start, until it ends.

A tmux session, a ``nohup`` job, a run over ssh, a queue entry: none of them
is a child of agentica, so nothing reports their end. Run this command with
``execute(background=True)`` and the job gets what a background command gets:
a card with its latest output, ``/ps``, Stop, and a completion report.

The contract is carried by exit codes, never by reading output:

- ``--done`` runs every ``--interval`` seconds. Exit 0: the job has finished.
  Exit 1: still running. Anything else (ssh's 255, a timeout): the check
  failed, which says nothing about the job; it is retried, and after
  ``--max-failures`` failures in a row the watch exits ``LOST_CONTACT_EXIT``.
- ``--progress`` runs while the job is running; its new lines go to the log,
  so the last one is the card's progress line.
- ``--result`` runs once after ``--done``; its output goes to the log and its
  exit code becomes the watch's exit code (``exit $(cat run.exit)``).

Stopping the watch (Stop in the UI, the session ending) ends only the watch.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple

LOST_CONTACT_EXIT = 75
# The job already finished, but ``--result`` did not return an exit code.
# Distinct from ``LOST_CONTACT_EXIT``: that one means the job's state is unknown.
RESULT_UNKNOWN_EXIT = 124
DEFAULT_MAX_FAILURES = 10
DEFAULT_PROBE_TIMEOUT = 120.0

# The probe in flight, so a stop can take it down too: it runs in its own
# process group, out of reach of the signal the registry sends this one.
_current_probe: Optional[subprocess.Popen] = None


def _run(command: str, timeout: float) -> Tuple[Optional[int], str]:
    """Run one probe; ``None`` exit code means it timed out.

    Own process group, killed whole on timeout: a hung ``ssh`` is a grandchild
    of the shell and would otherwise outlive the kill and pin the pipe.
    """
    global _current_probe
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )
    _current_probe = proc
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_probe(proc)
        out, _ = proc.communicate()
        return None, out.decode("utf-8", errors="replace")
    finally:
        _current_probe = None
    return proc.returncode, out.decode("utf-8", errors="replace")


def _kill_probe(proc: subprocess.Popen) -> None:
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def new_lines(previous: List[str], current: List[str]) -> List[str]:
    """Lines of ``current`` not already printed as the tail of ``previous``.

    A progress probe is usually ``tail -n K log``: consecutive outputs overlap.
    Printing only what follows the longest overlap keeps the log a readable
    stream instead of K copies of every line.
    """
    for k in range(min(len(previous), len(current)), 0, -1):
        if previous[-k:] == current[:k]:
            return current[k:]
    return current


def _emit(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def _stopped(signum, _frame) -> None:
    if _current_probe is not None:
        _kill_probe(_current_probe)
    _emit("[watch stopped; the job itself was not touched]")
    # Die by the signal so the registry records a stop, not a job failure.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def watch(spec: dict) -> int:
    done = spec["done"]
    progress = spec.get("progress") or ""
    result = spec.get("result") or ""
    interval = float(spec.get("interval", 30))
    if interval <= 0:
        interval = 30
    max_failures = int(spec.get("max_failures", DEFAULT_MAX_FAILURES))
    probe_timeout = float(spec.get("probe_timeout", DEFAULT_PROBE_TIMEOUT))

    failures = 0
    shown: List[str] = []
    while True:
        rc, out = _run(done, probe_timeout)
        if rc == 0:
            break
        if rc == 1:
            failures = 0
            if progress:
                prc, pout = _run(progress, probe_timeout)
                if prc == 0:
                    lines = [line for line in pout.splitlines() if line.strip()]
                    fresh = new_lines(shown, lines)
                    if fresh:
                        _emit("\n".join(fresh))
                    shown = lines
        else:
            failures += 1
            why = "timed out" if rc is None else f"exit {rc}"
            tail = " | ".join(out.strip().splitlines()[-2:])
            _emit(
                f"[{time.strftime('%H:%M:%S')}] probe failed ({why}), job state unknown, "
                f"retrying ({failures}/{max_failures})" + (f": {tail}" if tail else "")
            )
            if failures >= max_failures:
                _emit(
                    f"[lost contact: the done probe failed {failures} times in a row. "
                    "The job may still be running; fix the connection and watch it again.]"
                )
                return LOST_CONTACT_EXIT
        time.sleep(interval)

    if not result:
        _emit("[job finished]")
        return 0
    rc, out = _run(result, probe_timeout)
    if out.strip():
        _emit("[result]")
        _emit(out.rstrip("\n"))
    if rc is None:
        _emit("[job finished, but the result probe timed out; its exit code is unknown]")
        return RESULT_UNKNOWN_EXIT
    _emit(f"[job finished: exit {rc}]")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentica watch",
        description="Track a job agentica did not start, by exit-code checks. "
        "Run it with execute(background=True).",
    )
    parser.add_argument("--done", required=True,
                        help="shell check: exit 0 = finished, 1 = running, other = check failed (retried)")
    parser.add_argument("--progress", default="", help="shell command printing recent output")
    parser.add_argument("--result", default="",
                        help="shell command run once at the end; its exit code is the job's")
    parser.add_argument("--interval", type=float, default=30, help="seconds between checks (default 30)")
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES,
                        help=f"failed checks in a row before exiting {LOST_CONTACT_EXIT} (default {DEFAULT_MAX_FAILURES})")
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT,
                        help=f"seconds one check may take (default {DEFAULT_PROBE_TIMEOUT:g})")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _stopped)
    signal.signal(signal.SIGINT, _stopped)
    return watch(vars(args))


if __name__ == "__main__":
    sys.exit(main())
