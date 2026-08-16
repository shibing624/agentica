# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Subprocess teardown must not hang, orphan, or reach stderr.

The CLI runs every turn in its own ``asyncio.run()``. A subprocess transport
left open outlives that loop, and its destructor then raises
``RuntimeError: Event loop is closed``, which CPython prints to ``sys.stderr`` —
and ``prompt_toolkit.patch_stdout`` points ``sys.stderr`` at the TUI transcript,
so the user reads an asyncio traceback in the middle of an answer.

Every case here is built on the shape that produced the original report:
``cmd &`` inside the shell, where the shell exits at once and the child it
backgrounded keeps the write end of our PIPEs — so waiting for EOF waits for
the child, and killing "the process" kills something already gone.
"""
from __future__ import annotations

import asyncio
import gc
import io
import os
import signal
import subprocess
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
from agentica.utils.async_utils import (
    DRAIN_TIMEOUT_SECONDS,
    close_subprocess_transport,
    terminate_subprocess,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")


def _holder_script(tmp_path: Path, *, escape_group: bool = False) -> str:
    """Write a script that holds the inherited PIPE write end open for a while.

    ``escape_group``: the holder puts itself in a new session first, so no
    ``killpg`` of ours can reach it and EOF can never arrive.
    """
    name = "holder_escaped.sh" if escape_group else "holder.sh"
    body = "#!/bin/sh\n"
    if escape_group:
        body += "exec python3 -c 'import os, time; os.setsid(); time.sleep(45)'\n"
    else:
        body += "sleep 45\n"
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def _pids_running(script: str) -> list[int]:
    """PIDs whose argv is exactly this script (`/bin/sh <script>`).

    Matched on argv rather than a substring: the pytest process's own command
    line can contain the path, which would read as a survivor that never was.
    """
    out = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True
    ).stdout
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) in (2, 3) and parts[-1] == script:
            pids.append(int(parts[0]))
    return pids


def _kill_holders(*scripts: str) -> None:
    for script in scripts:
        for pid in _pids_running(script):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


async def _spawn_then_lose_the_shell(script: str) -> asyncio.subprocess.Process:
    """Spawn `script & sleep`, and return once our own child has been reaped."""
    proc = await asyncio.create_subprocess_shell(
        f"{script} & sleep 0.05",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while proc.returncode is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert proc.returncode is not None, "the shell should have exited on its own"
    return proc


def _wait_gone(script: str, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pids_running(script):
            return True
        time.sleep(0.05)
    return False


# Teardown must finish on its own terms. The holder sleeps far longer than this,
# so a run that exceeds it did not clean up — it waited the stray process out,
# which is the hang being tested for.
_TEARDOWN_BUDGET = DRAIN_TIMEOUT_SECONDS + 3


@pytest.mark.asyncio
async def test_group_is_signalled_even_after_our_own_child_was_reaped(tmp_path):
    """The signal must not be gated on ``process.returncode``.

    The shell is gone but the group it led still holds our pipes. Gating on our
    own child's exit code skipped the kill entirely, which left the work running
    and the drain below waiting for an EOF that could not come.
    """
    holder = _holder_script(tmp_path)
    try:
        proc = await _spawn_then_lose_the_shell(holder)
        assert _pids_running(holder), "the backgrounded child should still be alive"

        await asyncio.wait_for(
            terminate_subprocess(proc, process_group=True), timeout=_TEARDOWN_BUDGET
        )

        assert _wait_gone(holder), "terminate_subprocess left the group running"
    finally:
        _kill_holders(holder)


@pytest.mark.asyncio
async def test_teardown_gives_up_on_a_drain_that_can_never_finish(tmp_path):
    """EOF is best effort: the holder escaped the group, so no signal reaches it.

    Waiting for EOF unbounded here is what pinned a turn for as long as the
    stray process lived.
    """
    holder = _holder_script(tmp_path, escape_group=True)
    try:
        proc = await _spawn_then_lose_the_shell(holder)

        await asyncio.wait_for(
            terminate_subprocess(proc, process_group=True), timeout=_TEARDOWN_BUDGET
        )
        assert proc._transport.is_closing(), "the transport must be closed regardless"
    finally:
        _kill_holders(holder)


@pytest.mark.asyncio
async def test_grace_period_escalates_to_sigkill_and_closes(tmp_path):
    holder = _holder_script(tmp_path)
    try:
        proc = await _spawn_then_lose_the_shell(holder)
        await asyncio.wait_for(
            terminate_subprocess(proc, process_group=True, grace_period=0.3),
            timeout=_TEARDOWN_BUDGET,
        )
        assert _wait_gone(holder)
        assert proc._transport.is_closing()
    finally:
        _kill_holders(holder)


def _unraisable_after_a_turn(coro_factory) -> list[str]:
    """Run one CLI-shaped turn and return what CPython could not raise.

    Watching ``sys.stderr`` does not work here: pytest's ``unraisableexception``
    plugin installs its own ``sys.unraisablehook`` and turns these into warnings,
    so the bytes never reach the stream and the assertion passes on broken code.
    The hook is the real interface — CPython calls it *instead of* printing, and
    the default hook writes to ``sys.stderr``, which in the CLI is the TUI.

    The GC pass matters too: while an exception is still referenced its
    ``__traceback__`` keeps the subprocess alive, which is why the leak surfaced
    on the turn *after* the one that caused it.
    """
    recorded: list[str] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda u: recorded.append(f"{u.exc_type.__name__}: {u.exc_value}")
    try:
        try:
            asyncio.run(coro_factory())
        except asyncio.CancelledError:
            pass  # deliberately not bound: binding would keep the traceback
        gc.collect()
        gc.collect()
    finally:
        sys.unraisablehook = previous_hook
    return recorded


def test_double_ctrl_c_during_a_hung_command_leaves_nothing_on_stderr(tmp_path):
    """The reported failure, end to end: Ctrl+C, then Ctrl+C again.

    The second cancel lands inside the cleanup's own drain. Before the fix that
    abandoned the transport with a live reader, and the next turn printed
    ``Exception ignored in: BaseSubprocessTransport.__del__`` into the TUI.
    """
    holder = _holder_script(tmp_path)
    tool = BuiltinExecuteTool(work_dir=str(tmp_path), timeout=300)

    async def turn():
        task = asyncio.create_task(tool.execute(command=f"{holder} & sleep 0.05"))
        await asyncio.sleep(0.08)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        assert _unraisable_after_a_turn(turn) == []
    finally:
        _kill_holders(holder)


def test_timeout_with_an_orphan_pipe_holder_returns_and_kills_the_group(tmp_path):
    """The timeout must be the timeout, not the start of an unbounded wait."""
    holder = _holder_script(tmp_path)
    tool = BuiltinExecuteTool(work_dir=str(tmp_path), timeout=0.05)

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(tool.execute(command=f"{holder} & sleep 0.05"))
        assert time.monotonic() - started < 10, "cleanup hung past the timeout"
        assert _wait_gone(holder), "the timed-out command was left running"
    finally:
        _kill_holders(holder)


def test_plain_command_still_returns_its_output(tmp_path):
    tool = BuiltinExecuteTool(work_dir=str(tmp_path))
    buf = io.StringIO()
    with redirect_stderr(buf):
        result = asyncio.run(tool.execute(command="echo hello-from-test"))
        gc.collect()
    assert "hello-from-test" in result
    assert buf.getvalue() == ""


def test_close_subprocess_transport_tolerates_none_and_repeat_calls(tmp_path):
    close_subprocess_transport(None)

    async def go():
        proc = await asyncio.create_subprocess_shell(
            "true", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        close_subprocess_transport(proc)
        close_subprocess_transport(proc)

    asyncio.run(go())
