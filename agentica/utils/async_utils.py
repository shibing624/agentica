# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async-first utilities.

Provides run_sync() for bridging async-first code
to synchronous callers.
"""

import asyncio
import os
import signal
import threading
from collections.abc import Coroutine
from typing import Optional, TypeVar

T = TypeVar("T")


# How long a teardown drain may wait for EOF on the pipes. Bounded because EOF
# is not guaranteed to arrive at all: see terminate_subprocess.
DRAIN_TIMEOUT_SECONDS = 2.0


def close_subprocess_transport(process: Optional[asyncio.subprocess.Process]) -> None:
    """Close an asyncio subprocess transport while its event loop is still alive.

    A pipe transport that reached EOF has already closed itself, so its
    destructor is a no-op. One that did not — a killed process, a cancelled
    turn, a write end still held by something we left behind — still has a
    reader registered on the loop, and ``BaseSubprocessTransport.__del__``
    closes it from whatever GC pass happens to collect it. The CLI runs each
    turn in its own ``asyncio.run()``, so that pass falls after the loop was
    shut: ``call_soon`` raises ``RuntimeError: Event loop is closed`` and
    CPython prints ``Exception ignored in: ...__del__`` to ``sys.stderr``,
    which ``prompt_toolkit.patch_stdout`` has redirected into the TUI
    transcript. Closing on the live loop is what makes ``__del__`` a no-op.
    """
    if process is None:
        return
    transport = process._transport
    if transport is None or transport.is_closing():
        return
    transport.close()


async def terminate_subprocess(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool = False,
    grace_period: float = 0.0,
) -> None:
    """Signal, drain and close a subprocess that did not finish on its own.

    Two properties this must hold, both learned from a command that hung for
    the rest of the session:

    * **The signal cannot be gated on our own child's exit code.** ``cmd &``
      inside the shell leaves a grandchild holding the write end of our PIPEs,
      and the shell exits at once — so the process we spawned is already reaped
      while the group it led is still running. Skipping ``killpg`` in that state
      left the grandchild alive *and* the drain below waiting for an EOF that
      could never come.
    * **Every drain is bounded.** Even after ``killpg`` the write end may be
      held by a process outside the group, so waiting for EOF is best effort.
      The transport is closed either way.
    """

    def stop(*, force: bool) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            if process_group and os.name != "nt":
                # pid == pgid: every caller spawns with start_new_session=True.
                # Deliberately not guarded by process.returncode — that guard
                # was the bug. The group outlives its leader here, and once no
                # member is left this just raises ProcessLookupError.
                os.killpg(process.pid, sig)
            elif process.returncode is None:
                if force:
                    process.kill()
                else:
                    process.terminate()
        except (ProcessLookupError, PermissionError):
            pass

    try:
        if grace_period > 0:
            stop(force=False)
            if await _drain(process, grace_period):
                return
            stop(force=True)
        else:
            stop(force=True)
        await _drain(process, DRAIN_TIMEOUT_SECONDS)
    finally:
        close_subprocess_transport(process)


async def _drain(process: asyncio.subprocess.Process, timeout: float) -> bool:
    """Read the pipes to EOF and collect the exit code. False if the wait ran out."""
    try:
        await asyncio.wait_for(process.communicate(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


def run_sync(coro: Coroutine[None, None, T]) -> T:
    """Run an async coroutine from a synchronous context.

    Handles three scenarios:
    1. No running event loop -> asyncio.run()
    2. Inside an event loop (Jupyter / nested) -> new thread + new event loop
    3. Keyboard interrupt -> clean cancellation
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run directly
        return asyncio.run(coro)

    # Already inside an event loop, run in a separate thread
    result: T = None  # type: ignore
    exception: BaseException = None  # type: ignore

    def _run_in_thread() -> None:
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except BaseException as e:
            exception = e

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    return result
