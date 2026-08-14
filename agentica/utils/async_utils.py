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


def close_subprocess_transport(process: Optional[asyncio.subprocess.Process]) -> None:
    """Mark an asyncio subprocess transport closed while its loop is still alive.

    ``Process.communicate()`` drains the pipes and waits for the exit code; it
    does not set ``BaseSubprocessTransport._closed``. The destructor then calls
    ``close()``, which does ``loop.call_soon`` on a loop ``asyncio.run()`` has
    already shut — CPython prints ``Exception ignored in:
    BaseSubprocessTransport.__del__`` / ``RuntimeError: Event loop is closed``
    to stderr. The CLI TUI only patches stdout, so that traceback lands in the
    transcript. Closing here makes ``__del__`` a no-op.
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
    """Terminate and fully reap an asyncio subprocess on its live event loop.

    ``Process.wait()`` only observes the exit code. ``communicate()`` also
    drains stdout/stderr. ``close_subprocess_transport`` then marks the
    transport closed so ``__del__`` cannot call back into a shut loop.
    """

    def stop(*, force: bool) -> None:
        if process.returncode is not None:
            return
        try:
            if process_group and os.name != "nt":
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.killpg(process.pid, sig)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    try:
        if grace_period > 0:
            stop(force=False)
            try:
                await asyncio.shield(
                    asyncio.wait_for(process.communicate(), timeout=grace_period)
                )
                return
            except asyncio.TimeoutError:
                stop(force=True)
        else:
            stop(force=True)

        # Ctrl+C cancels the agent task. Without shield, that cancel also
        # aborts this communicate() — the process is killed but its PIPE
        # transports stay open, and their __del__ later dumps
        # "Event loop is closed" into the next turn's TUI.
        await asyncio.shield(process.communicate())
    finally:
        close_subprocess_transport(process)


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
