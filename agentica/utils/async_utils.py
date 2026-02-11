# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async-first utilities.

Provides run_sync() for bridging async-first code
to synchronous callers.
"""

import asyncio
import threading
from typing import TypeVar, Coroutine

T = TypeVar("T")


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

