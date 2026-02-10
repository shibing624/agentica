# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async-first utilities.

Provides run_sync() and iter_over_async() for bridging async-first code
to synchronous callers.
"""

import asyncio
import threading
from typing import TypeVar, Coroutine, AsyncIterator, Iterator

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


def iter_over_async(ait: AsyncIterator[T]) -> Iterator[T]:
    """Convert an AsyncIterator to a synchronous Iterator.

    Creates a dedicated event loop in a background thread to drive the
    async iterator, yielding items back to the synchronous caller.
    """
    sentinel = object()
    loop = asyncio.new_event_loop()

    async def _get_next() -> T:
        try:
            return await ait.__anext__()
        except StopAsyncIteration:
            return sentinel  # type: ignore

    try:
        while True:
            item = loop.run_until_complete(_get_next())
            if item is sentinel:
                break
            yield item
    finally:
        # Clean up: close the async iterator if it has aclose()
        if hasattr(ait, "aclose"):
            try:
                loop.run_until_complete(ait.aclose())
            except Exception:
                pass
        loop.close()
