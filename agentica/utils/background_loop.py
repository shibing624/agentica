# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Process-wide event loop for hook side work that outlives a turn.

Sync callers (``run_sync`` / ``run_stream_sync``, and so the CLI) drive every
turn through a fresh ``asyncio.run``. A task created on that loop is cancelled
the moment the turn returns, so fire-and-forget work such as memory extraction
never reached its LLM call. Work submitted here runs on one daemon thread whose
loop lives as long as the process; callers await it from any loop through
:func:`wait_all`.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine, Iterable
from typing import Any, Optional

_loop: Optional[asyncio.AbstractEventLoop] = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="agentica-background-loop", daemon=True,
            ).start()
            _loop = loop
        return _loop


def submit(coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
    """Schedule ``coro`` on the background loop; returns a thread-safe future."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


async def wait_all(futures: Iterable[concurrent.futures.Future]) -> None:
    """Await background futures from whichever loop the caller is on."""
    pending = list(futures)
    if pending:
        await asyncio.gather(*(asyncio.wrap_future(f) for f in pending), return_exceptions=True)
