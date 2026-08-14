# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Asyncio subprocess transports must be closed on the live loop.
"""
from __future__ import annotations

import asyncio
import gc
import io
from contextlib import redirect_stderr

from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
from agentica.utils.async_utils import close_subprocess_transport


def _stderr_after_gc(run) -> str:
    buf = io.StringIO()
    with redirect_stderr(buf):
        run()
        gc.collect()
        gc.collect()
    return buf.getvalue()


def test_close_subprocess_transport_after_communicate_keeps_stderr_clean():
    """CLI runs each turn in asyncio.run(); leftover transports __del__ after
    the loop closes and dump 'Event loop is closed' to stderr."""

    async def go():
        proc = await asyncio.create_subprocess_shell(
            "true",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        close_subprocess_transport(proc)

    stderr = _stderr_after_gc(lambda: asyncio.run(go()))
    assert "Event loop is closed" not in stderr
    assert "Exception ignored" not in stderr


def test_execute_does_not_dump_closed_loop_traceback_to_stderr(tmp_path):
    tool = BuiltinExecuteTool(work_dir=str(tmp_path))
    stderr = _stderr_after_gc(lambda: asyncio.run(tool.execute("true")))
    assert "Event loop is closed" not in stderr
    assert "Exception ignored" not in stderr
    assert "BaseSubprocessTransport" not in stderr


def test_cancelling_a_running_execute_does_not_dump_closed_loop_on_next_gc(tmp_path):
    """Ctrl+C while execute is blocked in communicate(): the next turn's
    asyncio.run() must not inherit an unclosed transport from this loop."""
    tool = BuiltinExecuteTool(work_dir=str(tmp_path), timeout=30)

    async def cancel_after_start():
        task = asyncio.create_task(tool.execute("sleep 30"))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    stderr = _stderr_after_gc(lambda: asyncio.run(cancel_after_start()))
    assert "Event loop is closed" not in stderr
    assert "Exception ignored" not in stderr
    assert "BaseSubprocessTransport" not in stderr
