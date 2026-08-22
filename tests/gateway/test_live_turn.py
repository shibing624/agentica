# -*- coding: utf-8 -*-
"""Web chat run registry: reconnect, seq replay, cancel waits for the lock."""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

from agentica.gateway.services import live_turn
from agentica.gateway.services.live_turn import cancel_and_wait


@pytest.fixture(autouse=True)
def _reset_hub():
    live_turn.reset()
    yield
    live_turn.reset()


def test_subscribe_replays_then_live():
    turn = live_turn.start("s1", owner="default")
    turn.publish({"event": "content", "data": "a"})
    turn.publish({"event": "content", "data": "b"})
    q = turn.subscribe()
    evs = [q.get_nowait(), q.get_nowait()]
    assert [e["seq"] for e in evs] == [1, 2]
    assert [e["data"] for e in evs] == ["a", "b"]
    turn.publish({"event": "content", "data": "c"})
    assert q.get_nowait()["seq"] == 3


def test_subscribe_after_skips_old_seq():
    turn = live_turn.start("s1", owner="default")
    turn.publish({"event": "content", "data": "a"})
    turn.publish({"event": "content", "data": "b"})
    turn.publish({"event": "content", "data": "c"})
    q = turn.subscribe(after=2)
    ev = q.get_nowait()
    assert ev["seq"] == 3
    assert ev["data"] == "c"


def test_unsubscribe_does_not_cancel_and_late_subscriber_replays():
    turn = live_turn.start("s1", owner="default")
    q1 = turn.subscribe()
    turn.publish({"event": "content", "data": "hello"})
    turn.unsubscribe(q1)
    turn.publish({"event": "content", "data": "world"})
    assert live_turn.active("s1") is turn
    q2 = turn.subscribe()
    assert [q2.get_nowait()["data"] for _ in range(2)] == ["hello", "world"]


def test_owner_mismatch_is_not_owned():
    turn = live_turn.start("s1", owner="alice")
    assert live_turn.owned(turn, "alice")
    assert not live_turn.owned(turn, "bob")


def test_cancel_of_missing_run_is_completed():
    svc = MagicMock()
    result = asyncio.run(cancel_and_wait(svc, session_id="nope", owner="default"))
    assert result["status"] == "completed"
    assert result["cancelled"] is False


def test_cancel_of_already_done_run_is_idempotent():
    turn = live_turn.start("s1", owner="default")
    turn.finish("completed")
    svc = MagicMock()
    svc.has_cached_session.return_value = False
    svc.is_session_active.return_value = False
    result = asyncio.run(cancel_and_wait(svc, run_id=turn.run_id, owner="default"))
    assert result["status"] == "completed"
    assert result["cancelled"] is False


def test_cancel_wrong_owner_raises():
    live_turn.start("s1", owner="alice")
    svc = MagicMock()
    with pytest.raises(PermissionError):
        asyncio.run(cancel_and_wait(svc, session_id="s1", owner="bob"))


def test_cancel_and_wait_releases_lock_and_finishes_task():
    turn = live_turn.start("s1", owner="default")
    started = asyncio.Event()
    released = asyncio.Event()

    async def runner():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            released.set()
            turn.publish({"event": "aborted", "data": {}})
            turn.finish("cancelled")
            raise

    turn.task = None

    async def scenario():
        turn.task = asyncio.create_task(runner())
        await started.wait()
        lock = asyncio.Lock()
        await lock.acquire()

        svc = MagicMock()
        svc.has_cached_session.return_value = False
        svc.is_session_active.side_effect = lambda sid: lock.locked()
        svc.cancel_session.side_effect = lambda sid: turn.task.cancel() or True

        async def unlock_when_cancelled():
            await released.wait()
            lock.release()

        asyncio.create_task(unlock_when_cancelled())
        result = await cancel_and_wait(svc, run_id=turn.run_id, owner="default", timeout=2)
        assert result["status"] == "cancelled"
        assert turn.done
        assert not lock.locked()

    asyncio.run(scenario())
