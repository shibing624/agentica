# -*- coding: utf-8 -*-
"""Web chat run registry: reconnect, seq replay, cancel waits for the lock."""
import asyncio
import os
import time
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")

from agentica.gateway.services import live_turn
from agentica.gateway.services.live_turn import cancel_and_wait


@pytest.fixture(autouse=True)
def _reset_hub():
    live_turn.reset()
    yield
    live_turn.reset()


def test_subscribe_replays_then_live():
    turn = live_turn.start("s1", owner="default")
    turn.publish({"event": "tool_call", "data": {"name": "a"}})
    turn.publish({"event": "tool_call", "data": {"name": "b"}})
    replay = turn.replay()
    assert [e["seq"] for e in replay] == [1, 2]
    assert [e["data"]["name"] for e in replay] == ["a", "b"]
    q = turn.subscribe()
    turn.publish({"event": "tool_call", "data": {"name": "c"}})
    assert q.get_nowait()["seq"] == 3


def test_subscribe_after_skips_old_seq():
    turn = live_turn.start("s1", owner="default")
    turn.publish({"event": "tool_call", "data": {"name": "a"}})
    turn.publish({"event": "tool_call", "data": {"name": "b"}})
    turn.publish({"event": "tool_call", "data": {"name": "c"}})
    replay = turn.replay(after=2)
    assert len(replay) == 1
    assert replay[0]["seq"] == 3
    assert replay[0]["data"]["name"] == "c"


def test_content_deltas_coalesce_in_buffer():
    turn = live_turn.start("s1", owner="default")
    turn.publish({"event": "content", "data": "a"})
    turn.publish({"event": "content", "data": "b"})
    replay = turn.replay()
    assert len(replay) == 1
    assert replay[0]["seq"] == 2
    assert replay[0]["data"] == "ab"


def test_unsubscribe_does_not_cancel_and_late_subscriber_replays():
    turn = live_turn.start("s1", owner="default")
    q1 = turn.subscribe()
    turn.publish({"event": "tool_call", "data": {"name": "hello"}})
    turn.unsubscribe(q1)
    turn.publish({"event": "tool_call", "data": {"name": "world"}})
    assert live_turn.active("s1", owner="default") is turn
    replay = turn.replay()
    assert [e["data"]["name"] for e in replay] == ["hello", "world"]


def test_owner_mismatch_is_not_owned():
    turn = live_turn.start("s1", owner="alice")
    assert live_turn.owned(turn, "alice")
    assert not live_turn.owned(turn, "bob")


def test_same_session_id_is_partitioned_by_owner():
    alice = live_turn.start("s1", owner="alice")
    bob = live_turn.start("s1", owner="bob")
    assert alice is not bob
    assert live_turn.active("s1", owner="alice") is alice
    assert live_turn.active("s1", owner="bob") is bob
    live_turn.drop("s1", owner="bob")
    assert live_turn.active("s1", owner="alice") is alice
    assert live_turn.get_run(alice.run_id) is alice
    assert live_turn.get_run(bob.run_id) is None


def test_start_drops_previous_finished_run():
    first = live_turn.start("s1", owner="default")
    first.finish("completed")
    rid = first.run_id
    second = live_turn.start("s1", owner="default")
    assert live_turn.get_run(rid) is None
    assert live_turn.active("s1", owner="default") is second


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


def test_cancel_wrong_owner_by_run_id_raises():
    turn = live_turn.start("s1", owner="alice")
    svc = MagicMock()
    with pytest.raises(PermissionError):
        asyncio.run(cancel_and_wait(svc, run_id=turn.run_id, owner="bob"))


def test_cancel_other_owners_session_id_is_noop():
    turn = live_turn.start("s1", owner="alice")
    svc = MagicMock()
    result = asyncio.run(cancel_and_wait(svc, session_id="s1", owner="bob"))
    assert result["status"] == "completed"
    assert live_turn.active("s1", owner="alice") is turn


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
        svc.is_session_active.side_effect = lambda sid, owner=None: lock.locked()
        svc.cancel_session.side_effect = lambda sid, owner=None: turn.task.cancel() or True

        async def unlock_when_cancelled():
            await released.wait()
            lock.release()

        asyncio.create_task(unlock_when_cancelled())
        result = await cancel_and_wait(svc, run_id=turn.run_id, owner="default", timeout=2)
        assert result["status"] == "cancelled"
        assert turn.done
        assert not lock.locked()

    asyncio.run(scenario())


def test_gc_keeps_finished_run_inside_retain_window():
    turn = live_turn.start("s1", owner="default")
    rid = turn.run_id
    turn.finish("completed")
    live_turn.gc()
    assert live_turn.get_run(rid) is turn


def test_gc_drops_finished_run_after_retain_window():
    turn = live_turn.start("s1", owner="default")
    rid = turn.run_id
    turn.finish("completed")
    turn.finished_at = time.monotonic() - live_turn.RETAIN_AFTER_DONE_S - 1
    live_turn.gc()
    assert live_turn.get_run(rid) is None
    assert live_turn.get("s1", owner="default") is None


def test_drop_session_removes_retained_runs():
    turn = live_turn.start("s1", owner="default")
    turn.finish("completed")
    live_turn.drop("s1", owner="default")
    assert live_turn.get_run(turn.run_id) is None


def test_buffer_caps_event_count(monkeypatch):
    monkeypatch.setattr(live_turn, "MAX_BUFFER_EVENTS", 3)
    turn = live_turn.start("s1", owner="default")
    for i in range(6):
        turn.publish({"event": "tool_call", "data": {"i": i}})
    assert len(turn.events) == 3
    assert [e["data"]["i"] for e in turn.events] == [3, 4, 5]


def test_slow_subscriber_is_dropped(monkeypatch):
    monkeypatch.setattr(live_turn, "SUB_QUEUE_MAX", 2)
    turn = live_turn.start("s1", owner="default")
    q = turn.subscribe()
    for i in range(5):
        turn.publish({"event": "tool_call", "data": {"i": i}})
    assert q._dropped.is_set()
    assert q not in turn._subs


def test_sse_keepalive_when_queue_is_idle(monkeypatch):
    from agentica.gateway.routes import chat as chat_mod

    monkeypatch.setattr(chat_mod, "_SSE_KEEPALIVE_S", 0.05)
    turn = live_turn.start("k", owner="default")

    async def take_two():
        gen = chat_mod._sse_from_turn(turn)
        first = await gen.__anext__()
        second = await gen.__anext__()
        await gen.aclose()
        return first, second

    first, second = asyncio.run(take_two())
    assert first.startswith(": keepalive")
    assert second.startswith(": keepalive")
