# -*- coding: utf-8 -*-
"""Regression tests for per-session inbound channel message queuing.

IM users (WeChat, etc.) often send several messages in quick succession
before the agent has replied. Without serialization the second message hits
the session run-lock ("already has an active run") and is dropped. These
tests verify that rapid-fire messages for the same session are queued and
processed one at a time, in order, with none lost.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from agentica.gateway.channels.base import ChannelType, Message
from agentica.gateway import main as gw_main
from agentica.gateway.services.agent_service import ChatResult


def _make_message(content: str, sender: str = "userA") -> Message:
    return Message(
        channel=ChannelType.WECHAT,
        channel_id=f"chat:{sender}",
        sender_id=sender,
        sender_name=sender,
        content=content,
        message_id=f"m-{content}",
    )


@pytest.fixture()
def patched_deps():
    """Patch gateway deps + ws so no real agent/channel is touched."""
    router = MagicMock()
    router.route = MagicMock(return_value="main")
    # Route session id purely from sender so same user -> same session.
    router.get_session_id = MagicMock(side_effect=lambda msg, agent_id: f"sess:{msg.sender_id}")

    channel_manager = MagicMock()
    channel_manager.send = AsyncMock()

    agent_service = MagicMock()

    with patch.object(gw_main.deps, "agent_service", agent_service), \
         patch.object(gw_main.deps, "channel_manager", channel_manager), \
         patch.object(gw_main.deps, "message_router", router), \
         patch.object(gw_main.ws.ws_manager, "broadcast", AsyncMock()):
        # Reset module-level queue registries between tests.
        gw_main._channel_queues.clear()
        gw_main._channel_workers.clear()
        yield agent_service, channel_manager


@pytest.mark.asyncio
async def test_rapid_messages_are_queued_and_processed_in_order(patched_deps):
    agent_service, channel_manager = patched_deps

    processed_order: list[str] = []
    release = asyncio.Event()
    first_started = asyncio.Event()

    async def slow_chat(message, session_id, user_id):
        # Block the first message so the next two arrive while it "runs".
        if not first_started.is_set():
            first_started.set()
            await release.wait()
        processed_order.append(message)
        return ChatResult(content=f"reply:{message}", session_id=session_id)

    agent_service.chat = AsyncMock(side_effect=slow_chat)

    # Fire three messages for the same session in quick succession.
    await gw_main._handle_channel_message(_make_message("m1"))
    await first_started.wait()  # ensure worker is busy on m1
    await gw_main._handle_channel_message(_make_message("m2"))
    await gw_main._handle_channel_message(_make_message("m3"))

    # All three should be queued (m1 in-flight, m2/m3 pending) — none dropped.
    release.set()

    # Wait for the worker to drain.
    for _ in range(200):
        if len(processed_order) == 3:
            break
        await asyncio.sleep(0.01)

    assert processed_order == ["m1", "m2", "m3"]
    assert agent_service.chat.await_count == 3
    # Worker self-cleans once the queue drains.
    await asyncio.sleep(0.02)
    assert "sess:userA" not in gw_main._channel_workers


@pytest.mark.asyncio
async def test_queue_full_drops_excess(patched_deps):
    agent_service, _ = patched_deps

    release = asyncio.Event()

    async def blocking_chat(message, session_id, user_id):
        await release.wait()
        return ChatResult(content="ok", session_id=session_id)

    agent_service.chat = AsyncMock(side_effect=blocking_chat)

    # First message starts the worker (in-flight). Then fill the queue to cap.
    await gw_main._handle_channel_message(_make_message("start"))
    for i in range(gw_main._MAX_CHANNEL_QUEUE + 5):
        await gw_main._handle_channel_message(_make_message(f"q{i}"))

    queue = gw_main._channel_queues["sess:userA"]
    # Queue holds at most the cap (the in-flight message already left the queue).
    assert queue.qsize() <= gw_main._MAX_CHANNEL_QUEUE

    release.set()
    for _ in range(200):
        if not gw_main._channel_workers.get("sess:userA"):
            break
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_distinct_sessions_run_concurrently(patched_deps):
    agent_service, _ = patched_deps

    both_running = asyncio.Event()
    running = set()

    async def chat(message, session_id, user_id):
        running.add(session_id)
        if len(running) == 2:
            both_running.set()
        await both_running.wait()
        return ChatResult(content="ok", session_id=session_id)

    agent_service.chat = AsyncMock(side_effect=chat)

    await gw_main._handle_channel_message(_make_message("hi", sender="userA"))
    await gw_main._handle_channel_message(_make_message("hi", sender="userB"))

    # Different sessions are not serialized against each other.
    await asyncio.wait_for(both_running.wait(), timeout=2.0)
    assert running == {"sess:userA", "sess:userB"}
