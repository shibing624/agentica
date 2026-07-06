"""Unit tests for WeChatChannel (inline WxBotClient is mocked)."""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


def test_channel_type_enum_has_wechat():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.WECHAT.value == "wechat"


def test_wxbotclient_extract_text_concatenates():
    from agentica.gateway.channels.wechat import WxBotClient
    msg = {
        "item_list": [
            {"type": 1, "text_item": {"text": "hello"}},
            {"type": 1, "text_item": {"text": "world"}},
            {"type": 2, "image_item": {}},
        ]
    }
    assert WxBotClient.extract_text(msg) == "hello\nworld"


def test_wxbotclient_is_user_msg():
    from agentica.gateway.channels.wechat import WxBotClient
    assert WxBotClient.is_user_msg({"message_type": 1}) is True
    assert WxBotClient.is_user_msg({"message_type": 2}) is False


@pytest.mark.asyncio
async def test_on_native_message_emits_unified_message(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel
    from agentica.gateway.channels.base import ChannelType

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()

    received = []

    async def handler(msg):
        received.append(msg)

    ch.set_handler(handler)

    bot = MagicMock()
    msg_payload = {
        "message_id": "wx-1",
        "message_type": 1,
        "from_user_id": "wx-user-1",
        "context_token": "ctx-token-1",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }
    bot.extract_text.return_value = "hi"

    ch._on_native_message(bot, msg_payload)

    # call_soon_threadsafe schedules on the loop; yield to let it run
    await asyncio.sleep(0.05)

    assert len(received) == 1
    m = received[0]
    assert m.channel == ChannelType.WECHAT
    assert m.channel_id == "wx-user-1"
    assert m.content == "hi"
    assert m.metadata["context_token"] == "ctx-token-1"
    assert ch._user_ctx["wx-user-1"] == "ctx-token-1"


@pytest.mark.asyncio
async def test_on_native_message_dedupes(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = "hi"
    msg_payload = {
        "message_id": "dup",
        "message_type": 1,
        "from_user_id": "u",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }

    ch._on_native_message(bot, msg_payload)
    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_on_native_message_allowlist_blocks(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(
        token_file=str(tmp_path / "tok.json"),
        allowed_users=["someone-else"],
    )
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = "hi"
    msg_payload = {
        "message_id": "m",
        "message_type": 1,
        "from_user_id": "blocked",
        "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
    }

    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert received == []


@pytest.mark.asyncio
async def test_send_uses_cached_context_and_splits(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._bot = MagicMock()
    ch._user_ctx["u1"] = "ctx-A"

    text = "z" * (WeChatChannel.SPLIT_LIMIT * 2 + 1)
    ok = await ch.send("u1", text)

    assert ok is True
    assert ch._bot.send_text.call_count == 3
    # All three calls should pass the cached context_token
    for call in ch._bot.send_text.call_args_list:
        args, _ = call
        assert args[0] == "u1"
        assert args[2] == "ctx-A"


@pytest.mark.asyncio
async def test_send_when_not_connected_returns_false(monkeypatch, tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    assert await ch.send("u1", "hi") is False
