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


def test_base_info_carries_bot_agent_and_version():
    from agentica.gateway.channels.wechat import WxBotClient, _BOT_AGENT, _VERSION

    bi = WxBotClient(token="t")._base_info()
    assert bi["channel_version"] == _VERSION
    assert bi["bot_agent"] == _BOT_AGENT
    assert bi["bot_agent"].startswith("Agentica/")


def test_aes_ecb_roundtrip():
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    key = os.urandom(16)
    plain = b"wechat clawbot media payload " * 7
    cipher = bot._aes_ecb_encrypt(plain, key)
    assert len(cipher) % 16 == 0
    assert cipher != plain
    assert bot._aes_ecb_decrypt(cipher, key) == plain


def test_send_typing_uses_status_field(monkeypatch):
    from agentica.gateway.channels.wechat import WxBotClient, _TYPING, _CANCEL

    bot = WxBotClient(token="t")
    captured = {}

    def fake_post(ep, body, timeout=10):
        captured["ep"] = ep
        captured["body"] = body
        return {}

    monkeypatch.setattr(bot, "_post", fake_post)
    bot.send_typing("u1", typing_ticket="TICK", cancel=False)
    assert captured["ep"] == "ilink/bot/sendtyping"
    assert captured["body"]["ilink_user_id"] == "u1"
    assert captured["body"]["typing_ticket"] == "TICK"
    assert captured["body"]["status"] == _TYPING
    assert "typing_status" not in captured["body"]

    bot.send_typing("u1", typing_ticket="TICK", cancel=True)
    assert captured["body"]["status"] == _CANCEL


def test_send_typing_fetches_ticket_from_getconfig(monkeypatch):
    from agentica.gateway.channels.wechat import WxBotClient

    bot = WxBotClient(token="t")
    captured = {}
    eps = []

    def fake_post(ep, body, timeout=10):
        eps.append(ep)
        if ep == "ilink/bot/getconfig":
            return {"typing_ticket": "GOT"}
        captured["body"] = body
        return {}

    monkeypatch.setattr(bot, "_post", fake_post)
    bot.send_typing("u2")
    assert "ilink/bot/getconfig" in eps
    assert captured["body"]["typing_ticket"] == "GOT"


def test_extract_media_pulls_cdn_refs():
    from agentica.gateway.channels.wechat import WxBotClient

    msg = {
        "item_list": [
            {"type": 2, "image_item": {"media": {"encrypt_query_param": "e1", "aes_key": "k1"}}},
            {"type": 4, "file_item": {"media": {"encrypt_query_param": "e2", "aes_key": "k2"}}},
            {"type": 1, "text_item": {"text": "hi"}},
        ]
    }
    media = WxBotClient.extract_media(msg)
    assert len(media) == 2
    assert media[0]["encrypt_query_param"] == "e1"
    assert media[1]["aes_key"] == "k2"


@pytest.mark.asyncio
async def test_on_native_message_attaches_media(tmp_path):
    from agentica.gateway.channels.wechat import WeChatChannel

    ch = WeChatChannel(token_file=str(tmp_path / "tok.json"))
    ch._main_loop = asyncio.get_running_loop()
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    bot = MagicMock()
    bot.extract_text.return_value = ""
    msg_payload = {
        "message_id": "m-media",
        "message_type": 1,
        "from_user_id": "u",
        "item_list": [
            {"type": 2, "image_item": {"media": {"encrypt_query_param": "e", "aes_key": "k"}}},
        ],
    }
    ch._on_native_message(bot, msg_payload)
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].metadata["media"][0]["encrypt_query_param"] == "e"
