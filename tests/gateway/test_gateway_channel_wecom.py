"""Unit tests for WeComChannel (wecom_aibot_sdk is mocked)."""
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


def _install_fake_wecom(monkeypatch, *, populate_globals: bool = True):
    """Install a stub ``wecom_aibot_sdk`` module.

    When ``populate_globals=True`` (the default for tests that exercise
    code paths after ``connect()`` would have run), also populate the
    module-level SDK refs so ``send()`` can use ``generate_req_id``.
    Tests targeting ``_ensure_wecom_sdk`` itself should pass
    ``populate_globals=False``.
    """
    fake = types.ModuleType("wecom_aibot_sdk")
    fake.WSClient = MagicMock()
    fake.generate_req_id = lambda prefix="": f"{prefix}-rid"
    monkeypatch.setitem(sys.modules, "wecom_aibot_sdk", fake)

    from agentica.gateway.channels import wecom as wm
    if populate_globals:
        monkeypatch.setattr(wm, "WSClient", fake.WSClient)
        monkeypatch.setattr(wm, "generate_req_id", fake.generate_req_id)
    else:
        monkeypatch.setattr(wm, "WSClient", None)
        monkeypatch.setattr(wm, "generate_req_id", None)


def test_channel_type_enum_has_wecom():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.WECOM.value == "wecom"


def test_lazy_import_error_points_to_extras(monkeypatch):
    from agentica.gateway.channels import wecom as wm
    monkeypatch.setattr(wm, "WSClient", None)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wecom_aibot_sdk":
            raise ImportError("No module named 'wecom_aibot_sdk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"agentica\[wecom\]"):
        wm._ensure_wecom_sdk()


@pytest.mark.asyncio
async def test_on_text_emits_unified_message_and_caches_frame(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel
    from agentica.gateway.channels.base import ChannelType

    ch = WeComChannel(bot_id="b", secret="s")
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    frame = MagicMock()
    frame.body = {
        "msgid": "m-1",
        "chatid": "c-1",
        "from": {"userid": "u-1", "name": "Alice"},
        "text": {"content": "hello"},
    }

    await ch._on_text(frame)

    assert len(received) == 1
    m = received[0]
    assert m.channel == ChannelType.WECOM
    assert m.channel_id == "c-1"
    assert m.sender_id == "u-1"
    assert m.sender_name == "Alice"
    assert m.content == "hello"
    assert ch._chat_frames["c-1"] is frame


@pytest.mark.asyncio
async def test_on_text_dedupes(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel

    ch = WeComChannel(bot_id="b", secret="s")
    received = []
    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    frame = MagicMock()
    frame.body = {"msgid": "dup", "chatid": "c", "from": {"userid": "u"}, "text": {"content": "hi"}}
    await ch._on_text(frame)
    await ch._on_text(frame)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_on_text_allowlist_blocks(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel

    ch = WeComChannel(bot_id="b", secret="s", allowed_users=["someone_else"])
    received = []
    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    frame = MagicMock()
    frame.body = {"msgid": "m", "chatid": "c", "from": {"userid": "blocked"}, "text": {"content": "hi"}}
    await ch._on_text(frame)
    assert received == []


@pytest.mark.asyncio
async def test_send_uses_cached_frame(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel

    ch = WeComChannel(bot_id="b", secret="s")
    ch._client = MagicMock()
    ch._client.reply_stream = AsyncMock()
    ch._chat_frames["c-1"] = "fake-frame"

    ok = await ch.send("c-1", "hello")
    assert ok is True
    ch._client.reply_stream.assert_awaited_once()
    args, kwargs = ch._client.reply_stream.await_args
    assert args[0] == "fake-frame"
    assert args[2] == "hello"
    assert kwargs.get("finish") is True


@pytest.mark.asyncio
async def test_send_no_cached_frame_returns_false(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel

    ch = WeComChannel(bot_id="b", secret="s")
    ch._client = MagicMock()
    ch._client.reply_stream = AsyncMock()

    ok = await ch.send("unknown-chat", "hello")
    assert ok is False
    ch._client.reply_stream.assert_not_called()


@pytest.mark.asyncio
async def test_send_splits_long_text(monkeypatch):
    _install_fake_wecom(monkeypatch)
    from agentica.gateway.channels.wecom import WeComChannel

    ch = WeComChannel(bot_id="b", secret="s")
    ch._client = MagicMock()
    ch._client.reply_stream = AsyncMock()
    ch._chat_frames["c-1"] = "fake-frame"

    text = "y" * (WeComChannel.SPLIT_LIMIT * 2 + 1)
    await ch.send("c-1", text)
    assert ch._client.reply_stream.await_count == 3
