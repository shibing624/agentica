"""Unit tests for QQChannel (qq-botpy SDK is mocked).

Requires the [gateway] extras for FastAPI / lark-oapi imports.
"""
import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock LLM key per project convention
os.environ.setdefault("OPENAI_API_KEY", "test-key")

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


def _install_fake_botpy(monkeypatch):
    """Install a stub ``botpy`` package so the SDK import succeeds."""
    fake_botpy = types.ModuleType("botpy")
    fake_botpy.Intents = MagicMock()
    fake_botpy.Client = type("Client", (), {"__init__": lambda self, **kw: None})

    fake_msg = types.ModuleType("botpy.message")
    fake_msg.C2CMessage = type("C2CMessage", (), {})
    fake_msg.GroupMessage = type("GroupMessage", (), {})

    monkeypatch.setitem(sys.modules, "botpy", fake_botpy)
    monkeypatch.setitem(sys.modules, "botpy.message", fake_msg)

    # Reset module-level cached SDK refs so _ensure_qq_sdk re-imports.
    from agentica.gateway.channels import qq as qq_mod
    monkeypatch.setattr(qq_mod, "botpy", None)
    monkeypatch.setattr(qq_mod, "C2CMessage", None)
    monkeypatch.setattr(qq_mod, "GroupMessage", None)


def test_channel_type_enum_has_qq():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.QQ.value == "qq"


def test_lazy_import_error_points_to_extras(monkeypatch):
    """When qq-botpy is not installed, the error should mention the extras."""
    monkeypatch.setitem(sys.modules, "botpy", None)
    from agentica.gateway.channels import qq as qq_mod
    monkeypatch.setattr(qq_mod, "botpy", None)

    # Patch __import__ to raise ImportError for botpy
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "botpy" or name.startswith("botpy."):
            raise ImportError("No module named 'botpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"agentica\[qq\]"):
        qq_mod._ensure_qq_sdk()


@pytest.mark.asyncio
async def test_on_message_c2c_emits_unified_message(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel
    from agentica.gateway.channels.base import ChannelType, Message

    ch = QQChannel(app_id="appid", app_secret="secret", allowed_users=[])
    received = []

    async def handler(msg: Message):
        received.append(msg)

    ch.set_handler(handler)

    data = MagicMock()
    data.id = "msg-1"
    data.content = "hello"
    data.author = MagicMock(user_openid="u1", id="u1")

    await ch._on_message(data, is_group=False)

    assert len(received) == 1
    m = received[0]
    assert m.channel == ChannelType.QQ
    assert m.channel_id == "u1"
    assert m.sender_id == "u1"
    assert m.content == "hello"
    assert m.metadata["is_group"] is False
    assert ch._last_msg_id["u1"] == "msg-1"


@pytest.mark.asyncio
async def test_on_message_group_uses_group_prefix(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="appid", app_secret="secret")
    received = []

    async def handler(msg):
        received.append(msg)

    ch.set_handler(handler)

    data = MagicMock()
    data.id = "msg-2"
    data.content = "hi group"
    data.author = MagicMock(member_openid="member-1", id="member-1")
    data.group_openid = "grp-1"

    await ch._on_message(data, is_group=True)
    assert len(received) == 1
    assert received[0].channel_id == "group:grp-1"
    assert received[0].sender_id == "member-1"
    assert received[0].metadata["is_group"] is True


@pytest.mark.asyncio
async def test_on_message_dedupes_by_id(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y")
    received = []

    async def handler(msg):
        received.append(msg)

    ch.set_handler(handler)

    data = MagicMock()
    data.id = "dup"
    data.content = "first"
    data.author = MagicMock(user_openid="u1", id="u1")

    await ch._on_message(data, is_group=False)
    await ch._on_message(data, is_group=False)

    assert len(received) == 1


@pytest.mark.asyncio
async def test_on_message_allowlist_blocks(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y", allowed_users=["allowed_user"])
    received = []

    async def handler(msg):
        received.append(msg)

    ch.set_handler(handler)

    data = MagicMock()
    data.id = "msg-3"
    data.content = "hi"
    data.author = MagicMock(user_openid="blocked", id="blocked")

    await ch._on_message(data, is_group=False)
    assert received == []


@pytest.mark.asyncio
async def test_send_c2c_calls_post_c2c(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y")
    ch._client = MagicMock()
    ch._client.api.post_c2c_message = AsyncMock()
    ch._client.api.post_group_message = AsyncMock()
    ch._last_msg_id["u1"] = "msg-id-cached"

    ok = await ch.send("u1", "hello", )

    assert ok is True
    ch._client.api.post_c2c_message.assert_awaited_once()
    ch._client.api.post_group_message.assert_not_called()
    kwargs = ch._client.api.post_c2c_message.await_args.kwargs
    assert kwargs["openid"] == "u1"
    assert kwargs["content"] == "hello"
    assert kwargs["msg_id"] == "msg-id-cached"


@pytest.mark.asyncio
async def test_send_group_calls_post_group(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y")
    ch._client = MagicMock()
    ch._client.api.post_c2c_message = AsyncMock()
    ch._client.api.post_group_message = AsyncMock()

    ok = await ch.send("group:grp-1", "hi", msg_id="m-1")
    assert ok is True
    ch._client.api.post_group_message.assert_awaited_once()
    ch._client.api.post_c2c_message.assert_not_called()
    kwargs = ch._client.api.post_group_message.await_args.kwargs
    assert kwargs["group_openid"] == "grp-1"
    assert kwargs["msg_id"] == "m-1"


@pytest.mark.asyncio
async def test_send_splits_long_text(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y")
    ch._client = MagicMock()
    ch._client.api.post_c2c_message = AsyncMock()

    long_text = "x" * (QQChannel.SPLIT_LIMIT * 2 + 5)
    await ch.send("u1", long_text)
    assert ch._client.api.post_c2c_message.await_count == 3


@pytest.mark.asyncio
async def test_send_when_not_connected_returns_false(monkeypatch):
    _install_fake_botpy(monkeypatch)
    from agentica.gateway.channels.qq import QQChannel

    ch = QQChannel(app_id="x", app_secret="y")
    assert await ch.send("u1", "hi") is False
