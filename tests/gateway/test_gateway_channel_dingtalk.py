"""Unit tests for DingTalkChannel (dingtalk-stream is mocked)."""
import os
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


def _install_fake_dingtalk(monkeypatch):
    fake = types.ModuleType("dingtalk_stream")

    class _Ack:
        STATUS_OK = "OK"

    class _CB:
        def __init__(self): pass

    fake.AckMessage = _Ack
    fake.CallbackHandler = _CB
    fake.Credential = MagicMock()
    fake.DingTalkStreamClient = MagicMock()

    fake_chatbot = types.ModuleType("dingtalk_stream.chatbot")

    class _CMsg:
        TOPIC = "ChatbotMessage"

        @classmethod
        def from_dict(cls, d):
            self = cls()
            text = (d.get("text") or {}).get("content", "")
            self.text = MagicMock(content=text)
            self.sender_staff_id = d.get("senderStaffId", "")
            self.sender_id = d.get("senderId", "")
            self.sender_nick = d.get("senderNick", "")
            self.message_id = d.get("msgId", "")
            self.extensions = d.get("extensions", {})
            return self

    fake_chatbot.ChatbotMessage = _CMsg

    monkeypatch.setitem(sys.modules, "dingtalk_stream", fake)
    monkeypatch.setitem(sys.modules, "dingtalk_stream.chatbot", fake_chatbot)

    from agentica.gateway.channels import dingtalk as dt
    monkeypatch.setattr(dt, "AckMessage", None)
    monkeypatch.setattr(dt, "CallbackHandler", None)
    monkeypatch.setattr(dt, "Credential", None)
    monkeypatch.setattr(dt, "DingTalkStreamClient", None)
    monkeypatch.setattr(dt, "ChatbotMessage", None)


def test_channel_type_enum_has_dingtalk():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.DINGTALK.value == "dingtalk"


def test_lazy_import_error_points_to_extras(monkeypatch):
    from agentica.gateway.channels import dingtalk as dt
    monkeypatch.setattr(dt, "AckMessage", None)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dingtalk_stream" or name.startswith("dingtalk_stream."):
            raise ImportError("No module named 'dingtalk_stream'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"agentica\[dingtalk\]"):
        dt._ensure_dingtalk_sdk()


@pytest.mark.asyncio
async def test_on_message_one_to_one(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel
    from agentica.gateway.channels.base import ChannelType

    ch = DingTalkChannel(client_id="cid", client_secret="cs")
    received = []
    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    await ch._on_message(
        "hello", "u1", "Alice",
        conversation_type="1", conversation_id=None, message_id="m1",
    )

    assert len(received) == 1
    m = received[0]
    assert m.channel == ChannelType.DINGTALK
    assert m.channel_id == "u1"
    assert m.metadata["is_group"] is False


@pytest.mark.asyncio
async def test_on_message_group_uses_prefix(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel

    ch = DingTalkChannel(client_id="cid", client_secret="cs")
    received = []
    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    await ch._on_message(
        "hi", "u1", "Alice",
        conversation_type="2", conversation_id="conv-x", message_id="m2",
    )

    assert received[0].channel_id == "group:conv-x"
    assert received[0].metadata["is_group"] is True


@pytest.mark.asyncio
async def test_on_message_allowlist_blocks(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel

    ch = DingTalkChannel(client_id="c", client_secret="s", allowed_users=["other"])
    received = []
    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    await ch._on_message("hi", "blocked", "B", conversation_type="1")
    assert received == []


@pytest.mark.asyncio
async def test_get_access_token_caches(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel

    ch = DingTalkChannel(client_id="c", client_secret="s")

    fake_resp = MagicMock()
    fake_resp.json.return_value = {"accessToken": "tok-1", "expireIn": 3600}
    fake_resp.raise_for_status = MagicMock()

    with patch("agentica.gateway.channels.dingtalk.requests.post", return_value=fake_resp) as mock_post:
        t1 = await ch._get_access_token()
        t2 = await ch._get_access_token()

    assert t1 == "tok-1"
    assert t2 == "tok-1"
    assert mock_post.call_count == 1  # cached on the second call


@pytest.mark.asyncio
async def test_send_one_to_one_endpoint(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel

    ch = DingTalkChannel(client_id="cid", client_secret="cs")
    ch._access_token = "tok"
    ch._token_expiry = time.time() + 3600

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.text = "{}"
    fake_resp.json.return_value = {}

    with patch("agentica.gateway.channels.dingtalk.requests.post", return_value=fake_resp) as mock_post:
        ok = await ch.send("u1", "hello")

    assert ok is True
    assert mock_post.call_count == 1
    url = mock_post.call_args.args[0]
    assert url == DingTalkChannel.O2O_SEND_URL
    payload = mock_post.call_args.kwargs["json"]
    assert payload["userIds"] == ["u1"]
    assert payload["robotCode"] == "cid"


@pytest.mark.asyncio
async def test_send_group_endpoint(monkeypatch):
    _install_fake_dingtalk(monkeypatch)
    from agentica.gateway.channels.dingtalk import DingTalkChannel

    ch = DingTalkChannel(client_id="cid", client_secret="cs")
    ch._access_token = "tok"
    ch._token_expiry = time.time() + 3600

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.text = "{}"
    fake_resp.json.return_value = {}

    with patch("agentica.gateway.channels.dingtalk.requests.post", return_value=fake_resp) as mock_post:
        ok = await ch.send("group:conv-x", "hello")

    assert ok is True
    url = mock_post.call_args.args[0]
    assert url == DingTalkChannel.GROUP_SEND_URL
    payload = mock_post.call_args.kwargs["json"]
    assert payload["openConversationId"] == "conv-x"
