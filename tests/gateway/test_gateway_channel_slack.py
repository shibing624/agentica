"""Unit tests for SlackChannel (slack_sdk is mocked)."""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")


class _FakeSocketModeRequest:
    def __init__(self, type=None, envelope_id=None, payload=None):
        self.type = type
        self.envelope_id = envelope_id
        self.payload = payload


class _FakeSocketModeResponse:
    def __init__(self, envelope_id=None, payload=None):
        self.envelope_id = envelope_id
        self.payload = payload


class _FakeSocketModeClient:
    def __init__(self, *args, **kwargs):
        self.socket_mode_request_listeners = []
        self.closed = False

    def send_socket_mode_response(self, *args, **kwargs):
        pass

    def connect(self):
        pass

    def close(self):
        self.closed = True


def _install_fake_slack(monkeypatch):
    """Install stub ``slack_sdk`` modules so connect() runs without the SDK."""
    fake_sdk = types.ModuleType("slack_sdk")
    fake_sdk.WebClient = MagicMock()

    fake_sm = types.ModuleType("slack_sdk.socket_mode")
    fake_sm.SocketModeClient = _FakeSocketModeClient

    fake_req = types.ModuleType("slack_sdk.socket_mode.request")
    fake_req.SocketModeRequest = _FakeSocketModeRequest
    fake_sm.request = fake_req

    fake_resp = types.ModuleType("slack_sdk.socket_mode.response")
    fake_resp.SocketModeResponse = _FakeSocketModeResponse
    fake_sm.response = fake_resp

    monkeypatch.setitem(sys.modules, "slack_sdk", fake_sdk)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode", fake_sm)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode.request", fake_req)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode.response", fake_resp)


def test_channel_type_enum_has_slack():
    from agentica.gateway.channels.base import ChannelType
    assert ChannelType.SLACK.value == "slack"


@pytest.mark.asyncio
async def test_connect_returns_false_without_tokens():
    from agentica.gateway.channels.slack import SlackChannel
    ch = SlackChannel(bot_token=None, app_token=None)
    assert await ch.connect() is False


@pytest.mark.asyncio
async def test_connect_registers_listener_and_emits_message(monkeypatch):
    _install_fake_slack(monkeypatch)
    from agentica.gateway.channels.slack import SlackChannel
    from agentica.gateway.channels.base import ChannelType

    ch = SlackChannel(bot_token="xoxb-x", app_token="xapp-x")
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    assert await ch.connect() is True
    assert ch.is_connected is True
    assert len(ch._client.socket_mode_request_listeners) == 1

    listener = ch._client.socket_mode_request_listeners[0]
    req = _FakeSocketModeRequest(
        type="events_api",
        envelope_id="env-1",
        payload={"event": {
            "type": "message",
            "channel": "C123",
            "user": "U1",
            "text": "hello slack",
            "client_msg_id": "m1",
            "ts": "1.0",
        }},
    )
    listener(ch._client, req)
    await _yield()

    assert len(received) == 1
    msg = received[0]
    assert msg.channel == ChannelType.SLACK
    assert msg.channel_id == "C123"
    assert msg.sender_id == "U1"
    assert msg.content == "hello slack"
    assert msg.message_id == "m1"


@pytest.mark.asyncio
async def test_listener_ignores_bot_messages(monkeypatch):
    _install_fake_slack(monkeypatch)
    from agentica.gateway.channels.slack import SlackChannel

    ch = SlackChannel(bot_token="xoxb-x", app_token="xapp-x")
    received = []

    async def handler(msg):
        received.append(msg)
    ch.set_handler(handler)

    await ch.connect()

    listener = ch._client.socket_mode_request_listeners[0]
    # bot_id present -> ignored
    req = _FakeSocketModeRequest(
        type="events_api",
        envelope_id="e",
        payload={"event": {"type": "message", "channel": "C1", "user": "U1",
                            "text": "hi", "bot_id": "B1"}},
    )
    listener(ch._client, req)
    await _yield()
    assert received == []


@pytest.mark.asyncio
async def test_send_splits_and_calls_chat_postmessage(monkeypatch):
    _install_fake_slack(monkeypatch)
    from agentica.gateway.channels.slack import SlackChannel

    ch = SlackChannel(bot_token="xoxb-x", app_token="xapp-x")
    await ch.connect()

    ok = await ch.send("C123", "short msg")
    assert ok is True
    # WebClient(token=...).chat_postMessage called
    assert ch._web.chat_postMessage.called


async def _yield():
    import asyncio
    await asyncio.sleep(0.05)
