# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Slack channel implementation using Slack Socket Mode.

Socket Mode lets the gateway receive Slack events over a WebSocket managed by
Slack's `slack_sdk`, so **no public webhook / HTTP endpoint is required**. A
bot token (`SLACK_BOT_TOKEN`, `xoxb-...`) plus an app-level token
(`SLACK_APP_TOKEN`, `xapp-...`, enabled for Socket Mode) are enough. The
inbound listener runs in a background thread and dispatches messages to the
main asyncio event loop via ``run_coroutine_threadsafe``.
"""
import asyncio
import threading
from typing import Optional, List

from agentica.utils.log import logger

from .base import Channel, ChannelType, Message
from ..config import settings


def _ensure_slack_sdk():
    """Ensure the ``slack_sdk`` package is importable (lazy)."""
    try:
        import slack_sdk  # noqa: F401
    except ImportError:
        raise ImportError("slack_sdk not installed. Run: pip install 'agentica[slack]'")


class SlackChannel(Channel):
    """Slack messaging channel (Socket Mode, no public endpoint needed)."""

    SPLIT_LIMIT = 3000

    def __init__(
        self,
        bot_token: Optional[str] = None,
        app_token: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
        allowed_channels: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.slack_allowed_users or [])
        self.bot_token = bot_token or settings.slack_bot_token
        self.app_token = app_token or settings.slack_app_token
        self.allowed_channels = allowed_channels or settings.slack_allowed_channels or []
        self._client = None
        self._web = None
        self._thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SLACK

    def _allow_channel(self, ch: str) -> bool:
        """Restrict to ``allowed_channels`` when it is non-empty."""
        if not self.allowed_channels:
            return True
        return ch in self.allowed_channels

    async def connect(self) -> bool:
        """Connect via Slack Socket Mode (runs in a background thread)."""
        if not self.bot_token or not self.app_token:
            logger.warning("Slack: Missing SLACK_BOT_TOKEN / SLACK_APP_TOKEN, skipped")
            return False

        try:
            _ensure_slack_sdk()
            from slack_sdk import WebClient
            from slack_sdk.socket_mode import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse

            self._main_loop = asyncio.get_running_loop()
            self._web = WebClient(token=self.bot_token)
            self._client = SocketModeClient(app_token=self.app_token, web_client=self._web)
            channel = self

            def _on_request(client, req: SocketModeRequest):
                # Acknowledge the envelope so Slack stops retrying it.
                client.send_socket_mode_response(
                    SocketModeResponse(envelope_id=req.envelope_id)
                )
                if req.type != "events_api":
                    return
                event = req.payload.get("event", {})
                etype = event.get("type")
                if etype not in ("message", "app_mention"):
                    return
                # Skip bot echoes, channel-join notices, edits, etc.
                if event.get("subtype") or event.get("bot_id"):
                    return
                ch = event.get("channel", "")
                if not channel._allow_channel(ch):
                    return
                text = (event.get("text") or "").strip()
                if not text:
                    return
                if not channel.check_allowlist(event.get("user", "")):
                    logger.debug(f"Slack: user {event.get('user')} not in allowlist")
                    return

                message = Message(
                    channel=ChannelType.SLACK,
                    channel_id=ch,
                    sender_id=event.get("user", ""),
                    sender_name=event.get("user", ""),
                    content=text,
                    message_id=event.get("client_msg_id") or event.get("ts", ""),
                    metadata={"ts": event.get("ts"), "thread_ts": event.get("thread_ts")},
                )
                asyncio.run_coroutine_threadsafe(
                    channel._emit_message(message), channel._main_loop
                )

            self._client.socket_mode_request_listeners.append(_on_request)
            self._thread = threading.Thread(
                target=self._client.connect,
                daemon=True,
                name="slack-socket",
            )
            self._thread.start()

            self._connected = True
            logger.info("Slack: Connected (Socket Mode)")
            return True

        except ImportError as e:
            logger.error(f"Slack: SDK not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Slack: Connect failed: {e}")
            return False

    async def disconnect(self):
        """Close the Socket Mode connection."""
        self._connected = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info("Slack: Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        """Send a text message to a Slack conversation (channel_id).

        ``channel_id`` is the Slack conversation id (``D...`` for DMs,
        ``C...`` for channels). Pass ``thread_ts`` in kwargs to reply inside a
        thread.
        """
        if not self._web:
            logger.warning("Slack: Not connected")
            return False
        thread_ts = kwargs.get("thread_ts")
        try:
            for chunk in self.split_text(content, self.SPLIT_LIMIT):
                await asyncio.to_thread(
                    self._web.chat_postMessage,
                    channel=channel_id,
                    text=chunk,
                    **({"thread_ts": thread_ts} if thread_ts else {}),
                )
            return True
        except Exception as e:
            logger.error(f"Slack: Send error: {e} channel_id={channel_id}")
            return False
