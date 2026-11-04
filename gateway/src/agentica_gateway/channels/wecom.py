# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
WeCom (Enterprise WeChat) channel implementation using wecom_aibot_sdk.

Connects to WeCom's AI Bot platform via the SDK's ``WSClient`` (event-based
WebSocket). Replies require the original incoming ``frame``, so the channel
caches the latest frame per ``channel_id`` for use in :meth:`send`.
"""
import asyncio
from collections import deque
from typing import Optional, List

from agentica.utils.log import logger

from .base import Channel, ChannelType, Message
from ..config import settings

# wecom_aibot_sdk globals (lazy-imported to avoid hard dependency)
WSClient = None
generate_req_id = None


def _ensure_wecom_sdk():
    """Ensure the wecom_aibot_sdk has been imported (lazy).

    Raises:
        ImportError: If ``wecom_aibot_sdk`` is not installed.
    """
    global WSClient, generate_req_id
    if WSClient is None:
        try:
            from wecom_aibot_sdk import WSClient as _WSClient, generate_req_id as _gen
            WSClient = _WSClient
            generate_req_id = _gen
        except ImportError:
            raise ImportError(
                "WeCom SDK not installed. Run: pip install 'agentica[wecom]'"
            )


class WeComChannel(Channel):
    """WeCom (Enterprise WeChat) messaging channel.

    Replies are sent via ``client.reply_stream(frame, ...)``, which requires
    the original incoming ``frame``. We cache the latest frame per
    ``channel_id`` so :meth:`send` can locate it without the caller knowing
    about WeCom-specific concepts.
    """

    SPLIT_LIMIT = 1200

    def __init__(
        self,
        bot_id: Optional[str] = None,
        secret: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.wecom_allowed_users or [])
        self.bot_id = bot_id or settings.wecom_bot_id
        self.secret = secret or settings.wecom_secret
        self._client = None
        self._client_task: Optional[asyncio.Task] = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._chat_frames: dict = {}

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WECOM

    async def connect(self) -> bool:
        """Establish the WebSocket connection to WeCom AI Bot platform."""
        if not self.bot_id or not self.secret:
            logger.warning("WeCom: Missing bot_id/secret, skipped")
            return False

        try:
            _ensure_wecom_sdk()

            self._client = WSClient({
                "bot_id": self.bot_id,
                "secret": self.secret,
                "reconnect_interval": 1000,
                "max_reconnect_attempts": -1,
                "heartbeat_interval": 30000,
            })

            for event, handler in {
                "connected": self._on_connected,
                "authenticated": self._on_authenticated,
                "disconnected": self._on_disconnected,
                "error": self._on_error,
                "message.text": self._on_text,
            }.items():
                self._client.on(event, handler)

            self._client_task = asyncio.create_task(self._client.connect_async())
            self._connected = True
            logger.info("WeCom: Connected")
            return True

        except ImportError as e:
            logger.error(f"WeCom: SDK not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"WeCom: Connect failed: {e}")
            return False

    async def disconnect(self):
        """Cancel the background WSClient task."""
        self._connected = False
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass
            self._client_task = None
        logger.info("WeCom: Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:  # noqa: ARG002
        """Reply to a WeCom chat using the cached inbound ``frame``."""
        if not self._client:
            logger.warning("WeCom: Not connected")
            return False

        frame = self._chat_frames.get(channel_id)
        if frame is None:
            logger.warning(f"WeCom: no frame cached for chat: {channel_id}")
            return False

        try:
            for chunk in self.split_text(content, self.SPLIT_LIMIT):
                await self._client.reply_stream(
                    frame, generate_req_id("stream"), chunk, finish=True,
                )
            return True
        except Exception as e:
            logger.error(f"WeCom: Send error: {e} channel_id={channel_id}")
            return False

    async def _on_connected(self, frame):  # noqa: ARG002
        logger.info("WeCom: connected")

    async def _on_authenticated(self, frame):  # noqa: ARG002
        logger.info("WeCom: authenticated")

    async def _on_disconnected(self, frame):  # noqa: ARG002
        logger.info("WeCom: disconnected")

    async def _on_error(self, frame):
        logger.error(f"WeCom: error: {frame}")

    async def _on_text(self, frame) -> None:
        """Convert a native WeCom text frame to the unified :class:`Message`."""
        try:
            if hasattr(frame, "body"):
                body = frame.body
            elif isinstance(frame, dict):
                body = frame.get("body", frame)
            else:
                body = {}
            if not isinstance(body, dict):
                return

            msg_id = body.get("msgid") or f"{body.get('chatid', '')}_{body.get('sendertime', '')}"
            if msg_id in self._processed_ids:
                return
            self._processed_ids.append(msg_id)

            from_info = body.get("from", {})
            if not isinstance(from_info, dict):
                from_info = {}
            sender_id = str(from_info.get("userid", "") or "unknown")
            chat_id = str(body.get("chatid", "") or sender_id)

            text_block = body.get("text", {}) or {}
            content = str(text_block.get("content", "") or "").strip()
            if not content:
                return

            if not self.check_allowlist(sender_id):
                logger.debug(f"WeCom: User {sender_id} not in allowlist")
                return

            self._chat_frames[chat_id] = frame

            message = Message(
                channel=ChannelType.WECOM,
                channel_id=chat_id,
                sender_id=sender_id,
                sender_name=str(from_info.get("name", "") or sender_id),
                content=content,
                message_id=str(msg_id),
                metadata={
                    "chat_id": chat_id,
                    "from": from_info,
                },
            )
            await self._emit_message(message)

        except Exception as e:
            logger.error(f"WeCom: Message error: {e}")
