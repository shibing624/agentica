# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
QQ channel implementation using qq-botpy.

Connects to the QQ Open Platform via the ``qq-botpy`` SDK's intents-based
WebSocket client. Supports both C2C (one-to-one direct messages) and group
@-messages. Sending requires the original ``msg_id`` per QQ API requirements,
so the channel caches the latest inbound message ID per ``channel_id``.
"""
import asyncio
import itertools
import threading
from collections import deque
from typing import Optional, List

from agentica.utils.log import logger

from .base import Channel, ChannelType, Message
from ..config import settings

# qq-botpy SDK globals (lazy-imported to avoid hard dependency)
botpy = None
C2CMessage = None
GroupMessage = None


def _ensure_qq_sdk():
    """Ensure the qq-botpy SDK has been imported (lazy).

    Raises:
        ImportError: If ``qq-botpy`` is not installed.
    """
    global botpy, C2CMessage, GroupMessage
    if botpy is None:
        try:
            import botpy as _botpy
            from botpy.message import C2CMessage as _C2C, GroupMessage as _Group
            botpy = _botpy
            C2CMessage = _C2C
            GroupMessage = _Group
        except ImportError:
            raise ImportError(
                "QQ SDK not installed. Run: pip install 'agentica[qq]'"
            )


class QQChannel(Channel):
    """QQ messaging channel.

    Uses ``qq-botpy``'s ``Client`` to connect to the QQ Open Platform via
    intents-based WebSocket. C2C messages route on the user's ``openid``;
    group @-messages route on ``group_openid`` and use a ``"group:"`` prefix
    in ``channel_id`` so :meth:`send` can pick the right API.
    """

    SPLIT_LIMIT = 1500

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.qq_allowed_users or [])
        self.app_id = app_id or settings.qq_app_id
        self.app_secret = app_secret or settings.qq_app_secret
        self._client = None
        self._client_task: Optional[asyncio.Task] = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._msg_seq = itertools.count(2)
        self._seq_lock = threading.Lock()
        # Cache latest inbound msg_id per channel_id (required by QQ API for
        # passive replies). Updated on every incoming message.
        self._last_msg_id: dict = {}

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.QQ

    def _build_intents(self):
        """Build intents covering both C2C and group@ messages.

        Newer ``qq-botpy`` exposes ``public_messages`` / ``direct_message``
        as constructor kwargs. Fall back to attribute-based configuration
        for older releases.
        """
        try:
            return botpy.Intents(public_messages=True, direct_message=True)
        except Exception:
            intents = botpy.Intents.none() if hasattr(botpy.Intents, "none") else botpy.Intents()
            for attr in (
                "public_messages", "public_guild_messages",
                "direct_message", "direct_messages",
                "c2c_message", "c2c_messages",
                "group_at_message", "group_at_messages",
            ):
                if hasattr(intents, attr):
                    try:
                        setattr(intents, attr, True)
                    except Exception:
                        pass
            return intents

    def _next_msg_seq(self) -> int:
        """Thread-safe monotonically increasing message sequence number."""
        with self._seq_lock:
            return next(self._msg_seq)

    async def connect(self) -> bool:
        """Connect to the QQ gateway in a background task."""
        if not self.app_id or not self.app_secret:
            logger.warning("QQ: Missing app_id/app_secret, skipped")
            return False

        try:
            _ensure_qq_sdk()

            channel = self

            class _QQBot(botpy.Client):
                def __init__(self):
                    super().__init__(intents=channel._build_intents(), ext_handlers=False)

                async def on_ready(self):
                    name = getattr(getattr(self, "robot", None), "name", "QQBot")
                    logger.info(f"QQ: bot ready ({name})")

                async def on_c2c_message_create(self, message):
                    await channel._on_message(message, is_group=False)

                async def on_group_at_message_create(self, message):
                    await channel._on_message(message, is_group=True)

                async def on_direct_message_create(self, message):
                    await channel._on_message(message, is_group=False)

            self._client = _QQBot()
            self._client_task = asyncio.create_task(self._start_client())
            self._connected = True
            logger.info("QQ: Connected")
            return True

        except ImportError as e:
            logger.error(f"QQ: SDK not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"QQ: Connect failed: {e}")
            return False

    async def _start_client(self):
        """Run ``client.start()`` in a background task with reconnect."""
        while self._connected:
            try:
                await self._client.start(appid=self.app_id, secret=self.app_secret)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"QQ: client error: {e}")
            if not self._connected:
                break
            logger.info("QQ: reconnect in 5s...")
            await asyncio.sleep(5)

    async def disconnect(self):
        """Cancel the background client task."""
        self._connected = False
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass
            self._client_task = None
        logger.info("QQ: Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        """Send a text reply to a QQ chat.

        ``channel_id`` may be prefixed with ``"group:"`` to denote a group
        chat; otherwise it is treated as a C2C ``openid``. The original
        inbound ``msg_id`` is required by the QQ API; it is taken from
        ``kwargs["msg_id"]`` if provided, otherwise from the per-channel
        cache populated by :meth:`_on_message`.
        """
        if not self._client:
            logger.warning("QQ: Not connected")
            return False

        is_group = channel_id.startswith("group:")
        target_id = channel_id[len("group:"):] if is_group else channel_id
        msg_id = kwargs.get("msg_id") or self._last_msg_id.get(channel_id)

        try:
            api = self._client.api
            send_method = api.post_group_message if is_group else api.post_c2c_message
            id_kw = "group_openid" if is_group else "openid"
            for chunk in self.split_text(content, self.SPLIT_LIMIT):
                await send_method(**{
                    id_kw: target_id,
                    "msg_type": 0,
                    "content": chunk,
                    "msg_id": msg_id,
                    "msg_seq": self._next_msg_seq(),
                })
            return True
        except Exception as e:
            logger.error(f"QQ: Send error: {e} channel_id={channel_id}")
            return False

    async def _on_message(self, data, is_group: bool = False) -> None:
        """Convert a native QQ message to the unified :class:`Message`."""
        try:
            msg_id = getattr(data, "id", None)
            if msg_id and msg_id in self._processed_ids:
                return
            if msg_id:
                self._processed_ids.append(msg_id)

            content = (getattr(data, "content", "") or "").strip()
            if not content:
                return

            author = getattr(data, "author", None)
            user_attr = "member_openid" if is_group else "user_openid"
            user_id = str(
                getattr(author, user_attr, "") or getattr(author, "id", "") or "unknown"
            )

            if is_group:
                group_openid = str(getattr(data, "group_openid", "") or "")
                channel_id = f"group:{group_openid}" if group_openid else user_id
            else:
                channel_id = user_id

            if not self.check_allowlist(user_id):
                logger.debug(f"QQ: User {user_id} not in allowlist")
                return

            # Cache latest msg_id so send() can reply correctly
            if msg_id:
                self._last_msg_id[channel_id] = msg_id

            message = Message(
                channel=ChannelType.QQ,
                channel_id=channel_id,
                sender_id=user_id,
                sender_name=user_id,
                content=content,
                message_id=str(msg_id) if msg_id else "",
                metadata={
                    "is_group": is_group,
                    "msg_id": msg_id,
                },
            )
            await self._emit_message(message)

        except Exception as e:
            logger.error(f"QQ: Message error: {e}")
