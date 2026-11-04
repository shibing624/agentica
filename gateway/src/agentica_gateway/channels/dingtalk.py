# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
DingTalk channel implementation using dingtalk-stream.

Connects to DingTalk's robot platform via the ``dingtalk-stream`` SDK
(WebSocket long connection) for receiving messages, and uses HTTP API
calls (with cached ``accessToken``) for sending replies. Group chats are
addressed by ``"group:<openConversationId>"`` channel IDs; one-to-one
chats are addressed by ``sender_staff_id``.
"""
import asyncio
import json
import time
from typing import Optional, List

import requests

from agentica.utils.log import logger

from .base import Channel, ChannelType, Message
from ..config import settings

# dingtalk-stream globals (lazy-imported to avoid hard dependency)
AckMessage = None
CallbackHandler = None
Credential = None
DingTalkStreamClient = None
ChatbotMessage = None


def _ensure_dingtalk_sdk():
    """Ensure the dingtalk-stream SDK has been imported (lazy).

    Raises:
        ImportError: If ``dingtalk-stream`` is not installed.
    """
    global AckMessage, CallbackHandler, Credential, DingTalkStreamClient, ChatbotMessage
    if AckMessage is None:
        try:
            from dingtalk_stream import (
                AckMessage as _Ack,
                CallbackHandler as _CB,
                Credential as _Cred,
                DingTalkStreamClient as _Client,
            )
            from dingtalk_stream.chatbot import ChatbotMessage as _CMsg
            AckMessage = _Ack
            CallbackHandler = _CB
            Credential = _Cred
            DingTalkStreamClient = _Client
            ChatbotMessage = _CMsg
        except ImportError:
            raise ImportError(
                "DingTalk SDK not installed. Run: pip install 'agentica[dingtalk]'"
            )


class DingTalkChannel(Channel):
    """DingTalk messaging channel.

    Inbound: ``DingTalkStreamClient`` + ``CallbackHandler`` on
    ``ChatbotMessage.TOPIC``. Outbound: HTTP POST to the appropriate batch
    send endpoint with a cached ``accessToken``.
    """

    SPLIT_LIMIT = 1800
    OAUTH_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    GROUP_SEND_URL = "https://api.dingtalk.com/v1.0/robot/groupMessages/send"
    O2O_SEND_URL = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.dingtalk_allowed_users or [])
        self.client_id = client_id or settings.dingtalk_client_id
        self.client_secret = client_secret or settings.dingtalk_client_secret
        self._client = None
        self._client_task: Optional[asyncio.Task] = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.DINGTALK

    async def connect(self) -> bool:
        """Start the DingTalk stream client in a background task."""
        if not self.client_id or not self.client_secret:
            logger.warning("DingTalk: Missing client_id/client_secret, skipped")
            return False

        try:
            _ensure_dingtalk_sdk()

            self._client = DingTalkStreamClient(Credential(self.client_id, self.client_secret))
            self._client.register_callback_handler(ChatbotMessage.TOPIC, _make_handler(self))

            self._connected = True
            self._client_task = asyncio.create_task(self._start_client())
            logger.info("DingTalk: Connected")
            return True

        except ImportError as e:
            logger.error(f"DingTalk: SDK not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"DingTalk: Connect failed: {e}")
            return False

    async def _start_client(self):
        """Run the stream client with reconnect-on-error loop."""
        while self._connected:
            try:
                await self._client.start()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"DingTalk: stream error: {e}")
            if not self._connected:
                break
            logger.info("DingTalk: reconnect in 5s...")
            await asyncio.sleep(5)

    async def disconnect(self):
        """Cancel the background stream client task."""
        self._connected = False
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass
            self._client_task = None
        logger.info("DingTalk: Disconnected")

    async def _get_access_token(self) -> Optional[str]:
        """Fetch (and cache) the DingTalk OAuth2 ``accessToken``."""
        async with self._token_lock:
            if self._access_token and time.time() < self._token_expiry:
                return self._access_token

            def _fetch():
                resp = requests.post(
                    self.OAUTH_URL,
                    json={"appKey": self.client_id, "appSecret": self.client_secret},
                    timeout=20,
                )
                resp.raise_for_status()
                return resp.json()

            try:
                data = await asyncio.to_thread(_fetch)
            except Exception as e:
                logger.error(f"DingTalk: token error: {e}")
                return None
            self._access_token = data.get("accessToken")
            self._token_expiry = time.time() + int(data.get("expireIn", 7200)) - 60
            return self._access_token

    async def _send_one(self, channel_id: str, msg_key: str, msg_param: dict) -> bool:
        """POST one message to the appropriate send endpoint."""
        token = await self._get_access_token()
        if not token:
            return False

        if channel_id.startswith("group:"):
            url = self.GROUP_SEND_URL
            payload = {
                "robotCode": self.client_id,
                "openConversationId": channel_id[len("group:"):],
                "msgKey": msg_key,
                "msgParam": json.dumps(msg_param, ensure_ascii=False),
            }
        else:
            url = self.O2O_SEND_URL
            payload = {
                "robotCode": self.client_id,
                "userIds": [channel_id],
                "msgKey": msg_key,
                "msgParam": json.dumps(msg_param, ensure_ascii=False),
            }
        headers = {"x-acs-dingtalk-access-token": token}

        def _post():
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            result = resp.json() if "json" in resp.headers.get("content-type", "") else {}
            errcode = result.get("errcode")
            if errcode not in (None, 0):
                raise RuntimeError(f"API errcode={errcode}: {resp.text[:300]}")
            return True

        try:
            return await asyncio.to_thread(_post)
        except Exception as e:
            logger.error(f"DingTalk: send error: {e} channel_id={channel_id}")
            return False

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:  # noqa: ARG002
        """Send a Markdown message to the given DingTalk chat.

        Long messages are split into chunks of :attr:`SPLIT_LIMIT`
        characters; each chunk is delivered as a ``sampleMarkdown`` message.
        """
        ok = True
        for chunk in self.split_text(content, self.SPLIT_LIMIT):
            sent = await self._send_one(
                channel_id,
                "sampleMarkdown",
                {"text": chunk, "title": "Agent Reply"},
            )
            ok = ok and sent
        return ok

    async def _on_message(
        self,
        content: str,
        sender_id: str,
        sender_name: str,
        conversation_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> None:
        """Convert callback-extracted fields to the unified :class:`Message`."""
        try:
            if not content:
                return
            if not self.check_allowlist(sender_id):
                logger.debug(f"DingTalk: User {sender_id} not in allowlist")
                return

            is_group = conversation_type == "2" and conversation_id
            channel_id = f"group:{conversation_id}" if is_group else sender_id

            message = Message(
                channel=ChannelType.DINGTALK,
                channel_id=channel_id,
                sender_id=sender_id,
                sender_name=sender_name or sender_id,
                content=content,
                message_id=message_id or "",
                metadata={
                    "is_group": bool(is_group),
                    "conversation_id": conversation_id,
                },
            )
            await self._emit_message(message)
        except Exception as e:
            logger.error(f"DingTalk: Message error: {e}")


def _make_handler(channel: DingTalkChannel):
    """Build a ``CallbackHandler`` subclass instance bound to ``channel``.

    Defined as a factory because ``CallbackHandler`` is only available
    after :func:`_ensure_dingtalk_sdk` runs, which happens inside
    :meth:`DingTalkChannel.connect`.
    """

    class _Handler(CallbackHandler):
        async def process(self, message):
            try:
                data = message.data or {}
                chatbot_msg = ChatbotMessage.from_dict(data)
                text_obj = getattr(chatbot_msg, "text", None)
                text = (getattr(text_obj, "content", "") or "").strip()

                if not text:
                    extensions = getattr(chatbot_msg, "extensions", None) or {}
                    if isinstance(extensions, dict):
                        recognition = ((extensions.get("content") or {}).get("recognition") or "").strip()
                        if recognition:
                            text = recognition
                if not text:
                    text = str((data.get("text", {}) or {}).get("content", "") or "").strip()

                sender_id = str(
                    getattr(chatbot_msg, "sender_staff_id", None)
                    or getattr(chatbot_msg, "sender_id", None)
                    or "unknown"
                )
                sender_name = getattr(chatbot_msg, "sender_nick", None) or "Unknown"
                conv_type = data.get("conversationType")
                conv_id = data.get("conversationId") or data.get("openConversationId")
                msg_id = data.get("msgId") or getattr(chatbot_msg, "message_id", None)

                await channel._on_message(
                    text, sender_id, sender_name,
                    conversation_type=conv_type,
                    conversation_id=conv_id,
                    message_id=str(msg_id) if msg_id else None,
                )
            except Exception as e:
                logger.error(f"DingTalk: callback error: {e}")
            return AckMessage.STATUS_OK, "OK"

    return _Handler()
