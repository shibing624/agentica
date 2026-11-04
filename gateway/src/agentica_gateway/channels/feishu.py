# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Feishu (Lark) channel implementation using WebSocket long-polling.

Connects to the Feishu Open Platform via the ``lark_oapi`` SDK's WebSocket
client. Incoming messages are received on a dedicated background thread and
forwarded to the main asyncio event loop via ``call_soon_threadsafe``.
"""
import json
import asyncio
import threading
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from agentica.utils.log import logger

from .base import Channel, ChannelType, Message
from ..config import settings

# Feishu SDK globals (lazy-imported to avoid hard dependency)
lark = None
CreateMessageRequest = None
CreateMessageRequestBody = None
_lark_executor = None


def _get_lark_executor():
    """Return the dedicated thread pool for Feishu SDK initialization.

    A single-threaded executor is used to isolate the Feishu SDK's internal
    event loop from the main application's event loop.
    """
    global _lark_executor
    if _lark_executor is None:
        _lark_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lark-ws")
    return _lark_executor


def _init_lark_in_thread():
    """Import and initialize ``lark_oapi`` in a dedicated thread.

    The Feishu SDK creates its own event loop internally; running the import
    in a separate thread prevents conflicts with the main asyncio loop.
    """
    global lark, CreateMessageRequest, CreateMessageRequestBody

    # Create a thread-local event loop for the Feishu SDK
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    import lark_oapi
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest as LarkCreateMessageRequest,
        CreateMessageRequestBody as LarkCreateMessageRequestBody,
    )
    lark = lark_oapi
    CreateMessageRequest = LarkCreateMessageRequest
    CreateMessageRequestBody = LarkCreateMessageRequestBody


def _ensure_lark_sdk():
    """Ensure the Feishu SDK has been imported (lazy, thread-safe)."""
    global lark
    if lark is None:
        # Initialize in a dedicated thread to avoid event loop conflicts
        executor = _get_lark_executor()
        future = executor.submit(_init_lark_in_thread)
        future.result()  # Block until import completes


class FeishuChannel(Channel):
    """Feishu (Lark) messaging channel.

    Uses the SDK's WebSocket client for receiving messages in real-time.
    Outgoing messages are sent via the IM v1 ``create_message`` API.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
        allowed_groups: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.feishu_allowed_users or [])
        self.app_id = app_id or settings.feishu_app_id
        self.app_secret = app_secret or settings.feishu_app_secret
        self.allowed_groups = allowed_groups or settings.feishu_allowed_groups or []
        self._client = None
        self._ws_client = None
        self._ws_thread = None
        self._main_loop = None

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.FEISHU

    async def connect(self) -> bool:
        """Establish the WebSocket connection to Feishu Open Platform."""
        if not self.app_id or not self.app_secret:
            logger.warning("Feishu: Missing credentials, skipped")
            return False

        try:
            _ensure_lark_sdk()

            self._client = (
                lark.Client.builder()
                .app_id(self.app_id)
                .app_secret(self.app_secret)
                .build()
            )

            # Save reference to the main event loop for cross-thread dispatch
            self._main_loop = asyncio.get_running_loop()

            # Build the event dispatcher for incoming messages
            event_handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_message)
                .register_p1_customized_event("im.message.message_read_v1", lambda _: None)
                .build()
            )

            # Create the WebSocket long-polling client
            self._ws_client = lark.ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.WARNING,
            )

            # Run the WebSocket client in a background daemon thread
            # to avoid blocking the main asyncio event loop
            self._ws_thread = threading.Thread(
                target=self._run_ws_client,
                daemon=True,
            )
            self._ws_thread.start()

            self._connected = True
            logger.info("Feishu: Connected")
            return True

        except ImportError as e:
            logger.error(f"Feishu: SDK not installed ({e}). Run: pip install \"agentica-gateway[feishu]\"")
            return False
        except Exception as e:
            logger.error(f"Feishu: Connect failed: {e}")
            return False

    def _run_ws_client(self):
        """Run the Feishu WebSocket client (called in a background thread)."""
        try:
            # The lark_oapi SDK manages its own module-level event loop internally
            self._ws_client.start()
        except Exception as e:
            logger.error(f"Feishu: WebSocket error: {e}")
            self._connected = False

    async def disconnect(self):
        """Mark the channel as disconnected."""
        self._connected = False
        logger.info("Feishu: Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:  # noqa: ARG002
        """Send a text message to a Feishu chat.

        Long messages are automatically split into chunks of 4000 characters
        to stay within the Feishu API's per-message size limit.
        """
        if not self._client:
            logger.warning("Feishu: Not connected")
            return False

        try:
            _ensure_lark_sdk()

            # Split long messages into chunks
            for chunk in self.split_text(content, 4000):
                request = (
                    CreateMessageRequest.builder()
                    .receive_id_type("chat_id")
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(channel_id)
                        .msg_type("text")
                        .content(json.dumps({"text": chunk}))
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.message.create(request)

                if not response.success():
                    logger.error(f"Feishu: Send failed: code={response.code} msg={response.msg} channel_id={channel_id}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Feishu: Send error: {e} channel_id={channel_id}")
            return False

    def _on_message(self, data) -> None:
        """Handle an incoming Feishu message (sync callback, runs in the WS thread).

        Parses the message, applies the user allowlist filter, converts to the
        unified ``Message`` format, and dispatches to the main event loop.
        """
        try:
            msg = data.event.message
            sender = data.event.sender

            # Only handle text messages
            if msg.message_type != "text":
                return

            content = json.loads(msg.content)
            text = content.get("text", "").strip()
            if not text:
                return

            user_id = sender.sender_id.user_id if sender.sender_id else "feishu_user"
            open_id = sender.sender_id.open_id if sender.sender_id else ""

            logger.debug(f"[feishu] {user_id}: {text[:100]}")

            # User allowlist check (via base class)
            if not self.check_allowlist(user_id):
                logger.debug(f"Feishu: User {user_id} not in allowlist")
                return

            # Convert to unified message format
            message = Message(
                channel=ChannelType.FEISHU,
                channel_id=msg.chat_id,
                sender_id=user_id,
                sender_name=open_id,
                content=text,
                message_id=msg.message_id,
                metadata={
                    "chat_type": msg.chat_type,
                    "open_id": open_id,
                }
            )

            # Dispatch to the main event loop using call_soon_threadsafe
            # since this callback runs in the WebSocket background thread.
            # Wrap in ensure_future with error handler to avoid silent task failures.
            if self._message_handler and self._main_loop:
                def _dispatch():
                    fut = asyncio.ensure_future(self._emit_message(message))
                    fut.add_done_callback(FeishuChannel._log_dispatch_error)

                self._main_loop.call_soon_threadsafe(_dispatch)

        except Exception as e:
            logger.error(f"Feishu: Message error: {e} message_id={getattr(getattr(data, 'event', None), 'message', None) and data.event.message.message_id or 'unknown'}")

    @staticmethod
    def _log_dispatch_error(fut: asyncio.Future) -> None:
        """Log errors from dispatched message tasks (prevents silent failures)."""
        if fut.exception():
            logger.error(f"Feishu: Dispatch error: {fut.exception()}")
