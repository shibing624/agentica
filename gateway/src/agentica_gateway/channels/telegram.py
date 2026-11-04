# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Telegram channel implementation using python-telegram-bot.

Uses long-polling (``start_polling``) to receive incoming messages.
Outgoing messages are sent via the Bot API's ``send_message`` method.
"""
import asyncio
from typing import Optional, List

from .base import Channel, ChannelType, Message
from ..config import settings

# Telegram SDK globals (lazy-imported to avoid hard dependency)
telegram = None
Application = None


def _ensure_telegram_sdk():
    """Ensure the Telegram SDK has been imported (lazy).

    Raises:
        ImportError: If ``python-telegram-bot`` is not installed.
    """
    global telegram, Application
    if telegram is None:
        try:
            from telegram import Bot, Update
            from telegram.ext import Application as TelegramApplication, MessageHandler, filters
            telegram = type('telegram', (), {'Bot': Bot, 'Update': Update, 'MessageHandler': MessageHandler, 'filters': filters})()
            Application = TelegramApplication
        except ImportError:
            raise ImportError(
                "Telegram SDK not installed. Run: pip install python-telegram-bot"
            )


class TelegramChannel(Channel):
    """Telegram messaging channel.

    Uses ``python-telegram-bot``'s ``Application`` to receive messages via
    long-polling and send replies through the Bot API.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or [])
        self.bot_token = bot_token or settings.telegram_bot_token
        self._app = None
        self._bot = None
        self._polling_task: Optional[asyncio.Task] = None

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.TELEGRAM

    async def connect(self) -> bool:
        """Start the Telegram bot and begin polling for updates."""
        if not self.bot_token:
            print("[Telegram] Missing bot token, skipped")
            return False

        try:
            _ensure_telegram_sdk()

            # Build the Application with the provided bot token
            self._app = Application.builder().token(self.bot_token).build()
            self._bot = self._app.bot

            # Register a handler for all non-command text messages
            self._app.add_handler(
                telegram.MessageHandler(
                    telegram.filters.TEXT & ~telegram.filters.COMMAND,
                    self._on_message,
                )
            )

            # Start polling in a tracked background task
            self._polling_task = asyncio.create_task(self._start_polling())

            self._connected = True
            print("[Telegram] Connected")
            return True

        except ImportError as e:
            print(f"[Telegram] SDK not installed: {e}")
            return False
        except Exception as e:
            print(f"[Telegram] Connect failed: {e}")
            return False

    async def _start_polling(self):
        """Initialize the application and start update polling."""
        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
        except Exception as e:
            print(f"[Telegram] Polling error: {e}")
            self._connected = False

    async def disconnect(self):
        """Stop polling and shut down the Telegram application."""
        # Cancel the polling task if still running
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except (asyncio.CancelledError, Exception):
                pass
            self._polling_task = None

        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                print(f"[Telegram] Disconnect error: {e}")

        self._connected = False
        print("[Telegram] Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        """Send a text message to a Telegram chat.

        Long messages are automatically split into chunks of 4000 characters
        to stay within Telegram's 4096-character per-message limit.
        """
        if not self._bot:
            print("[Telegram] Not connected")
            return False

        try:
            # Split long messages (Telegram limit: 4096 characters)
            for chunk in self.split_text(content, 4000):
                await self._bot.send_message(
                    chat_id=int(channel_id),
                    text=chunk,
                    parse_mode=kwargs.get("parse_mode", "Markdown"),
                )
            return True

        except Exception as e:
            print(f"[Telegram] Send error: {e} channel_id={channel_id}")
            return False

    async def _on_message(self, update, context) -> None:  # noqa: ARG002
        """Handle an incoming Telegram message.

        Applies the user allowlist filter, converts to the unified ``Message``
        format, and forwards to the registered handler.
        """
        try:
            msg = update.message
            if not msg or not msg.text:
                return

            user = msg.from_user
            user_id = str(user.id) if user else ""

            # User allowlist check (via base class)
            if not self.check_allowlist(user_id):
                print(f"[Telegram] User {user_id} not in allowlist")
                return

            # Convert to unified message format
            message = Message(
                channel=ChannelType.TELEGRAM,
                channel_id=str(msg.chat_id),
                sender_id=user_id,
                sender_name=user.username or user.first_name or "",
                content=msg.text,
                message_id=str(msg.message_id),
                timestamp=msg.date.timestamp() if msg.date else 0,
                metadata={
                    "chat_type": msg.chat.type,
                    "first_name": user.first_name if user else "",
                    "last_name": user.last_name if user else "",
                }
            )

            # Forward to the registered handler
            if self._message_handler:
                await self._emit_message(message)

        except Exception as e:
            print(f"[Telegram] Message error: {e} sender_id={user_id if 'user_id' in dir() else 'unknown'}")
