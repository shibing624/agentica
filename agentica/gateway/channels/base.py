# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Abstract base class and shared data models for messaging channels.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List
from enum import Enum


class ChannelType(Enum):
    """Supported messaging channel types."""
    WEB = "web"
    FEISHU = "feishu"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WECHAT = "wechat"     # personal WeChat (ilinkai)
    DINGTALK = "dingtalk"
    QQ = "qq"
    WECOM = "wecom"       # Enterprise WeChat (WeCom)
    SLACK = "slack"       # Slack (Socket Mode)


@dataclass
class Message:
    """Unified message format shared across all channels.

    Every channel implementation converts its native message format into
    this dataclass before forwarding it to the message handler.
    """
    channel: ChannelType
    channel_id: str          # Conversation/chat ID within the channel
    sender_id: str           # Unique identifier of the message sender
    sender_name: str         # Human-readable display name of the sender
    content: str             # Text content of the message
    message_id: str          # Unique identifier of the message itself
    timestamp: float = 0
    reply_to: Optional[str] = None
    attachments: List[Any] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Channel(ABC):
    """Abstract base class that all channel implementations must extend.

    A channel represents a connection to an external messaging platform
    (Feishu, Telegram, Discord, etc.). Subclasses must implement
    ``channel_type``, ``connect()``, ``disconnect()``, and ``send()``.

    Common functionality (text splitting, allowlist checking) is provided
    here to avoid duplication across channel implementations.
    """

    def __init__(self, allowed_users: Optional[List[str]] = None):
        self._message_handler: Optional[Callable[[Message], Any]] = None
        self._connected = False
        self.allowed_users: List[str] = allowed_users or []

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type enum value for this implementation."""
        pass

    @property
    def is_connected(self) -> bool:
        """Whether this channel currently has an active connection."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """Establish the connection to the external messaging platform.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Gracefully close the connection to the external messaging platform."""
        pass

    @abstractmethod
    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        """Send a text message to the specified conversation.

        Args:
            channel_id: The target conversation/chat ID.
            content: The text content to send.
            **kwargs: Channel-specific options (e.g. ``parse_mode`` for Telegram).

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        pass

    def set_handler(self, handler: Callable[[Message], Any]):
        """Set the callback that will be invoked when this channel receives a message.

        Args:
            handler: An async callback ``(Message) -> Any`` to process incoming messages.
        """
        self._message_handler = handler

    async def _emit_message(self, message: Message):
        """Forward a received message to the registered handler.

        Args:
            message: The incoming message in unified format.
        """
        if self._message_handler:
            await self._message_handler(message)

    def check_allowlist(self, user_id: str) -> bool:
        """Check whether a user is permitted by the allowlist.

        If ``allowed_users`` is empty, all users are permitted.

        Args:
            user_id: The sender's user ID to check.

        Returns:
            True if the user is allowed, False otherwise.
        """
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    @staticmethod
    def split_text(text: str, max_len: int) -> List[str]:
        """Split text into chunks of at most ``max_len`` characters.

        Messaging platforms impose per-message size limits (Feishu: 4000,
        Telegram: 4096, Discord: 2000). This utility splits long messages
        into compliant chunks.

        Args:
            text: The text to split.
            max_len: Maximum characters per chunk.

        Returns:
            A list of text chunks. Returns ``[""]`` for empty input.
        """
        if not text:
            return [""]
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
