# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Channel manager for multi-channel message routing and lifecycle management.
"""
from typing import Dict, Optional, Callable, Any, Union

from agentica.utils.log import logger

from ..channels.base import Channel, ChannelType, Message


class ChannelManager:
    """Manages the lifecycle and message dispatch of all registered channels.

    Channels (Feishu, Telegram, Discord, etc.) register themselves here.
    The manager provides a single entry point for connecting, disconnecting,
    and sending messages across all channels, plus a unified message handler
    callback that receives incoming messages regardless of source.
    """

    def __init__(self):
        self.channels: Dict[ChannelType, Channel] = {}
        self._message_handler: Optional[Callable[[Message], Any]] = None

    def register(self, channel: Channel):
        """Register a channel instance and wire up its message handler.

        Args:
            channel: The channel instance to register. Its internal handler
                     is set to ``self._on_message`` so incoming messages are
                     forwarded to the unified handler.
        """
        channel.set_handler(self._on_message)
        self.channels[channel.channel_type] = channel
        logger.info(f"Channel registered: {channel.channel_type.value}")

    def set_handler(self, handler: Callable[[Message], Any]):
        """Set the unified message handler that receives all incoming messages.

        Args:
            handler: An async callback ``(Message) -> Any`` invoked whenever
                     any registered channel receives a message.
        """
        self._message_handler = handler

    async def _on_message(self, message: Message):
        """Internal callback invoked by individual channels on incoming messages."""
        if self._message_handler:
            await self._message_handler(message)

    async def connect_all(self):
        """Connect all registered channels.

        Each channel's ``connect()`` is called independently; failures in
        one channel do not prevent others from connecting.
        """
        for channel in self.channels.values():
            try:
                await channel.connect()
            except Exception as e:
                logger.error(f"Failed to connect {channel.channel_type.value}: {e}")

    async def disconnect_all(self):
        """Disconnect all registered channels.

        Each channel's ``disconnect()`` is called independently; failures in
        one channel do not prevent others from disconnecting.
        """
        for channel in self.channels.values():
            try:
                await channel.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect {channel.channel_type.value}: {e}")

    async def send(
        self,
        channel_type: Union[ChannelType, str],
        channel_id: str,
        content: str,
        **kwargs
    ) -> bool:
        """Send a message to a specific channel.

        Args:
            channel_type: Target channel, either a ``ChannelType`` enum value
                          or its string representation (e.g. ``"feishu"``).
            channel_id: The conversation/chat ID within the target channel.
            content: The text content to send.
            **kwargs: Additional keyword arguments forwarded to the channel's
                      ``send()`` method (e.g. ``parse_mode`` for Telegram).

        Returns:
            True if the message was sent successfully, False otherwise
            (unknown channel type, channel not registered, or not connected).
        """
        # Accept string channel types for convenience (e.g. from API requests)
        if isinstance(channel_type, str):
            try:
                channel_type = ChannelType(channel_type)
            except ValueError:
                logger.warning(f"Unknown channel type: {channel_type}")
                return False

        channel = self.channels.get(channel_type)
        if not channel:
            logger.warning(f"Channel not registered: {channel_type.value}")
            return False

        if not channel.is_connected:
            logger.warning(f"Channel not connected: {channel_type.value}")
            return False

        return await channel.send(channel_id, content, **kwargs)

    def get_status(self) -> Dict[str, dict]:
        """Return the connection status of all registered channels.

        Returns:
            A dict keyed by channel type string (e.g. ``"feishu"``) with
            value dicts containing at least ``{"connected": bool}``.
        """
        return {
            ct.value: {
                "connected": ch.is_connected,
            }
            for ct, ch in self.channels.items()
        }

    def get_channel(self, channel_type: ChannelType) -> Optional[Channel]:
        """Look up a registered channel by type.

        Args:
            channel_type: The channel type to look up.

        Returns:
            The ``Channel`` instance if registered, or ``None``.
        """
        return self.channels.get(channel_type)

    def list_channels(self) -> list:
        """List all registered channel type strings.

        Returns:
            A list of channel type values, e.g. ``["feishu", "telegram"]``.
        """
        return [ct.value for ct in self.channels.keys()]
