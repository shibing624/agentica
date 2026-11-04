# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Discord channel implementation using discord.py.

Connects to Discord's gateway via the ``discord.Client`` WebSocket client.
Incoming messages trigger the ``on_message`` event; outgoing messages are
sent via the channel's ``send()`` method.
"""
import asyncio
from typing import Optional, List

from .base import Channel, ChannelType, Message
from ..config import settings

# Discord SDK global (lazy-imported to avoid hard dependency)
discord = None


def _ensure_discord_sdk():
    """Ensure the Discord SDK has been imported (lazy).

    Raises:
        ImportError: If ``discord.py`` is not installed.
    """
    global discord
    if discord is None:
        try:
            import discord
        except ImportError:
            raise ImportError(
                "Discord SDK not installed. Run: pip install discord.py"
            )


class DiscordChannel(Channel):
    """Discord messaging channel.

    Uses ``discord.py``'s ``Client`` to connect to the Discord gateway,
    receive messages in real-time, and send replies.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
        allowed_guilds: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or [])
        self.bot_token = bot_token or settings.discord_bot_token
        self.allowed_guilds = allowed_guilds or []
        self._client = None
        self._ready_event = asyncio.Event()
        self._client_task: Optional[asyncio.Task] = None

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.DISCORD

    async def connect(self) -> bool:
        """Connect to the Discord gateway and wait until the bot is ready."""
        if not self.bot_token:
            print("[Discord] Missing bot token, skipped")
            return False

        try:
            _ensure_discord_sdk()

            # Configure intents (message_content is required for reading text)
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True

            # Create the Discord client
            self._client = discord.Client(intents=intents)

            # Register event handlers
            @self._client.event
            async def on_ready():
                print(f"[Discord] Logged in as {self._client.user}")
                self._ready_event.set()

            @self._client.event
            async def on_message(message):
                await self._on_message(message)

            # Start the client in a tracked background task
            self._client_task = asyncio.create_task(self._start_client())

            # Wait for the client to become ready (timeout: 30s)
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                print("[Discord] Connection timeout")
                return False

            self._connected = True
            print("[Discord] Connected")
            return True

        except ImportError as e:
            print(f"[Discord] SDK not installed: {e}")
            return False
        except Exception as e:
            print(f"[Discord] Connect failed: {e}")
            return False

    async def _start_client(self):
        """Start the Discord client (runs in a background task)."""
        try:
            await self._client.start(self.bot_token)
        except Exception as e:
            print(f"[Discord] Client error: {e}")
            self._connected = False

    async def disconnect(self):
        """Gracefully close the Discord client connection."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                print(f"[Discord] Disconnect error: {e}")

        # Cancel the client task if still running
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass
            self._client_task = None

        self._connected = False
        print("[Discord] Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:  # noqa: ARG002
        """Send a text message to a Discord channel or DM.

        Long messages are automatically split into chunks of 1900 characters
        to stay within Discord's 2000-character per-message limit.

        If the ``channel_id`` does not match a known channel, the method
        attempts to create a DM channel with the user of that ID.
        """
        if not self._client:
            print("[Discord] Not connected")
            return False

        try:
            channel = self._client.get_channel(int(channel_id))
            if not channel:
                # Attempt to open a DM channel with the user
                try:
                    user = await self._client.fetch_user(int(channel_id))
                    channel = await user.create_dm()
                except Exception:
                    print(f"[Discord] Channel not found: {channel_id}")
                    return False

            # Split long messages (Discord limit: 2000 characters)
            for chunk in self.split_text(content, 1900):
                await channel.send(chunk)

            return True

        except Exception as e:
            print(f"[Discord] Send error: {e} channel_id={channel_id}")
            return False

    async def _on_message(self, message) -> None:
        """Handle an incoming Discord message.

        Filters out bot messages, applies user and guild allowlists,
        converts to the unified ``Message`` format, and forwards to the
        registered handler.
        """
        try:
            # Ignore messages from the bot itself
            if message.author == self._client.user:
                return

            # Ignore messages from other bots
            if message.author.bot:
                return

            user_id = str(message.author.id)

            # User allowlist check (via base class)
            if not self.check_allowlist(user_id):
                return

            # Guild (server) allowlist check
            if self.allowed_guilds and message.guild:
                if str(message.guild.id) not in self.allowed_guilds:
                    return

            # Convert to unified message format
            msg = Message(
                channel=ChannelType.DISCORD,
                channel_id=str(message.channel.id),
                sender_id=user_id,
                sender_name=message.author.display_name,
                content=message.content,
                message_id=str(message.id),
                timestamp=message.created_at.timestamp() if message.created_at else 0,
                metadata={
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "guild_name": message.guild.name if message.guild else None,
                    "channel_name": message.channel.name if hasattr(message.channel, 'name') else "DM",
                }
            )

            # Forward to the registered handler
            if self._message_handler:
                await self._emit_message(msg)

        except Exception as e:
            print(f"[Discord] Message error: {e} sender_id={message.author.id if hasattr(message, 'author') else 'unknown'}")
