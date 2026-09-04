# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Relay IM messages between a chat app and this machine's live CLI sessions.

From an IM app (WeCom, WeChat, Feishu, Telegram, ...) the user addresses one of
the agentica CLI sessions running on this machine and talks to it directly:

    @list                 which CLI sessions are live right now
    @nlp-f1 rerun arm 3   send that line to the session named nlp-f1
    rerun arm 4           ... and to the same session again (it stays pinned)
    @off                  stop talking to it; back to the gateway's own agent

**There is no new protocol.** The bridge is just another peer on the existing
file-based channel (``agentica/peers.py``): it owns a ``PeerSession`` per IM
user, so a CLI sees the phone in ``list_agents`` like any other session and
answers it with the ``send_message`` tool it already has. Everything the channel
already guarantees — mailbox ordering, backpressure, the repeat/rate brakes,
delivery at a tool-batch boundary rather than mid-tool — is inherited rather
than reimplemented.

Two consequences of that design decide the rest of this module:

**A relayed line is the user, so it carries the user's authority.** Sends use
``from_kind="user"``: the receiving CLI treats it as if it were typed in that
terminal (the bundled ``multi-agent`` skill). That is the point — the user *is* typing
it, from a phone. This is a personal-assistant gateway, so the bridge is on by
default and adds no gate of its own; a channel's ``allowed_users`` (when set)
already filters every inbound message before it can reach the bridge.

**The bridge lives in the same ``AGENTICA_HOME`` as the CLI or it sees nothing
at all.** The peers tree is per-install state under ``AGENTICA_CACHE_DIR``, so
a gateway started with a different ``AGENTICA_HOME`` finds an empty ``live/``
forever and every reply would be an unexplained "no sessions". Nothing can
assert this away, so the failure is made visible instead: the startup log and
the empty-list reply both name the directory that was searched.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from agentica.peers import (
    PeerInfo,
    PeerMessage,
    PeerMessageRefused,
    PeerSession,
    list_live_peers,
    match_peers,
    peers_root,
)
from agentica.utils.log import logger

from ..channels.base import Channel, ChannelType, Message

# How often the bridge looks for replies addressed to its IM users. Presence is
# refreshed on the same tick; `PeerSession.heartbeat` throttles that to
# HEARTBEAT_INTERVAL on its own.
POLL_INTERVAL = 1.0

# One live record + mailbox per IM user talking to the bridge. A cap keeps a
# group chat full of strangers from publishing a peer per member.
MAX_ENDPOINTS = 20

# IM platforms reject long messages; each reply is split into chunks. Well
# under every platform limit (Discord's 2000 is the smallest).
CHUNK_CHARS = 1800

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_LIST_WORDS = frozenset({"list", "ls", "agents", "peers", "sessions"})
_OFF_WORDS = frozenset({"off", "none", "stop", "-", "."})

USAGE = (
    "Talking to this machine's CLI sessions:\n"
    "  @list            list live sessions\n"
    "  @<name> <text>   send to that session (it stays the target)\n"
    "  @off             stop; back to the gateway agent"
)


@dataclass(frozen=True)
class BridgeCommand:
    """One parsed IM line. ``kind`` is ``list`` / ``send`` / ``pin`` / ``off``."""

    kind: str
    target: str = ""
    text: str = ""


def parse_bridge_line(line: str, *, pinned: Optional[str] = None) -> Optional[BridgeCommand]:
    """Classify one IM line, or return None when it is not for the bridge.

    ``None`` means "the gateway's own agent should answer this", which is what
    an unprefixed line means until the user has pinned a session. Pinning is
    what makes the bridge usable from a phone — nobody retypes ``@nlp-f1``
    before every line of a conversation — and unlike a hidden mode it announces
    itself on every reply, because each one is labelled with the session that
    sent it.
    """
    text = (line or "").strip()
    if not text.startswith("@"):
        if pinned:
            return BridgeCommand(kind="send", target=pinned, text=text) if text else None
        return None

    head, _, rest = text[1:].strip().partition(" ")
    head = head.strip()
    rest = rest.strip()
    lowered = head.casefold()
    if not head or lowered in _LIST_WORDS:
        return BridgeCommand(kind="list")
    if lowered in _OFF_WORDS:
        return BridgeCommand(kind="off")
    if not rest:
        return BridgeCommand(kind="pin", target=head)
    return BridgeCommand(kind="send", target=head, text=rest)


class _Endpoint:
    """One IM user's end of the peer channel: their peer session and reply route."""

    def __init__(self, session: PeerSession, channel: ChannelType, channel_id: str) -> None:
        self.session = session
        self.channel = channel
        self.channel_id = channel_id
        self.pinned: Optional[str] = None


class PeerBridge:
    """Relays between IM users and this machine's live CLI sessions."""

    def __init__(
        self,
        channel_manager,
        *,
        poll_interval: float = POLL_INTERVAL,
        gateway_peer_ids: Optional[Callable[[], set]] = None,
    ) -> None:
        self._channel_manager = channel_manager
        self._poll_interval = poll_interval
        # The gateway agent's own peers (agent_peers.py). Excluded from every
        # listing and lookup: relaying a line into the agent that is already
        # answering this chat would echo it straight back to the phone, and an
        # unprefixed line reaches that agent anyway.
        self._gateway_peer_ids = gateway_peer_ids
        self._endpoints: Dict[Tuple[str, str], _Endpoint] = {}
        self._task: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        logger.info(
            f"Peer bridge started — relaying to this machine's live CLI sessions "
            f"(peers dir: {peers_root()}; send `@list` from a chat)"
        )
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for endpoint in self._endpoints.values():
            endpoint.session.unpublish()
        self._endpoints.clear()

    # -- inbound -----------------------------------------------------------

    async def handle(self, message: Message) -> bool:
        """Relay ``message`` to a CLI session; False means "not for the bridge".

        False leaves the message to the gateway's own agent, so enabling the
        bridge takes nothing away from an IM user who never types ``@``.
        """
        endpoint = self._endpoints.get(self._key(message))
        command = parse_bridge_line(message.content, pinned=endpoint.pinned if endpoint else None)
        if command is None:
            return False

        if command.kind == "list":
            await self._reply(message, self._render_sessions())
            return True

        if command.kind == "off":
            if endpoint is not None:
                endpoint.pinned = None
            await self._reply(message, f"Stopped. Plain messages go to the gateway agent again.\n\n{USAGE}")
            return True

        endpoint = self._endpoint_for(message)
        if endpoint is None:
            await self._reply(
                message,
                f"The bridge is already relaying for {MAX_ENDPOINTS} chats; "
                "send `@off` from one of them first.",
            )
            return True

        if command.kind == "pin":
            await self._reply(message, self._pin(endpoint, command.target))
            return True

        await self._reply(message, self._send(endpoint, command.target, command.text))
        return True

    def _pin(self, endpoint: _Endpoint, target: str) -> str:
        peers = self._resolve(target)
        if len(peers) != 1:
            return self._render_sessions(problem=target, matched=len(peers))
        endpoint.pinned = peers[0].name
        return (
            f"Talking to {peers[0].name} — send text and it lands there.\n"
            f"{peers[0].describe()}"
        )

    def _send(self, endpoint: _Endpoint, target: str, text: str) -> str:
        peers = self._resolve(target)
        if len(peers) != 1:
            # An unknown or ambiguous name is answered with the live listing:
            # on a phone the next thing the user needs is the name to retry
            # with, not a report that they got it wrong.
            return self._render_sessions(problem=target, matched=len(peers))
        try:
            # Addressed by peer id: the name was already resolved here, against
            # a candidate set the channel does not know to narrow.
            message = endpoint.session.send(peers[0].peer_id, text, from_kind="user")
        except PeerMessageRefused as e:
            # The channel's own refusals (unread backlog, size) already say what
            # to do about them; a paraphrase would only lose that.
            return f"Not delivered: {e}"
        endpoint.pinned = message.to_name
        return f"Sent to {message.to_name}. It lands between that session's tool calls."

    def _resolve(self, target: str) -> List[PeerInfo]:
        """Live CLI sessions ``target`` could name — every candidate, so that
        "unknown" and "ambiguous" stay different answers."""
        own = self._own_peer_ids()
        return [peer for peer in match_peers(target) if peer.peer_id not in own]

    def _own_peer_ids(self) -> set:
        own = {endpoint.session.peer_id for endpoint in self._endpoints.values()}
        if self._gateway_peer_ids is not None:
            own |= self._gateway_peer_ids()
        return own

    def _live_sessions(self) -> List[PeerInfo]:
        """Live CLI sessions, never this process's own peers.

        Each IM user is published as a peer so a CLI can answer it, which also
        makes it a candidate in every listing and every address lookup. Nobody
        wants to send their own phone a message, and one endpoint's name would
        shadow a real session's. The gateway agent's own peers are excluded for
        the same reason.
        """
        own = self._own_peer_ids()
        return [peer for peer in list_live_peers() if peer.peer_id not in own]

    def _render_sessions(self, *, problem: str = "", matched: int = 0) -> str:
        peers = self._live_sessions()
        if not peers:
            # Naming the directory is the only way an AGENTICA_HOME mismatch
            # between the gateway and the CLI becomes visible: the symptom is
            # identical to having no session open.
            return (
                f"No live agentica CLI session found on this machine "
                f"(searched {peers_root()}). Open one, or check that the "
                f"gateway and the CLI share the same AGENTICA_HOME.\n\n{USAGE}"
            )
        lines: List[str] = []
        if problem:
            lines.append(
                f"'{problem}' matches {matched} live sessions."
                if matched
                else f"No live session matches '{problem}'."
            )
        lines.append(f"{len(peers)} live session(s):")
        for peer in peers:
            state = "running a turn" if peer.busy else "idle"
            lines.append(f"  {peer.name} — {state} — {peer.cwd}")
            if peer.task:
                lines.append(f"      {peer.task}")
        lines.append("")
        lines.append(USAGE)
        return "\n".join(lines)

    # -- outbound ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A poll failure must not end the loop: the next tick is the
                # retry, and the alternative is a bridge that silently stops.
                logger.error(f"Peer bridge poll error: {e}")

    async def _drain_once(self) -> None:
        for endpoint in list(self._endpoints.values()):
            messages = await asyncio.to_thread(self._refresh_and_drain, endpoint)
            for message in messages:
                await self._push(endpoint, message)

    @staticmethod
    def _refresh_and_drain(endpoint: _Endpoint) -> List[PeerMessage]:
        """Republish presence and take this endpoint's mail (both hit the disk)."""
        endpoint.session.heartbeat()
        return endpoint.session.drain()

    async def _push(self, endpoint: _Endpoint, message: PeerMessage) -> None:
        body = f"{message.from_name} ›\n{message.text}"
        for chunk in Channel.split_text(body, CHUNK_CHARS):
            await self._channel_manager.send(endpoint.channel, endpoint.channel_id, chunk)

    # -- endpoints ---------------------------------------------------------

    @staticmethod
    def _key(message: Message) -> Tuple[str, str]:
        return message.channel.value, message.sender_id

    def _endpoint_for(self, message: Message) -> Optional[_Endpoint]:
        """This IM user's endpoint, created on first use. None when full."""
        key = self._key(message)
        endpoint = self._endpoints.get(key)
        if endpoint is not None:
            # A user can start a new conversation (a group vs a DM); replies
            # follow wherever they last spoke from.
            endpoint.channel_id = message.channel_id
            return endpoint
        if len(self._endpoints) >= MAX_ENDPOINTS:
            return None
        name = _endpoint_name(message)
        session = PeerSession(name=name)
        who = message.sender_name or message.sender_id
        session.publish(task=f"relaying {message.channel.value} messages for {who}")
        self._endpoints[key] = _Endpoint(session, message.channel, message.channel_id)
        logger.info(f"Peer bridge endpoint {name} [peer={session.peer_id}] published")
        return self._endpoints[key]

    async def _reply(self, message: Message, text: str) -> None:
        for chunk in Channel.split_text(text, CHUNK_CHARS):
            await self._channel_manager.send(message.channel, message.channel_id, chunk)


def _endpoint_name(message: Message) -> str:
    """Addressable name a CLI sees, e.g. ``wecom-xuming``.

    Derived from ``sender_id`` rather than the display name because it must be
    unique within the channel, which is exactly what a sender id already is.
    """
    slug = _SLUG_RE.sub("-", (message.sender_id or "user").casefold()).strip("-")[:24]
    return f"{message.channel.value}-{slug or 'user'}"
