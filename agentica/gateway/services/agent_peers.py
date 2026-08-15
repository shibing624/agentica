# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Peer-channel identity for the gateway's *own* agent sessions.

``peer_bridge.py`` gives the **user** reach into this machine's CLI sessions
from a chat app: ``@nlp-f1 rerun arm 3``. This module gives the same reach to
the **gateway's own agent**, which until now had none: ``PeerMessagingTool``
was wired only in ``cli/runtime.py``, so a plain (un-``@``-prefixed) line like
"让三个会话都把改动提交了" reached an agent that could not see a single one of
them. Everything the README promises about a team of sessions was reachable
from a terminal and from an ``@`` command, and unreachable from the one surface
a user has when they are not at the machine: a sentence typed into IM.

**Still no new protocol.** Each gateway chat session gets a ``PeerSession`` on
the same file-based channel (``agentica/peers.py``), published under a short
name like the CLI (``wechat-agentica-41``: channel + cwd folder + two id
chars), so a CLI sees it in ``list_agents`` and answers it with the
``send_message`` tool it already has. That symmetry buys the reverse direction
for free: a CLI session can message ``wechat-agentica-41`` to reach the user's
phone without knowing anything about the gateway.

Two things this module owns beyond publishing:

**Replies have to find their way back to the phone.** A CLI answers into this
session's mailbox. Mid-turn the Runner drains it (``agent.peer_session``) and
the model sees it; between turns nobody would, so the poll loop drains and
pushes it to the IM conversation this session belongs to. The route is
*recorded* by the caller that already knows it
(``main.py::_process_channel_message``) rather than parsed back out of a
session id — ``agent:{agent_id}:{channel}:{channel_id}`` is unambiguous to
build and not to split. A web-UI session has no IM route, so its mail is left
in the mailbox for the next turn rather than dropped.

**A live record must not outlive the session it describes.** Agents are cached
LRU, so eviction, session deletion and shutdown all end sessions without
telling anyone, and a stale ``live/`` record is worse than no record: it is an
addressable name whose mailbox nobody reads. The loop asks ``is_live`` and
unpublishes whatever the agent cache has forgotten.

**And the name it comes back under must be the one it left with**, which is why
the peer id is derived from the chat session id rather than minted fresh
(``_stable_peer_id``). See that function: "re-publish under a new id, peers are
addressed by name" was the original reasoning and it was wrong, because the
name embeds the id.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Callable, Dict, List, Optional, Tuple

from agentica.peers import PeerMessage, PeerSession, default_peer_name
from agentica.utils.log import logger

from ..channels.base import Channel, ChannelType
from ..config import settings

# Same tick as the bridge: presence refresh (throttled to HEARTBEAT_INTERVAL by
# PeerSession.heartbeat itself) plus a mailbox look.
POLL_INTERVAL = 1.0

# IM platforms reject long messages; Discord's 2000 is the smallest limit.
CHUNK_CHARS = 1800

# One line of "what this session is working on", as CLI peers publish. Enough
# for another session to tell two conversations apart, not enough to leak a
# whole prompt into every listing.
MAX_TASK_CHARS = 200

def _gateway_peer_name(channel: str, cwd: Optional[str], peer_id: str) -> str:
    """Short addressable name: channel + cwd folder + two id chars.

    A CLI publishes ``<folder>-<xx>`` (``agentica-41``). Putting the channel
    in front (``wechat-agentica-41``) keeps this out of that prefix space and
    out of a bridge endpoint's ``<channel>-<sender>`` space, without stuffing
    a WeChat openid into a name the model has to type as a ``send_message``
    target.
    """
    return f"{channel}-{default_peer_name(cwd, peer_id)}"


def _stable_peer_id(session_id: str) -> str:
    """A peer id that survives agent-cache eviction, deletion and restart.

    A CLI mints its peer id once per process, so the address a worker was told
    to answer (``agentica-41``) stays valid for as long as that terminal is
    open. The gateway had no such anchor: ``session_for`` runs again after
    every LRU eviction or restart, and a fresh uuid moved the *name* too
    (``default_peer_name`` embeds the first two characters), so a worker
    reporting back an hour later addressed a session that no longer exists —
    with the user then relaying the result by hand. The chat session id is what
    identifies this conversation, so the identity is derived from it and the
    name becomes a property of the conversation rather than of the cache.
    """
    return hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8]


def _task_line(text: str) -> str:
    one_line = " ".join((text or "").split())
    return one_line[:MAX_TASK_CHARS] or "gateway agent"


class GatewayAgentPeers:
    """The gateway agent's end of the peer channel, one ``PeerSession`` per chat session.

    ``is_live`` / ``is_busy`` are the agent cache and the per-session run lock,
    injected rather than imported: this class must not reach back into
    ``AgentService`` (which builds it into every agent it creates), and the two
    predicates are the only things it needs from there.
    """

    def __init__(
        self,
        *,
        channel_manager=None,
        is_live: Optional[Callable[[str], bool]] = None,
        is_busy: Optional[Callable[[str], bool]] = None,
        poll_interval: float = POLL_INTERVAL,
    ) -> None:
        self._channel_manager = channel_manager
        self._is_live = is_live
        self._is_busy = is_busy
        self._poll_interval = poll_interval
        self._sessions: Dict[str, PeerSession] = {}
        self._routes: Dict[str, Tuple[ChannelType, str]] = {}
        self._task: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Gateway agent peer channel started — the gateway's own agent can now "
            "list_agents / send_message this machine's CLI sessions"
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for session in list(self._sessions.values()):
            session.unpublish()
        self._sessions.clear()
        self._routes.clear()

    # -- presence ----------------------------------------------------------

    def session_for(self, session_id: str, *, cwd: Optional[str] = None) -> PeerSession:
        """This chat session's peer identity, published on first use.

        Called while building the session's agent (in a worker thread), so it
        does its own small presence write instead of waiting for the loop —
        a session that is about to send must be addressable for the reply.

        Rebuilt after an eviction it publishes the same id and name as before
        (``_stable_peer_id``), so an answer addressed to the name handed out
        earlier still lands here.
        """
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        route = self._routes.get(session_id)
        channel = route[0].value if route else "web"
        cwd_str = str(cwd or settings.base_dir)
        peer_id = _stable_peer_id(session_id)
        session = PeerSession(
            peer_id=peer_id,
            name=_gateway_peer_name(channel, cwd_str, peer_id),
            cwd=cwd_str,
            session_id=session_id,
            workspace_path=str(settings.workspace_path),
            model_provider=settings.model_provider or None,
            model_name=settings.model_name or None,
        )
        session.publish(task="gateway agent (web UI / chat channels)")
        self._sessions[session_id] = session
        logger.info(f"Gateway agent peer {session.name} [peer={session.peer_id}] published")
        return session

    def note_route(self, session_id: str, channel: ChannelType, channel_id: str) -> None:
        """Remember where this session's replies go, before its agent exists.

        Recorded on the inbound path so ``session_for`` can name the peer after
        the channel it serves (``wechat-agentica-41``), and so the loop can
        push a CLI's answer to the right chat. The conversation id is the
        reply address, not part of the published name.
        """
        self._routes[session_id] = (channel, channel_id)

    def note_turn(self, session_id: str, text: str) -> None:
        """Publish what this session is working on, as a CLI peer does.

        Only touches an already-registered session: a turn on a session with no
        peer identity (cron, or a service built without this channel) is not a
        reason to create one.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.heartbeat(task=_task_line(text), busy=True)

    def peer_ids(self) -> set:
        """Peer ids the gateway itself owns — the bridge excludes them from
        ``@list`` and from addressing, so the phone cannot relay a line into
        the very agent that is answering it."""
        return {session.peer_id for session in self._sessions.values()}

    def _unpublish(self, session_id: str) -> None:
        """Take a session off the peer directory, keeping its reply route.

        An eviction is not the end of the conversation — the next message
        rebuilds the agent — and the route is what names the peer
        (``wecom-agentica-64``). Dropping it here republished the same session
        as ``web-agentica-64``, so the stable peer id above bought nothing: the
        address a worker holds moved anyway.
        """
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.unpublish()
            logger.debug(f"Gateway agent peer {session.name} unpublished")

    def forget(self, session_id: str) -> None:
        """Drop a session's peer identity and its reply route (deleted session)."""
        self._unpublish(session_id)
        self._routes.pop(session_id, None)

    # -- outbound ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A failed tick must not end the loop: the next one is the
                # retry, and the alternative is presence that silently stops.
                logger.error(f"Gateway agent peer poll error: {e}")

    async def _tick(self) -> None:
        for session_id, session in list(self._sessions.items()):
            busy = self._is_busy(session_id) if self._is_busy else False
            live = self._is_live(session_id) if self._is_live else True
            if not live and not busy:
                self._unpublish(session_id)
                continue
            route = self._routes.get(session_id)
            messages = await asyncio.to_thread(self._refresh_and_drain, session, busy, route)
            for message in messages:
                await self._push(route, message)

    @staticmethod
    def _refresh_and_drain(
        session: PeerSession,
        busy: bool,
        route: Optional[Tuple[ChannelType, str]],
    ) -> List[PeerMessage]:
        """Republish presence and, when it is ours to deliver, take the mail.

        Two cases leave the mailbox alone. **Mid-turn**: the Runner drains it
        between tool batches and the model acts on it, which is strictly better
        than telling the user something the agent it is talking to has not
        seen. **No IM route** (a web-UI session): draining here would consume a
        message with nowhere to put it, so it waits for the next turn instead.
        """
        session.heartbeat(busy=busy)
        if busy or route is None:
            return []
        return session.drain()

    async def _push(self, route: Optional[Tuple[ChannelType, str]], message: PeerMessage) -> None:
        if route is None or self._channel_manager is None:
            return
        channel, channel_id = route
        body = f"{message.from_name} ›\n{message.text}"
        for chunk in Channel.split_text(body, CHUNK_CHARS):
            await self._channel_manager.send(channel, channel_id, chunk)
