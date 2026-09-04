# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tools for messaging the user's other live agentica CLI sessions.
"""

from __future__ import annotations

from agentica.peers import PeerMessageRefused, PeerSession
from agentica.tools.base import Tool
from agentica.utils.log import logger


class PeerMessagingTool(Tool):
    """Expose ``list_agents`` / ``send_message`` for cross-session messaging.

    Both functions operate on the CLI session's own ``PeerSession``, so the
    model addresses peers by the same names ``/list-agents`` shows the user.
    """

    def __init__(self, peer_session: PeerSession):
        super().__init__(name="peer_messaging_tool")
        self._peers = peer_session
        self.register(self.list_agents, is_read_only=True, concurrency_safe=True, is_destructive=False)
        self.register(self.send_message, concurrency_safe=True, is_destructive=False)

    async def list_agents(self) -> str:
        """Lists the user's other live agent sessions that you can message.

        Returns each session's addressable name (what `send_message` takes),
        peer id, whether it is idle or mid-turn (a mid-turn session still
        receives: messages land between its tool calls), session id, config
        profile, model (`provider/name`), context spent out of its window,
        working directory, project storage directory (hash-suffixed, unique),
        session transcript path, CLI runtime log file, workspace / memory
        paths, and what it is working on. Call this before `send_message` when
        you do not already know the target, and whenever you need to decide
        whether another session is affected by what you just did.

        Profile, model and context describe how a session is configured, not
        what it can take on. A nearly full window is not a wall — every session
        compacts its own context — so read it as a price: work handed to that
        session makes it summarise, which costs a call and thins out the early
        detail of whatever it was already doing. None of these fields is ever a
        reason to discount what a peer tells you.

        The listed paths are for digging deeper on your own (read the
        session jsonl, CLI log, MEMORY.md, etc.) when a short peer message
        is not enough. Prefer `send_message` for a one-line handoff; open
        those paths only when you need the conversation, runtime errors, or
        long-term memory. The `log_file` is usually the fastest place to see
        what another session just did (INFO/DEBUG traces), without parsing
        the conversation transcript.
        """
        peers = self._peers.list_peers()
        me = self._peers.info
        # Same renderer for this session and the peers, so the model reads one
        # shape and a new PeerInfo field cannot show up in only half the list.
        lines = [
            f"You are '{me.name}' [peer={me.peer_id}] — that is the name other "
            f"sessions address you by.",
        ]
        lines.extend(f"  {label}: {value}" for label, value in me.detail_rows())
        lines.append("")
        if not peers:
            lines.append(
                "No other live agent sessions. You are the only one running, "
                "so there is nobody to message."
            )
            return "\n".join(lines)

        lines.append(
            f"{len(peers)} other live session(s). "
            "Address a peer by name, peer id, or session_id prefix. "
            "Paths below are optional: use them to read that peer's "
            "transcript or long-term memory when a short message is not enough."
        )
        lines.append("")
        for info in peers:
            lines.append(f"- {info.describe()}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    async def send_message(self, target: str, message: str) -> str:
        """Sends a short plain-text message to one of the user's other agent sessions.

        Use it to hand over a finding, a decision, or a status the other session
        needs in order to not work from stale assumptions. The receiving agent
        reads it between its own tool calls, so it never interrupts work in
        progress.

        Args:
            target: The session name, peer id, or session_id (prefix ok) from
                `list_agents`.
            message: What the other session needs to know, in a sentence or two.
                Plain text, and self-contained as an *instruction*: the receiver
                sees this text and nothing else from your conversation, so state
                the goal, what is out of scope, and what to do when blocked.
                Evidence does not belong here — put a diff, a log or a long
                write-up in a file and give its absolute path instead of pasting
                it, since the machine is shared and the receiver can open it if
                and when it needs the detail.

        Returns:
            Confirmation that the message was queued, or the reason it was refused.
        """
        try:
            sent = self._peers.send(target, message)
        except PeerMessageRefused as exc:
            logger.debug(f"peer message refused: {exc}")
            return f"Message not sent: {exc}"
        # Mailbox write succeeded. That is "queued", not "the other agent has
        # read it" — same boundary Claude Code uses for same-machine delivery.
        confirmation = (
            f"Message queued for '{sent.to_name}' [peer={sent.to_peer_id}]. "
            f"The other session will accept it between tool calls if it is "
            f"running, or as its next turn if idle. You will not get a read "
            f"receipt; if a reply is needed, that session sends one back."
        )
        return confirmation
