# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Mid-run steering and peer-message injection into the tool loop
"""

from typing import (
    List,
    TYPE_CHECKING,
)


from agentica.utils.log import logger
from agentica.model.message import Message

if TYPE_CHECKING:
    from agentica.agent import Agent



class SteerMixin:
    """Extracted Runner methods."""

    @staticmethod
    def _inject_steering(messages: List[Message], agent: "Agent") -> None:
        """Flush pending user steering into the message list before an inference.

        Guidance buffered via ``agent.steer()`` (possibly from another thread)
        must reach the model on its very next call. We PREFER to fold it into the
        most recent tool result rather than appending a standalone user message:

        - It keeps a single trailing tool/user turn, so providers that enforce
          strict role alternation (Anthropic treats tool results as user-role)
          never see two consecutive user turns — both on this call and on later
          history replays, since ``messages`` is the persisted run transcript.
        - When there is no trailing tool result to fold into (e.g. the very first
          inference of a run), we fall back to appending a user message.

        Draining here — right before each inference — guarantees delivery: if a
        run ends before the buffer is flushed, the leftover guidance survives on
        the agent and is delivered at the start of the next run.
        """
        drained = agent._drain_steer()
        if not drained:
            return
        marker = "\n\n".join(f"[User guidance received while you were working]\n{guidance}" for guidance in drained)
        last = messages[-1] if messages else None
        if last is not None and last.role == "tool":
            existing = last.content.rstrip() if isinstance(last.content, str) else ""
            last.content = f"{existing}\n\n{marker}" if existing else marker
            logger.debug("Folded steering guidance into the latest tool result")
        else:
            messages.append(Message(role="user", content=marker))
            logger.debug("Injected steering guidance as a user message")

    @staticmethod
    def _inject_peer_messages(messages: List[Message], agent: "Agent") -> None:
        """Flush messages from the user's other sessions before an inference.

        Same boundary and same folding rules as ``_inject_steering`` (see there
        for why a trailing tool result is preferred over a second user turn):
        draining only between tool batches is what guarantees a running tool is
        never interrupted mid-call.

        The mailbox is a shared directory, so a failure to read it is an I/O
        edge we absorb — a peer message is never worth killing the user's turn.
        """
        peers = agent.peer_session
        if peers is None:
            return
        try:
            drained = peers.drain()
        except OSError:
            logger.warning("draining the peer mailbox failed", exc_info=True)
            return
        if not drained:
            return
        from agentica.peers import format_for_model

        marker = format_for_model(drained)
        last = messages[-1] if messages else None
        if last is not None and last.role == "tool":
            existing = last.content.rstrip() if isinstance(last.content, str) else ""
            last.content = f"{existing}\n\n{marker}" if existing else marker
        else:
            messages.append(Message(role="user", content=marker))
        logger.debug(f"Injected {len(drained)} peer message(s) into the run")

