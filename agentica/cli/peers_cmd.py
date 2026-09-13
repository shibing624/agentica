# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ``agentica peers list|send`` — message a live session from a shell.

This is the entry point for a process that is **not** an agentica session: a
desktop app, a pet, a notification script, a keybinding. It is a thin shell over
``agentica/peers.py``, which owns the mailbox format, the addressing rules and
the limits — nothing about the wire is decided here.

Why it exists rather than each consumer writing the markdown itself: the mailbox
frontmatter is an internal format, and every external writer of it is a silent
breakage waiting for the next field change. It is also one rule —
``from_kind: user`` — away from being wrong in a way that changes the receiving
session's authority. One implementation, in the package that owns the format.
"""

from __future__ import annotations

import socket
from typing import List

from agentica.peers import (
    DELIVERY_QUEUE,
    DELIVERY_STEER,
    PeerInfo,
    PeerMessageRefused,
    list_live_peers,
    send_to_live_peer,
)
from agentica.utils.log import logger


def _default_from_name() -> str:
    """Who to attribute a message to when the caller does not say.

    The hostname, because on the machine running this it is unambiguous and the
    user can map it to "the thing I installed". It deliberately is not a peer
    name: an external sender has no mailbox, so presenting it as a session would
    invite a reply that has nowhere to land.
    """
    return socket.gethostname() or "external"


def _render_peers(peers: List[PeerInfo]) -> str:
    if not peers:
        return (
            "No live agentica sessions.\n\n"
            "A session appears here while its interactive CLI is running. "
            "Start one with 'agentica', then run this again."
        )
    lines = [f"{len(peers)} live session(s):", ""]
    for info in peers:
        # Name and peer id are what --to accepts; the rest is what a human or a
        # desktop app needs to pick between them without asking.
        task = (info.task or "").strip()
        lines.append(f"  {info.name}  [peer={info.peer_id}]")
        lines.append(f"    cwd:  {info.cwd}")
        if info.session_id:
            lines.append(f"    session: {info.session_id}")
        if info.busy:
            lines.append("    busy:  mid-turn (a message lands between its tool calls)")
        if task:
            lines.append(f"    doing: {task}")
        lines.append("")
    lines.append("Send with: agentica peers send --to <name|peer_id> --text '...'")
    return "\n".join(lines)


def run_peers_command(args, con) -> int:
    """Run ``peers list`` / ``peers send``. Returns a process exit code."""
    command = getattr(args, "peers_command", None)

    if command == "list":
        # markup=False: every line here carries user data (peer ids in brackets,
        # names and paths derived from the filesystem). Rich would read
        # ``[peer=53a1]`` as a style tag and print the name without its id — the
        # one address a caller can always use — and a path containing square
        # brackets would silently lose part of itself. Verified by
        # scripts/verify_peers_send_e2e.py, which could not find the id.
        con.print(_render_peers(list_live_peers()), markup=False)
        return 0

    if command != "send":
        con.print(f"Unknown peers subcommand: {command!r}")
        return 2

    text = getattr(args, "text_flag", None)
    positional = getattr(args, "text", None)
    if text is not None and positional is not None:
        # Concatenating two texts would send a message the caller did not write,
        # and picking one silently would drop the other.
        con.print("Use either --text or a positional message, not both.")
        return 2
    text = text if text is not None else positional
    if not text or not text.strip():
        con.print("No message text. Pass it positionally or with --text '...'.")
        return 2

    from_name = (getattr(args, "from_name", None) or _default_from_name()).strip()
    delivery = getattr(args, "delivery", DELIVERY_STEER)

    try:
        sent = send_to_live_peer(
            args.to,
            text,
            from_name=from_name,
            from_kind="user",
            delivery=delivery,
        )
    except PeerMessageRefused as exc:
        # The reason is the whole message: "no such session" and "be more
        # specific" have different fixes, and peers.py already words them.
        con.print(f"Not sent: {exc}")
        logger.debug(f"peers send refused: {exc}")
        return 1

    if sent.delivery == DELIVERY_QUEUE:
        when = "as its next turn, once its current run finishes"
    else:
        when = "between its tool calls if it is running, or as its next turn if idle"
    con.print(
        f"Sent to {sent.to_name} [{sent.to_peer_id}] as the user ({sent.delivery}). "
        f"It arrives {when}.",
        markup=False,
    )
    return 0
