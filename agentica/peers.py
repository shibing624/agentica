# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cross-session peer messaging between live agentica CLI sessions.

Two sessions running in separate terminals exchange short plain-text messages
so the user does not have to copy-paste findings between them. Everything is
plain files under ``$AGENTICA_CACHE_DIR/peers`` — no daemon, socket or broker:

    peers/live/<peer_id>.json       one record per live session (discovery)
    peers/mailbox/<peer_id>/*.md    one markdown file per undelivered message

Discovery reads ``live/`` and drops records whose process is gone or whose
heartbeat went stale. Delivery is a *pull*: the receiving session drains its
own mailbox directory between tool calls (see ``Runner._inject_peer_messages``)
or while idle (the CLI's peer loop), so a running tool is never interrupted.

The tree is deliberately user-level rather than project-scoped. Coordinating
two worktrees of the same repository is the main use case and those have
different working directories, so scoping by cwd would hide exactly the peers
that most need to talk.

A message is plain text. It never carries conversation history or files — to
move a whole conversation, resume or fork the session instead.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger

# A peer heartbeats while it lives; a record older than this is treated as a
# crashed session even when its pid still resolves (pids get reused).
HEARTBEAT_INTERVAL = 30.0
STALE_AFTER = 150.0

# Loop brake. Every message carries the hop count of the exchange that produced
# it, so a ping-pong between two sessions dies on its own instead of burning
# tokens forever.
MAX_HOP = 3

# Backpressure: a session that never reads its mailbox must not accumulate
# unbounded work for whenever it finally does.
MAX_UNREAD = 50

# A message is injected straight into the receiver's context, so the only thing
# this bounds is how much of someone else's window one send can spend. Generous
# enough for a full handoff write-up; past it, the filesystem is shared and a
# path costs the receiver nothing.
MAX_MESSAGE_CHARS = 40_000


def peers_root() -> Path:
    return Path(AGENTICA_CACHE_DIR) / "peers"


def live_dir() -> Path:
    return peers_root() / "live"


def mailbox_dir(peer_id: str) -> Path:
    return peers_root() / "mailbox" / peer_id


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        # Best effort: a restrictive umask or an exotic filesystem is not a
        # reason to disable messaging.
        pass


def new_peer_id() -> str:
    return uuid.uuid4().hex[:8]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def default_peer_name(cwd: Optional[str], peer_id: str) -> str:
    """Derive a short addressable name from the working directory.

    Mirrors what a user would call the session ("agentica-3f"): the folder they
    are working in, plus a couple of characters so two sessions in sibling
    checkouts of the same project stay distinguishable.
    """
    folder = Path(os.path.realpath(os.path.expanduser(cwd or os.getcwd()))).name
    slug = _SLUG_RE.sub("-", folder.lower()).strip("-") or "session"
    return f"{slug}-{peer_id[:2]}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user: alive, just not ours to signal.
        return True
    except OSError:
        return False
    return True


@dataclass
class PeerInfo:
    """One live session, as published for other sessions to discover."""

    peer_id: str
    name: str
    pid: int
    cwd: str
    session_id: Optional[str] = None
    git_branch: Optional[str] = None
    # One line of "what this session is working on", so a sending agent can
    # pick a target on its own instead of guessing from the name.
    task: Optional[str] = None
    # Session transcript lives under project_dir; long-term memory under
    # memory_path. Published so another agent can dig deeper than a short
    # peer message when it decides that is worth it.
    project_dir: Optional[str] = None
    workspace_path: Optional[str] = None
    memory_path: Optional[str] = None
    updated_at: float = 0.0

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.updated_at)

    @property
    def alive(self) -> bool:
        return self.age <= STALE_AFTER and _pid_alive(self.pid)

    @property
    def project_slug(self) -> Optional[str]:
        """Basename of ``project_dir`` (hash-suffixed, unique per cwd)."""
        if not self.project_dir:
            return None
        return Path(self.project_dir).name

    @property
    def session_log_path(self) -> Optional[str]:
        if not self.project_dir or not self.session_id:
            return None
        return str(Path(self.project_dir) / f"{self.session_id}.jsonl")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        return cls(
            peer_id=str(data.get("peer_id") or ""),
            name=str(data.get("name") or ""),
            pid=int(data.get("pid") or 0),
            cwd=str(data.get("cwd") or ""),
            session_id=data.get("session_id") or None,
            git_branch=data.get("git_branch") or None,
            task=data.get("task") or None,
            project_dir=data.get("project_dir") or None,
            workspace_path=data.get("workspace_path") or None,
            memory_path=data.get("memory_path") or None,
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def describe(self) -> str:
        """Multi-line listing used by ``list_agents`` and ``/list-agents``.

        Keep the addressable name on the first line; put the diggable paths
        underneath so a model (or a human scanning the terminal) can resume
        the session or read its memory without guessing directory names.
        """
        lines = [f"{self.name} [peer={self.peer_id}]"]
        if self.session_id:
            lines.append(f"  session_id: {self.session_id}")
        lines.append(f"  cwd: {self.cwd}")
        if self.project_dir:
            lines.append(f"  project: {self.project_dir}")
        if self.session_log_path:
            lines.append(f"  session_log: {self.session_log_path}")
        if self.workspace_path:
            lines.append(f"  workspace: {self.workspace_path}")
        if self.memory_path:
            lines.append(f"  memory: {self.memory_path}")
        if self.git_branch:
            lines.append(f"  branch: {self.git_branch}")
        if self.task:
            lines.append(f"  working on: {self.task}")
        return "\n".join(lines)


def _write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)


class PeerSession:
    """This CLI session's end of the peer channel: presence, inbox and sending.

    Identity is the CLI *process*, not the session log: ``/resume`` swaps the
    session id underneath but in-flight messages addressed to this terminal
    must still land, so both the live record and the mailbox are keyed by
    ``peer_id``.

    One object owns all three concerns because they share state — the hop count
    of a reply depends on what arrived from that same peer earlier.
    """

    def __init__(
        self,
        *,
        peer_id: Optional[str] = None,
        name: Optional[str] = None,
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
        git_branch: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
        memory_path: Optional[str] = None,
        on_drain: Optional[Any] = None,
    ) -> None:
        from agentica.compression.tool_result_storage import get_project_dir

        self.peer_id = peer_id or new_peer_id()
        self._user_id = user_id
        # Optional CLI/SDK hook fired after a successful drain, so the UI can
        # show the accepted message whether the Runner or the idle loop took it.
        self.on_drain = on_drain
        resolved_cwd = os.path.realpath(os.path.expanduser(cwd or os.getcwd()))
        self.info = PeerInfo(
            peer_id=self.peer_id,
            name=name or default_peer_name(resolved_cwd, self.peer_id),
            pid=os.getpid(),
            cwd=resolved_cwd,
            session_id=session_id,
            git_branch=git_branch,
            project_dir=get_project_dir(resolved_cwd, user_id=user_id),
            workspace_path=workspace_path,
            memory_path=memory_path,
        )
        self._last_publish = 0.0
        # Hop depth of the last message received from each peer, so a reply
        # continues that exchange's count instead of restarting at 1.
        self._inbound_hop: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def path(self) -> Path:
        return live_dir() / f"{self.peer_id}.json"

    # -- presence ----------------------------------------------------------

    def publish(self, **updates: Any) -> None:
        """Write (or refresh) this session's live record."""
        from agentica.compression.tool_result_storage import get_project_dir

        for key, value in updates.items():
            if value is not None and hasattr(self.info, key):
                setattr(self.info, key, value)
        if updates.get("cwd"):
            resolved = os.path.realpath(os.path.expanduser(str(updates["cwd"])))
            self.info.cwd = resolved
            self.info.project_dir = get_project_dir(resolved, user_id=self._user_id)
            # Addressable name tracks the directory this process is working in.
            self.info.name = default_peer_name(resolved, self.peer_id)
        if "user_id" in updates and updates["user_id"] is not None:
            self._user_id = updates["user_id"]
            self.info.project_dir = get_project_dir(self.info.cwd, user_id=self._user_id)
        self.info.updated_at = time.time()
        _ensure_private_dir(live_dir())
        _write_private_json(self.path, self.info.to_dict())
        self._last_publish = self.info.updated_at

    def heartbeat(self, **updates: Any) -> None:
        """Refresh the record when the heartbeat interval has elapsed."""
        if time.time() - self._last_publish < HEARTBEAT_INTERVAL and not updates:
            return
        self.publish(**updates)

    def unpublish(self) -> None:
        self.path.unlink(missing_ok=True)
        box = mailbox_dir(self.peer_id)
        if box.exists():
            for item in box.glob("*.md"):
                item.unlink(missing_ok=True)
            try:
                box.rmdir()
            except OSError:
                pass

    # -- messaging ---------------------------------------------------------

    def list_peers(self) -> List[PeerInfo]:
        return list_live_peers(exclude_peer_id=self.peer_id)

    def unread_count(self) -> int:
        return unread_count(self.peer_id)

    def drain(self) -> List[PeerMessage]:
        """Take every pending message, recording hop depth per sender."""
        messages = drain_inbox(self.peer_id)
        for message in messages:
            if message.from_peer_id:
                self._inbound_hop[message.from_peer_id] = max(
                    self._inbound_hop.get(message.from_peer_id, 0), message.hop
                )
        if messages and self.on_drain is not None:
            self.on_drain(messages)
        return messages

    def send(self, target: str, text: str, *, from_kind: str = "agent") -> PeerMessage:
        """Resolve ``target`` among live peers and deliver ``text`` to it."""
        info = resolve_peer(target, exclude_peer_id=self.peer_id)
        if info is None:
            raise PeerMessageRefused(
                f"no live session matches '{target}'; call list_agents to see current names"
            )
        return send_message(
            info,
            text=text,
            from_name=self.info.name,
            from_peer_id=self.peer_id,
            # A relayed user instruction is not part of an agent-to-agent
            # exchange, so it does not consume that exchange's hop budget.
            hop=1 if from_kind == "user" else self._inbound_hop.get(info.peer_id, 0) + 1,
            from_kind=from_kind,
        )


def list_live_peers(exclude_peer_id: Optional[str] = None) -> List[PeerInfo]:
    """Return live peers, newest heartbeat first, reaping dead records."""
    directory = live_dir()
    if not directory.exists():
        return []
    peers: List[PeerInfo] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        info = PeerInfo.from_dict(data)
        if not info.peer_id:
            continue
        if not info.alive:
            # The owner is gone and cannot clean up after itself.
            path.unlink(missing_ok=True)
            continue
        if exclude_peer_id and info.peer_id == exclude_peer_id:
            continue
        peers.append(info)
    peers.sort(key=lambda p: p.updated_at, reverse=True)
    return peers


def resolve_peer(target: str, *, exclude_peer_id: Optional[str] = None) -> Optional[PeerInfo]:
    """Find one live peer by peer_id, session_id, exact name, or unique prefix."""
    needle = (target or "").strip()
    if not needle:
        return None
    peers = list_live_peers(exclude_peer_id=exclude_peer_id)
    for info in peers:
        if needle == info.peer_id:
            return info
    session_exact = [p for p in peers if p.session_id and needle == p.session_id]
    if len(session_exact) == 1:
        return session_exact[0]
    if len(needle) >= 8:
        session_prefixed = [
            p for p in peers
            if p.session_id and p.session_id.startswith(needle)
        ]
        if len(session_prefixed) == 1:
            return session_prefixed[0]
        if len(session_prefixed) > 1:
            return None
    lowered = needle.casefold()
    exact = [p for p in peers if p.name.casefold() == lowered]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    prefixed = [p for p in peers if p.name.casefold().startswith(lowered)]
    if len(prefixed) == 1:
        return prefixed[0]
    return None


class PeerMessageRefused(Exception):
    """A send was refused by a channel limit (hop, backpressure, size)."""


@dataclass
class PeerMessage:
    """One plain-text message handed from a sending session to a receiver.

    ``from_kind`` separates the two things that can put a message in a mailbox,
    because they carry different authority: ``"agent"`` is another session's
    model talking (no authority — see the receiving-side policy), ``"user"`` is
    the human relaying an instruction with ``/send-message``. Only a process running
    as this user can write to the mailbox (mode 0700), so ``"user"`` is as
    trustworthy as a line typed into this terminal.
    """

    text: str
    from_name: str
    from_peer_id: str
    to_peer_id: str
    hop: int = 1
    created_at: str = ""
    from_kind: str = "agent"
    path: Optional[str] = field(default=None, compare=False)

    @property
    def from_user(self) -> bool:
        return self.from_kind == "user"

    def render(self) -> str:
        """Serialize to markdown with a small YAML-ish frontmatter."""
        return (
            "---\n"
            f"from_name: {self.from_name}\n"
            f"from_peer_id: {self.from_peer_id}\n"
            f"from_kind: {self.from_kind}\n"
            f"to_peer_id: {self.to_peer_id}\n"
            f"hop: {self.hop}\n"
            f"created_at: {self.created_at}\n"
            "---\n\n"
            f"{self.text.strip()}\n"
        )

    @classmethod
    def parse(cls, raw: str, *, path: Optional[str] = None) -> Optional["PeerMessage"]:
        """Parse a mailbox file.

        The frontmatter is written by ``render`` and holds only slugs, an int
        and a timestamp, so a flat ``key: value`` split is enough — no YAML
        dependency and no way for message text (which lives strictly after the
        closing delimiter) to be mistaken for a field.
        """
        if not raw.startswith("---\n"):
            return None
        _, _, rest = raw.partition("---\n")
        head, delimiter, body = rest.partition("\n---")
        if not delimiter:
            return None
        fields: Dict[str, str] = {}
        for line in head.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
        text = body.lstrip("\n").strip()
        if not text:
            return None
        try:
            hop = int(fields.get("hop", "1"))
        except ValueError:
            hop = 1
        return cls(
            text=text,
            from_name=fields.get("from_name", "unknown"),
            from_peer_id=fields.get("from_peer_id", ""),
            to_peer_id=fields.get("to_peer_id", ""),
            hop=hop,
            created_at=fields.get("created_at", ""),
            # Anything but an explicit "user" is treated as the unprivileged
            # case, so a malformed or truncated header cannot grant authority.
            from_kind="user" if fields.get("from_kind") == "user" else "agent",
            path=path,
        )


def unread_count(peer_id: str) -> int:
    box = mailbox_dir(peer_id)
    if not box.exists():
        return 0
    return sum(1 for _ in box.glob("*.md"))


def send_message(
    target: PeerInfo,
    *,
    text: str,
    from_name: str,
    from_peer_id: str,
    hop: int = 1,
    from_kind: str = "agent",
) -> PeerMessage:
    """Drop a message into ``target``'s mailbox.

    Raises ``PeerMessageRefused`` when a channel limit says no, so the caller
    (a tool) can report the reason to the model instead of failing silently.
    """
    body = (text or "").strip()
    if not body:
        raise PeerMessageRefused("message text is empty")
    if len(body) > MAX_MESSAGE_CHARS:
        raise PeerMessageRefused(
            f"message is {len(body)} chars, over the {MAX_MESSAGE_CHARS} limit; "
            "put the long version in a file and send the path instead"
        )
    if hop > MAX_HOP:
        raise PeerMessageRefused(
            f"this exchange already went {hop - 1} hops (limit {MAX_HOP}); "
            "stop relaying and report back to the user instead"
        )
    pending = unread_count(target.peer_id)
    if pending >= MAX_UNREAD:
        raise PeerMessageRefused(
            f"{target.name} has {pending} unread messages (limit {MAX_UNREAD}); it is not reading them"
        )

    message = PeerMessage(
        text=body,
        from_name=from_name,
        from_peer_id=from_peer_id,
        to_peer_id=target.peer_id,
        hop=hop,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        from_kind=from_kind,
    )
    box = mailbox_dir(target.peer_id)
    _ensure_private_dir(box)
    # The nanosecond stamp leads the name so a lexicographic sort of the
    # directory *is* send order, even for several messages inside one second and
    # across sending processes. A readable stamp and a random suffix follow, for
    # eyeballing the directory and for collision safety.
    now_ns = time.time_ns()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_ns / 1e9))
    path = box / f"{now_ns:019d}-{stamp}-{uuid.uuid4().hex[:4]}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(message.render(), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    # Rename last so a reader never sees a half-written message.
    os.replace(tmp, path)
    message.path = str(path)
    logger.debug(f"peer message sent to {target.name} [{target.peer_id}]: {path}")
    return message


def drain_inbox(peer_id: str) -> List[PeerMessage]:
    """Take every pending message for ``peer_id``, oldest first.

    Files are removed as they are read: a message is delivered exactly once,
    and an unparsable file is discarded rather than blocking the mailbox.
    """
    box = mailbox_dir(peer_id)
    if not box.exists():
        return []
    messages: List[PeerMessage] = []
    for path in sorted(box.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            # Leave it in place: a transient read error must not destroy a
            # message that a later drain could still deliver.
            continue
        path.unlink(missing_ok=True)
        message = PeerMessage.parse(raw, path=str(path))
        if message is None:
            logger.warning(f"discarded unparsable peer message: {path}")
            continue
        messages.append(message)
    return messages


def format_for_model(messages: List[PeerMessage]) -> str:
    """Render drained messages as the text injected into the receiving agent.

    The header states who is speaking, because the two cases differ in what the
    receiver may act on: a relayed user instruction is the human talking and
    carries their authority, while another session's agent carries none.
    """
    blocks = []
    for message in messages:
        if message.from_user:
            header = (
                f"[Your user sent this from their other session '{message.from_name}' "
                f"(reply_to={message.from_peer_id})]"
            )
        else:
            header = (
                f"[Message from another agent session: {message.from_name} "
                f"(reply_to={message.from_peer_id})]"
            )
        blocks.append(f"{header}\n{message.text}")
    return "\n\n".join(blocks)


def format_for_cli(messages: List[PeerMessage], *, delivery: str) -> str:
    """Render drained messages for the receiving terminal.

    ``delivery`` is a short clause such as ``starting a turn`` or
    ``will reach the agent between tool calls`` — the CLI knows which path
    accepted the message; this helper only formats it.
    """
    blocks = []
    for message in messages:
        who = "your user via" if message.from_user else "agent"
        blocks.append(
            f"✉ Accepted peer message from {who} {message.from_name} "
            f"[peer={message.from_peer_id}] — {delivery}\n"
            f"  {message.text}"
        )
    return "\n".join(blocks)
