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
import hashlib
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.git_state import GitState

# A peer heartbeats while it lives; a record older than this is treated as a
# crashed session even when its pid still resolves (pids get reused).
HEARTBEAT_INTERVAL = 30.0
STALE_AFTER = 150.0

# Backpressure: a session that never reads its mailbox must not accumulate
# unbounded work for whenever it finally does.
MAX_UNREAD = 50

# Two agents left alone will keep replying to each other, and neither of them is
# the user whose windows they are spending. Counting the exchange and cutting it
# off at N is the wrong brake: a handoff that legitimately needs a few more
# rounds gets refused, and the refusal lands on whatever the user asked for
# next. What is never legitimate is saying the same thing twice, or filling a
# peer's context faster than anyone could act on it — so the brake is a repeat
# check plus a generous rate limit, both per target peer and both cleared the
# moment a human joins in (``PeerSession.note_user_turn``).
RATE_WINDOW_SECONDS = 300.0
MAX_SENDS_PER_WINDOW = 20

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


def _text_digest(text: str) -> str:
    """Identity of a message for the repeat check: whitespace and case do not
    make a reworded point out of the same one."""
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


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


def _guess_cli_log_file(pid: int) -> Optional[str]:
    """Best-effort CLI log path for a live peer that did not publish ``log_file``.

    CLI logs are ``~/.agentica/logs/YYYYMMDD-<pid>.log``. When the other
    process is an older agentica that never advertised the path, the pid on
    its live record is still enough to find the file on the shared machine.
    """
    if pid <= 0:
        return None
    from agentica.config import AGENTICA_HOME

    log_dir = Path(AGENTICA_HOME) / "logs"
    if not log_dir.is_dir():
        return None
    matches = sorted(
        log_dir.glob(f"*-{pid}.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(matches[0]) if matches else None


@dataclass
class PeerInfo:
    """One live session, as published for other sessions to discover."""

    peer_id: str
    name: str
    pid: int
    cwd: str
    session_id: Optional[str] = None
    git_branch: Optional[str] = None
    # Where this session sits in the repository (agentica/git_state.py).
    # Published because the alternative is asking: "did you already touch that
    # file?" and "are you behind main?" cost a message, a turn on both sides,
    # and an answer that goes stale the moment a third session commits. Git
    # knows; presence carries it.
    head_sha: Optional[str] = None
    base_ref: Optional[str] = None
    # The repository this session's directory belongs to (its main checkout).
    # Equal across worktrees of one repo, which is how "another session has that
    # same file dirty" is decided without comparing bare filenames.
    repo_root: Optional[str] = None
    ahead: Optional[int] = None
    behind: Optional[int] = None
    dirty_files: List[str] = field(default_factory=list)
    dirty_count: Optional[int] = None
    # One line of "what this session is working on", so a sending agent can
    # pick a target on its own instead of guessing from the name.
    task: Optional[str] = None
    # Session transcript lives under project_dir; long-term memory under
    # memory_path. Published so another agent can dig deeper than a short
    # peer message when it decides that is worth it.
    project_dir: Optional[str] = None
    workspace_path: Optional[str] = None
    memory_path: Optional[str] = None
    # CLI runtime log (~/.agentica/logs/YYYYMMDD-<pid>.log) — easier to grep
    # than reconstructing the conversation from session_log.
    log_file: Optional[str] = None
    log_level: Optional[str] = None
    # Which profile / model this CLI session is actually running — the same
    # pair the status bar shows. Empty profile_name means a flag replaced the
    # model and no profile describes the session.
    profile_name: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    # Whether this session is mid-turn, and how much of its window is spent.
    # Neither bounds what it can take on — every session compacts its own
    # context, and a message reaches a busy one between its tool calls — so
    # both read as the price of sending, not as capacity.
    busy: bool = False
    context_tokens: Optional[int] = None
    context_window: Optional[int] = None
    updated_at: float = 0.0

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.updated_at)

    @property
    def alive(self) -> bool:
        return self.age <= STALE_AFTER and _pid_alive(self.pid)

    @property
    def git_state(self) -> "GitState":
        """The published git position, rendered by the module that collects it.

        Reassembled rather than stored as one blob: the live record stays flat
        and readable, and an older record that never published these fields
        yields an empty state instead of a parse error.
        """
        from agentica.git_state import GitState

        return GitState(
            branch=self.git_branch or "",
            head_sha=self.head_sha or "",
            repo_root=self.repo_root or "",
            base_ref=self.base_ref or "",
            ahead=self.ahead or 0,
            behind=self.behind or 0,
            dirty_files=tuple(self.dirty_files or ()),
            # None (never published) must stay None: it is "unknown", not
            # "clean". See GitState.dirty_count.
            dirty_count=self.dirty_count,
        )

    @property
    def project_slug(self) -> Optional[str]:
        """Basename of ``project_dir`` (hash-suffixed, unique per cwd)."""
        resolved = self.resolved_project_dir
        if not resolved:
            return None
        return Path(resolved).name

    @property
    def resolved_project_dir(self) -> Optional[str]:
        """Published ``project_dir``, or the deterministic path for ``cwd``."""
        if self.project_dir:
            return self.project_dir
        if not self.cwd:
            return None
        from agentica.project_store import project_base_dir

        return project_base_dir(self.cwd)

    @property
    def session_log_path(self) -> Optional[str]:
        project = self.resolved_project_dir
        if not project or not self.session_id:
            return None
        return str(Path(project) / f"{self.session_id}.jsonl")

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
            head_sha=data.get("head_sha") or None,
            base_ref=data.get("base_ref") or None,
            repo_root=data.get("repo_root") or None,
            ahead=data.get("ahead") or None,
            behind=data.get("behind") or None,
            dirty_files=list(data.get("dirty_files") or []),
            dirty_count=data.get("dirty_count") or None,
            task=data.get("task") or None,
            project_dir=data.get("project_dir") or None,
            workspace_path=data.get("workspace_path") or None,
            memory_path=data.get("memory_path") or None,
            log_file=data.get("log_file") or None,
            log_level=data.get("log_level") or None,
            profile_name=data.get("profile_name") or None,
            model_provider=data.get("model_provider") or None,
            model_name=data.get("model_name") or None,
            busy=bool(data.get("busy")),
            context_tokens=data.get("context_tokens") or None,
            context_window=data.get("context_window") or None,
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def detail_rows(self) -> List[Tuple[str, str]]:
        """``(label, value)`` pairs for this session, in display order.

        The single source for both the plain text a model reads via
        ``list_agents`` and the styled ``/list-agents`` listing — the two
        drifted apart while they each hand-rolled the same field list.

        Self and other peers use the same field set. When a live record is
        from an older agentica that omitted paths, fill what is deterministic
        on this machine (project from cwd, session_log, mailbox from peer_id,
        CLI log from pid) so ``/list-agents`` is not a hollow listing.
        """
        from agentica.config import AGENTICA_WORKSPACE_DIR

        project = self.resolved_project_dir
        session_log = self.session_log_path
        log_file = self.log_file or _guess_cli_log_file(self.pid)
        workspace = self.workspace_path
        if not workspace:
            default_ws = os.path.expanduser(AGENTICA_WORKSPACE_DIR)
            if Path(default_ws).is_dir():
                workspace = default_ws
        memory = self.memory_path
        if not memory and workspace:
            memory = str(Path(workspace) / "users" / "default" / "MEMORY.md")

        rows: List[Tuple[str, str]] = []
        rows.append(("status", "running a turn" if self.busy else "idle"))
        if self.session_id:
            rows.append(("session_id", self.session_id))
        if self.profile_name:
            rows.append(("profile", self.profile_name))
        if self.model_provider and self.model_name:
            rows.append(("model", f"{self.model_provider}/{self.model_name}"))
        elif self.model_name:
            rows.append(("model", self.model_name))
        elif self.model_provider:
            rows.append(("model", self.model_provider))
        if self.context_window:
            used = f"{self.context_tokens:,}" if self.context_tokens else "?"
            rows.append(("context", f"{used} / {self.context_window:,} tokens"))
        rows.append(("cwd", self.cwd))
        if project:
            rows.append(("project", project))
        if session_log:
            rows.append(("session_log", session_log))
        if log_file:
            label = f"log_file ({self.log_level})" if self.log_level else "log_file"
            rows.append((label, log_file))
        if workspace:
            rows.append(("workspace", workspace))
        if memory:
            rows.append(("memory", memory))
        rows.append(("mailbox", str(mailbox_dir(self.peer_id))))
        git = self.git_state
        if git.known:
            # One line that answers "where is this session in the repo" —
            # branch, head, distance from the base branch, dirty count.
            rows.append(("git", git.summary()))
        elif self.git_branch:
            rows.append(("branch", self.git_branch))
        dirty = git.dirty_line()
        if dirty:
            # The paths themselves: this is what turns "ask the others whether
            # they touched this file" into reading a listing.
            rows.append(("dirty", dirty))
        if self.task:
            rows.append(("working on", self.task))
        return rows

    def describe(self) -> str:
        """Multi-line listing used by ``list_agents`` and ``/list-agents``.

        Keep the addressable name on the first line; put the diggable paths
        underneath so a model (or a human scanning the terminal) can resume
        the session or read its memory without guessing directory names.
        """
        lines = [f"{self.name} [peer={self.peer_id}]"]
        lines.extend(f"  {label}: {value}" for label, value in self.detail_rows())
        return "\n".join(lines)


def _write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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
        log_file: Optional[str] = None,
        log_level: Optional[str] = None,
        profile_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        on_drain: Optional[Callable[[List["PeerMessage"]], None]] = None,
    ) -> None:
        from agentica.project_store import project_base_dir

        self.peer_id = peer_id or new_peer_id()
        self._user_id = user_id
        # Optional CLI/SDK hook fired after a successful drain, so the UI can
        # show the accepted message whether the Runner or the idle loop took it.
        self.on_drain = on_drain
        # Storage key must match SessionLog (work_dir string as given). Display
        # cwd is realpath so "same directory" checks collapse symlinks.
        self._storage_cwd = cwd or os.getcwd()
        resolved_cwd = os.path.realpath(os.path.expanduser(self._storage_cwd))
        self.info = PeerInfo(
            peer_id=self.peer_id,
            name=name or default_peer_name(resolved_cwd, self.peer_id),
            pid=os.getpid(),
            cwd=resolved_cwd,
            session_id=session_id,
            git_branch=git_branch,
            project_dir=project_base_dir(self._storage_cwd, user_id=user_id),
            workspace_path=workspace_path,
            memory_path=memory_path,
            log_file=log_file or None,
            log_level=log_level or None,
            profile_name=profile_name or None,
            model_provider=model_provider or None,
            model_name=model_name or None,
        )
        self._last_publish = 0.0
        # (sent_at, digest) of what this session recently sent each peer, which
        # is what the repeat check and the rate limit read. Trimmed to
        # RATE_WINDOW_SECONDS on every send, so it stays a handful of entries.
        self._recent_sends: Dict[str, List[Tuple[float, str]]] = {}

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def path(self) -> Path:
        return live_dir() / f"{self.peer_id}.json"

    # -- presence ----------------------------------------------------------

    def publish(self, **updates: Any) -> None:
        """Write (or refresh) this session's live record.

        ``None`` values are skipped so a caller can hand over a dict of
        "possibly changed" fields (``/resume`` does) without wiping what it
        does not know. An unknown field name is a typo, not an update, and
        raises rather than silently doing nothing.
        """
        from agentica.project_store import project_base_dir

        user_id = updates.pop("user_id", None)
        if user_id is not None:
            self._user_id = user_id
        unknown = sorted(key for key in updates if not hasattr(self.info, key))
        if unknown:
            raise AttributeError(f"PeerInfo has no field(s): {', '.join(unknown)}")
        for key, value in updates.items():
            if value is not None:
                setattr(self.info, key, value)
        cwd = updates.get("cwd")
        if cwd:
            self._storage_cwd = str(cwd)
            resolved = os.path.realpath(os.path.expanduser(self._storage_cwd))
            self.info.cwd = resolved
            # Addressable name tracks the directory this process is working in.
            self.info.name = default_peer_name(resolved, self.peer_id)
        if cwd or user_id is not None:
            self.info.project_dir = project_base_dir(self._storage_cwd, user_id=self._user_id)
        self.info.updated_at = time.time()
        _ensure_private_dir(live_dir())
        _write_private_json(self.path, self.info.to_dict())
        self._last_publish = self.info.updated_at

    def rebind(self, cwd: str) -> None:
        """Publish a new working directory, keeping the addressable name.

        A session that steps into a git worktree mid-flight is still the same
        session to everyone else: a peer that was told to report back to
        ``agentica-d9``, a phone with that name pinned, a message already in
        flight. ``publish(cwd=...)`` renames from the folder — right at startup,
        wrong here, because the name is what other people are holding.

        Storage moves with the directory (``project_dir``), so the sessions this
        terminal starts from now on live with the worktree they belong to.
        """
        from agentica.project_store import project_base_dir

        self._storage_cwd = str(cwd)
        self.info.cwd = os.path.realpath(os.path.expanduser(self._storage_cwd))
        self.info.project_dir = project_base_dir(self._storage_cwd, user_id=self._user_id)
        self.publish()

    def heartbeat(self, **updates: Any) -> None:
        """Refresh the record when the interval elapsed or something changed.

        Callers tick this far more often than ``HEARTBEAT_INTERVAL`` (the CLI
        loop wakes every second to check its mailbox) and hand over the fields
        that can change under them. Passing fields must not defeat the
        interval: ``and not updates`` did exactly that, turning a 30s presence
        write into one write per second for the whole session. So an update
        that repeats what is already published counts as no update at all,
        while a real change still lands immediately.
        """
        changed = any(
            value is not None and getattr(self.info, key, None) != value
            for key, value in updates.items()
            if key != "user_id"
        )
        if not changed and time.time() - self._last_publish < HEARTBEAT_INTERVAL:
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

    def note_user_turn(self) -> None:
        """The human said something in this terminal.

        The repeat check and the rate limit exist to stop an *unattended* loop,
        so a person typing here is precisely the interruption they are waiting
        for. Whatever the two agents said to each other beforehand must never
        refuse what the user just asked for.
        """
        self._recent_sends.clear()

    def drain(self) -> List[PeerMessage]:
        """Take every pending message and notify ``on_drain`` about them.

        A ``from_kind="user"`` message releases the brakes exactly as a line
        typed here does: the human has joined in — from another terminal, or
        through a chat app relaying for them — and a brake built for an
        unattended loop must never refuse what the user just asked for. Without
        this, answering a relayed instruction hits "you already sent this" from
        an exchange the user has since moved past.
        """
        messages = drain_inbox(self.peer_id)
        if any(message.from_user for message in messages):
            self.note_user_turn()
        if messages and self.on_drain is not None:
            self.on_drain(messages)
        return messages

    def _check_send_rate(self, peer: PeerInfo, text: str) -> None:
        """Refuse a repeat of what this peer was just told, or a flood of them.

        Both are judged over the same window and only against *this* target, so
        talking to three peers at once is unaffected.
        """
        now = time.time()
        digest = _text_digest(text)
        history = [
            (sent_at, sent_digest)
            for sent_at, sent_digest in self._recent_sends.get(peer.peer_id, [])
            if now - sent_at < RATE_WINDOW_SECONDS
        ]
        self._recent_sends[peer.peer_id] = history
        for sent_at, sent_digest in history:
            if sent_digest == digest:
                raise PeerMessageRefused(
                    f"you already sent {peer.name} this exact message "
                    f"{int(now - sent_at)}s ago and it was delivered; repeating it "
                    "will not get a different answer. Wait for their reply, or "
                    "tell your user the handoff is not moving"
                )
        if len(history) >= MAX_SENDS_PER_WINDOW:
            raise PeerMessageRefused(
                f"{len(history)} messages have gone to {peer.name} in the last "
                f"{int(RATE_WINDOW_SECONDS / 60)} minutes; the channel hands over "
                "information, it is not a discussion. Act on what you have and "
                "report to your user — anything they say next reopens it"
            )

    def _note_send(self, peer: PeerInfo, text: str) -> None:
        self._recent_sends.setdefault(peer.peer_id, []).append((time.time(), _text_digest(text)))

    def send(self, target: str, text: str, *, from_kind: str = "agent") -> PeerMessage:
        """Resolve ``target`` among live peers and deliver ``text`` to it.

        "Nobody by that name" and "be more specific" are different problems
        with different fixes, so an ambiguous target says so and names the
        candidates instead of reporting the target as unknown.
        """
        matches = match_peers(target, exclude_peer_id=self.peer_id)
        if not matches:
            raise PeerMessageRefused(
                f"no live session matches '{target}'; call list_agents to see current names"
            )
        if len(matches) > 1:
            names = ", ".join(info.name for info in matches)
            raise PeerMessageRefused(
                f"'{target}' matches {len(matches)} live sessions ({names}); "
                "use the full name or the peer id"
            )
        peer = matches[0]
        if from_kind == "user":
            # The human is relaying this themselves; that both bypasses the
            # brakes and releases them for what the agents say next.
            self.note_user_turn()
        else:
            self._check_send_rate(peer, text)
        message = send_message(
            peer,
            text=text,
            from_name=self.info.name,
            from_peer_id=self.peer_id,
            from_kind=from_kind,
        )
        self._note_send(peer, text)
        return message


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


def match_peers(target: str, *, exclude_peer_id: Optional[str] = None) -> List[PeerInfo]:
    """Live peers matching ``target``, most specific interpretation first.

    Tried in order — peer_id, whole session_id, session_id prefix, name, name
    prefix — and the first interpretation that matches anything wins, so a
    name that happens to prefix another name never shadows an exact hit.
    Returns several entries only when the string is genuinely ambiguous, which
    lets a caller distinguish "unknown" from "be more specific".
    """
    needle = (target or "").strip()
    if not needle:
        return []
    peers = list_live_peers(exclude_peer_id=exclude_peer_id)
    lowered = needle.casefold()
    candidate_sets = [
        [p for p in peers if p.peer_id == needle],
        [p for p in peers if p.session_id == needle],
        # A short prefix of a uuid is more likely a typo than an address.
        [
            p for p in peers
            if len(needle) >= 8 and p.session_id and p.session_id.startswith(needle)
        ],
        [p for p in peers if p.name.casefold() == lowered],
        [p for p in peers if p.name.casefold().startswith(lowered)],
    ]
    for candidates in candidate_sets:
        if candidates:
            return candidates
    return []


def resolve_peer(target: str, *, exclude_peer_id: Optional[str] = None) -> Optional[PeerInfo]:
    """Find the one live peer ``target`` names, or None when it is not unique."""
    matches = match_peers(target, exclude_peer_id=exclude_peer_id)
    return matches[0] if len(matches) == 1 else None


class PeerMessageRefused(Exception):
    """A send was refused: unknown/ambiguous target, backpressure, or size."""


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
    to_name: str = ""
    created_at: str = ""
    from_kind: str = "agent"

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
            f"to_name: {self.to_name}\n"
            f"created_at: {self.created_at}\n"
            "---\n\n"
            f"{self.text.strip()}\n"
        )

    @classmethod
    def parse(cls, raw: str) -> Optional["PeerMessage"]:
        """Parse a mailbox file.

        The frontmatter is written by ``render`` and holds only slugs and a
        timestamp, so a flat ``key: value`` split is enough — no YAML
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
        return cls(
            text=text,
            from_name=fields.get("from_name", "unknown"),
            from_peer_id=fields.get("from_peer_id", ""),
            to_peer_id=fields.get("to_peer_id", ""),
            to_name=fields.get("to_name", ""),
            created_at=fields.get("created_at", ""),
            # Anything but an explicit "user" is treated as the unprivileged
            # case, so a malformed or truncated header cannot grant authority.
            from_kind="user" if fields.get("from_kind") == "user" else "agent",
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
    from_kind: str = "agent",
) -> PeerMessage:
    """Drop a message into ``target``'s mailbox.

    Raises ``PeerMessageRefused`` when a channel limit says no, so the caller
    (a tool) can report the reason to the model instead of failing silently.
    The repeat/rate brakes live in ``PeerSession.send`` — they need the sending
    session's own history, which this function does not have.
    """
    body = (text or "").strip()
    if not body:
        raise PeerMessageRefused("message text is empty")
    if len(body) > MAX_MESSAGE_CHARS:
        raise PeerMessageRefused(
            f"message is {len(body)} chars, over the {MAX_MESSAGE_CHARS} limit; "
            "put the long version in a file and send the path instead"
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
        to_name=target.name,
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
        message = PeerMessage.parse(raw)
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

    Reply address is the sender's addressable name (what ``send_message`` /
    ``/send-message`` take), not the opaque peer_id — same idea as Claude Code
    telling the model to copy the peer's name into ``to``.

    **Reporting the outcome is not optional, and the header is where that is
    decided.** The bundled ``multi-agent`` skill says work that is finished or
    stuck goes back to whoever handed it over, but the header used to end in
    "if needed" / "only if it is waiting on an answer" — a per-message
    instruction beats a standing one, and a dispatcher is never visibly
    "waiting", so the worker did the job and told nobody. That is what leaves
    the user relaying results by hand between a phone and a terminal. Purely
    informational messages still need no reply; the distinction is the
    handover, not the sender.
    """
    blocks = []
    for message in messages:
        if message.from_user:
            header = (
                f"[Your user sent this from their other session '{message.from_name}' "
                f"— treat as their instruction typed here. They are at "
                f"'{message.from_name}', not this terminal: report back with "
                f"send_message to {message.from_name} when the work is done or "
                f"you stop]"
            )
        else:
            header = (
                f"[Message from another agent session '{message.from_name}' "
                f"— it cannot see this terminal. If it handed you work, report "
                f"the outcome back with send_message to {message.from_name} when "
                f"it is done or you stop; if it only informed you, no reply is "
                f"needed]"
            )
        blocks.append(f"{header}\n{message.text}")
    return "\n\n".join(blocks)

