# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The session list — one reading of which sessions exist and which are shown.

The Web sidebar, the IM ``/project`` and ``/session`` pickers and the IM
default-session choice used to scan the project directories each on their
own. They disagreed: the Web folded sessions older than 30 days but exempted
gateway/IM ones, the IM showed every file on disk ("共有 116 个会话"), and each
inferred ``source`` and the title in its own way. This module is the one place
that answers, from disk facts plus the live state the caller hands in:

* which sessions are listed at all (``visible``);
* which of those the default view shows and how many it folds away;
* every row field (source / origin / title / unread / background jobs).

A surface only renders the rows. It does not re-filter by age, re-derive the
source, or count files.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional

from agentica.config.prefs import normalize_cli_prefs
from agentica.session.log import SessionLog, is_session_unread, iso_timestamp
from agentica.session.rows import strip_image_markers
from agentica.tools.background_processes import count_running_session_jobs

# Per-job sessions scheduled runs wrote before runs were delivered into
# conversations. Their logs are still on disk; they were never conversations.
LEGACY_CRON_SESSION_PREFIX = "scheduled_"

# IM channel values stored in the sidecar ``source``.
IM_SESSION_SOURCES = frozenset({
    "wechat", "feishu", "wecom", "telegram", "dingtalk", "qq", "discord", "slack",
})

# Sources a person converses through on the gateway (Web/Desktop, IM, a cron
# job delivered into a conversation). ``cli`` is listed only when the caller
# asks for CLI sessions. ``sdk`` and ``legacy`` (a log older than the
# ``source`` stamp, which may be an SDK script's) are never listed.
_GATEWAY_SOURCES = frozenset({"gateway", "cron"}) | IM_SESSION_SOURCES

_SYSTEM_TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders")

DEFAULT_RECENT_DAYS = 30
_PREVIEW_CACHE_MAX = 10000
_preview_cache: "OrderedDict[str, tuple]" = OrderedDict()
_preview_lock = threading.Lock()


def is_system_temp_path(path: str) -> bool:
    """Whether ``path`` is inside the OS temp dir (``/tmp``, ``/var/folders``, ``$TMPDIR``).

    ``$AGENTICA_HOME`` is never temp, even when it lives under one (tests, a
    throwaway home): the web's own "temporary workspace" is
    ``$AGENTICA_HOME/tmp/web-chats``, a place the user chose to chat in.
    """
    if not path:
        return False
    try:
        real = os.path.realpath(os.path.expanduser(path))
        home = os.path.realpath(os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica")))
        roots = {os.path.realpath(r) for r in _SYSTEM_TEMP_ROOTS}
        roots.add(os.path.realpath(tempfile.gettempdir()))
    except (OSError, ValueError):
        return False
    if real == home or real.startswith(home + os.sep):
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def im_source_of_session_id(session_id: str) -> Optional[str]:
    """The channel of a legacy ``agent:<id>:<channel>:<chat_id>`` session id."""
    if not session_id or not session_id.startswith("agent:"):
        return None
    parts = session_id.split(":", 3)
    if len(parts) >= 3 and parts[2] in IM_SESSION_SOURCES:
        return parts[2]
    return None


def source_of(entry: Mapping[str, Any]) -> str:
    """Where the latest turn came in. Sidecar first; then what older logs carry."""
    source = entry.get("source")
    if source:
        return str(source)
    from_sid = im_source_of_session_id(str(entry.get("session_id") or ""))
    if from_sid:
        return from_sid
    if entry.get("cli"):
        return "cli"
    if entry.get("last_read_at"):
        # Only the Web marks a session read; a log with no source but a read
        # mark was a Web chat written before sources were stamped.
        return "gateway"
    return "legacy"


def origin_of(entry: Mapping[str, Any], source: str) -> str:
    """Where the session was started (a channel-minted id is its own origin)."""
    return im_source_of_session_id(str(entry.get("session_id") or "")) or entry.get("origin") or source


def session_preview(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """``SessionLog.session_preview`` cached by file version (size, mtime)."""
    path = str(entry["path"])
    version = (int(entry.get("size_bytes") or 0), int(entry.get("mtime_ns") or 0))
    with _preview_lock:
        cached = _preview_cache.get(path)
        if cached is not None and cached[0] == version:
            _preview_cache.move_to_end(path)
            return cached[1]
    preview = SessionLog.session_preview(path)
    with _preview_lock:
        _preview_cache[path] = (version, preview)
        _preview_cache.move_to_end(path)
        while len(_preview_cache) > _PREVIEW_CACHE_MAX:
            _preview_cache.popitem(last=False)
    return preview


def forget_preview(path: str) -> None:
    with _preview_lock:
        _preview_cache.pop(str(path), None)


@dataclass
class LiveState:
    """What only the running process knows, handed in by the caller.

    ``running``: sessions with a turn in flight here (a gateway LiveTurn, an
    agent run lock, a CLI peer that is attached). ``registries``: in-memory
    background registries by session id (jobs started by this process that
    have no record file yet). ``approval_modes``: tiers set in this process
    that the sidecar may not hold yet.
    """

    running: Collection[str] = ()
    registries: Mapping[str, Any] = field(default_factory=dict)
    approval_modes: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Catalog:
    rows: List[Dict[str, Any]]
    #: Listed sessions the ``recent`` view folded away (older than the window),
    #: as ``{session_id, work_dir, mtime}`` only: no transcript was read for them.
    folded: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> List[Dict[str, Any]]:
        """What a surface shows by default: listed and not archived."""
        return [r for r in self.rows if not r["archived"]]

    @property
    def hidden(self) -> int:
        """Sessions a surface's default view does not show (older + archived)."""
        return len(self.folded) + sum(1 for r in self.rows if r["archived"])


def _row(entry: Mapping[str, Any], project_work_dir: str, live: LiveState, bg_running: int) -> Dict[str, Any]:
    sid = entry["session_id"]
    running = sid in live.running
    source = source_of(entry)
    preview = session_preview(entry)
    first_user = strip_image_markers(preview.get("first_user", "") or "")
    cron_title = str(preview.get("cron_job_name") or "").strip()
    first_ts = entry.get("first_timestamp")
    if not first_ts and entry.get("mtime"):
        first_ts = iso_timestamp(entry["mtime"])
    last_ts = entry.get("last_timestamp")
    last_read_at = entry.get("last_read_at")
    return {
        "session_id": sid,
        "name": SessionLog.display_title(name=entry.get("name"), first_user=first_user, cron_job_name=cron_title),
        "preview": first_user or cron_title,
        "user_count": preview.get("user_count", 0),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "last_read_at": last_read_at,
        "mtime": float(entry.get("mtime") or 0),
        "unread": (not running) and is_session_unread(last_ts, last_read_at),
        "size_bytes": int(entry.get("size_bytes") or 0),
        "archived": bool(entry.get("archived")),
        "pinned": bool(entry.get("pinned")),
        "project_dir": entry.get("work_dir") or project_work_dir,
        "work_dir": entry.get("workspace_dir") or entry.get("work_dir") or project_work_dir,
        "running": running,
        "background_jobs": bg_running,
        "source": source,
        "origin": origin_of(entry, source),
        "approval_mode": live.approval_modes.get(sid) or normalize_cli_prefs(entry.get("cli")).get("permissions"),
    }


def _candidates(user_id: Optional[str], work_dir: Optional[str], fallback_base_dir: Optional[str], fallback_work_dir: str):
    """One disk entry per session id across the owner's projects (largest file wins)."""
    if work_dir:
        real = os.path.realpath(os.path.expanduser(work_dir))
        projects = [{"base_dir": None, "work_dir": real}]
    else:
        projects = SessionLog.list_projects(user_id=user_id)
        if not projects and fallback_base_dir:
            projects = [{"base_dir": fallback_base_dir, "work_dir": fallback_work_dir}]
    out: Dict[str, tuple] = {}
    for project in projects:
        project_work_dir = project.get("work_dir") or ""
        for entry in SessionLog.list_sessions(
            base_dir=project.get("base_dir"), work_dir=project_work_dir or None, user_id=user_id,
        ):
            sid = entry["session_id"]
            if sid.startswith(LEGACY_CRON_SESSION_PREFIX):
                continue
            previous = out.get(sid)
            if previous is None or int(entry.get("size_bytes") or 0) > int(previous[0].get("size_bytes") or 0):
                out[sid] = (entry, project_work_dir)
    return out.values()


def list_catalog(
    *,
    user_id: Optional[str],
    work_dir: Optional[str] = None,
    scope: str = "recent",
    recent_days: int = DEFAULT_RECENT_DAYS,
    include_cli: bool = False,
    hide_temp_dirs: bool = False,
    workspace_dir: str = "",
    live: Optional[LiveState] = None,
    fallback_base_dir: Optional[str] = None,
    now: Optional[float] = None,
) -> Catalog:
    """The session list for ``user_id`` (one project when ``work_dir`` is given).

    Listed at all: sessions with live work; otherwise gateway/IM/cron ones, and
    CLI/legacy ones when ``include_cli``. Never listed: legacy ``scheduled_``
    sessions, unnamed 0-byte stubs without live work, and (``hide_temp_dirs``) sessions
    in a system temp dir outside ``workspace_dir`` that hold no user turn and
    no cron run.

    ``scope="recent"`` folds listed sessions whose log was last written more
    than ``recent_days`` ago into ``hidden``, unless they have live work or are
    pinned or archived (the archive view needs archived ones). ``scope="all"``
    folds nothing. Rows are newest-created first; each carries ``mtime`` for a
    surface that orders by last activity instead.
    """
    if scope not in ("all", "recent"):
        raise ValueError(f"Unknown session scope: {scope}")
    if recent_days < 1:
        raise ValueError("recent_days must be positive")
    live = live or LiveState()
    cutoff = (time.time() if now is None else now) - recent_days * 86400
    keep_dir = os.path.realpath(workspace_dir) if workspace_dir else ""

    def _hidden_temp(path: str) -> bool:
        if not is_system_temp_path(path):
            return False
        real = os.path.realpath(path)
        return not (keep_dir and (real == keep_dir or real.startswith(keep_dir + os.sep)))

    rows: List[Dict[str, Any]] = []
    folded: List[Dict[str, Any]] = []
    for entry, project_work_dir in _candidates(user_id, work_dir, fallback_base_dir, workspace_dir):
        sid = entry["session_id"]
        bg_dir = Path(entry["path"]).parent / sid / "background"
        bg_running = count_running_session_jobs(bg_dir, live.registries.get(sid))
        has_live_work = sid in live.running or bg_running > 0
        source = source_of(entry)
        origin = origin_of(entry, source)
        if not has_live_work:
            # A 0-byte log is a stub something opened and never wrote to; one
            # created on purpose (a scheduled run titled before it starts)
            # already carries its name in the sidecar.
            if int(entry.get("size_bytes") or 0) <= 0 and not entry.get("name"):
                continue
            via_gateway = source in _GATEWAY_SOURCES or origin in _GATEWAY_SOURCES
            if not via_gateway and not (include_cli and source == "cli"):
                continue
            if hide_temp_dirs and _hidden_temp(entry.get("work_dir") or project_work_dir):
                preview = session_preview(entry)
                if not preview.get("user_count") and not preview.get("cron_job_name"):
                    continue
            if (
                scope == "recent"
                and not entry.get("archived")
                and not entry.get("pinned")
                and float(entry.get("mtime") or 0) < cutoff
            ):
                folded.append({
                    "session_id": sid,
                    "work_dir": entry.get("work_dir") or project_work_dir,
                    "mtime": float(entry.get("mtime") or 0),
                })
                continue
        rows.append(_row(entry, project_work_dir, live, bg_running))
    rows.sort(key=lambda row: str(row.get("first_timestamp") or ""), reverse=True)
    return Catalog(rows=rows, folded=folded)
