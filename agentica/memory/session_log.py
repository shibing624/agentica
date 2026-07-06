# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Append-only JSONL session log with UUID chain and compact boundary.

Mirrors CC's sessionStorage.ts:
- Entry types use role as type: "user", "assistant", "system", "tool"
- Each entry has uuid + parent_uuid forming a chain
- compact_boundary sets parent_uuid=null to break the chain
- Each entry carries session_id, cwd, version, git_branch
- timestamp uses ISO string format (CC convention)
- Default storage: <AGENTICA_PROJECT_DIR>/<cwd-name>/<session_id>.jsonl
- load() replays from the last compact_boundary
- Large file optimization: only parse bytes after the last boundary

JSONL format (CC-aligned):
    {"type":"user","uuid":"...","parent_uuid":null,"session_id":"...","cwd":"...","timestamp":"2026-04-02T07:32:26.046Z","version":"1.3.3","git_branch":"main","content":"..."}
    {"type":"assistant","uuid":"...","parent_uuid":"<prev>","timestamp":"...","content":"...","model":"gpt-4o","usage":{...}}
    {"type":"compact_boundary","uuid":"...","parent_uuid":null,"timestamp":"...","summary":"..."}
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agentica.utils.log import logger

# Large file optimization threshold (5MB, same as CC's SKIP_PRECOMPACT_THRESHOLD)
_LARGE_FILE_THRESHOLD = 5 * 1024 * 1024


def _get_default_base_dir() -> str:
    """Get default session storage directory: <AGENTICA_PROJECTS_DIR>/<cwd-name>/
    """
    from agentica.compression.tool_result_storage import get_project_dir
    return get_project_dir(os.getcwd())


def _iso_now() -> str:
    """Return current time as ISO 8601 string with milliseconds (CC convention)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class SessionLog:
    """Append-only JSONL session log with UUID chain. Enables session resume.

    Mirrors CC's sessionStorage.ts core design:
    - Each entry has uuid + parent_uuid forming a linked list
    - compact_boundary breaks the chain (parent_uuid=null)
    - Each entry stamped with session_id, cwd, version, git_branch
    - timestamp uses ISO 8601 string format
    - Default path: <AGENTICA_PROJECTS_DIR>/<cwd-name>/<session_id>.jsonl
    - Large files: only read bytes after last compact_boundary
    - load() returns messages ready to inject into WorkingMemory
    """

    def __init__(
        self,
        session_id: str,
        base_dir: Optional[str] = None,
        search_index: Optional[Any] = None,
    ):
        self.session_id = session_id
        self.base_dir = Path(base_dir) if base_dir else Path(_get_default_base_dir())
        self.path = self.base_dir / f"{session_id}.jsonl"
        self.meta_path = self.base_dir / f"{session_id}.meta.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_uuid: Optional[str] = None
        self._cwd: str = os.getcwd()
        self._version: str = self._get_version()
        self._git_branch: Optional[str] = self._get_git_branch()
        # Optional search index for dual-write (FTS5 acceleration).
        # If set, each append() also writes to the search index.
        self._search_index = search_index
        self._search_index_healthy: bool = True

    @staticmethod
    def _get_version() -> str:
        try:
            from agentica.version import __version__
            return __version__
        except Exception:
            return "unknown"

    @staticmethod
    def _get_git_branch() -> Optional[str]:
        import subprocess
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=3,
            )
            branch = result.stdout.strip()
            return branch if branch else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Append operations (write-only, atomic per line)
    # ------------------------------------------------------------------

    def append(self, role: str, content: str, **meta: Any) -> str:
        """Append a message entry. Returns the generated uuid.

        Args:
            role: "user", "assistant", "system", or "tool"
            content: Message content
            **meta: Extra fields (tool_name, tool_call_id, is_error, model, usage, etc.)
        """
        entry_uuid = str(uuid4())
        self._append({
            "type": role,
            "uuid": entry_uuid,
            "parent_uuid": self._last_uuid,
            "session_id": self.session_id,
            "cwd": self._cwd,
            "timestamp": _iso_now(),
            "version": self._version,
            "git_branch": self._git_branch,
            "content": content,
            **meta,
        })
        self._last_uuid = entry_uuid
        self._write_search_index_entry(role, content)
        return entry_uuid

    def append_compact_boundary(self, summary: str) -> str:
        """Mark a compaction boundary. Breaks the UUID chain (parent_uuid=null).

        On resume, all entries before the last boundary are discarded.
        The summary becomes the starting context.
        """
        entry_uuid = str(uuid4())
        self._append({
            "type": "compact_boundary",
            "uuid": entry_uuid,
            "parent_uuid": None,  # breaks the chain — CC convention
            "session_id": self.session_id,
            "cwd": self._cwd,
            "timestamp": _iso_now(),
            "version": self._version,
            "git_branch": self._git_branch,
            "summary": summary,
        })
        self._last_uuid = entry_uuid
        self._write_search_index_entry("compact_boundary", summary)
        return entry_uuid

    # ------------------------------------------------------------------
    # Goal entries (standing-goal loop; see agentica/goals.py)
    # ------------------------------------------------------------------
    # ``type="goal"`` entries snapshot the GoalState. They break the
    # parent_uuid chain (parent_uuid=None) like compact_boundary so they
    # NEVER show up in load() / _build_messages() (which whitelists only
    # user/assistant/system/tool).

    def append_goal(self, goal_state: Any) -> str:
        """Append a goal state snapshot. Returns entry uuid.

        ``goal_state`` may be a ``GoalState`` dataclass or any object with a
        ``to_dict()`` method. We keep the import lazy to avoid a hard cycle
        between memory and goals modules.
        """
        if hasattr(goal_state, "to_dict"):
            payload = goal_state.to_dict()
        elif isinstance(goal_state, dict):
            payload = dict(goal_state)
        else:
            raise TypeError(
                f"append_goal expected GoalState or dict, got {type(goal_state).__name__}"
            )

        entry_uuid = str(uuid4())
        self._append({
            "type": "goal",
            "uuid": entry_uuid,
            "parent_uuid": None,  # break chain like compact_boundary
            "session_id": self.session_id,
            "cwd": self._cwd,
            "timestamp": _iso_now(),
            "version": self._version,
            "git_branch": self._git_branch,
            "goal": payload,
        })
        # Do NOT update _last_uuid — goal entries are out-of-band and must
        # not interfere with the conversation UUID chain.
        return entry_uuid

    def load_goal(self) -> Optional[Dict[str, Any]]:
        """Read the LAST goal entry's payload, or None.

        Tail-reverse scan: streams the file in chunks from the end and
        stops at the first ``"type": "goal"`` line. Big-file friendly —
        a goal-less session pays only one stat() + small tail read.
        """
        if not self.path.exists():
            return None

        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                chunk_size = 64 * 1024
                pos = file_size
                tail = b""
                while pos > 0:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    tail = f.read(read_size) + tail
                    # Scan tail lines from newest to oldest.
                    lines = tail.splitlines()
                    # If we haven't reached SOF, the first line may be partial.
                    start_idx = 0 if pos == 0 else 1
                    for line in reversed(lines[start_idx:]):
                        if b'"type":"goal"' not in line and b'"type": "goal"' not in line:
                            continue
                        try:
                            entry = json.loads(line.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if entry.get("type") == "goal":
                            return entry.get("goal")
                    # Keep partial first line for next iteration.
                    if pos > 0 and lines:
                        tail = lines[0]
                    else:
                        tail = b""
        except OSError as exc:
            logger.warning("SessionLog.load_goal failed (%s): %s", self.path, exc)
            return None

        return None

    # ------------------------------------------------------------------
    # Load / Resume
    # ------------------------------------------------------------------

    def load(self, resume_at: Optional[str] = None) -> List[Dict[str, Any]]:
        """Replay JSONL log for session resume.

        Args:
            resume_at: Optional UUID — truncate the conversation at this message
                       (inclusive). Mirrors CC's --resume-session-at <uuid>.
                       Messages after this UUID are discarded (forms a fork point).

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        if not self.path.exists():
            return []

        file_size = self.path.stat().st_size
        if file_size > _LARGE_FILE_THRESHOLD and resume_at is None:
            return self._load_large_file()

        return self._load_full(resume_at=resume_at)

    def _load_full(self, resume_at: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load entire file (small files < 5MB), optionally truncated at resume_at."""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: List[Dict] = []
        last_boundary_idx = -1
        last_boundary_summary: Optional[str] = None

        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
                if entry.get("type") == "compact_boundary":
                    last_boundary_idx = len(entries) - 1
                    last_boundary_summary = entry.get("summary", "")
            except json.JSONDecodeError:
                continue

        # CC's --resume-session-at: slice(0, target_index + 1)
        if resume_at is not None:
            cut_idx = -1
            for i, e in enumerate(entries):
                if e.get("uuid") == resume_at:
                    cut_idx = i
                    break
            if cut_idx >= 0:
                entries = entries[:cut_idx + 1]
                # Recalculate boundary after truncation
                last_boundary_idx = -1
                last_boundary_summary = None
                for i, e in enumerate(entries):
                    if e.get("type") == "compact_boundary":
                        last_boundary_idx = i
                        last_boundary_summary = e.get("summary", "")

        if entries:
            self._last_uuid = entries[-1].get("uuid")

        return self._build_messages(entries, last_boundary_idx, last_boundary_summary)

    def _load_large_file(self) -> List[Dict[str, Any]]:
        """Large file optimization: only parse lines after the last compact_boundary."""
        last_boundary_offset = -1
        last_boundary_summary: Optional[str] = None

        with open(self.path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if '"compact_boundary"' in line:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "compact_boundary":
                            last_boundary_offset = offset
                            last_boundary_summary = entry.get("summary", "")
                    except json.JSONDecodeError:
                        pass

        entries: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            if last_boundary_offset >= 0:
                f.seek(last_boundary_offset)
                f.readline()  # skip the boundary line itself
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        if entries:
            self._last_uuid = entries[-1].get("uuid")

        return self._build_messages(entries, -1, last_boundary_summary)

    def _build_messages(
        self,
        entries: List[Dict],
        last_boundary_idx: int,
        last_boundary_summary: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build message list from parsed entries."""
        messages: List[Dict[str, Any]] = []

        if last_boundary_summary is not None:
            messages.append({
                "role": "user",
                "content": f"[Resumed session — previous context summary]\n\n{last_boundary_summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have the conversation context. Continuing.",
            })

        replay_fields = (
            "tool_call_id", "tool_calls", "tool_name", "tool_args", "is_error",
            "reasoning_content", "finish_reason", "provider_data", "metrics",
            "model", "usage",
        )
        start_from = last_boundary_idx + 1 if last_boundary_idx >= 0 else 0
        for entry in entries[start_from:]:
            entry_type = entry.get("type", "")
            if entry_type in ("user", "assistant", "system", "tool"):
                msg = {
                    "role": entry_type,
                    "content": entry.get("content", ""),
                }
                for key in replay_fields:
                    if key in entry and entry[key] is not None:
                        msg[key] = entry[key]
                messages.append(msg)

        logger.debug(
            f"SessionLog.load({self.session_id}): "
            f"{len(entries)} post-boundary entries, "
            f"resumed with {len(messages)} messages"
        )
        return messages

    # ------------------------------------------------------------------
    # Session listing (for /resume command)
    # ------------------------------------------------------------------

    @classmethod
    def list_sessions(cls, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available sessions for resume.

        Returns list of dicts sorted by mtime descending (most recent first).
        """
        base = Path(base_dir) if base_dir else Path(_get_default_base_dir())
        if not base.exists():
            return []

        sessions = []
        for f in sorted(base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            session_id = f.stem
            stat = f.stat()
            last_timestamp = None
            try:
                with open(f, "rb") as fh:
                    fh.seek(max(0, stat.st_size - 4096))
                    tail = fh.read().decode("utf-8", errors="replace")
                    lines = tail.strip().splitlines()
                    if lines:
                        last_entry = json.loads(lines[-1])
                        last_timestamp = last_entry.get("timestamp")
            except Exception:
                pass

            meta_path = base / f"{session_id}.meta.json"
            sessions.append({
                "session_id": session_id,
                "path": str(f),
                "size_bytes": stat.st_size,
                "last_timestamp": last_timestamp,
                "name": cls._read_meta_name(meta_path),
                "archived": cls._read_meta_archived(meta_path),
            })

        return sessions

    @classmethod
    def session_preview(cls, path: str, max_chars: int = 200) -> Dict[str, Any]:
        """Lightweight preview of a session file for the /resume picker.

        Returns ``{"first_user": str, "user_count": int}`` — the first user
        message (the task that started the session, truncated) and the number
        of user turns. Reads the file once line-by-line so even multi-MB logs
        stay cheap; malformed lines are skipped. This is what makes the resume
        list show *what* a session was about instead of just an opaque id.
        """
        first_user = ""
        user_count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        entry = json.loads(s)
                    except Exception:
                        continue
                    if entry.get("type") != "user":
                        continue
                    user_count += 1
                    if not first_user:
                        content = entry.get("content") or ""
                        if isinstance(content, str):
                            first_user = content[:max_chars]
        except Exception:
            pass
        return {"first_user": first_user, "user_count": user_count}

    # ---- sidecar metadata (session name) -----------------------------------
    #
    # Each session may have an optional sidecar JSON at
    # ``<base_dir>/<session_id>.meta.json`` carrying user-supplied metadata.
    # Today the only field is ``name`` (a short, human-friendly title shown
    # in /resume picker and /status), but the schema is open so we can add
    # more fields later without touching the JSONL data file.
    #
    # The sidecar is owned by SessionLog so all downstream callers (CLI,
    # TUI, future web UI) get one unified API and never hand-roll the path
    # or JSON shape — that's what
    # ``fix_at_library_layer_for_downstream`` mandates.

    @staticmethod
    def _read_meta(meta_path: Path) -> Dict[str, Any]:
        """Return sidecar metadata, or an empty dict if missing/malformed."""
        try:
            if not meta_path.exists():
                return {}
            with open(meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _read_meta_name(meta_path: Path) -> Optional[str]:
        """Return the ``name`` field from a sidecar file, or ``None`` if the
        file is missing / malformed / empty. Never raises — callers treat
        a missing name as "no name set" and fall back to a preview."""
        try:
            data = SessionLog._read_meta(meta_path)
            name = data.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
            return None
        except Exception:
            return None

    @staticmethod
    def _read_meta_archived(meta_path: Path) -> bool:
        """Return whether a session has been archived in sidecar metadata."""
        return bool(SessionLog._read_meta(meta_path).get("archived"))

    def _write_meta(self, updates: Dict[str, Any]) -> None:
        """Merge sidecar metadata updates and persist atomically."""
        from datetime import datetime, timezone

        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._read_meta(self.meta_path)
        payload.update(updates)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.meta_path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise

    def get_name(self) -> Optional[str]:
        """Return the human-friendly name for THIS session, or ``None``
        if the user has never renamed it."""
        return self._read_meta_name(self.meta_path)

    def set_name(self, name: str) -> None:
        """Set (or overwrite) the human-friendly name for this session.

        Writes ``<session_id>.meta.json`` atomically: write to a temp file
        first then ``os.replace`` so a crash mid-write can never leave a
        corrupt sidecar. Empty / whitespace-only names raise ``ValueError``
        — for "clear the name" callers should use :meth:`clear_name`.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("session name must be a non-empty string")
        name = name.strip()
        self._write_meta({"name": name})

    def set_archived(self, archived: bool = True) -> None:
        """Set the archived flag in sidecar metadata."""
        self._write_meta({"archived": bool(archived)})

    def clear_name(self) -> bool:
        """Delete the sidecar file. Returns ``True`` if a file was removed,
        ``False`` if there was nothing to clear. Never raises on a missing
        file — clearing an already-empty name is a no-op success path from
        the caller's POV."""
        try:
            if self.meta_path.exists():
                payload = self._read_meta(self.meta_path)
                if "name" not in payload:
                    return False
                payload.pop("name", None)
                if payload:
                    from datetime import datetime, timezone

                    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                    tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, ensure_ascii=False, indent=2)
                    os.replace(tmp, self.meta_path)
                else:
                    self.meta_path.unlink()
                return True
        except Exception:
            pass
        return False

    @classmethod
    def rename_session(
        cls,
        session_id: str,
        name: str,
        base_dir: Optional[str] = None,
    ) -> bool:
        """Convenience classmethod to rename a session by id without
        instantiating the full ``SessionLog`` (no JSONL must exist either —
        the sidecar can be written ahead of any logged turn).

        Returns ``True`` on success. Raises ``ValueError`` for empty names.
        """
        base = Path(base_dir) if base_dir else Path(_get_default_base_dir())
        log = cls(session_id=session_id, base_dir=str(base))
        log.set_name(name)
        return True

    @classmethod
    def archive_session(
        cls,
        session_id: str,
        archived: bool = True,
        base_dir: Optional[str] = None,
    ) -> bool:
        """Archive or unarchive a session by id using sidecar metadata."""
        base = Path(base_dir) if base_dir else Path(_get_default_base_dir())
        log = cls(session_id=session_id, base_dir=str(base))
        log.set_archived(archived)
        return True

    # ------------------------------------------------------------------
    # Fork: create a new session branching from a specific message
    # ------------------------------------------------------------------

    def fork(self, new_session_id: str, at_uuid: Optional[str] = None) -> "SessionLog":
        """Fork this session into a new one, optionally truncated at at_uuid.

        Mirrors CC's --fork-session: copies entries (up to at_uuid) into a new
        JSONL file with re-stamped session_id. The original file is untouched
        (append-only principle — old branches stay on disk).

        Args:
            new_session_id: The session_id for the forked session.
            at_uuid: Optional UUID to truncate at (inclusive). None = copy all.

        Returns:
            A new SessionLog instance for the forked session.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Session log not found: {self.path}")

        new_log = SessionLog(
            new_session_id,
            base_dir=str(self.base_dir),
            search_index=self._search_index if self._search_index_healthy else None,
        )

        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Re-stamp session_id (CC convention on fork)
            entry["session_id"] = new_session_id
            new_log._append(entry)
            new_log._last_uuid = entry.get("uuid")
            new_log._index_existing_entry(entry)

            # Stop at the fork point
            if at_uuid and entry.get("uuid") == at_uuid:
                break

        logger.debug(f"Forked session {self.session_id} → {new_session_id}"
                    f"{f' at {at_uuid}' if at_uuid else ''}")
        return new_log

    # ------------------------------------------------------------------
    # List user messages (for query-granularity resume picker)
    # ------------------------------------------------------------------

    def list_user_messages(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List user messages from the session log (most recent first).

        Used by /resume to show resumable query points.

        Returns:
            List of dicts with uuid, content (truncated), timestamp.
        """
        if not self.path.exists():
            return []

        user_msgs = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "user":
                    content = entry.get("content", "")
                    user_msgs.append({
                        "uuid": entry.get("uuid", ""),
                        "content": content[:100] + ("..." if len(content) > 100 else ""),
                        "timestamp": entry.get("timestamp", ""),
                    })
            except json.JSONDecodeError:
                continue

        # Most recent first, limited
        return list(reversed(user_msgs[-limit:]))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Check if the session log file exists."""
        return self.path.exists()

    @property
    def search_index_healthy(self) -> bool:
        """Whether dual-write search indexing is still healthy."""
        return self._search_index_healthy

    def entry_count(self) -> int:
        """Count total entries in the log."""
        if not self.path.exists():
            return 0
        return sum(1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _append(self, entry: Dict) -> None:
        """Append a single JSON entry as a new line (atomic write).

        Disk errors are logged but not propagated — session logging is non-critical.
        """
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"SessionLog write failed ({self.path}): {e}")

    def _index_existing_entry(self, entry: Dict[str, Any]) -> None:
        """Index an existing JSONL entry using the same read-model rules."""
        entry_type = entry.get("type")
        if entry_type in ("user", "assistant", "system", "tool"):
            self._write_search_index_entry(entry_type, entry.get("content", ""))
        elif entry_type == "compact_boundary":
            self._write_search_index_entry("compact_boundary", entry.get("summary", ""))

    def _write_search_index_entry(self, role: str, content: str) -> None:
        """Best-effort dual-write to the search index.

        JSONL remains the canonical session store. If the search index fails,
        we keep the primary write, mark the index unhealthy, and stop dual-write
        attempts for this SessionLog instance to avoid silent repeated drift.
        """
        if self._search_index is None or not self._search_index_healthy:
            return
        try:
            self._search_index.index_message(self.session_id, role, content)
        except Exception as e:
            self._search_index_healthy = False
            logger.warning(
                "Search index write disabled for session %s after failure: %s",
                self.session_id,
                e,
            )
