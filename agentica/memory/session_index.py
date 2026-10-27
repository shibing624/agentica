# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cross-session index of user questions and tracked intents.

The session JSONL stays the source of truth; this SQLite file is a derived
read model, one per user at ``<projects>/<user>/session_index.db``:

- ``turns``: every human-written user turn of every session (automated turns
  and our own window preambles excluded), with ``session_id``, ``work_dir``
  and ``uuid`` so a hit leads back to the raw transcript.
- ``intents``: what the user asked for across sessions and where it ended up
  (``open`` / ``done`` / ``dropped``). Written by the intent tracker, read by
  per-turn session recall.
- ``intent_pending``: turns waiting for the intent tracker. Durable, so a
  session that exits before its batch is processed is picked up later.
- ``files``: byte offset already indexed per transcript, so backfill is
  incremental (transcripts are append-only).

Full-text search uses FTS5 over a pre-tokenized ``terms`` column: ASCII
tokens plus overlapping CJK bigrams (the same split ``search_session`` uses),
so 工单号 finds 「工单 ZX-41827」 without a CJK tokenizer.
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from agentica.config.project import projects_root, read_project_file
from agentica.memory.session_search import (
    score_content,
    search_terms,
    strip_window_preamble,
)
from agentica.utils.log import logger

INDEX_FILENAME = "session_index.db"
# Stored text per user turn. Pasted logs can be huge; the ask sits at the end,
# so keep both ends.
TURN_MAX_CHARS = 4_000
# FTS candidates re-scored in Python per query.
FTS_CANDIDATES = 200
INTENT_STATUSES = ("open", "done", "dropped")

_USER_LINE_PREFIX = '{"type": "user"'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    work_dir TEXT NOT NULL,
    ts REAL NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_turns_work_dir ON turns(work_dir, ts);
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(terms, tokenize = 'unicode61');

CREATE TABLE IF NOT EXISTS intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    work_dir TEXT NOT NULL,
    session_id TEXT NOT NULL,
    uuids TEXT NOT NULL,
    ask TEXT NOT NULL,
    outcome TEXT NOT NULL,
    status TEXT NOT NULL,
    artifacts TEXT NOT NULL,
    next_step TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intents_work_dir ON intents(work_dir, status, updated);
CREATE VIRTUAL TABLE IF NOT EXISTS intents_fts USING fts5(terms, tokenize = 'unicode61');

CREATE TABLE IF NOT EXISTS intent_pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    work_dir TEXT NOT NULL,
    ts REAL NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    claim TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_session ON intent_pending(session_id, claim);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    offset INTEGER NOT NULL
);
"""


def index_terms(text: str) -> str:
    """Space-joined FTS terms: the tokens ``search_terms`` yields, minus the whole-text phrase."""
    return " ".join(sorted(t for t in search_terms(text) if not any(ch.isspace() for ch in t)))


def fts_query(terms: Iterable[str]) -> str:
    """OR of quoted terms. Quoting makes punctuation inside a term literal."""
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms if t.strip()]
    return " OR ".join(quoted)


def clip_both_ends(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head].rstrip() + " … " + text[-(limit - head):].lstrip()


def _iso_to_epoch(value: Any) -> float:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def human_user_text(entry: Dict[str, Any]) -> str:
    """Prose a person typed in a ``user`` row, or "" if it is not one."""
    if entry.get("type") != "user" or entry.get("automated") is True:
        return ""
    content = entry.get("content")
    if not isinstance(content, str):
        return ""
    return strip_window_preamble(content).strip()


class SessionIndex:
    """Per-user SQLite read model. Thread-safe through one lock per instance.

    Get the shared instance with :meth:`for_user`; tests may construct one
    directly on a temp path.
    """

    _instances: Dict[str, "SessionIndex"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, db_path: str, projects_dir: Optional[str] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Root whose transcripts ``backfill`` scans; defaults to the db's dir.
        self.projects_dir = Path(projects_dir) if projects_dir else self.db_path.parent
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._backfill_thread: Optional[threading.Thread] = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    @classmethod
    def for_user(cls, user_id: Optional[str] = None) -> "SessionIndex":
        """Process-wide instance for ``user_id``'s projects root."""
        root = projects_root(user_id)
        path = os.path.join(root, INDEX_FILENAME)
        with cls._instances_lock:
            inst = cls._instances.get(path)
            if inst is None:
                inst = cls(path, projects_dir=root)
                cls._instances[path] = inst
            return inst

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        with self._instances_lock:
            for key, inst in list(self._instances.items()):
                if inst is self:
                    del self._instances[key]

    # ------------------------------------------------------------------
    # User turns
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_turn(conn: sqlite3.Connection, uuid: str, session_id: str, work_dir: str, ts: float,
                     content: str) -> None:
        content = clip_both_ends(content, TURN_MAX_CHARS)
        cur = conn.execute(
            "INSERT OR IGNORE INTO turns (uuid, session_id, work_dir, ts, content) VALUES (?, ?, ?, ?, ?)",
            (uuid, session_id, work_dir, ts, content),
        )
        if cur.rowcount:
            conn.execute("INSERT INTO turns_fts (rowid, terms) VALUES (?, ?)", (cur.lastrowid, index_terms(content)))

    def index_user_turn(self, entry: Dict[str, Any]) -> None:
        """Index one session-log ``user`` row (no-op for automated / chrome rows)."""
        text = human_user_text(entry)
        if not text:
            return
        with self._lock:
            self._insert_turn(
                self._conn, entry["uuid"], entry["session_id"], entry.get("cwd") or "",
                _iso_to_epoch(entry.get("timestamp")), text,
            )
            self._conn.commit()

    @staticmethod
    def _turn_row(r: Sequence[Any]) -> Dict[str, Any]:
        return {"uuid": r[0], "session_id": r[1], "work_dir": r[2], "ts": r[3], "content": r[4]}

    def recent_sessions_questions(
        self,
        work_dir: str,
        exclude_session: Optional[str],
        sessions: int,
        per_session: int,
    ) -> List[Dict[str, Any]]:
        """Last ``per_session`` asks of the newest ``sessions`` other sessions in ``work_dir``.

        Grouped by session, newest session first; asks oldest-first inside a
        session so the group reads in order.
        """
        with self._lock:
            sids = [r[0] for r in self._conn.execute(
                "SELECT session_id, MAX(ts) AS last FROM turns WHERE work_dir = ? AND session_id != ? "
                "GROUP BY session_id ORDER BY last DESC LIMIT ?",
                (work_dir, exclude_session or "", sessions),
            )]
            out: List[Dict[str, Any]] = []
            for sid in sids:
                rows = self._conn.execute(
                    "SELECT uuid, session_id, work_dir, ts, content FROM turns WHERE session_id = ? "
                    "ORDER BY ts DESC LIMIT ?",
                    (sid, per_session),
                ).fetchall()
                out.extend(self._turn_row(r) for r in reversed(rows))
        return out

    def search_turns(
        self,
        query: str,
        *,
        work_dir: Optional[str] = None,
        exclude_session: Optional[str] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """User turns ranked by ``score_content`` (same scoring as ``search_session``)."""
        q = (query or "").strip()
        terms = search_terms(q)
        match = fts_query(terms)
        if not match:
            return []
        sql = (
            "SELECT t.uuid, t.session_id, t.work_dir, t.ts, t.content FROM turns_fts f "
            "JOIN turns t ON t.id = f.rowid WHERE turns_fts MATCH ?"
        )
        params: List[Any] = [match]
        if work_dir:
            sql += " AND t.work_dir = ?"
            params.append(work_dir)
        if exclude_session:
            sql += " AND t.session_id != ?"
            params.append(exclude_session)
        sql += " ORDER BY bm25(turns_fts) LIMIT ?"
        params.append(FTS_CANDIDATES)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        scored = []
        for r in rows:
            row = self._turn_row(r)
            row["score"] = score_content(row["content"], q, terms)
            if row["score"] > 0:
                scored.append(row)
        scored.sort(key=lambda x: (-x["score"], -x["ts"]))
        return scored[:limit]

    def find_turn(self, uuid_prefix: str) -> List[Dict[str, Any]]:
        if not uuid_prefix:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT uuid, session_id, work_dir, ts, content FROM turns WHERE uuid LIKE ? LIMIT 5",
                (uuid_prefix.replace("%", "") + "%",),
            ).fetchall()
        return [self._turn_row(r) for r in rows]

    def turn_uuid(self, session_id: str, text: str) -> Optional[str]:
        """Newest uuid of ``session_id``'s user turn whose stored text is ``text``."""
        stored = clip_both_ends(text.strip(), TURN_MAX_CHARS)
        with self._lock:
            row = self._conn.execute(
                "SELECT uuid FROM turns WHERE session_id = ? AND content = ? ORDER BY ts DESC LIMIT 1",
                (session_id, stored),
            ).fetchone()
        return row[0] if row else None

    def count_turns(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    # ------------------------------------------------------------------
    # Backfill from transcripts
    # ------------------------------------------------------------------

    def backfill(self) -> int:
        """Index user turns of every transcript under ``projects_dir``. Incremental.

        Uses its own connection so it can run on a worker thread while the
        shared connection keeps serving queries. Returns rows inserted.
        """
        if not self.projects_dir.is_dir():
            return 0
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        inserted = 0
        try:
            for project in self.projects_dir.iterdir():
                if not project.is_dir():
                    continue
                default_cwd = read_project_file(project).get("work_dir") or ""
                for path in project.glob("*.jsonl"):
                    inserted += self._backfill_file(conn, path, default_cwd)
        finally:
            conn.close()
        return inserted

    def _backfill_file(self, conn: sqlite3.Connection, path: Path, default_cwd: str) -> int:
        key = str(path)
        row = conn.execute("SELECT offset FROM files WHERE path = ?", (key,)).fetchone()
        offset = row[0] if row else 0
        size = path.stat().st_size
        if size < offset:
            offset = 0  # truncated / rewritten transcript
        if size == offset:
            return 0
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(size - offset)
        end = chunk.rfind(b"\n")
        if end < 0:
            return 0
        inserted = 0
        for raw in chunk[:end].split(b"\n"):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith(_USER_LINE_PREFIX):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = human_user_text(entry)
            if not text or not entry.get("uuid"):
                continue
            before = conn.total_changes
            self._insert_turn(
                conn, entry["uuid"], entry.get("session_id") or path.stem,
                entry.get("cwd") or default_cwd, _iso_to_epoch(entry.get("timestamp")), text,
            )
            inserted += 1 if conn.total_changes > before else 0
        conn.execute(
            "INSERT INTO files (path, offset) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET offset = excluded.offset",
            (key, offset + end + 1),
        )
        conn.commit()
        return inserted

    def start_backfill(self) -> None:
        """Run :meth:`backfill` once per instance on a daemon thread."""
        with self._lock:
            if self._backfill_thread is not None:
                return
            self._backfill_thread = threading.Thread(
                target=self._backfill_worker, name="agentica-session-index-backfill", daemon=True,
            )
            self._backfill_thread.start()

    def _backfill_worker(self) -> None:
        started = time.time()
        try:
            n = self.backfill()
        except (OSError, sqlite3.Error) as e:
            # Background I/O boundary: a missing / locked file must not kill
            # the process; per-turn dual-write keeps the index growing.
            logger.warning(f"Session index backfill stopped: {e}")
            return
        logger.debug(f"Session index backfill: {n} user turn(s) in {time.time() - started:.1f}s")

    def wait_backfill(self, timeout: Optional[float] = None) -> None:
        thread = self._backfill_thread
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------
    # Intents
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_terms(ask: str, outcome: str, next_step: str, artifacts: List[str]) -> str:
        return index_terms(" ".join([ask, outcome, next_step, " ".join(artifacts)]))

    @staticmethod
    def _intent_row(r: Sequence[Any]) -> Dict[str, Any]:
        return {
            "id": r[0], "created": r[1], "updated": r[2], "work_dir": r[3], "session_id": r[4],
            "uuids": json.loads(r[5]), "ask": r[6], "outcome": r[7], "status": r[8],
            "artifacts": json.loads(r[9]), "next_step": r[10],
        }

    _INTENT_COLS = "id, created, updated, work_dir, session_id, uuids, ask, outcome, status, artifacts, next_step"

    def add_intent(
        self,
        *,
        work_dir: str,
        session_id: str,
        ask: str,
        outcome: str = "",
        status: str = "open",
        artifacts: Optional[List[str]] = None,
        next_step: str = "",
        uuids: Optional[List[str]] = None,
        now: Optional[float] = None,
    ) -> int:
        if status not in INTENT_STATUSES:
            raise ValueError(f"status must be one of {INTENT_STATUSES}")
        ts = now if now is not None else time.time()
        arts = list(artifacts or [])
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO intents (created, updated, work_dir, session_id, uuids, ask, outcome, status, "
                "artifacts, next_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, ts, work_dir, session_id, json.dumps(uuids or []), ask, outcome, status,
                 json.dumps(arts, ensure_ascii=False), next_step),
            )
            intent_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO intents_fts (rowid, terms) VALUES (?, ?)",
                (intent_id, self._intent_terms(ask, outcome, next_step, arts)),
            )
            self._conn.commit()
        return intent_id

    def get_intent(self, intent_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._INTENT_COLS} FROM intents WHERE id = ?", (intent_id,)
            ).fetchone()
        return self._intent_row(r) if r else None

    def update_intent(
        self,
        intent_id: int,
        *,
        outcome: Optional[str] = None,
        status: Optional[str] = None,
        next_step: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
        add_uuids: Optional[List[str]] = None,
        now: Optional[float] = None,
    ) -> bool:
        current = self.get_intent(intent_id)
        if current is None:
            return False
        if status is not None and status not in INTENT_STATUSES:
            raise ValueError(f"status must be one of {INTENT_STATUSES}")
        merged = {
            "outcome": current["outcome"] if outcome is None else outcome,
            "status": current["status"] if status is None else status,
            "next_step": current["next_step"] if next_step is None else next_step,
            "artifacts": current["artifacts"] if artifacts is None else
            list(dict.fromkeys(current["artifacts"] + list(artifacts))),
            "uuids": list(dict.fromkeys(current["uuids"] + list(add_uuids or []))),
        }
        with self._lock:
            self._conn.execute(
                "UPDATE intents SET updated = ?, outcome = ?, status = ?, next_step = ?, artifacts = ?, uuids = ? "
                "WHERE id = ?",
                (now if now is not None else time.time(), merged["outcome"], merged["status"],
                 merged["next_step"], json.dumps(merged["artifacts"], ensure_ascii=False),
                 json.dumps(merged["uuids"]), intent_id),
            )
            self._conn.execute("DELETE FROM intents_fts WHERE rowid = ?", (intent_id,))
            self._conn.execute(
                "INSERT INTO intents_fts (rowid, terms) VALUES (?, ?)",
                (intent_id, self._intent_terms(current["ask"], merged["outcome"], merged["next_step"],
                                               merged["artifacts"])),
            )
            self._conn.commit()
        return True

    def open_intents(self, work_dir: str, since: float, limit: int) -> List[Dict[str, Any]]:
        """``open`` intents of ``work_dir`` updated after ``since``, newest first."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._INTENT_COLS} FROM intents WHERE work_dir = ? AND status = 'open' "
                "AND updated >= ? ORDER BY updated DESC LIMIT ?",
                (work_dir, since, limit),
            ).fetchall()
        return [self._intent_row(r) for r in rows]

    def search_intents(
        self,
        query: str,
        *,
        exclude_session: Optional[str] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Intents across all projects ranked by ``score_content`` over ask/outcome/next step."""
        q = (query or "").strip()
        terms = search_terms(q)
        match = fts_query(terms)
        if not match:
            return []
        sql = (
            f"SELECT {', '.join('i.' + c.strip() for c in self._INTENT_COLS.split(','))} FROM intents_fts f "
            "JOIN intents i ON i.id = f.rowid WHERE intents_fts MATCH ?"
        )
        params: List[Any] = [match]
        if exclude_session:
            sql += " AND i.session_id != ?"
            params.append(exclude_session)
        sql += " ORDER BY bm25(intents_fts) LIMIT ?"
        params.append(FTS_CANDIDATES)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        scored = []
        for r in rows:
            row = self._intent_row(r)
            blob = " ".join([row["ask"], row["outcome"], row["next_step"], " ".join(row["artifacts"])])
            row["score"] = score_content(blob, q, terms)
            if row["score"] > 0:
                scored.append(row)
        scored.sort(key=lambda x: (-x["score"], -x["updated"]))
        return scored[:limit]

    # ------------------------------------------------------------------
    # Durable buffer for the intent tracker
    # ------------------------------------------------------------------

    def add_pending(self, session_id: str, work_dir: str, user_text: str, assistant_text: str) -> int:
        """Queue one finished turn; returns this session's unclaimed queue length."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO intent_pending (session_id, work_dir, ts, user_text, assistant_text) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, work_dir, time.time(), user_text, assistant_text),
            )
            self._conn.commit()
            return self._conn.execute(
                "SELECT COUNT(*) FROM intent_pending WHERE session_id = ? AND claim IS NULL", (session_id,)
            ).fetchone()[0]

    def stale_pending_sessions(self, work_dir: str, exclude_session: Optional[str]) -> List[str]:
        """Other sessions of ``work_dir`` that left turns unprocessed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM intent_pending WHERE work_dir = ? AND session_id != ? "
                "AND claim IS NULL",
                (work_dir, exclude_session or ""),
            ).fetchall()
        return [r[0] for r in rows]

    def claim_pending(self, session_id: str, token: str) -> List[Dict[str, Any]]:
        """Atomically take this session's unclaimed turns (oldest first)."""
        with self._lock:
            self._conn.execute(
                "UPDATE intent_pending SET claim = ? WHERE session_id = ? AND claim IS NULL",
                (token, session_id),
            )
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT work_dir, ts, user_text, assistant_text FROM intent_pending WHERE claim = ? ORDER BY id",
                (token,),
            ).fetchall()
        return [{"work_dir": r[0], "ts": r[1], "user": r[2], "assistant": r[3]} for r in rows]

    def finish_claim(self, token: str, succeeded: bool) -> None:
        """Drop processed turns, or release them for a later retry."""
        with self._lock:
            if succeeded:
                self._conn.execute("DELETE FROM intent_pending WHERE claim = ?", (token,))
            else:
                self._conn.execute("UPDATE intent_pending SET claim = NULL WHERE claim = ?", (token,))
            self._conn.commit()
