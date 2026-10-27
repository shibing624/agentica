# -*- coding: utf-8 -*-
"""Search this session's JSONL (including rows before compact_boundary) and earlier sessions.

Codex's history-notes extension (ChatGPT backend, default off) has
search_contents plus a read_item pager for huge server-side items. We
keep search only: snippets plus a newest-user-question index. A local
JSONL row is usually shorter than the snippet; a second tool that
reprints type/timestamp/content is schema tax that invites reading
「hi」. Notes stay on the filesystem; this tool does not write them.

``scope=project|all`` reads the cross-session index (user questions and
tracked work items); ``session_id`` opens another session's transcript, the
raw evidence behind a recalled summary line.
"""
import time
import weakref
from pathlib import Path
from typing import List, Optional

from agentica.compression.new_window import notes_path_for
from agentica.memory.session_log import SessionLog, local_turn_stamp
from agentica.memory.session_recall import intent_line, turn_line
from agentica.memory.session_search import (
    score_content,
    search_terms,
    snippet_for,
)
from agentica.tools.base import Tool

_SEARCH_QUERY_DESC = (
    "Literal substring; a Chinese phrase also matches its bigrams (工单号 finds "
    "工单 ZX-41827). Prefer ids, paths, numbers. Hits are ranked by relevance and "
    "also cover this session's notes file. Single-session results end with the "
    "newest user questions; an empty query returns only those."
)

_SEARCH_ROLE_DESC = "Only hits of this role. tool + uuid recovers an earlier tool result."

_SEARCH_LIMIT_DESC = "Max hits, 1-20."

_SEARCH_UUID_DESC = (
    "Full text of one entry: a uuid or 8-char prefix from a hit, or the tool-call "
    "id in an evicted-result placeholder."
)

_SEARCH_SCOPE_DESC = (
    "session (default) | project: earlier sessions in this directory (user "
    "questions and tracked work items; empty query lists their last asks) | all: "
    "every project. Pass a hit's session id as session_id with uuid for the original."
)

_SEARCH_SESSION_ID_DESC = "Search or read another session (id or prefix) instead of this one."

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": _SEARCH_QUERY_DESC},
        "role": {
            "type": "string",
            "enum": ["user", "assistant", "tool"],
            "description": _SEARCH_ROLE_DESC,
        },
        "limit": {
            "type": "integer",
            "description": _SEARCH_LIMIT_DESC,
        },
        "uuid": {
            "type": "string",
            "description": _SEARCH_UUID_DESC,
        },
        "scope": {
            "type": "string",
            "enum": ["session", "project", "all"],
            "description": _SEARCH_SCOPE_DESC,
        },
        "session_id": {
            "type": "string",
            "description": _SEARCH_SESSION_ID_DESC,
        },
    },
}

# Empty-query listing for scope=project: newest other sessions x their last asks.
_PROJECT_INDEX_SESSIONS = 5
_PROJECT_INDEX_ASKS = 4


class BuiltinContextTool(Tool):
    """JSONL history across compact_boundary and, through the session index, across sessions."""

    def __init__(self):
        super().__init__(name="builtin_context_tool")
        # Weak, like every other tool-held agent back-reference. A plain
        # attribute keeps Agent alive through
        # Agent -> Model -> functions -> Function.entrypoint -> tool -> Agent
        # (``Function.entrypoint`` is a bound method, so it holds the tool),
        # which the memory-leak tests pin.
        self._agent_ref: Optional[weakref.ReferenceType] = None
        self.register(
            self.search_session,
            concurrency_safe=True,
            is_read_only=True,
            parameters_override=_SEARCH_SCHEMA,
        )

    def set_agent(self, agent) -> None:
        self._agent_ref = weakref.ref(agent) if agent is not None else None

    @property
    def _agent(self):
        return self._agent_ref() if self._agent_ref is not None else None

    def clone(self) -> "BuiltinContextTool":
        return BuiltinContextTool()

    def _session_log(self):
        agent = self._agent
        if agent is None:
            return None
        return agent._session_log

    def _format_hit_line(self, hit: dict, *, with_type: bool) -> str:
        stamp = local_turn_stamp(hit.get("timestamp"))
        body = hit["snippet"]
        if with_type:
            body = f"{hit.get('type', '')}: {body}"
        if hit.get("automated"):
            body = f"(auto) {body}"
        prefix = " ".join(p for p in (stamp, f"[{hit['uuid'][:8]}]") if p)
        return f"- {prefix} {body}"

    def _format_user_questions(self, questions) -> str:
        if not questions:
            return ""
        lines = [f"Recent user questions (newest first, {len(questions)}):"]
        for hit in questions:
            lines.append(self._format_hit_line(hit, with_type=False))
        return "\n".join(lines)

    @staticmethod
    def _notes_hit(slog, query: str):
        path = notes_path_for(slog)
        if not path:
            return None
        file = Path(path)
        if not file.is_file():
            return None
        text = file.read_text(encoding="utf-8")
        if not text.strip():
            return None
        terms = search_terms(query)
        if score_content(text, query, terms) <= 0:
            return None
        return snippet_for(text, query, list(terms))

    def _open_session(self, current, needle: str):
        """``(SessionLog, "")`` for the session ``needle`` names, or ``(None, error)``."""
        if current.session_id.startswith(needle):
            return current, ""
        found = SessionLog.find_sessions(needle, user_id=self._agent.user_id)
        if not found:
            return None, f"No session found for id {needle!r}."
        ids = sorted({m["session_id"] for m in found})
        if len(ids) > 1:
            return None, (
                f"Session id {needle!r} matches {len(ids)} sessions ({', '.join(ids[:5])}); "
                "pass a longer prefix."
            )
        return SessionLog(found[0]["session_id"], base_dir=found[0]["base_dir"]), ""

    @staticmethod
    def _search_across(slog, query: str, scope: str, cap: int) -> str:
        index = slog.search_index
        if index is None:
            return "No cross-session index on this agent; only scope=session is available."
        index.start_backfill()
        here = slog.cwd
        project = here if scope == "project" else None
        if not query:
            if project is None:
                return "scope=all needs a query."
            rows = index.recent_sessions_questions(here, slog.session_id, _PROJECT_INDEX_SESSIONS, _PROJECT_INDEX_ASKS)
            if not rows:
                return "No earlier sessions indexed for this project."
            lines = [f"Last asks of the {_PROJECT_INDEX_SESSIONS} most recent other sessions in this project:"]
            lines.extend(turn_line(r, here) for r in rows)
            return "\n".join(lines)
        intents = [
            i for i in index.search_intents(query, exclude_session=slog.session_id, limit=cap * 2)
            if project is None or i["work_dir"] == project
        ][:cap]
        turns = index.search_turns(query, work_dir=project, exclude_session=slog.session_id, limit=cap)
        if not intents and not turns:
            return f"No earlier-session matches for {query!r} (scope={scope})."
        lines: List[str] = []
        if intents:
            lines.append(f"Tracked work items ({len(intents)}):")
            lines.extend(intent_line(i, here, with_status=True) for i in intents)
        if turns:
            if lines:
                lines.append("")
            lines.append(f"User questions ({len(turns)}):")
            lines.extend(turn_line(r, here) for r in turns)
        return "\n".join(lines)

    @staticmethod
    def _read_entry(slog, handle: str, cross: bool) -> str:
        matches = slog.find_entries(handle)
        if not matches and cross and slog.search_index is not None:
            rows = slog.search_index.find_turn(handle)
            if len(rows) == 1:
                row = rows[0]
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["ts"]))
                return (
                    f"{stamp} [{row['uuid']}] user (session {row['session_id']}, {row['work_dir']}):\n"
                    f"{row['content']}\n\nPass session_id={row['session_id'][:8]!r} to search that session."
                )
        if not matches:
            return f"No session-log entry found for uuid {handle!r}."
        if len(matches) > 1:
            ids = ", ".join(m["uuid"] for m in matches[:5])
            return f"uuid {handle!r} matches {len(matches)} entries ({ids}); pass a longer prefix."
        entry = matches[0]
        stamp = local_turn_stamp(entry.get("timestamp"))
        header = " ".join(p for p in (stamp, f"[{entry['uuid']}]", f"{entry['type']}:") if p)
        return f"{header}\n{entry['content']}"

    async def search_session(
        self,
        query: str = "",
        role: str = "",
        limit: int = 8,
        uuid: str = "",
        scope: str = "session",
        session_id: str = "",
    ) -> str:
        """Search this session's transcript (including turns before the last compact boundary) or earlier sessions."""
        current = self._session_log()
        if current is None:
            return "No session log on this agent; nothing to search."
        scope = (scope or "session").strip().lower()
        if scope not in ("session", "project", "all"):
            return "scope must be session, project or all."
        slog = current
        needle = (session_id or "").strip()
        if needle:
            slog, error = self._open_session(current, needle)
            if slog is None:
                return error
        if uuid:
            return self._read_entry(slog, uuid, cross=not needle)
        q = (query or "").strip()
        cap = min(20, max(1, int(limit)))
        if scope != "session" and not needle:
            return self._search_across(current, q, scope, cap)
        questions = slog.list_user_questions()
        q_block = self._format_user_questions(questions)
        if not q:
            return q_block or "No user questions in this session log."
        hits = slog.search_entries(q, limit=cap, role=role)
        notes = None if (role or "").strip() or slog is not current else self._notes_hit(slog, q)
        if not hits and not notes:
            head = f"No session-log matches for {q!r}."
            return f"{head}\n\n{q_block}" if q_block else head
        total = len(hits) + (1 if notes else 0)
        lines = [f"{total} hit(s) for {q!r}:"]
        for hit in hits:
            lines.append(self._format_hit_line(hit, with_type=True))
        if notes:
            lines.append(f"- notes: {notes}")
        if q_block:
            lines.append("")
            lines.append(q_block)
        return "\n".join(lines)
