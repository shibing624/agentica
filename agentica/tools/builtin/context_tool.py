# -*- coding: utf-8 -*-
"""Search this session's JSONL, including rows before compact_boundary.

Codex's history-notes extension (ChatGPT backend, default off) has
search_contents plus a read_item pager for huge server-side items. We
keep search only: snippets plus a newest-user-question index. A local
JSONL row is usually shorter than the snippet; a second tool that
reprints type/timestamp/content is schema tax that invites reading
「hi」. Notes stay on the filesystem; this tool does not write them.
"""
import weakref
from pathlib import Path
from typing import Optional

from agentica.compression.new_window import notes_path_for
from agentica.memory.session_search import (
    score_content,
    search_terms,
    snippet_for,
)
from agentica.memory.session_log import local_turn_stamp
from agentica.tools.base import Tool

_SEARCH_QUERY_DESC = (
    "Literal substring to find in item content. "
    "A Chinese question is also split into overlapping bigrams: 工单号 matches "
    "工单 ZX-41827. Prefer distinctive tokens — ticket ids, file paths, numbers. "
    "Generic words (the, dump, 什么) are ignored. Hits are ranked by relevance; "
    "equal scores prefer later turns. Also searches model-authored "
    "<session>.notes.md (standing state, not a transcript copy). "
    "Every result includes the newest user questions (up to 20, truncated, "
    "with timestamps). Empty query returns only that index."
)

_SEARCH_ROLE_DESC = (
    "Restrict keyword hits to this role. Omit to search user, assistant, "
    "and tool. Does not change the user-question index."
)

_SEARCH_LIMIT_DESC = (
    "Maximum number of matching items to return, 1-20."
)

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
    },
}


class BuiltinContextTool(Tool):
    """JSONL history across compact_boundary. Not a second conversation store."""

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
        if stamp:
            return f"- {stamp} {body}"
        return f"- {body}"

    def _format_user_questions(self, questions) -> str:
        if not questions:
            return ""
        lines = [f"Recent user questions (newest first, {len(questions)}):"]
        for hit in questions:
            lines.append(self._format_hit_line(hit, with_type=False))
        return "\n".join(lines)

    def _notes_hit(self, query: str):
        slog = self._session_log()
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

    async def search_session(
        self,
        query: str = "",
        role: str = "",
        limit: int = 8,
    ) -> str:
        """Search this session JSONL, including turns before the last compact boundary.

        query: literal substring, plus CJK bigrams so 工单号 matches 工单 ZX-41827.
        Prefer ticket ids, paths, numbers. Hits ranked by relevance.
        Every result includes the newest user questions (up to 20, truncated).
        Also searches the session notes file. Empty query returns only the index.
        role: restrict keyword hits to user | assistant | tool.
        limit: max keyword hits 1-20.
        """
        slog = self._session_log()
        if slog is None:
            return "No session log on this agent; nothing to search."
        q = (query or "").strip()
        cap = min(20, max(1, int(limit)))
        questions = slog.list_user_questions()
        q_block = self._format_user_questions(questions)
        if not q:
            return q_block or "No user questions in this session log."
        hits = slog.search_entries(q, limit=cap, role=role)
        notes = None if (role or "").strip() else self._notes_hit(q)
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
