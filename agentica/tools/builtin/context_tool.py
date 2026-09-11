# -*- coding: utf-8 -*-
"""Local analogue of Codex history.search_contents + a user-question index.

The history-notes extension (ChatGPT + Codex backend) exposes search and
list against a server store. We have the session JSONL, including rows
before the last compact_boundary. Notes stay on the filesystem; this
tool does not write them.

Keyword search keeps a CJK bigram path so 工单号 hits 工单 ZX-41827.
Every result also carries the newest user questions (role=user, newest
first, capped). That is the browse — not a phrase list that guesses
intent. Empty query returns only the index.
"""
from agentica.tools.base import Tool

_SEARCH_QUERY_DESC = (
    "Literal substring to find in item content (history.search_contents). "
    "A Chinese question is also split into overlapping bigrams: 工单号 matches "
    "工单 ZX-41827. Prefer distinctive tokens — ticket ids, file paths, numbers. "
    "Generic words (the, dump, 什么) are ignored. Hits are ranked by relevance; "
    "equal scores prefer later turns. Every result also includes the newest "
    "user questions (up to 20, truncated). Empty query returns only that index."
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

_READ_ITEM_ID_DESC = (
    "Item id from a search_session hit (the item_id= field). Pass unchanged."
)

_READ_OFFSET_DESC = (
    "Zero-based character offset at which reading starts."
)

_READ_LIMIT_DESC = (
    "Maximum number of characters to return from the item body."
)

_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string", "description": _READ_ITEM_ID_DESC},
        "offset_chars": {
            "type": "integer",
            "description": _READ_OFFSET_DESC,
        },
        "limit_chars": {
            "type": "integer",
            "description": _READ_LIMIT_DESC,
        },
    },
    "required": ["item_id"],
}


class BuiltinContextTool(Tool):
    """JSONL history across compact_boundary. Not a second conversation store."""

    def __init__(self):
        super().__init__(name="builtin_context_tool")
        self._agent = None
        self.register(
            self.search_session,
            concurrency_safe=True,
            is_read_only=True,
            parameters_override=_SEARCH_SCHEMA,
        )
        self.register(
            self.read_session_item,
            concurrency_safe=True,
            is_read_only=True,
            parameters_override=_READ_SCHEMA,
        )

    def set_agent(self, agent) -> None:
        self._agent = agent

    def clone(self) -> "BuiltinContextTool":
        return BuiltinContextTool()

    def _session_log(self):
        agent = self._agent
        if agent is None:
            return None
        return agent._session_log

    def _format_user_questions(self, questions) -> str:
        if not questions:
            return ""
        lines = [f"Recent user questions (newest first, {len(questions)}):"]
        for hit in questions:
            lines.append(f"- item_id={hit['uuid']}: {hit['snippet']}")
        return "\n".join(lines)

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
        Empty query returns only that index.
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
        if not hits:
            head = f"No session-log matches for {q!r}."
            return f"{head}\n\n{q_block}" if q_block else head
        lines = [f"{len(hits)} hit(s) for {q!r}:"]
        for hit in hits:
            lines.append(
                f"- item_id={hit['uuid']} type={hit['type']}: {hit['snippet']}"
            )
        if q_block:
            lines.append("")
            lines.append(q_block)
        return "\n".join(lines)

    async def read_session_item(
        self,
        item_id: str,
        offset_chars: int = 0,
        limit_chars: int = 8000,
    ) -> str:
        """Read one session-log entry by item_id, including pre-boundary rows.

        item_id: id from a search_session hit. Pass unchanged.
        offset_chars / limit_chars: slice the body (Codex history.read_item).
        """
        slog = self._session_log()
        if slog is None:
            return "No session log on this agent; nothing to read."
        key = (item_id or "").strip()
        if not key:
            raise ValueError("item_id cannot be empty.")
        entry = slog.read_entry(key)
        if entry is None:
            return f"No session-log entry {key!r}."
        start = max(0, int(offset_chars))
        cap = max(1, int(limit_chars))
        return slog.format_entry(entry, offset_chars=start, limit_chars=cap)
