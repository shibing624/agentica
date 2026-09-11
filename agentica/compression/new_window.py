# -*- coding: utf-8 -*-
"""Replace the activity window with a fresh context window (no LLM summary).

Codex #29743 / compact_token_budget.rs: /compact and auto-compact install
initial context locally instead of asking a summarizer. Same session, same
JSONL; a compact_boundary with an empty summary is the cut.

Keeps the two agentica invariants Layer 2 already had:
- the system prompt (otherwise the rest of the turn has no instructions)
- the trailing user turn on auto-compact (otherwise the pending question
  vanishes and providers reject an assistant-prefill continuation)

``keep_trailing_turn=False`` drops the pending question (tests / a caller
that already persisted notes and wants a blank page). Production auto and
``/compact`` keep the tail.

Budget / notes text is folded into a single user message so we never emit
consecutive user roles (Bedrock and some gateways 400 on that).
"""
from pathlib import Path
from typing import List, Optional

from agentica.compression.evict import trailing_user_turn_start
from agentica.compression.token_budget import full_window_text
from agentica.model.message import Message


def notes_path_for(session_log) -> Optional[str]:
    """``<session_id>.notes.md`` next to the JSONL. Not a second transcript."""
    if session_log is None:
        return None
    path = getattr(session_log, "path", None)
    if path is None:
        return None
    return str(Path(path).with_name(f"{Path(path).stem}.notes.md"))


def notes_excerpt(
    notes_path: Optional[str],
    limit: int = 4000,
    notes_text: Optional[str] = None,
) -> Optional[str]:
    """Inject existing notes into a new window so the first request is not empty.

    Codex issue #43335: a new window that only names the notes file leaves
    the first LLM call with no task state.
    """
    text = notes_text
    if not text and notes_path:
        path = Path(notes_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
    if not text:
        return None
    if not text.strip():
        return None
    if len(text) > limit:
        text = text[:limit] + "\n…[truncated]"
    label = notes_path or "session.notes.md"
    return f"<session_notes path=\"{label}\">\n{text}\n</session_notes>"


def _preamble(
    window_id: int,
    tokens_left: int,
    notes_path: Optional[str],
    *,
    continuation: bool,
    notes_text: Optional[str] = None,
) -> str:
    parts = [full_window_text(window_id, tokens_left, notes_path)]
    excerpt = notes_excerpt(notes_path, notes_text=notes_text)
    if excerpt:
        parts.append(excerpt)
    if continuation:
        parts.append(
            "New context window started without a conversation summary. "
            "Continue from session notes and search_session."
        )
    return "\n\n".join(parts)


def start_new_context_window(
    messages: List[Message],
    *,
    window_id: int,
    tokens_left: int,
    notes_path: Optional[str] = None,
    notes_text: Optional[str] = None,
    keep_trailing_turn: bool = True,
) -> List[Message]:
    """Rewrite ``messages`` in place into the new window. Returns the tail kept."""
    preserved_system = [m for m in messages if m.role == "system"]
    preserved_tail: List[Message] = []
    if keep_trailing_turn:
        tail_start = trailing_user_turn_start(messages)
        preserved_tail = [m for m in messages[tail_start:] if m.role != "system"]

    preamble = _preamble(
        window_id,
        tokens_left,
        notes_path,
        continuation=not keep_trailing_turn,
        notes_text=notes_text,
    )

    rebuilt = list(preserved_system)
    if preserved_tail and preserved_tail[0].role == "user":
        first = preserved_tail[0]
        content = first.content if isinstance(first.content, str) else str(first.content or "")
        first.content = f"{preamble}\n\n{content}"
        rebuilt.extend(preserved_tail)
    else:
        rebuilt.append(Message(role="user", content=preamble))
        rebuilt.extend(preserved_tail)

    messages.clear()
    messages.extend(rebuilt)
    return preserved_tail
