# -*- coding: utf-8 -*-
"""Session notes that survive a context-window rollover.

Codex TokenBudget (#33255 / history-notes) does not regex-extract a summary.
The model authors notes (here: the existing file tools, path in
``<context_window>``). Auto-compact first injects a fallback prompt and
gives a short buffer so that write can happen. Only if the file is still
empty at the cut do we persist a local transcript digest — a skim of the
dropped span, not a fact list — so the first hop is not Codex #43335.
"""
from pathlib import Path
from typing import List, Optional, Sequence

from agentica.model.message import Message


_PAD_MARK_PREFIX = "[pad "
_LONG_HEAD = 400
_LONG_TAIL = 400
_USER_LINE = 300
_ASSISTANT_LINE = 400
_MAX_USERS = 30
_MAX_ASSISTANTS = 15
_MAX_LONG = 8

# After the 0.95 Layer 2 trigger, wait this extra share of the working
# window for the model to write notes (Codex auto_compact_fallback_buffer).
AUTO_COMPACT_FALLBACK_BUFFER_RATIO = 0.04


def fallback_buffer_tokens(working_window: int) -> int:
    if working_window <= 0:
        return 0
    return max(1, int(working_window * AUTO_COMPACT_FALLBACK_BUFFER_RATIO))


def can_author_notes(functions) -> bool:
    """True when this agent can write the notes file itself."""
    if not functions:
        return False
    return "write_file" in functions or "apply_patch" in functions


def notes_are_ready(notes_path: Optional[str]) -> bool:
    if not notes_path:
        return False
    path = Path(notes_path)
    if not path.is_file():
        return False
    return bool(path.read_text(encoding="utf-8").strip())


def _content(message: Message) -> str:
    raw = message.content
    if isinstance(raw, str):
        return raw
    return str(raw or "")


def _is_pad(text: str) -> bool:
    head = text.lstrip()[:80]
    return head.startswith(_PAD_MARK_PREFIX) or _PAD_MARK_PREFIX in head


def _clip(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head].rstrip() + "\n…\n" + text[-tail:].lstrip()


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    if len(line) > limit:
        return line[:limit] + "…"
    return line


def compose_transcript_digest(messages: Sequence[Message]) -> str:
    """Local skim of the span a window cut is about to drop. Not a summary."""
    users: List[str] = []
    assistants: List[str] = []
    long_excerpts: List[str] = []
    for m in messages:
        if m.role == "system":
            continue
        text = _content(m)
        if not text.strip() or _is_pad(text):
            continue
        if m.role == "user":
            line = _one_line(text, _USER_LINE)
            if line and line not in users:
                users.append(line)
        elif m.role == "assistant" and not m.tool_calls:
            line = _one_line(text, _ASSISTANT_LINE)
            if line and line not in assistants:
                assistants.append(line)
        if len(text) > _LONG_HEAD + _LONG_TAIL:
            excerpt = _clip(text, _LONG_HEAD, _LONG_TAIL)
            long_excerpts.append(
                f"### {m.role} chars={len(text)}\n{excerpt}"
            )

    parts = [
        "# Session notes (transcript digest)",
        "",
        "The model did not update this file before the window reset. "
        "This is a local skim of the dropped span, not a summary. "
        "Search the session log for anything missing.",
        "",
    ]
    if users:
        parts.append("## User turns")
        parts.extend(f"- {x}" for x in users[-_MAX_USERS:])
        parts.append("")
    if assistants:
        parts.append("## Assistant")
        parts.extend(f"- {x}" for x in assistants[-_MAX_ASSISTANTS:])
        parts.append("")
    if long_excerpts:
        parts.append("## Long excerpts")
        parts.extend(long_excerpts[-_MAX_LONG:])
        parts.append("")
    if not users and not assistants and not long_excerpts:
        parts.append("No dropped turns to skim. Search the session log.")
        parts.append("")
    return "\n".join(parts)


def persist_notes(notes_path: str, notes_text: str) -> None:
    """Write a digest only when the model left the file empty. Never clobber."""
    path = Path(notes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = notes_text.strip()
    if not incoming:
        return
    if path.is_file() and path.read_text(encoding="utf-8").strip():
        return
    path.write_text(incoming + "\n", encoding="utf-8")


def ensure_rollover_notes(
    messages: Sequence[Message],
    notes_path: Optional[str],
    window_id: int,
) -> str:
    """Prefer model-authored notes; otherwise persist and return a digest."""
    del window_id  # kept so call sites stay a single signature
    if notes_are_ready(notes_path):
        return Path(notes_path).read_text(encoding="utf-8")
    text = compose_transcript_digest(messages)
    if notes_path:
        persist_notes(notes_path, text)
        path = Path(notes_path)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return text
