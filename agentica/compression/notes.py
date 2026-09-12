# -*- coding: utf-8 -*-
"""Standing session notes vs the one-shot dropped-span skim.

Two files, two jobs — do not copy the transcript into notes.md.

- ``<session>.jsonl`` is the log. ``search_session`` reads it.
- ``<session>.notes.md`` is standing state the model writes with the
  existing file tools (goals, constraints, IDs, decisions). Codex
  TokenBudget (#33255) never regex-extracts this and never writes a
  transcript into it.

A local digest exists only for Codex #43335: the first hop after a cut
must not be an empty path. It is injected as ``<dropped_span>``, not
written to notes.md. Writing it there flipped ``notes_are_ready``
(no more fallback nudge). The same skim still rides the preserved
tail into JSONL; ``rank_entries`` scores ``strip_window_preamble``
so that chrome is not a second hit. Digest lines have no timestamp
— ``Message.created_at`` is not the JSONL ``timestamp`` after
``load()``. ``search_session`` may still hit model-authored notes.
"""
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agentica.compression.evict import carries_tool_results, tool_result_blocks
from agentica.memory.session_search import strip_window_preamble
from agentica.model.message import Message


_PAD_MARK_PREFIX = "[pad "
_LONG_HEAD = 400
_LONG_TAIL = 400
_USER_LINE = 300
_ASSISTANT_LINE = 400
_TOOL_ARGS_LINE = 200
_TOOL_RESULT_LINE = 300
_MAX_TURNS = 40
_MAX_TOOLS = 20
# Stay under notes_excerpt's 4000-char inject cap.
_DIGEST_BUDGET = 3600

# These two tags are *injected by us* and carry no user prose: a previous
# digest, and the model's own notes. ``strip_window_preamble`` deliberately
# peels them so the search index does not score our chrome as a second hit.
# The digest needs the opposite — what a previous cut already skimmed is the
# only copy of those turns when there is no session log, so it is carried
# forward instead of discarded.
_DROPPED_SPAN_OPEN = "<dropped_span>"
_DROPPED_SPAN_CLOSE = "</dropped_span>"
_SESSION_NOTES_OPEN = "<session_notes"
_SESSION_NOTES_CLOSE = "</session_notes>"

# After the 0.95 Layer 2 trigger, wait this extra share of the working
# window for the model to write notes (Codex auto_compact_fallback_buffer).
AUTO_COMPACT_FALLBACK_BUFFER_RATIO = 0.04


def _tag_body(text: str, open_tag: str, close_tag: str) -> str:
    start = text.find(open_tag)
    if start < 0:
        return ""
    end = text.find(close_tag, start)
    if end < 0:
        return ""
    return text[start + len(open_tag):end]


def carried_rows(content: str) -> List[str]:
    """Digest rows a previous cut already skimmed, for carry-forward.

    Both bodies are ours, not the user's: ``<dropped_span>`` is the prior
    digest and ``<session_notes>`` the model's file. ``strip_window_preamble``
    peels them on purpose (the search index must not score our chrome), but
    the digest must not, or the only surviving copy of those turns is thrown
    away on the next cut — for an SDK agent with no session log, the copy in
    the prompt *is* the history.

    Only the ``- `` rows are taken, dropping the previous header and section
    titles: re-emitting those would nest one digest inside the next and the
    text would drift into header soup within a few cuts. The rows come back in
    their original order, so prepending keeps the span chronological.
    """
    rows: List[str] = []
    for open_tag, close_tag in (
        (_DROPPED_SPAN_OPEN, _DROPPED_SPAN_CLOSE),
        (_SESSION_NOTES_OPEN, _SESSION_NOTES_CLOSE),
    ):
        body = _tag_body(content, open_tag, close_tag)
        if not body:
            continue
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and stripped != "- ":
                rows.append(stripped)
    return rows


def fallback_buffer_tokens(working_window: int) -> int:
    if working_window <= 0:
        return 0
    return max(1, int(working_window * AUTO_COMPACT_FALLBACK_BUFFER_RATIO))


def can_author_notes(functions, notes_path: Optional[str] = None) -> bool:
    """True when this agent can actually write the notes file.

    Two conditions, not one. The tools must exist (``write_file`` /
    ``apply_patch``), **and** there must be a path to write them to — the path
    comes from ``notes_path_for(agent._session_log)``, which is None for an
    SDK agent built without ``session_id``. Checking only the tools made the
    runner postpone a window cut to ask for a file that had nowhere to go.
    """
    if not functions or not notes_path:
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
    return ""


def _block_text(block) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        inner = block.get("content", block.get("text", ""))
        if isinstance(inner, str):
            return inner
        if isinstance(inner, list):
            return " ".join(_block_text(x) for x in inner)
        return str(inner or "")
    return str(block or "")


def _is_pad(text: str) -> bool:
    head = text.lstrip()[:80]
    return head.startswith(_PAD_MARK_PREFIX) or _PAD_MARK_PREFIX in head


def _clip(text: str, head: int, tail: int, marker: str = "\n…\n") -> str:
    if len(text) <= head + tail:
        return text
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def clip_head_tail(text: str, limit: int, marker: str = "\n…\n") -> str:
    """Keep both ends of an over-long block. The tail carries the ask.

    Head-only clipping lost it: an assistant turn that reasoned out loud and
    ended with 「要我把行号一并订正吗？」 reached the next window as reasoning
    with no question, so the user's "ok" had no visible antecedent. Tool
    results already kept both ends (``_clip``); turns did not.

    The marker is charged against ``limit``, so the result is never longer
    than the caller asked for.
    """
    body = max(2, limit - len(marker))
    head = max(1, body * 2 // 3)
    return _clip(text, head, max(1, body - head), marker)


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    return clip_head_tail(line, limit, marker=" … ")


def _line(body: str) -> str:
    return f"- {body}"


def _tool_call_rows(message: Message) -> List[str]:
    rows: List[str] = []
    for call in message.tool_calls or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            fn = call if isinstance(call, dict) else {}
        name = fn.get("name") or "tool"
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        rows.append(f"{name} args: {_one_line(args, _TOOL_ARGS_LINE)}")
    return rows


def _tool_result_line(name: str, text: str) -> str:
    label = name or "tool"
    if len(text) > _LONG_HEAD + _LONG_TAIL:
        return f"{label} result chars={len(text)}:\n{_clip(text, _LONG_HEAD, _LONG_TAIL)}"
    return f"{label} result: {_one_line(text, _TOOL_RESULT_LINE)}"


def _collect(messages: Sequence[Message]) -> Tuple[List[str], List[str]]:
    turns: List[str] = []
    tools: List[str] = []
    for message in messages:
        if message.role == "system":
            continue
        if carries_tool_results(message):
            if message.role == "tool":
                text = _content(message)
                if text.strip() and not _is_pad(text):
                    tools.append(_line(
                        _tool_result_line(message.tool_name or "tool", text),
                    ))
            for block in tool_result_blocks(message):
                text = _block_text(block)
                if not text.strip() or _is_pad(text):
                    continue
                tools.append(_line(_tool_result_line("tool", text)))
            continue
        if message.role == "user":
            raw = _content(message)
            # Carried rows ride in front of this turn's own prose, keeping the
            # span chronological: earlier windows, then this row's question.
            turns.extend(carried_rows(raw))
            text = strip_window_preamble(raw)
            if not text.strip() or _is_pad(text):
                continue
            turns.append(_line(f"user: {_one_line(text, _USER_LINE)}"))
            continue
        if message.role == "assistant":
            text = _content(message)
            if text.strip() and not _is_pad(text):
                turns.append(_line(f"assistant: {_one_line(text, _ASSISTANT_LINE)}"))
            for row in _tool_call_rows(message):
                tools.append(_line(row))
    return turns, tools


def _fit(lines: List[str], limit: int, budget: int) -> List[str]:
    chosen = lines[-limit:] if len(lines) > limit else list(lines)
    used = sum(len(x) + 1 for x in chosen)
    while chosen and used > budget:
        used -= len(chosen[0]) + 1
        chosen.pop(0)
    return chosen


def compose_transcript_digest(messages: Sequence[Message]) -> str:
    """Local chronological skim of the span a window cut is about to drop."""
    turns, tools = _collect(messages)
    header = [
        "# Dropped span",
        "",
        "Not session notes — that file is still empty. "
        "Chronological skim of what left this window. "
        "Write goals, constraints, IDs, and decisions to the notes file. "
        "Use search_session for anything missing.",
        "",
    ]
    header_size = sum(len(x) + 1 for x in header)
    turn_budget = max(800, _DIGEST_BUDGET - header_size)
    turns = _fit(turns, _MAX_TURNS, turn_budget)
    leftover = max(400, _DIGEST_BUDGET - header_size - sum(len(x) + 1 for x in turns))
    tools = _fit(tools, _MAX_TOOLS, leftover)

    parts = list(header)
    if turns:
        parts.append("## Turns")
        parts.extend(turns)
        parts.append("")
    if tools:
        parts.append("## Tools")
        parts.extend(tools)
        parts.append("")
    if not turns and not tools:
        parts.append("No dropped turns to skim. Use search_session.")
        parts.append("")
    return "\n".join(parts)


def rollover_handover(
    messages: Sequence[Message],
    notes_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """``(model_notes, dropped_span)``. Never writes the digest to disk."""
    if notes_are_ready(notes_path):
        return Path(notes_path).read_text(encoding="utf-8"), None
    return None, compose_transcript_digest(messages)
