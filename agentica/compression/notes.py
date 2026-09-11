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


def _clip(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head].rstrip() + "\n…\n" + text[-tail:].lstrip()


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    if len(line) > limit:
        return line[:limit] + "…"
    return line


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
            text = strip_window_preamble(_content(message))
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
