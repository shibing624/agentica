# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Per-turn recall of earlier sessions (``<session_recall>``).

Fact recall (``recall.py``) answers "what do I know about this user". This
answers "where did we leave off", in three slots with their own retrieval
and budget, none of which calls a model:

- **Open items** (first turn of a session): the project's unfinished
  requests from the intent log, newest first. A first message like
  "continue" or "push it" has no content to search with; the open loop is
  the context it refers to.
- **Last asks** (first turn): the final user questions of the most recent
  other sessions in this project, verbatim, from the session index.
- **Related earlier work** (any turn): intents and earlier user questions
  that share distinctive terms with this message. Other projects only when
  the message names that project's directory, so one repo's pending push is
  not offered while working in another.

Every line carries a marker (``[intent:12]``, ``[q:996ba8f8]``). A marker
still in the context is not recalled again, and ``search_session`` resolves
the session / uuid on it to the raw transcript, the evidence behind the
summary.
"""
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

from agentica.memory.intents import OPEN_INTENT_DAYS, process_pending_intents
from agentica.memory.recall import display_path
from agentica.memory.session_index import SessionIndex, clip_both_ends
from agentica.memory.session_search import search_terms, snippet_head
from agentica.model.message import Message
from agentica.utils.log import logger

SESSION_RECALL_OPEN = "<session_recall>"
SESSION_RECALL_CLOSE = "</session_recall>"

OPEN_ITEMS_LIMIT = 3
RECENT_SESSIONS = 2
RECENT_ASKS_PER_SESSION = 3
RELATED_LIMIT = 3
# Floor on ``match_strength`` for "related": one distinctive token (path, id,
# number), two named words (VPetMac + TalkConfig), or a word plus a couple of
# CJK bigrams. Two incidental bigrams (改动 + 推送) stay below it.
RELATED_MIN_STRENGTH = 3
LINE_MAX_CHARS = 240
TURN_MATCH_CHARS = 400
RECALL_MAX_CHARS = 2_000

_MARKER_RE = re.compile(r"\[(intent:\d+|q:[0-9a-f]{8})\]")
_ASCII_WORD_RE = re.compile(r"^[a-z][a-z0-9]{3,}$")

_HEADER = (
    "Context from this user's earlier sessions. Use it only if it bears on the request. "
    "Before acting on an open item, check the current state (git status, the file, the service): "
    "it may have been done since. search_session(session_id=..., uuid=...) returns the original turn."
)


def surfaced_markers(messages: Sequence[Message]) -> Set[str]:
    """Markers of session-recall lines still present in ``messages``."""
    found: Set[str] = set()
    for msg in messages:
        if msg.role == "user" and isinstance(msg.content, str) and SESSION_RECALL_OPEN in msg.content:
            found.update(_MARKER_RE.findall(msg.content))
    return found


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _minute(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _project_tag(work_dir: str, current: str) -> str:
    return "" if work_dir == current or not work_dir else f"{display_path(work_dir)} · "


def intent_line(item: Dict[str, Any], current_work_dir: str, with_status: bool) -> str:
    stale = item["status"] == "open" and item["updated"] < time.time() - OPEN_INTENT_DAYS * 86400
    body = item["ask"]
    if item["outcome"]:
        body += f" → {item['outcome']}"
    if item["status"] == "open" and item["next_step"] and not stale:
        body += f". Next: {item['next_step']}"
    if with_status:
        body += " [stale: open but untouched for weeks]" if stale else f" [{item['status']}]"
    body = snippet_head(body, LINE_MAX_CHARS)
    tag = _project_tag(item["work_dir"], current_work_dir)
    return f"- [intent:{item['id']}] {_day(item['updated'])} {tag}{body} (session {item['session_id'][:8]})"


def turn_line(row: Dict[str, Any], current_work_dir: str) -> str:
    tag = _project_tag(row["work_dir"], current_work_dir)
    text = snippet_head(row["content"], LINE_MAX_CHARS)
    return f"- [q:{row['uuid'][:8]}] {_minute(row['ts'])} {tag}(session {row['session_id'][:8]}) {text}"


def _names_project(query: str, work_dir: str) -> bool:
    name = os.path.basename(work_dir.rstrip("/")).casefold()
    return len(name) >= 3 and name in query.casefold()


def match_strength(text: str, query: str) -> int:
    """How much of ``query`` shows up in ``text``, per matched token.

    Distinctive tokens of 4+ chars (with digits, ``/``, ``_``, ``-``: ids,
    paths, hashes) count 3, ASCII words of 4+ letters 2, anything else
    (short numbers, short words, CJK bigrams) 1. The whole query as a phrase
    counts 3. Lexical only, no phrase lists.
    """
    blob = text.casefold()
    phrase = query.strip().casefold()
    strength = 3 if phrase and phrase in blob else 0
    for term in search_terms(query):
        if term == phrase or term not in blob:
            continue
        if len(term) >= 4 and (any(ch.isdigit() for ch in term) or any(ch in term for ch in "/_-")):
            strength += 3
        elif _ASCII_WORD_RE.match(term):
            strength += 2
        else:
            strength += 1
    return strength


def select_related(
    index: SessionIndex,
    query: str,
    *,
    work_dir: str,
    session_id: Optional[str],
    skip: Set[str],
) -> List[str]:
    """Related intents first, then earlier user questions of this project."""
    lines: List[str] = []
    for item in index.search_intents(query, exclude_session=session_id, limit=RELATED_LIMIT * 4):
        marker = f"intent:{item['id']}"
        if marker in skip:
            continue
        if item["work_dir"] != work_dir and not _names_project(query, item["work_dir"]):
            continue
        blob = " ".join([item["ask"], item["outcome"], item["next_step"], " ".join(item["artifacts"])])
        if match_strength(blob, query) < RELATED_MIN_STRENGTH:
            continue
        skip.add(marker)
        lines.append(intent_line(item, work_dir, with_status=True))
        if len(lines) >= RELATED_LIMIT:
            return lines
    seen_text: Set[str] = {query.strip().casefold()}
    for row in index.search_turns(query, work_dir=work_dir, exclude_session=session_id, limit=RELATED_LIMIT * 4):
        marker = f"q:{row['uuid'][:8]}"
        # A long turn is mostly pasted material with the ask at the end; match
        # on its two ends, or a pasted log would be "related" to everything.
        # A turn that only repeats this message ("git push" again) tells
        # nothing about how it went; intents carry outcomes, raw turns do not.
        ask = clip_both_ends(row["content"], TURN_MATCH_CHARS)
        if marker in skip or ask.strip().casefold() in seen_text:
            continue
        if match_strength(ask, query) < RELATED_MIN_STRENGTH:
            continue
        seen_text.add(ask.strip().casefold())
        skip.add(marker)
        lines.append(turn_line(row, work_dir))
        if len(lines) >= RELATED_LIMIT:
            break
    return lines


def render_session_recall(sections: List[tuple]) -> str:
    """``<session_recall>`` from ``(heading, lines)``; whole lines only, within budget."""
    out: List[str] = []
    used = len(_HEADER)
    for heading, lines in sections:
        kept: List[str] = []
        for line in lines:
            if used + len(line) + len(heading) > RECALL_MAX_CHARS:
                break
            kept.append(line)
            used += len(line) + 1
        if kept:
            used += len(heading)
            out.append(heading)
            out.extend(kept)
    if not out:
        return ""
    return "\n".join([SESSION_RECALL_OPEN, _HEADER, *out, SESSION_RECALL_CLOSE])


async def drain_stale_pending(
    index: SessionIndex,
    model: Any,
    work_dir: str,
    session_id: Optional[str],
    timeout: float,
) -> None:
    """Process turns earlier sessions of this project left queued (exited mid-batch).

    Runs before the first turn's recall so "where did we leave off" reflects
    the last session's tail. Bounded by ``timeout`` overall.
    """
    sids = index.stale_pending_sessions(work_dir, session_id)
    if not sids or model is None:
        return
    deadline = time.monotonic() + timeout if timeout > 0 else None
    for sid in sids:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            logger.debug(f"Session recall: {len(sids)} stale intent batch(es) left for later")
            return
        await process_pending_intents(index, model, sid, remaining or 0)


async def build_session_recall(
    index: SessionIndex,
    model: Any,
    *,
    query: str,
    work_dir: str,
    session_id: Optional[str],
    first_turn: bool,
    context: Sequence[Message],
    timeout: float,
) -> Optional[str]:
    """``<session_recall>`` block for this turn, or None when nothing applies."""
    index.start_backfill()
    if first_turn:
        await drain_stale_pending(index, model, work_dir, session_id, timeout)
    skip = surfaced_markers(context)
    sections: List[tuple] = []
    if first_turn:
        since = time.time() - OPEN_INTENT_DAYS * 86400
        open_items = [
            i for i in index.open_intents(work_dir, since, OPEN_ITEMS_LIMIT + len(skip))
            if f"intent:{i['id']}" not in skip and i["session_id"] != session_id
        ][:OPEN_ITEMS_LIMIT]
        skip.update(f"intent:{i['id']}" for i in open_items)
        sections.append(("Open items in this project:", [intent_line(i, work_dir, False) for i in open_items]))
        # A turn an open item was built from would only repeat it.
        cited = {u[:8] for i in open_items for u in i["uuids"]}
        asks = [
            r for r in index.recent_sessions_questions(work_dir, session_id, RECENT_SESSIONS, RECENT_ASKS_PER_SESSION)
            if f"q:{r['uuid'][:8]}" not in skip and r["uuid"][:8] not in cited
        ]
        skip.update(f"q:{r['uuid'][:8]}" for r in asks)
        sections.append(("Last asks in recent sessions of this project:", [turn_line(r, work_dir) for r in asks]))
    if query.strip():
        sections.append(("Related earlier work:", select_related(
            index, query, work_dir=work_dir, session_id=session_id, skip=skip,
        )))
    block = render_session_recall(sections)
    return block or None


async def session_recall_safe(*args: Any, **kwargs: Any) -> Optional[str]:
    """:func:`build_session_recall` that a broken local index cannot fail the turn with."""
    try:
        return await build_session_recall(*args, **kwargs)
    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Session recall skipped ({type(e).__name__}: {e})")
        return None
