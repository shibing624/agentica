# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Per-turn recall of earlier sessions (``<session_recall>``).

Fact recall (``recall.py``) answers "what do I know about this user". This
answers "where did we leave off", in three slots that share one budget:

- **Open items** (first turn of a session): the project's unfinished
  requests from the intent log, newest first. A first message like
  "continue" or "push it" has no content to search with; the open loop is
  the context it refers to.
- **Last asks** (first turn): the final user questions of the most recent
  other sessions in this project, verbatim, from the session index.
- **Related earlier work** (any turn): intents of any project that the
  per-turn recall selector (``recall.py``) picked for this message, from the
  most recent ones. The selector sees each intent's project next to the
  current one and decides; no keyword matching here.

Every line carries a marker (``[intent:12]``, ``[q:996ba8f8]``). A marker
still in the context is not recalled again, and ``search_session`` resolves
the session / uuid on it to the raw transcript, the evidence behind the
summary.
"""
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

from agentica.memory.intents import OPEN_INTENT_DAYS, process_pending_intents
from agentica.memory.recall import display_path
from agentica.memory.session_index import SessionIndex
from agentica.memory.session_search import snippet_head
from agentica.model.message import Message
from agentica.utils.log import logger

SESSION_RECALL_OPEN = "<session_recall>"
SESSION_RECALL_CLOSE = "</session_recall>"

OPEN_ITEMS_LIMIT = 3
RECENT_SESSIONS = 2
RECENT_ASKS_PER_SESSION = 3
RELATED_LIMIT = 3
# Intents offered to the selector, newest first; the selector request keeps
# whole lines within its own budget (recall.SELECT_WORK_MAX_CHARS).
RELATED_CANDIDATES = 60
LINE_MAX_CHARS = 240
RECALL_MAX_CHARS = 2_000

_MARKER_RE = re.compile(r"\[(intent:\d+|q:[0-9a-f]{8})\]")

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


def selector_line(item: Dict[str, Any], current_work_dir: str) -> str:
    """One earlier-work line as the recall selector sees it."""
    project = "this project" if item["work_dir"] == current_work_dir else display_path(item["work_dir"])
    body = item["ask"] + (f" → {item['outcome']}" if item["outcome"] else "")
    return snippet_head(f"{item['id']}: [{project}] [{item['status']}] {_day(item['updated'])} {body}", LINE_MAX_CHARS)


def turn_line(row: Dict[str, Any], current_work_dir: str) -> str:
    tag = _project_tag(row["work_dir"], current_work_dir)
    text = snippet_head(row["content"], LINE_MAX_CHARS)
    return f"- [q:{row['uuid'][:8]}] {_minute(row['ts'])} {tag}(session {row['session_id'][:8]}) {text}"


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


@dataclass
class SessionRecallDraft:
    """``<session_recall>`` before the selector has picked related earlier work.

    ``sections`` are the first-turn slots (open items, last asks);
    ``candidates`` are the intents the selector may pick for this message.
    """

    work_dir: str
    sections: List[tuple] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def selector_items(self) -> List[Dict[str, Any]]:
        return [{"id": c["id"], "line": selector_line(c, self.work_dir)} for c in self.candidates]

    def render(self, picked: Sequence[int]) -> Optional[str]:
        by_id = {c["id"]: c for c in self.candidates}
        related = [intent_line(by_id[i], self.work_dir, with_status=True) for i in picked if i in by_id]
        block = render_session_recall(self.sections + [("Related earlier work:", related)])
        return block or None


async def draft_session_recall(
    index: SessionIndex,
    model: Any,
    *,
    work_dir: str,
    session_id: Optional[str],
    first_turn: bool,
    context: Sequence[Message],
    timeout: float,
) -> SessionRecallDraft:
    """First-turn sections and the related-work candidates for this turn."""
    index.start_backfill()
    if first_turn:
        await drain_stale_pending(index, model, work_dir, session_id, timeout)
    skip = surfaced_markers(context)
    draft = SessionRecallDraft(work_dir=work_dir)
    if first_turn:
        since = time.time() - OPEN_INTENT_DAYS * 86400
        open_items = [
            i for i in index.open_intents(work_dir, since, OPEN_ITEMS_LIMIT + len(skip))
            if f"intent:{i['id']}" not in skip and i["session_id"] != session_id
        ][:OPEN_ITEMS_LIMIT]
        skip.update(f"intent:{i['id']}" for i in open_items)
        draft.sections.append(("Open items in this project:", [intent_line(i, work_dir, False) for i in open_items]))
        # A turn an open item was built from would only repeat it.
        cited = {u[:8] for i in open_items for u in i["uuids"]}
        asks = [
            r for r in index.recent_sessions_questions(work_dir, session_id, RECENT_SESSIONS, RECENT_ASKS_PER_SESSION)
            if f"q:{r['uuid'][:8]}" not in skip and r["uuid"][:8] not in cited
        ]
        draft.sections.append(("Last asks in recent sessions of this project:", [turn_line(r, work_dir) for r in asks]))
    draft.candidates = [
        i for i in index.recent_intents(session_id, RELATED_CANDIDATES) if f"intent:{i['id']}" not in skip
    ]
    return draft


async def draft_session_recall_safe(*args: Any, **kwargs: Any) -> Optional[SessionRecallDraft]:
    """:func:`draft_session_recall` that a broken local index cannot fail the turn with."""
    try:
        return await draft_session_recall(*args, **kwargs)
    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Session recall skipped ({type(e).__name__}: {e})")
        return None
