# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Per-turn long-term memory recall.

The system prompt carries only the session-start MEMORY.md index (title +
hook), frozen so the prompt prefix stays cache-stable. Bodies reach the model
only when it calls ``search_memory`` / ``read_file`` itself — which it skips
exactly when it does not know a relevant memory exists.

Recall pushes them instead: before each human turn an auxiliary model picks
the entries this message needs from the *live* index (so memories written by
other sessions after this one started are eligible), and their bodies are
prepended to the user message inside ``<memory_recall>``. The user's own
question stays at the very bottom — long-context models attend least to the
middle, so the fresh question reads last. Prepending to the newest user
message keeps the conversation append-only, so the provider cache is
untouched; earlier turns' blocks stay byte-identical in history.

The same call also picks earlier work: the most recent intents of every
project (``session_recall``), each shown with its project next to the
current one. Whether another project's work bears on this message is the
selector's call — it reads meaning, where a keyword overlap would offer one
repo's pending push whenever another repo says "push".

Dedup reads the context itself: a file whose block is still in the messages
being sent is not recalled again. After a window rollover drops that block,
the file becomes eligible again — no separate state to reset.
"""
import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agentica.model.message import Message
from agentica.prompts.memory import get_memory_recall_select_prompt
from agentica.utils.log import logger

MEMORY_RECALL_OPEN = "<memory_recall>"
MEMORY_RECALL_CLOSE = "</memory_recall>"

# Budgets for what one turn may add to the user message.
RECALL_MAX_CHARS = 6_000
RECALL_ENTRY_MAX_CHARS = 2_000
# Index text handed to the selector; the on-disk index is capped at 25KB.
SELECT_INDEX_MAX_CHARS = 12_000
# Earlier-work lines handed to the selector, newest kept.
SELECT_WORK_MAX_CHARS = 8_000
SELECT_QUERY_MAX_CHARS = 4_000
# Output cap for the selection call. The reply is a short JSON list, but a
# thinking model spends part of the cap on reasoning first; too small a cap
# leaves no visible answer.
SELECT_MAX_OUTPUT_TOKENS = 2_048
RECALL_MAX_CONSECUTIVE_FAILURES = 3

# Heading ends in the entry's path; the file name is its last segment. Matches
# both the full path written now and the bare ``memory/<file>`` of older
# blocks still in resumed history.
_RECALLED_FILE_RE = re.compile(r"^### .*\((?:[^()\n]*/)?([^/()\n]+)\)$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def surfaced_memory_files(messages: Sequence[Message]) -> Set[str]:
    """Memory files whose recall block is still in ``messages``."""
    surfaced: Set[str] = set()
    for msg in messages:
        if msg.role != "user" or not isinstance(msg.content, str):
            continue
        start = msg.content.find(MEMORY_RECALL_OPEN)
        if start < 0:
            continue
        end = msg.content.find(MEMORY_RECALL_CLOSE, start)
        block = msg.content[start:end if end >= 0 else len(msg.content)]
        surfaced.update(_RECALLED_FILE_RE.findall(block))
    return surfaced


@dataclass
class Selection:
    """What the selector picked: memory filenames and earlier-work ids."""

    files: List[str]
    work: List[int]


def _newest_within(lines: List[str], budget: int) -> List[str]:
    kept: List[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    return kept


def build_selection_request(
    query: str,
    entries: List[Dict[str, str]],
    limit: int,
    work: Sequence[Dict[str, Any]] = (),
    work_limit: int = 0,
    work_dir: str = "",
) -> List[Message]:
    """System + user messages for the selector. Newest lines are kept.

    ``work`` items are ``{"id", "line"}``, newest first, as rendered by
    ``session_recall.selector_line``.
    """
    q = query.strip()
    if len(q) > SELECT_QUERY_MAX_CHARS:
        q = q[-SELECT_QUERY_MAX_CHARS:]
    parts: List[str] = []
    if entries:
        lines = [f"{e['filename']}: {e['title']} — {e['hook']}" for e in entries]
        parts.append("Memories:\n" + "\n".join(_newest_within(lines, SELECT_INDEX_MAX_CHARS)))
    if work:
        lines = _newest_within([w["line"] for w in reversed(work)], SELECT_WORK_MAX_CHARS)
        parts.append(
            f"Earlier work (current project: {display_path(work_dir)}):\n" + "\n".join(reversed(lines))
        )
    parts.append(f"User message:\n{q}")
    return [
        Message(role="system", content=get_memory_recall_select_prompt(limit, work_limit if work else 0)),
        Message(role="user", content="\n\n".join(parts)),
    ]


def _picked(values: Any, allowed: Set, limit: int) -> Optional[List]:
    if not isinstance(values, list):
        return None
    picked: List = []
    for value in values:
        if value in allowed and value not in picked:
            picked.append(value)
        if len(picked) >= limit:
            break
    return picked


def parse_selection(
    text: str,
    allowed: Set[str],
    limit: int,
    allowed_work: Set[int] = frozenset(),
    work_limit: int = 0,
) -> Optional[Selection]:
    """The selector's picks, restricted to what was offered.

    ``None`` when the reply is not a JSON object with a ``files`` list (and a
    ``work`` list when earlier work was offered): the call failed to produce
    a selection. Empty lists when it validly chose nothing. A name or id not
    offered is dropped.
    """
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    files = _picked(data.get("files"), allowed, limit)
    work = _picked(data.get("work"), allowed_work, work_limit) if allowed_work else []
    if files is None or work is None:
        return None
    return Selection(files=files, work=work)


def display_path(origin_work_dir: str) -> str:
    """The origin directory as a path the model can open, home shortened to ``~``.

    Kept whole so the model can check it (``read_file`` / ``glob`` / shell all
    take ``~``); the system prompt already carries the current work_dir in
    full, so this exposes nothing new to the provider.
    """
    home = os.path.expanduser("~")
    if origin_work_dir == home or origin_work_dir.startswith(home + os.sep):
        return "~" + origin_work_dir[len(home):]
    return origin_work_dir


def render_recall_block(items: Sequence[Tuple[str, ...]], memory_dir: str) -> str:
    """``<memory_recall>`` block from ``(title, filename, body[, origin_work_dir])``.

    Each heading carries the entry's full path under ``memory_dir`` so the
    model can open the file as-is; a bare ``memory/<file>`` is relative to
    the user's workspace, not the work_dir, and does not resolve.

    The origin line is provenance, not a filter: memory is recalled per user
    across projects, and the model decides whether a memory written while
    working in another directory applies here.
    """
    parts: List[str] = []
    used = 0
    for item in items:
        title, filename, body = item[0], item[1], item[2]
        origin = item[3] if len(item) > 3 else ""
        if len(body) > RECALL_ENTRY_MAX_CHARS:
            body = body[:RECALL_ENTRY_MAX_CHARS].rstrip() + "\n…[truncated; read the file for the rest]"
        label = display_path(origin) if origin else ""
        origin_line = f"(written while working in {label})\n" if label else ""
        part = f"### {title} ({display_path(os.path.join(memory_dir, filename))})\n{origin_line}{body}"
        if parts and used + len(part) > RECALL_MAX_CHARS:
            break
        parts.append(part)
        used += len(part)
    if not parts:
        return ""
    return f"{MEMORY_RECALL_OPEN}\n" + "\n\n".join(parts) + f"\n{MEMORY_RECALL_CLOSE}"


async def select_recall(
    model,
    query: str,
    entries: List[Dict[str, str]],
    limit: int,
    timeout: float,
    work: Sequence[Dict[str, Any]] = (),
    work_limit: int = 0,
    work_dir: str = "",
) -> Optional[Selection]:
    """Ask ``model`` which memories and earlier work this message needs.

    ``None`` when the call fails, times out, or returns no parseable selection.
    The selector is a network side call that gates the user's turn, so a
    failure never fails the turn.
    """
    side = model.side_call_copy(max_output_tokens=SELECT_MAX_OUTPUT_TOKENS)
    request = build_selection_request(query, entries, limit, work, work_limit, work_dir)
    try:
        call = side.response(messages=request)
        response = await (asyncio.wait_for(call, timeout=timeout) if timeout > 0 else call)
    except asyncio.TimeoutError:
        logger.warning(f"Recall selection timed out after {timeout:.1f}s; turn continues without recall")
        return None
    except Exception as e:
        # Degradation boundary: provider/network errors must not fail the turn.
        logger.warning(f"Recall selection failed ({type(e).__name__}: {e}); turn continues without recall")
        return None
    finally:
        await side.close_client()
    picked = parse_selection(
        response.content or "", {e["filename"] for e in entries}, limit,
        {w["id"] for w in work}, work_limit,
    )
    if picked is None:
        logger.warning("Recall selection returned no parseable selection; turn continues without recall")
    return picked


@dataclass
class RecallBreaker:
    """Per-session circuit breaker for the recall selection call.

    A selector that keeps failing (endpoint down, every call timing out) would
    otherwise add ``recall_timeout`` seconds to every turn. After
    ``RECALL_MAX_CONSECUTIVE_FAILURES`` failures in a row, recall is skipped
    for the rest of the session; a new session id starts closed again. A
    successful selection — including one that picks nothing — resets the count.
    """

    session_id: Optional[str] = None
    failures: int = 0

    def is_open(self, session_id: Optional[str]) -> bool:
        if session_id != self.session_id:
            self.session_id = session_id
            self.failures = 0
        return self.failures >= RECALL_MAX_CONSECUTIVE_FAILURES

    def record(self, succeeded: bool) -> None:
        if succeeded:
            self.failures = 0
            return
        self.failures += 1
        if self.failures == RECALL_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"Memory recall selection failed {self.failures} times in a row; "
                f"recall is off for the rest of session {self.session_id}"
            )


async def build_recall(
    workspace,
    model,
    query: str,
    context: Sequence[Message],
    limit: int,
    timeout: float,
    breaker: RecallBreaker,
    session_id: Optional[str],
    work: Sequence[Dict[str, Any]] = (),
    work_limit: int = 0,
    work_dir: str = "",
) -> Tuple[Optional[str], List[int]]:
    """``(<memory_recall> block or None, picked earlier-work ids)`` for this turn.

    ``workspace`` is None when fact recall is off; ``work`` is empty when
    session recall is off. One selector call covers both.
    """
    if not query.strip() or breaker.is_open(session_id):
        return None, []
    entries: List[Dict[str, str]] = []
    if workspace is not None and limit > 0:
        surfaced = surfaced_memory_files(context)
        entries = [e for e in await workspace.list_memory_index_entries() if e["filename"] not in surfaced]
    if not work_limit:
        work = ()
    if not entries and not work:
        return None, []
    picked = await select_recall(model, query, entries, limit, timeout, work, work_limit, work_dir)
    breaker.record(picked is not None)
    if picked is None or not picked.files:
        return None, picked.work if picked else []
    titles = {e["filename"]: e["title"] for e in entries}
    items: List[Tuple[str, str, str, str]] = []
    for filename in picked.files:
        body, origin = await workspace.read_memory_entry(filename)
        if body:
            items.append((titles[filename], filename, body, origin))
    block = render_recall_block(items, str(workspace.get_user_memory_dir()))
    if block:
        logger.debug(f"Memory recall: {len(items)} entr(ies) -> {[i[1] for i in items]}")
    return block or None, picked.work
