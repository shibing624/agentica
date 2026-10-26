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
appended to the user message inside ``<memory_recall>``. Appending to the
newest user message keeps the conversation append-only, so the provider cache
is untouched; earlier turns' blocks stay byte-identical in history.

Dedup reads the context itself: a file whose block is still in the messages
being sent is not recalled again. After a window rollover drops that block,
the file becomes eligible again — no separate state to reset.
"""
import asyncio
import json
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
SELECT_QUERY_MAX_CHARS = 4_000

_RECALLED_FILE_RE = re.compile(r"^### .*\(memory/([^)\s]+)\)$", re.MULTILINE)
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


def build_selection_request(query: str, entries: List[Dict[str, str]], limit: int) -> List[Message]:
    """System + user messages for the selector. Newest index lines are kept."""
    lines: List[str] = []
    used = 0
    for entry in reversed(entries):
        line = f"{entry['filename']}: {entry['title']} — {entry['hook']}"
        if used + len(line) + 1 > SELECT_INDEX_MAX_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    lines.reverse()
    q = query.strip()
    if len(q) > SELECT_QUERY_MAX_CHARS:
        q = q[-SELECT_QUERY_MAX_CHARS:]
    user = "Memories:\n" + "\n".join(lines) + f"\n\nUser message:\n{q}"
    return [
        Message(role="system", content=get_memory_recall_select_prompt(limit)),
        Message(role="user", content=user),
    ]


def parse_selection(text: str, allowed: Set[str], limit: int) -> List[str]:
    """Filenames the selector returned, restricted to ``allowed``.

    Fails closed: anything that is not a JSON object with a ``files`` list
    selects nothing, and a name not in the offered index is dropped.
    """
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return []
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        return []
    picked: List[str] = []
    for name in files:
        if isinstance(name, str) and name in allowed and name not in picked:
            picked.append(name)
        if len(picked) >= limit:
            break
    return picked


def render_recall_block(items: List[Tuple[str, str, str]]) -> str:
    """``<memory_recall>`` block from ``(title, filename, body)`` triples."""
    parts: List[str] = []
    used = 0
    for title, filename, body in items:
        if len(body) > RECALL_ENTRY_MAX_CHARS:
            body = body[:RECALL_ENTRY_MAX_CHARS].rstrip() + "\n…[truncated; read the file for the rest]"
        part = f"### {title} (memory/{filename})\n{body}"
        if parts and used + len(part) > RECALL_MAX_CHARS:
            break
        parts.append(part)
        used += len(part)
    if not parts:
        return ""
    header = (
        "Long-term memories selected for this message. They were written in "
        "earlier sessions and may be outdated: verify file paths, flags and "
        "versions before relying on them. The current conversation wins on conflict."
    )
    return f"{MEMORY_RECALL_OPEN}\n{header}\n\n" + "\n\n".join(parts) + f"\n{MEMORY_RECALL_CLOSE}"


async def select_memories(
    model,
    query: str,
    entries: List[Dict[str, str]],
    limit: int,
    timeout: float,
) -> List[str]:
    """Ask ``model`` which entries this message needs. Empty on any failure.

    The selector is a network side call that gates the user's turn; when it
    fails or times out the turn proceeds without recall rather than failing.
    """
    side = model.side_call_copy()
    req = build_selection_request(query, entries, limit)
    try:
        call = side.response(messages=req)
        response = await (asyncio.wait_for(call, timeout=timeout) if timeout > 0 else call)
    except asyncio.TimeoutError:
        logger.warning(f"Memory recall selection timed out after {timeout:.1f}s; turn continues without recall")
        return []
    except Exception as e:
        # Degradation boundary: provider/network errors must not fail the turn.
        logger.warning(f"Memory recall selection failed ({type(e).__name__}: {e}); turn continues without recall")
        return []
    finally:
        await side.close_client()
    return parse_selection(response.content or "", {e["filename"] for e in entries}, limit)


async def build_memory_recall(
    workspace,
    model,
    query: str,
    context: Sequence[Message],
    limit: int,
    timeout: float,
) -> Optional[str]:
    """Recall block for this turn, or None when nothing is selected."""
    if limit <= 0 or not query.strip():
        return None
    surfaced = surfaced_memory_files(context)
    entries = [e for e in await workspace.list_memory_index_entries() if e["filename"] not in surfaced]
    if not entries:
        return None
    picked = await select_memories(model, query, entries, limit, timeout)
    if not picked:
        return None
    titles = {e["filename"]: e["title"] for e in entries}
    items: List[Tuple[str, str, str]] = []
    for filename in picked:
        body = await workspace.read_memory_body(filename)
        if body:
            items.append((titles[filename], filename, body))
    block = render_recall_block(items)
    if block:
        logger.debug(f"Memory recall: {len(items)} entr(ies) -> {[i[1] for i in items]}")
    return block or None
