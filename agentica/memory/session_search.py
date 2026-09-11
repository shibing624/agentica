# -*- coding: utf-8 -*-
"""Keyword search over session-log rows, plus a user-question index.

``search_session`` ranks by independent terms: CJK runs become overlapping
bigrams so 工单号 hits 「工单 ZX-41827」. That is search, not intent
detection — do not add language phrase lists to decide "the caller wants
a browse".

User questions are the index. Every search result carries the newest
user turns (cap + char budget). Empty query returns only that index.
Window preambles (``<context_window>``) are skipped because we injected
them, not because of the words they contain.
"""
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set

from agentica.compression.token_budget import WINDOW_CONTINUATION_MARK

ITEM_ROLES = ("user", "assistant", "tool")

# Newest-first user-question index attached to every search_session result.
USER_QUESTION_LIMIT = 20
USER_QUESTION_SNIPPET_CHARS = 160
USER_QUESTION_BUDGET_CHARS = 2400

_TOKEN = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]+")
_CJK = re.compile(r"^[\u4e00-\u9fff]+$")
_STOP = frozenset({
    "the", "a", "an", "is", "to", "of", "and", "or", "in", "for", "on",
    "this", "that", "with", "from", "what", "which", "how",
    "dump", "read", "background", "unrelated", "noted",
    "什么", "多少", "哪个", "哪些", "怎么", "如何", "这次", "只答",
})
_CONTEXT_WINDOW_OPEN = "<context_window>"


def search_terms(query: str) -> Set[str]:
    """Phrase plus independent tokens. CJK 工单号 → 工单号, 工单, 单号."""
    q = (query or "").strip()
    terms: Set[str] = set()
    if q:
        terms.add(q.casefold())
    for part in _TOKEN.findall(q):
        folded = part.casefold()
        if len(folded) < 2 or folded in _STOP:
            continue
        terms.add(folded)
        if _CJK.fullmatch(part) and len(part) >= 2:
            for i in range(len(part) - 1):
                terms.add(part[i:i + 2])
    if not terms and q:
        terms.add(q.casefold())
    return terms


def _distinctive(term: str) -> bool:
    return any(ch.isdigit() for ch in term) or "/" in term or "_" in term or "-" in term


def normalize_role(role: str) -> str:
    want = (role or "").strip().lower()
    if not want:
        return ""
    if want not in ITEM_ROLES:
        raise ValueError(f"role must be one of {', '.join(ITEM_ROLES)}")
    return want


def _entry_text(entry: Dict) -> str:
    content = entry.get("content", "")
    if not isinstance(content, str):
        return str(content)
    return content


def is_window_preamble(content: str) -> bool:
    return content.lstrip().startswith(_CONTEXT_WINDOW_OPEN)


_INJECTED_CLOSERS = ("</context_window>", "</session_notes>", "</dropped_span>")


def drop_shadowed_by_boundary(entries: List[Dict]) -> List[Dict]:
    """Drop pre-boundary rows that its own preserved tail re-logged.

    ``append_post_compact_messages`` re-appends the preserved tail after a
    ``compact_boundary`` — including a turn the runner already flushed
    mid-flight. ``load()`` replays only post-boundary rows, so the duplicate is
    invisible there, but both copies sit in this list.

    The shadowed row is the last ``user`` row before the boundary whose prose
    matches the first ``user`` row after it and which carries no preamble;
    ``assistant`` / ``tool`` rows in that same span were re-logged with the
    tail. The pre-boundary copies stay in the JSONL; this drops them from
    search/index only. Call this while ``compact_boundary`` rows are still
    in the list (``_conversation_rows``), not again after they are filtered.
    """
    drop: set = set()
    for pos, entry in enumerate(entries):
        if entry.get("type") != "compact_boundary":
            continue
        tail: List[Dict] = []
        for later in entries[pos + 1:]:
            if later.get("type") == "compact_boundary":
                break
            tail.append(later)
        tail_user = next((e for e in tail if e.get("type") == "user"), None)
        if tail_user is None:
            continue
        tail_prose = strip_window_preamble(_entry_text(tail_user))
        if not tail_prose.strip():
            continue
        shadow = next(
            (
                i
                for i in range(pos - 1, -1, -1)
                if entries[i].get("type") == "user"
                and not is_window_preamble(_entry_text(entries[i]))
                and _entry_text(entries[i]).strip() == tail_prose
            ),
            None,
        )
        if shadow is None:
            continue
        drop.add(shadow)
        for i in range(shadow + 1, pos):
            if entries[i].get("type") in ("assistant", "tool"):
                drop.add(i)
    if not drop:
        return entries
    return [e for i, e in enumerate(entries) if i not in drop]


def strip_window_preamble(content: str) -> str:
    """User prose after a folded new-window prefix. Unchanged otherwise.

    ``start_new_context_window`` puts ``<context_window>`` / notes / dropped
    span in front of the preserved user turn. The whole string starts with
    the oil-gauge, so a startswith skip throws away the question. Search
    scores this remainder; a chrome-only row (idle ``/compact``) is empty.
    """
    if not content or not is_window_preamble(content):
        return content
    rest = content
    for closer in _INJECTED_CLOSERS:
        idx = rest.find(closer)
        if idx >= 0:
            rest = rest[idx + len(closer):]
    mark_at = rest.find(WINDOW_CONTINUATION_MARK)
    if mark_at >= 0:
        rest = rest[mark_at + len(WINDOW_CONTINUATION_MARK):]
    return rest.lstrip()


def format_turn_stamp(value) -> str:
    """Compact UTC stamp for search_session hits (JSONL ``timestamp``)."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        if value <= 0:
            return ""
        return datetime.fromtimestamp(float(value), timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
    text = str(value).strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    return text[:16]


def snippet_head(content: str, width: int = USER_QUESTION_SNIPPET_CHARS) -> str:
    """One-line snippet that keeps both ends. The ask sits at the end.

    A long user turn is usually pasted material first and the question last
    (「…贴了一大段日志… 咋办？」). Head-only clipping left the paste and threw
    the question away, so the index that answers 「前面问了啥」 listed the
    setup instead of what was actually asked.
    """
    flat = content.strip().replace("\n", " ")
    if len(flat) <= width:
        return flat
    head = max(1, width * 2 // 3)
    tail = max(1, width - head)
    return flat[:head].rstrip() + " … " + flat[-tail:].lstrip()


def score_content(content: str, query: str, terms: Iterable[str]) -> int:
    blob = content.casefold()
    phrase = query.strip().casefold()
    score = 0
    if phrase and phrase in blob:
        score += 10
    for term in terms:
        if term == phrase:
            continue
        if term in blob:
            score += 3 if _distinctive(term) else 1
    return score


def snippet_for(content: str, query: str, terms: Sequence[str], width: int = 200) -> str:
    blob = content.casefold()
    phrase = query.strip().casefold()
    idx = blob.find(phrase) if phrase else -1
    if idx < 0:
        for term in terms:
            idx = blob.find(term)
            if idx >= 0:
                break
    if idx < 0:
        idx = 0
    start = max(0, idx - 40)
    flat = content.strip().replace("\n", " ")
    out = ("…" if start else "") + flat[start:start + width]
    if start + width < len(flat):
        out += "…"
    return out


def rank_entries(
    entries: List[Dict],
    query: str,
    limit: int,
) -> List[Dict]:
    """Score conversation rows; highest first, then later rows win ties."""
    q = (query or "").strip()
    terms = search_terms(q)
    if not q or not terms:
        return []
    scored: List[tuple] = []
    for i, entry in enumerate(entries):
        content = strip_window_preamble(_entry_text(entry))
        if not content.strip():
            continue
        sc = score_content(content, q, terms)
        if sc <= 0:
            continue
        scored.append((sc, i, entry, content))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    hits: List[Dict] = []
    for sc, _, entry, content in scored[:limit]:
        hits.append({
            "uuid": entry.get("uuid", ""),
            "type": entry.get("type", ""),
            "snippet": snippet_for(content, q, list(terms)),
            "score": sc,
            "timestamp": entry.get("timestamp", ""),
        })
    return hits


def list_user_questions(
    entries: List[Dict],
    limit: int = USER_QUESTION_LIMIT,
    snippet_width: int = USER_QUESTION_SNIPPET_CHARS,
    budget_chars: int = USER_QUESTION_BUDGET_CHARS,
) -> List[Dict]:
    """Newest user turns. Skip only the window preamble we inject."""
    cap = min(USER_QUESTION_LIMIT, max(1, int(limit)))
    hits: List[Dict] = []
    used = 0
    for entry in reversed(entries):
        if entry.get("type") != "user":
            continue
        text = strip_window_preamble(_entry_text(entry))
        if not text.strip():
            continue
        snippet = snippet_head(text, snippet_width)
        if hits and used + len(snippet) > budget_chars:
            break
        hits.append({
            "uuid": entry.get("uuid", ""),
            "type": "user",
            "snippet": snippet,
            "timestamp": entry.get("timestamp", ""),
        })
        used += len(snippet)
        if len(hits) >= cap:
            break
    return hits
