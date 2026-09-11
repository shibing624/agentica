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


def format_turn_stamp(value) -> str:
    """Compact UTC stamp shared by notes digest and search_session."""
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
    flat = content.strip().replace("\n", " ")
    if len(flat) <= width:
        return flat
    return flat[:width] + "…"


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
        content = _entry_text(entry)
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
        text = _entry_text(entry)
        if not text.strip() or is_window_preamble(text):
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
