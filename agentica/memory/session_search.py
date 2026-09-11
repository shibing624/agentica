# -*- coding: utf-8 -*-
"""Keyword search over session-log rows (not first-N substring).

``search_session`` used to require the exact query string and returned the
earliest hits in file order. A question like 工单号 then missed
「工单 ZX-41827」, and a generic word drowned in filler. Terms are split,
CJK runs become overlapping bigrams, and hits are scored.
"""
import re
from typing import Dict, Iterable, List, Sequence, Set

_TOKEN = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]+")
_CJK = re.compile(r"^[\u4e00-\u9fff]+$")
_STOP = frozenset({
    "the", "a", "an", "is", "to", "of", "and", "or", "in", "for", "on",
    "this", "that", "with", "from", "what", "which", "how",
    "dump", "read", "background", "unrelated", "noted",
    "什么", "多少", "哪个", "哪些", "怎么", "如何", "这次", "只答",
})


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
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content)
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
        })
    return hits
