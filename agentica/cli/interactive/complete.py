# -*- coding: utf-8 -*-
"""Slash-command completion: full list on ``/``, then fuzzy rank."""

from __future__ import annotations

from typing import Collection, Iterable, List, Optional, Sequence, Tuple

# (name, display, meta). ``name`` is what is inserted; ``display`` is the menu
# label (skills add ``(SkillName)``); ``meta`` is the one-line description.
SlashRow = Tuple[str, str, str]


def _body(text: str) -> str:
    text = text.lower()
    return text[1:] if text.startswith("/") else text


def _subsequence_score(query: str, name: str) -> Optional[int]:
    """Tighter, earlier subsequence matches score higher (less negative)."""
    pos = 0
    first: Optional[int] = None
    gaps = 0
    last: Optional[int] = None
    for ch in query:
        found = name.find(ch, pos)
        if found < 0:
            return None
        if first is None:
            first = found
        if last is not None:
            gaps += found - last - 1
        last = found
        pos = found + 1
    return -first - gaps - (len(name) - len(query))


def score_slash_command(query: str, name: str, meta: str = "") -> Optional[int]:
    """Rank one command against the typed text. ``None`` means hide it.

    Tiers, high to low: prefix of the command, substring of the name,
    subsequence of the name, then a description substring (only when the
    typed body is at least 3 characters, so ``/s`` does not match every
    help line that happens to contain an ``s``).
    """
    q = query.lower()
    n = name.lower()
    if not q.startswith("/"):
        return None
    q_body = _body(q)
    n_body = _body(n)
    # Bare `/` is a prefix of every command; keep the given list order so the
    # menu is the command list, not "shortest aliases first".
    if not q_body:
        return 0
    if n.startswith(q):
        return 4000 - (len(n) - len(q))
    if q_body in n_body:
        return 3000 - n_body.index(q_body) - (len(n_body) - len(q_body))
    sub = _subsequence_score(q_body, n_body)
    if sub is not None:
        return 2000 + sub
    if len(q_body) >= 3:
        hay = meta.lower()
        at = hay.find(q_body)
        if at >= 0:
            return 1000 - at
    return None


def rank_slash_commands(query: str, rows: Sequence[SlashRow]) -> List[SlashRow]:
    """Filter and rank ``rows`` for the completions menu.

    Bare ``/`` keeps the given order (the full command list). Ties keep
    that order as well.
    """
    scored: list[tuple[int, int, SlashRow]] = []
    for i, row in enumerate(rows):
        score = score_slash_command(query, row[0], row[2])
        if score is None:
            continue
        scored.append((score, -i, row))
    scored.sort(reverse=True)
    return [row for _score, _order, row in scored]


def is_slash_command_line(text: str, registered: Collection[str]) -> bool:
    """True when the first token is a registered ``/command`` or skill slug.

    A leading ``/`` is not enough: a Unix path (``/var/folders/...png``) starts
    the same way and is ordinary input — including mid-run steering. Idle
    dispatch already keys off the first token in ``COMMAND_HANDLERS``; this
    is the same test so a screenshot path cannot be queued as a "command".
    """
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return False
    return stripped.split()[0].lower() in registered


def slash_command_rows(
    registry: Iterable[tuple[str, str]],
    skill_rows: Iterable[SlashRow] = (),
) -> List[SlashRow]:
    """Build the candidate list: registry commands, then skill auto-commands.

    ``registry`` may be a one-shot generator (the TUI passes one). Derive
    the collision set from the rows already built, not by iterating it
    again — a second pass is empty, so ``/cron`` and ``/worktree`` leaked
    the bundled skills of the same slug. Dispatch prefers the real handler,
    so those extra rows inserted the same text and never ran the skill.
    """
    rows: List[SlashRow] = [(name, name, desc) for name, desc in registry]
    seen = {name for name, _display, _meta in rows}
    for name, display, meta in skill_rows:
        if name in seen:
            continue
        rows.append((name, display, meta))
        seen.add(name)
    return rows
