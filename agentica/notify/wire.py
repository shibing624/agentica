# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The payload discipline both external egresses share.

The notify sink and the hook egress put the same user-visible strings on a wire
that leaves the process, so "how much text, and which words count as a decision"
lives in exactly one place: two copies would drift, and the drift would be
invisible until a consumer rendered a half-prompt or accepted a fifth word.
"""

from __future__ import annotations

from typing import Any, Optional

#: How much of a prompt / question / answer goes on the wire. A consumer renders
#: a bubble, not a reader: a 40k-character answer would bloat every event and
#: still not be more useful there. The marker makes the cut visible, so a
#: consumer can tell "it said this much" from "it said 500 chars and more".
TEXT_LIMIT = 500
ELLIPSIS = "…"

#: Decisions a reply may carry back. Anything else is treated as "no decision"
#: rather than being coerced — guessing here would approve a command.
ALLOWED_DECISIONS = frozenset({"allow", "deny", "allow_prefix", "deny_prefix"})


def clip_text(value: Any) -> Optional[str]:
    """A short, wire-safe slice of user-visible text, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= TEXT_LIMIT:
        return text
    return text[:TEXT_LIMIT] + ELLIPSIS
