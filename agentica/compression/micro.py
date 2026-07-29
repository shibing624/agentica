# -*- coding: utf-8 -*-
"""
@description: Micro-compact — silent per-turn tool-result truncation.

Runs before every LLM call in the tool loop (zero cost, no LLM needed).
Mirrors CC's microCompact.ts time-based path:

* Keeps the most recent KEEP_RECENT tool-result messages intact.
* Replaces the *content* of older ones with a fixed placeholder so the
  model still sees the tool call happened, just not its full output.
* Never modifies the most recent `keep_recent` results so the model
  retains fresh context.

This is the cheapest possible compression: O(n) scan, in-place mutation,
no API call.  It should run every turn regardless of context size.
"""
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from agentica.model.message import Message

# Placeholder text — identical to CC's TIME_BASED_MC_CLEARED_MESSAGE
MICRO_COMPACT_PLACEHOLDER = "[Old tool result content cleared]"

# Minimum content length before we bother replacing (very short results
# are not worth the placeholder overhead).
_MIN_CONTENT_LEN = 80

# Default: keep this many most-recent tool results untouched.
DEFAULT_KEEP_RECENT = 5


def micro_compact(
    messages: "List[Message]",
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> int:
    """Replace old tool-result content with a short placeholder (in-place).

    Args:
        messages:    Full message list for the current turn (mutated in place).
        keep_recent: How many of the *most recent* tool-result messages to
                     leave untouched. 

    Returns:
        Number of messages whose content was replaced.
    """
    # Collect indices of tool-result messages that are not yet compacted.
    tool_indices: List[int] = [
        i for i, m in enumerate(messages)
        if m.role == "tool"
        and not m._micro_compacted
    ]

    if len(tool_indices) <= keep_recent:
        return 0

    # The oldest (len - keep_recent) entries are candidates for compaction.
    candidates = tool_indices[:-keep_recent]
    compacted = 0

    for idx in candidates:
        msg = messages[idx]
        content = msg.content
        if content is None:
            continue
        content_str = content if isinstance(content, str) else str(content)
        if len(content_str) <= _MIN_CONTENT_LEN:
            continue

        msg.content = MICRO_COMPACT_PLACEHOLDER
        # Mark so subsequent calls don't double-process this message.
        msg._micro_compacted = True
        compacted += 1

    return compacted
