# -*- coding: utf-8 -*-
"""
@description: Repair assistant tool_call / tool result pairing after messages are dropped.

Providers enforce a hard structural rule: every ``tool_call`` an assistant made
must be answered by a tool result, immediately after that assistant message and
in the order the calls were issued. Any path that removes messages by position
— the context-overflow FIFO drop is the one that still does — can break that
rule and get the whole request rejected. Layer 1 eviction cannot: it rewrites
content in place and never removes a message.

This repair only understands the OpenAI shape (one ``role="tool"`` message per
result). An Anthropic transcript keeps its results in ``tool_result`` blocks
inside a user message, so it looks to this code like every call went unanswered
— rebuilding it would inject a placeholder per call and corrupt a transcript
that was never broken. Those are left alone.
"""
from typing import Dict, List

from agentica.compression.evict import tool_result_blocks
from agentica.model.message import Message
from agentica.utils.log import logger


def sanitize_tool_pairs(messages: List[Message]) -> List[Message]:
    """Return a copy of ``messages`` with every tool_call/result pair repaired.

    Rebuilds the list in order: for each assistant message with tool_calls,
    inserts the existing result (matched by call_id) or a placeholder, in the
    original tool_calls order. Orphan tool results — those whose call_id no
    assistant message references — are dropped.

    Anthropic-shaped transcripts are returned untouched (see module docstring).
    """
    if any(tool_result_blocks(msg) for msg in messages):
        return list(messages)

    result_by_id: Dict[str, Message] = {}
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            result_by_id[msg.tool_call_id] = msg

    all_call_ids: set = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    all_call_ids.add(tc_id)

    rebuilt: List[Message] = []
    placeholder_count = 0
    orphan_count = 0

    for msg in messages:
        if msg.role == "tool":
            if msg.tool_call_id not in all_call_ids:
                orphan_count += 1
            # Results are re-inserted after their assistant message below.
            continue

        rebuilt.append(msg)

        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if not tc_id:
                    continue
                if tc_id in result_by_id:
                    rebuilt.append(result_by_id[tc_id])
                else:
                    rebuilt.append(Message(
                        role="tool",
                        tool_call_id=tc_id,
                        content="[Tool result removed during compression]",
                    ))
                    placeholder_count += 1

    if orphan_count:
        logger.debug(f"Removed {orphan_count} orphan tool results")
    if placeholder_count:
        logger.debug(f"Added {placeholder_count} placeholder tool results")

    return rebuilt
