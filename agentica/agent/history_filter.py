# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: History message filtering pipeline.

Used by ``PromptsMixin`` to apply ``HistoryConfig`` rules and the optional
``Agent.history_filter`` callable to multi-turn history before it's appended
to the model prompt.

Pipeline order (see ``HistoryConfig`` docstring):
    1. excluded_tools         -> drop matching tool messages + paired tool_calls
    2. assistant_max_chars    -> truncate long assistant content
    3. user-supplied callable -> final say
    4. consistency fix        -> strip orphan assistant.tool_calls

The original Message objects are never mutated; we copy via
``model_copy(update=...)`` whenever we change a field.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Callable, List, Optional

from agentica.agent.config import HistoryConfig
from agentica.model.message import Message


HistoryFilter = Callable[[List[Message]], List[Message]]


def _has_meaningful_content(content) -> bool:
    """True if ``content`` carries information (non-empty string OR non-empty multimodal list).

    Used to decide whether an assistant message should be kept after all its
    ``tool_calls`` are dropped. ``Message.content`` is typed
    ``Optional[Union[List[Any], str]]`` — calling ``.strip()`` on a list raises.
    """
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def apply_history_pipeline(
    history: List[Message],
    config: Optional[HistoryConfig],
    user_filter: Optional[HistoryFilter],
) -> List[Message]:
    """Apply config rules + user callable + consistency fix to a copy of history.

    Args:
        history: Messages returned from ``working_memory.get_messages_from_last_n_runs``.
        config: Declarative rules (``excluded_tools`` / ``assistant_max_chars``).
        user_filter: Optional user-supplied ``Callable[[List[Message]], List[Message]]``.
            Runs AFTER config rules, gets the final say.

    Returns:
        A new list (no in-place mutation of ``history``).
    """
    if not history:
        return list(history)

    out = list(history)

    if config is not None:
        if config.excluded_tools:
            out = _drop_excluded_tools(out, config.excluded_tools)
        if config.assistant_max_chars is not None and config.assistant_max_chars > 0:
            out = _truncate_assistant_content(out, config.assistant_max_chars)

    # Strip leaked <think>/<reasoning> blocks from replayed assistant turns.
    if config is None or config.scrub_reasoning:
        out = _scrub_reasoning_leak(out)

    if user_filter is not None:
        out = list(user_filter(out))

    out = _strip_orphan_tool_calls(out)
    return out


def _matches_any(name: Optional[str], patterns: List[str]) -> bool:
    if not name:
        return False
    return any(fnmatchcase(name, p) for p in patterns)


def _drop_excluded_tools(history: List[Message], patterns: List[str]) -> List[Message]:
    """Drop tool messages whose ``tool_name`` matches any glob pattern.

    Also strips the corresponding ``tool_calls`` entry from the preceding
    assistant message so the OpenAI API contract is preserved.
    """
    excluded_call_ids: set[str] = set()
    out: List[Message] = []
    for m in history:
        if m.role == "tool" and _matches_any(m.tool_name, patterns):
            if m.tool_call_id:
                excluded_call_ids.add(m.tool_call_id)
            continue
        out.append(m)

    if not excluded_call_ids:
        return out

    cleaned: List[Message] = []
    for m in out:
        if m.role == "assistant" and m.tool_calls:
            kept_calls = [tc for tc in m.tool_calls if tc.get("id") not in excluded_call_ids]
            if len(kept_calls) != len(m.tool_calls):
                if not kept_calls and not _has_meaningful_content(m.content):
                    # Assistant turn was purely tool-calls and all got dropped — drop the message too.
                    continue
                m = m.model_copy(update={"tool_calls": kept_calls or None})
        cleaned.append(m)
    return cleaned


def _scrub_reasoning_leak(history: List[Message]) -> List[Message]:
    """Remove leaked reasoning blocks from replayed assistant messages.

    No-op for the common case (no reasoning tags present). Only assistant
    messages are touched; user/tool content is left verbatim.
    """
    from agentica.think_scrubber import contains_reasoning_leak, sanitize_assistant_content_for_history

    out: List[Message] = []
    for m in history:
        if m.role == "assistant" and isinstance(m.content, str) and contains_reasoning_leak(m.content):
            m = m.model_copy(
                update={"content": sanitize_assistant_content_for_history(m.content)}
            )
        out.append(m)
    return out


def _truncate_assistant_content(history: List[Message], max_chars: int) -> List[Message]:
    out: List[Message] = []
    for m in history:
        if m.role == "assistant" and isinstance(m.content, str) and len(m.content) > max_chars:
            m = m.model_copy(update={"content": m.content[:max_chars] + "..."})
        out.append(m)
    return out


def _strip_orphan_tool_calls(history: List[Message]) -> List[Message]:
    """Remove tool_calls entries on assistant messages that have no matching tool result.

    Safety net for user-supplied filters that drop tool messages without
    cleaning the paired tool_calls. Without this, the next LLM API call
    would 400 with "tool_call_id has no matching tool message".
    """
    present_tool_call_ids: set[str] = {
        m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id
    }

    out: List[Message] = []
    for m in history:
        if m.role == "assistant" and m.tool_calls:
            kept = [tc for tc in m.tool_calls if tc.get("id") in present_tool_call_ids]
            if len(kept) != len(m.tool_calls):
                if not kept and not _has_meaningful_content(m.content):
                    continue
                m = m.model_copy(update={"tool_calls": kept or None})
        out.append(m)
    return out
