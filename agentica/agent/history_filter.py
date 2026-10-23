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

import json
import re
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


def _content_has_block_type(content, block_type: str) -> bool:
    """True if ``content`` is a list containing a dict block of ``block_type``.

    Anthropic serialises tool calls/results as *list content blocks*:
      - tool result -> role="user",  content=[{"type": "tool_result", "tool_use_id": ...}]
      - tool call   -> role="assistant", content=[..., {"type": "tool_use", "id": ...}]
    OpenAI-compatible providers instead use flat ``role="tool"`` messages and
    the ``assistant.tool_calls`` field. When history recorded under one wire
    format is replayed on the other provider, these list blocks are rejected
    (e.g. "unexpected tool_use_id found in tool_result blocks").
    """
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == block_type for b in content)


def _text_from_content_blocks(content) -> str:
    """Extract concatenated text from an Anthropic-style content-block list.

    Drops tool_use/tool_result/thinking blocks, keeps only ``{"type": "text", ...}``.
    Returns "" when there's no reusable text.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "\n".join(parts)


def _images_from_content_blocks(content) -> List[dict]:
    """Anthropic-shaped ``image`` blocks carried inside a content-block list.

    The same turn may instead arrive as ``Message.images``; both shapes have
    to survive a switch, or the new model answers a question it cannot see.
    """
    if not isinstance(content, list):
        return []
    return [
        b for b in content
        if isinstance(b, dict) and b.get("type") in ("image", "image_url")
    ]


def _merge_images(*groups) -> List[dict]:
    """Union of image blocks, deduplicated by identity then by value.

    A session may record the same image in both shapes; sending it twice
    makes some providers reject the request.
    """
    merged: List[dict] = []
    seen: List[dict] = []
    for group in groups:
        for image in group or ():
            if any(image is known for known in merged):
                continue
            if image in seen:
                continue
            merged.append(image)
            seen.append(image)
    return merged


def strip_all_tool_artifacts(messages: List[Message], *, drop_system: bool = False) -> List[Message]:
    """Reduce history to portable user/assistant text for a model switch.

    Switching models starts a new Q&A round. Thinking and tool rounds do not
    travel: their wire shapes disagree (OpenAI ``role=tool`` vs Anthropic
    ``tool_use`` / ``tool_result`` blocks), and a ``thinking.signature`` is
    bound to the model that issued it. What remains is what any provider can
    consume — user questions and assistant answers as plain strings.

    ``drop_system`` also removes system messages (recovery path; the system
    prompt is rebuilt fresh each run).
    """
    cleaned: List[Message] = []
    for m in messages:
        if m.role == "tool":
            continue
        if drop_system and m.role == "system":
            continue
        if m.role == "system":
            cleaned.append(m)
            continue
        if m.role == "user":
            if _content_has_block_type(m.content, "tool_result"):
                continue
            text = _text_from_content_blocks(m.content)
            images = _merge_images(m.images, _images_from_content_blocks(m.content))
            if images:
                cleaned.append(Message(role="user", content=text or None, images=images))
            elif isinstance(text, str) and text.strip():
                cleaned.append(Message(role="user", content=text))
            continue
        if m.role == "assistant":
            text = _text_from_content_blocks(m.content)
            if isinstance(text, str) and text.strip():
                cleaned.append(Message(role="assistant", content=text))
            continue
        cleaned.append(m)
    return cleaned


ELIDED_TOOLS_MARK = "<elided-tools>"
ELIDED_TOOLS_CLOSE = "</elided-tools>"
_ELIDED_BLOCK_RE = re.compile(
    re.escape(ELIDED_TOOLS_MARK) + r".*?" + re.escape(ELIDED_TOOLS_CLOSE),
    re.DOTALL,
)


def strip_elided_notice(text: str) -> str:
    """Drop the ``<elided-tools>`` block from text meant for human eyes.

    Sessions recorded before tool rounds survived a model switch carry this
    note in an assistant turn. Transcript replay (``/resume``, ``/history``)
    should show the turn's prose, not the marker.
    """
    if not isinstance(text, str) or ELIDED_TOOLS_MARK not in text:
        return text
    return _ELIDED_BLOCK_RE.sub("", text).strip()


def _tool_calls_from_blocks(content) -> List[dict]:
    """OpenAI-shape ``tool_calls`` for the ``tool_use`` blocks of a Claude turn."""
    return [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)},
        }
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    return _text_from_content_blocks(content)


def _portable_history(
    messages: List[Message],
    *,
    drop_system: bool,
    input_message_ids: Optional[set[int]] = None,
) -> tuple[List[Message], set[int]]:
    """Build portable messages and preserve which rebuilt items were inputs."""
    portable: List[Message] = []
    portable_input_ids: set[int] = set()

    def append(source: Message, converted: Message) -> None:
        portable.append(converted)
        if input_message_ids is not None and id(source) in input_message_ids:
            portable_input_ids.add(id(converted))

    tool_names: dict = {}
    answered: set = set()
    for m in messages:
        if m.role == "system":
            if not drop_system:
                append(m, m)
            continue
        if m.role == "assistant":
            tool_calls = m.tool_calls or _tool_calls_from_blocks(m.content) or None
            for tc in tool_calls or ():
                tool_names[tc["id"]] = tc["function"]["name"]
            text = _text_from_content_blocks(m.content)
            if not tool_calls and not text.strip():
                continue
            append(m, Message(role="assistant", content=text, tool_calls=tool_calls))
            continue
        if m.role == "tool":
            if m.tool_call_id in answered:
                continue
            answered.add(m.tool_call_id)
            append(
                m,
                Message(
                    role="tool",
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name or tool_names.get(m.tool_call_id),
                    content=_tool_result_text(m.content),
                    tool_call_error=m.tool_call_error,
                ),
            )
            continue
        # role == "user"
        if isinstance(m.content, list):
            for block in m.content:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                call_id = block["tool_use_id"]
                if call_id in answered:
                    continue
                answered.add(call_id)
                append(
                    m,
                    Message(
                        role="tool",
                        tool_call_id=call_id,
                        tool_name=tool_names.get(call_id),
                        content=_tool_result_text(block.get("content")),
                        tool_call_error=block.get("is_error") or None,
                    ),
                )
        text = _text_from_content_blocks(m.content)
        images = _merge_images(m.images, _images_from_content_blocks(m.content))
        if images:
            append(m, Message(role="user", content=text or None, images=images))
        elif text.strip():
            append(m, Message(role="user", content=text))
    return portable, portable_input_ids


def to_portable_history(messages: List[Message], *, drop_system: bool = False) -> List[Message]:
    """History in the one shape every provider adapter replays, tool rounds kept.

    The shared shape is the one SessionLog persists: ``assistant.tool_calls``
    answered by ``role="tool"`` messages. Claude's live turns carry the same
    round as ``tool_use`` / ``tool_result`` content blocks, which the OpenAI
    wires reject, so those are rewritten; the Claude adapter turns the shared
    shape back into blocks itself.

    What does not travel is what is bound to the model that produced it:
    thinking blocks (their signature only verifies on the issuing model),
    ``reasoning_content`` and the provider replay state
    (``provider_data`` / ``provider_checkpoint``). Each message is rebuilt from
    the portable fields only, so none of it can leak through.

    ``drop_system`` also removes system messages (the run stores; the system
    prompt is rebuilt fresh each run).
    """
    return _portable_history(messages, drop_system=drop_system)[0]


def to_portable_history_with_input_ids(
    messages: List[Message],
    input_message_ids: set[int],
) -> tuple[List[Message], set[int]]:
    """Portable history plus rebuilt IDs for messages that predated this turn."""
    return _portable_history(
        messages,
        drop_system=False,
        input_message_ids=input_message_ids,
    )


def make_history_portable(working_memory) -> None:
    """Rewrite a WorkingMemory's history with ``to_portable_history``, in place.

    Both stores have to be converted: ``runs[].response.messages`` is what the
    prompt builder replays to the model, and ``messages`` is what ``/history``,
    ``/compact`` and the memory hooks read. Converting one alone would make the
    two disagree about what the model can still see.
    """
    for run in working_memory.runs:
        if run.response is None or not run.response.messages:
            continue
        run.response.messages = to_portable_history(run.response.messages, drop_system=True)
    if working_memory.messages:
        working_memory.messages = to_portable_history(working_memory.messages)


def carries_signed_thinking(messages: List[Message]) -> bool:
    """True if an assistant turn replays ``thinking`` / ``redacted_thinking`` blocks.

    Their signature only verifies on the model that issued them.
    """
    return any(
        m.role == "assistant"
        and (
            _content_has_block_type(m.content, "thinking")
            or _content_has_block_type(m.content, "redacted_thinking")
        )
        for m in messages
    )


def model_identity(model) -> tuple:
    """What decides whether history written by one model replays on another.

    The adapter class fixes the wire shape, the id and endpoint fix who
    verifies thinking signatures and provider replay state. A profile switch
    that rebuilds the same model leaves all three unchanged.
    """
    return (type(model), model.id, model.base_url)


def ensure_history_portable(agent) -> bool:
    """Before a request: carry the history onto ``agent.model`` if another model wrote it.

    Called at run start, not on ``/model``: the rewrite drops signed thinking
    and provider replay state for good, so it waits until a request actually
    goes to the new model. A→B→A with no request in between leaves the
    history untouched, prompt cache and signatures included.

    ``agent._history_model_identity`` names the model the history in
    ``working_memory`` was written for. None means nothing model-bound is in
    it yet (fresh session, or history replayed from the session log, which is
    already in the shared shape). Returns True when the history was rewritten.
    """
    identity = model_identity(agent.model)
    written_for = agent._history_model_identity
    agent._history_model_identity = identity
    if written_for is None or written_for == identity:
        return False
    make_history_portable(agent.working_memory)
    return True


_CLAUDE_NATIVE_BLOCKS = frozenset({"tool_use", "tool_result", "thinking", "redacted_thinking"})


def _carries_claude_blocks(message: Message) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(b, dict) and b.get("type") in _CLAUDE_NATIVE_BLOCKS for b in message.content
    )


def claude_blocks_to_shared_shape(messages: List[Message]) -> List[Message]:
    """Request-side copy of ``messages`` with Claude's native blocks rewritten.

    Claude records a live tool round as ``tool_use`` / ``thinking`` blocks on
    the assistant and ``tool_result`` blocks on a user turn. ``/model`` and
    resume convert history ahead of time, but an in-turn fallback does not: a
    Claude primary that fails after a tool round hands that very array to the
    fallback model, and the OpenAI wires reject the blocks (chat: "type must
    not be null"; Responses: "Invalid value: 'thinking'").

    Only messages that carry such blocks are rebuilt (``to_portable_history``
    per message); every other message is passed through as the same object, so
    the wire's own replay state — Responses ``provider_data``, Chat
    ``reasoning_content`` handling — is untouched. Returns ``messages`` itself
    when nothing needs rewriting.
    """
    if not any(_carries_claude_blocks(m) for m in messages):
        return messages
    shared: List[Message] = []
    answered: set = set()
    for message in messages:
        rewritten = to_portable_history([message]) if _carries_claude_blocks(message) else [message]
        for m in rewritten:
            if m.role == "tool":
                # The runner can hold the same result twice (a role="tool"
                # pad and the native block); the wire takes one per call.
                if m.tool_call_id in answered:
                    continue
                answered.add(m.tool_call_id)
            shared.append(m)
    return shared


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
    from agentica.model.think_scrubber import contains_reasoning_leak, sanitize_assistant_content_for_history

    out: List[Message] = []
    for m in history:
        if m.role == "assistant" and isinstance(m.content, str) and contains_reasoning_leak(m.content):
            m = m.model_copy(update={"content": sanitize_assistant_content_for_history(m.content)})
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
    present_tool_call_ids: set[str] = {m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id}

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
