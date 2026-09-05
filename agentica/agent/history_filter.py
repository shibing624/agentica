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
from agentica.tools.patch import parse_patch_envelope


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
_ELIDED_WRITE_TOOLS = frozenset({"write_file", "apply_patch"})
_ELIDED_WRITE_LINES_CAP = 20
_ELIDED_BLOCK_RE = re.compile(
    re.escape(ELIDED_TOOLS_MARK) + r".*?" + re.escape(ELIDED_TOOLS_CLOSE),
    re.DOTALL,
)


def has_tool_artifacts(messages: List[Message]) -> bool:
    """True if ``messages`` still carries a tool round in either wire format.

    The notice below is only honest when something was actually dropped.
    ``supports_replayed_tool_history`` is a *provider* property, so the strip
    also runs over chat-only histories where there is nothing to warn about.
    """
    for m in messages:
        if m.role == "tool" or m.tool_calls:
            return True
        if _content_has_block_type(m.content, "tool_use"):
            return True
        if _content_has_block_type(m.content, "tool_result"):
            return True
    return False


def strip_elided_notice(text: str) -> str:
    """Drop the ``<elided-tools>`` block from text meant for human eyes.

    The notice is bookkeeping aimed at the next model. Transcript replay
    (``/resume``, ``/history``) should show the turn's prose, not the marker.
    """
    if not isinstance(text, str) or ELIDED_TOOLS_MARK not in text:
        return text
    return _ELIDED_BLOCK_RE.sub("", text).strip()


def _tool_call_name_and_args(tool_call: dict) -> tuple[str, dict]:
    fn = tool_call.get("function") or {}
    name = fn.get("name") or tool_call.get("name") or ""
    raw = fn.get("arguments") or tool_call.get("arguments") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    return name, raw if isinstance(raw, dict) else {}


def _write_target(name: str, args: dict) -> str:
    if name == "write_file":
        return str(args.get("file_path") or args.get("path") or "").strip()
    if name == "apply_patch":
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return ""
        try:
            return ", ".join(op.path for op in parse_patch_envelope(patch))
        except ValueError:
            return ""
    return ""


def _first_result_line(content) -> str:
    if not isinstance(content, str):
        return ""
    line = content.strip().splitlines()[0] if content.strip() else ""
    if len(line) > 100:
        return line[:99] + "…"
    return line


def summarize_elided_writes(messages: List[Message]) -> List[str]:
    """One line per write_file / apply_patch, from args + first result line.

    Execute / read results stay out: those bodies leak secrets and are not
    what the next model invents files from. Cap keeps a long session short.
    """
    pending: dict[str, tuple[str, str]] = {}
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                name, args = _tool_call_name_and_args(tc)
                if name not in _ELIDED_WRITE_TOOLS:
                    continue
                call_id = tc.get("id")
                if not call_id:
                    continue
                pending[call_id] = (name, _write_target(name, args))
        if m.role == "tool" and m.tool_call_id in pending:
            pending[m.tool_call_id] = (
                *pending[m.tool_call_id],
                _first_result_line(m.content),
            )
        if m.role == "user" and _content_has_block_type(m.content, "tool_result"):
            for block in m.content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                if call_id not in pending:
                    continue
                pending[call_id] = (
                    *pending[call_id],
                    _first_result_line(block.get("content")),
                )

    lines: List[str] = []
    for name, target, *rest in pending.values():
        result = rest[0] if rest else ""
        label = f"{name} {target}".strip() if target else name
        lines.append(f"- {label} → {result}" if result else f"- {label}")
    if len(lines) > _ELIDED_WRITE_LINES_CAP:
        omitted = len(lines) - _ELIDED_WRITE_LINES_CAP
        lines = lines[-_ELIDED_WRITE_LINES_CAP:]
        lines.insert(0, f"- … +{omitted} earlier writes omitted")
    return lines


def elided_tools_notice(messages: List[Message]) -> str:
    """Portable note after a /model or Claude-resume strip.

    Raw tool rounds cannot cross providers. Dropping them without a
    replacement leaves only assistant prose, and the next model treats
    「写好了：docs/foo.md（193 行）」 as a completed write.
    """
    writes = summarize_elided_writes(messages)
    parts = [
        ELIDED_TOOLS_MARK,
        "Tool calls were dropped after a model switch (provider wire format). "
        "Earlier assistant prose is not proof a file exists or a command ran. "
        "Use tools; do not narrate work as if it already happened.",
    ]
    if writes:
        parts.append("Writes that actually ran:")
        parts.extend(writes)
    parts.append(ELIDED_TOOLS_CLOSE)
    return "\n".join(parts)


def _append_elided_notice(messages: List[Message], notice_text: str) -> List[Message]:
    """Fold the notice into the trailing assistant turn, at most once.

    Merged rather than appended as its own message: a standalone assistant
    note after an assistant answer is two consecutive same-role turns, which
    Bedrock and several aggregating proxies reject with
    ``400 roles must alternate`` — the very failure the strip exists to avoid.
    Merging also makes the operation idempotent, so resume-then-``/model``
    cannot stack a second copy.
    """
    out = list(messages or [])
    for m in out:
        if m.role == "assistant" and isinstance(m.content, str) and ELIDED_TOOLS_MARK in m.content:
            return out
    if out and out[-1].role == "assistant" and isinstance(out[-1].content, str):
        prose = out[-1].content or ""
        joined = f"{prose}\n\n{notice_text}" if prose.strip() else notice_text
        out[-1] = out[-1].model_copy(update={"content": joined})
        return out
    out.append(Message(role="assistant", content=notice_text))
    return out


def strip_tool_artifacts_from_memory(working_memory) -> None:
    """Reduce a WorkingMemory's history to plain user/assistant text, in place.

    Both stores have to be cleaned: ``runs[].response.messages`` is what the
    prompt builder replays to the model, and ``messages`` is what ``/history``
    and ``/export`` show. Leaving the flat list untouched would make the two
    disagree about what the model can still see.

    When a tool round was actually dropped, a note naming the real writes is
    folded into the trailing assistant turn so the next model does not treat
    prior prose as proof. A history that never had tool rounds gets no note.
    """
    source: List[Message] = []
    for run in working_memory.runs:
        if run.response is None or not run.response.messages:
            continue
        source.extend(run.response.messages)
    if not source and working_memory.messages:
        source = list(working_memory.messages)
    notice_text = elided_tools_notice(source) if has_tool_artifacts(source) else ""

    for run in working_memory.runs:
        if run.response is None or not run.response.messages:
            continue
        run.response.messages = strip_all_tool_artifacts(run.response.messages, drop_system=True)
    if working_memory.messages:
        working_memory.messages = strip_all_tool_artifacts(
            working_memory.messages, drop_system=False
        )

    if not notice_text:
        return

    if working_memory.runs:
        last = working_memory.runs[-1]
        if last.response is not None:
            last.response.messages = _append_elided_notice(
                last.response.messages, notice_text
            )
    if working_memory.messages:
        working_memory.messages = _append_elided_notice(
            working_memory.messages, notice_text
        )


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
