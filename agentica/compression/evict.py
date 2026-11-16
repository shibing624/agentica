# -*- coding: utf-8 -*-
"""
@description: Layer 1 — the LLM-free way to make an oversized request fit.

There is one compression trigger, not two (the dsh design): the Layer 2 line,
``auto_compact_threshold`` — ``compact_token_limit`` or 95% of the window.
Below it, history is never rewritten, so the request grows as a strict prefix
and the provider cache keeps every token. Layer 1 used to fire on its own at
80% of the window and evict oldest-first down to 30%; every pass re-cached the
whole transcript from the first placeholder on, and since arguments and prose
are not evictable, each pass freed less than the last and fired sooner — the
cache paid again and again before Layer 2 ever ran.

At the line, the runner asks this module one question: would pruning old tool
results bring the request under ``PRUNE_TARGET_RATIO`` of the working window?
If so the prune is applied and the window stays — one prefix break, and the
next crossing is far away. If not, nothing is rewritten and Layer 2 cuts a new
window, which breaks the cache anyway; pruning first would only have been a
second break for nothing.

**Pruning keeps the head and the tail** of every result longer than
``PRUNE_MIN_CHARS`` (``PRUNE_HEAD_CHARS`` + ``PRUNE_TAIL_CHARS``, with a marker
in between) instead of replacing it with a placeholder. The model keeps what a
command printed first and how it ended — usually what it was working from —
and does not have to re-run the tool to see it again. A pruned result is under
``PRUNE_MIN_CHARS``, so it is never pruned twice. When the agent can read its
session log (``recoverable``), the marker quotes the tool-call id that
``search_session(uuid=...)`` resolves to the full output.

What is never rewritten:

* **The live tool round** (in-flight calls and the trailing result batch):
  the evidence the model has not read yet. The cutoff is by message position,
  not tool name — SDK ``tools=``, CLI ``--tools``, Web extra tools, and MCP
  share the round with builtins.
* **Tool-call arguments.** A stand-in in an old call — truncated prefix,
  string marker, or ``{"$evicted": N}`` — taught models to emit the stand-in
  as the payload of their next call (one session: 248 such calls, every one
  refused). Logs from before still carry markers; ``upgrade_legacy_omitted_args``
  keeps the oldest string form out of a slot where it could pass as content,
  and ``get_function_call`` refuses any call that carries one.
* **Results already bounded by Layer 0** (``<persisted-output>``): the path
  they carry is the only handle on output too large to re-read into context.

**The unit of pruning is a result, not a message**, because the two provider
shapes disagree about how results are packed:

* OpenAI-style — one result per ``role="tool"`` message.
* Anthropic-style — a whole round packed into the ``content`` list of a single
  ``role="user"`` message, as ``{"type": "tool_result", "tool_use_id": ...}``
  blocks (``AnthropicClaude.format_function_call_results``).
"""
import json
from typing import List, NamedTuple, Optional, TYPE_CHECKING

from agentica.compression.tool_call_args import (
    _LEGACY_OMITTED_PREFIX,
    rewrite_omitted_arg_leaves,
)
from agentica.utils.tokens import count_text_tokens, count_tokens

if TYPE_CHECKING:
    from agentica.model.message import Message

# Occupancy a prune must reach, as a fraction of the working window, to be
# worth its prefix break. Anything above it would cross the line again soon
# and pay another break; a new window is the cheaper answer then.
PRUNE_TARGET_RATIO = 0.5

# A result longer than this keeps only its head and tail. It must exceed
# head + marker + tail so a pruned result is never pruned again.
PRUNE_MIN_CHARS = 4096
PRUNE_HEAD_CHARS = 1000
PRUNE_TAIL_CHARS = 1000

_PRUNE_MARKER_OPEN = "[... "

# Anthropic packs a whole tool round into one user message as blocks of this type.
_TOOL_RESULT_BLOCK = "tool_result"

# Trailing user-turn share of the window at which reactive compact cannot help:
# Layer 2 keeps that turn intact, so an oversized single query must surface the
# provider error instead of burning an LLM summary that preserves the same text.
IRREDUCIBLE_PROMPT_RATIO = 0.9


def trailing_user_turn_start(messages: "List[Message]") -> int:
    """Index of the last real user message (not an Anthropic tool_result pack).

    A mid-turn injection (todo reminder, steering, peer message) is a user
    message too, but it belongs to the turn it landed in, not a new one. Read
    as the turn's start it left the question and the tool round before it in
    the dropped span, and the new window opened on the reminder alone.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "user" and not m._injected and not carries_tool_results(m):
            return i
    return len(messages)


def is_irreducible_prompt_too_long(
    messages: "List[Message]",
    *,
    context_window: int,
    model_id: str = "gpt-4o",
    ratio: float = IRREDUCIBLE_PROMPT_RATIO,
) -> bool:
    """True when the trailing user turn alone already fills the context window.

    Reactive compact / Layer 2 summarisation deliberately keeps that turn so
    providers do not see an assistant-prefill ending. If the user's message
    (plus any tool round attached to it) already exceeds ``ratio`` of the
    window, compacting history cannot make the retry fit — re-raise the
    provider's ``context_length_exceeded`` instead of hiding it behind a
    failed compact attempt.
    """
    if context_window <= 0 or not messages:
        return False
    start = trailing_user_turn_start(messages)
    if start >= len(messages):
        return False
    tokens = count_tokens(messages[start:], None, model_id, None)
    return tokens >= int(context_window * ratio)


def tool_result_blocks(msg: "Message") -> List[dict]:
    """The Anthropic-shaped ``tool_result`` blocks a user message carries, if any."""
    if msg.role != "user" or not isinstance(msg.content, list):
        return []
    return [
        block for block in msg.content
        if isinstance(block, dict) and block.get("type") == _TOOL_RESULT_BLOCK
    ]


def carries_tool_results(msg: "Message") -> bool:
    """True for either provider shape of "this message holds tool results"."""
    return msg.role == "tool" or bool(tool_result_blocks(msg))


def _last_batch_start(messages: "List[Message]", end: Optional[int] = None) -> int:
    """Index where the trailing run of tool results begins.

    The pipeline runs immediately before a model call, so that trailing run is
    the batch the model has not seen yet. Evicting from it guarantees a re-run,
    and if the request is still too big without it, the answer is summarisation
    rather than throwing away the turn's own evidence.

    ``end`` bounds the scan (exclusive) for callers that have already decided
    the transcript's real tail sits below some appended messages.
    """
    i = len(messages) if end is None else end
    while i > 0 and carries_tool_results(messages[i - 1]):
        i -= 1
    return i


def live_tool_round_start(messages: "List[Message]") -> int:
    """Start of the tool round that is still live: in-flight, or just returned.

    Compress runs before the next LLM call. The live round is:

    * a trailing assistant with ``tool_calls`` and no results yet (the tools
      are executing or have not been parsed), or
    * the assistant that issued the trailing result batch, plus those results.

    Layer 1 must not shrink arguments or evict results from this index on.
    The tool name does not matter: builtins, SDK ``tools=``, CLI ``--tools``,
    Web extra tools, and MCP calls all occupy the same round.

    Mid-turn injections (steering, peer messages) are skipped first. They are
    appended as a user message when there is no trailing ``role="tool"`` result
    to fold into — either because the round is still in flight, or because the
    provider packs results into the ``content`` of a user message (Anthropic).
    Reading them as the tail would put the cutoff above the round and hand
    Layer 1 the very arguments and results this function exists to protect.
    """
    n = len(messages)
    if n == 0:
        return 0
    end = n
    while end > 0 and messages[end - 1]._injected:
        end -= 1
    if end == 0:
        return 0
    last = messages[end - 1]
    if last.role == "assistant" and last.tool_calls:
        return end - 1
    batch = _last_batch_start(messages, end)
    if batch <= 0 or batch >= end:
        return batch
    prev = messages[batch - 1]
    if prev.role == "assistant" and prev.tool_calls:
        return batch - 1
    return batch


def _result_call_id(msg: "Message", block: Optional[dict]) -> Optional[str]:
    """The tool-call id a result answers, whichever shape holds it."""
    if block is not None:
        return block.get("tool_use_id") or None
    return msg.tool_call_id or None



def prune_text(text: str, handle: Optional[str] = None) -> str:
    """Head and tail of ``text`` around a marker naming what was cut."""
    cut = len(text) - PRUNE_HEAD_CHARS - PRUNE_TAIL_CHARS
    recover = f"; full output: search_session(uuid=\"{handle}\")" if handle else ""
    marker = f"{_PRUNE_MARKER_OPEN}{cut} chars pruned from the middle of this tool result{recover} ...]"
    return f"{text[:PRUNE_HEAD_CHARS]}\n\n{marker}\n\n{text[-PRUNE_TAIL_CHARS:]}"


class _Prune(NamedTuple):
    message: "Message"
    block: Optional[dict]
    text: str


def _prunable(messages: "List[Message]", recoverable: bool) -> List[_Prune]:
    """Every already-seen result long enough to prune, with its pruned text."""
    plan: List[_Prune] = []
    for msg in messages[:live_tool_round_start(messages)]:
        blocks: List[Optional[dict]] = [None] if msg.role == "tool" else list(tool_result_blocks(msg))
        for block in blocks:
            raw = block.get("content") if block is not None else msg.content
            if not isinstance(raw, str) or len(raw) <= PRUNE_MIN_CHARS:
                continue
            if "<persisted-output>" in raw:
                continue
            handle = _result_call_id(msg, block) if recoverable else None
            plan.append(_Prune(msg, block, prune_text(raw, handle)))
    return plan


class PruneResult(NamedTuple):
    """What a Layer 1 pass did: results rewritten and tokens freed."""
    pruned: int = 0
    freed_tokens: int = 0


def prune_tool_results(
    messages: "List[Message]",
    *,
    context_tokens: int,
    target_tokens: Optional[int],
    model_id: str = "gpt-4o",
    recoverable: bool = False,
) -> PruneResult:
    """Prune every already-seen long result, or none of them.

    Args:
        messages:       Full message list for the current request (mutated in place).
        context_tokens: Tokens the request occupies now, in ``model_id`` units.
        target_tokens:  The prune is applied only if it brings the request under
                        this; otherwise nothing changes and the caller cuts a new
                        window. None applies it unconditionally — for a caller
                        with no new window to fall back on.
        model_id:       Tokenizer for pricing what each prune frees.
        recoverable:    The model can read the session log (``search_session``),
                        so each marker quotes the tool-call id of the full output.
    """
    plan = _prunable(messages, recoverable)
    if not plan:
        return PruneResult()
    freed = 0
    for item in plan:
        old = item.block.get("content") if item.block is not None else item.message.content
        freed += count_text_tokens(old, model_id) - count_text_tokens(item.text, model_id)
    if target_tokens is not None and context_tokens - freed >= target_tokens:
        return PruneResult()
    for item in plan:
        if item.block is not None:
            item.block["content"] = item.text
        else:
            item.message.content = item.text
    return PruneResult(pruned=len(plan), freed_tokens=freed)


def _map_assistant_tool_args(msg: "Message", transform) -> int:
    """Apply ``transform`` to each tool-call arguments object in ``msg``.

    OpenAI stores JSON in ``tool_calls[].function.arguments`` (and sometimes
    on the tool_call itself). Anthropic also keeps ``tool_use.input``. A
    changed ``tool_use.input`` is written back onto the matching
    ``tool_calls`` entry. Sibling thinking is dropped when anything in the
    content list changes — mutating ``tool_use.input`` invalidates the
    thinking signature.
    """
    changed = 0
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            containers = [function] if isinstance(function, dict) else []
            containers.append(tool_call)
            for container in containers:
                raw = container.get("arguments")
                if not isinstance(raw, str):
                    continue
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                new = transform(parsed)
                if new != parsed:
                    container["arguments"] = json.dumps(new, ensure_ascii=False)
                    changed += 1
    if isinstance(msg.content, list):
        for block in msg.content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            new = transform(inp)
            if new == inp:
                continue
            block["input"] = new
            changed += 1
            call_id = block.get("id")
            dumped = json.dumps(new, ensure_ascii=False)
            if call_id and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if not isinstance(tool_call, dict) or tool_call.get("id") != call_id:
                        continue
                    function = tool_call.get("function")
                    if isinstance(function, dict):
                        function["arguments"] = dumped
                    if "arguments" in tool_call:
                        tool_call["arguments"] = dumped
        if changed:
            msg.content = [
                block
                for block in msg.content
                if not (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
            ]
    return changed


def _message_may_carry_legacy_omitted_arg(msg: "Message") -> bool:
    for tc in msg.tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict):
            raw = fn.get("arguments")
            if isinstance(raw, str) and _LEGACY_OMITTED_PREFIX in raw:
                return True
        raw_top = tc.get("arguments")
        if isinstance(raw_top, str) and _LEGACY_OMITTED_PREFIX in raw_top:
            return True
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if _LEGACY_OMITTED_PREFIX in str(block.get("input") or ""):
                    return True
    return False


def upgrade_legacy_omitted_args(messages: "List[Message]") -> int:
    """Rewrite leftover string markers in every assistant call, including live.

    Those strings are already in the transcript. Leaving them is how the
    model copies a marker into a new tool call. Fast-path substring check
    avoids parsing or touching clean assistant messages on normal steps.
    """
    changed = 0
    for msg in messages:
        if msg.role == "assistant" and _message_may_carry_legacy_omitted_arg(msg):
            changed += _map_assistant_tool_args(msg, rewrite_omitted_arg_leaves)
    return changed

