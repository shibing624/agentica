# -*- coding: utf-8 -*-
"""
@description: Layer 1 — the free, LLM-free way to make an oversized request fit.

Runs before every LLM call in the tool loop. It answers one question: is the
request too big, and if so, what is the cheapest thing to throw away?

Two rules, both learned from a real re-read loop:

* **Only act under pressure.** Evicting a result the window had room for is a
  pure loss — the model just re-runs the tool to get it back. Nothing happens
  below ``EVICT_THRESHOLD_RATIO``.
* **Evict oldest-first, only down to a target.** There is deliberately no
  "keep the last N results" knob: any fixed count loses to a batch of N+1
  parallel calls, which is exactly how a batch used to lose its first result
  before the model had ever seen it. Recent results survive because eviction
  stops once the request is small enough to reach them.

Evicted content is replaced by a placeholder naming the call, so the model can
re-issue it. It is deliberately *not* copied to disk first: recovering from a
spill file costs the same one tool call as re-running the original, and for a
file read the original path holds fresher content than the snapshot would.

Tool *results* are not the only bulk in a transcript: a ``write_file`` or
``apply_patch`` call carries its whole payload in the assistant message's
arguments, which no amount of result eviction can reach. Those strings are
shrunk in place, keeping the JSON valid so the provider still accepts the
transcript. When Layer 1 cannot get the request under target, the answer is
Layer 2 (``CompressionManager.auto_compact``), not more aggressive evicting.

**The unit of eviction is a result, not a message**, because the two provider
shapes disagree about how results are packed:

* OpenAI-style — one result per ``role="tool"`` message.
* Anthropic-style — a whole round packed into the ``content`` list of a single
  ``role="user"`` message, as ``{"type": "tool_result", "tool_use_id": ...}``
  blocks (``AnthropicClaude.format_function_call_results``).

Scanning only ``role="tool"`` meant Layer 1 never once ran on the Anthropic
path. Those blocks also carry no tool name, so the placeholder resolves the
call through an index of the assistant ``tool_calls`` that requested it.
"""
import json
import os
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

from agentica.compression.tool_call_args import shrink_tool_call_arguments_json
from agentica.utils.log import logger
from agentica.utils.tokens import count_message_tokens, count_tokens

if TYPE_CHECKING:
    from agentica.model.message import Message

# Occupancy at which eviction starts, as a fraction of the context window.
# 0.8 (not 0.9): from 0.9 the pass would have to free 40% of the window in tool
# results alone to reach the 0.5 target — text-heavy sessions can't, and Layer
# 2 then fires within the 5% gap, doubling prefix breaks. 0.8→0.95 leaves 15%.
# It also must not collide numerically with IRREDUCIBLE_PROMPT_RATIO (0.9).
EVICT_THRESHOLD_RATIO = 0.8

# The one user-facing knob of the compression policy (Reasonix /config
# compact_ratio): project-level override via AGENTICA_EVICT_THRESHOLD_RATIO
# (drop it in the project .env). Deliberately the ONLY exposed ratio — the
# Layer 2 trigger (0.95) and the 0.5 eviction target stay internal: touching
# the former reopens the layer-inversion trap, the latter is a separate
# "how hard to squeeze" question.
ENV_EVICT_THRESHOLD_RATIO = "AGENTICA_EVICT_THRESHOLD_RATIO"
_EVICT_THRESHOLD_MAX = 0.95  # strictly below AUTO_COMPACT_THRESHOLD_RATIO


def evict_threshold_ratio() -> float:
    """Effective evict trigger ratio: env override with validation, else 0.8."""
    raw = os.environ.get(ENV_EVICT_THRESHOLD_RATIO)
    if not raw:
        return EVICT_THRESHOLD_RATIO
    try:
        ratio = float(raw)
    except ValueError:
        logger.warning(f"{ENV_EVICT_THRESHOLD_RATIO}={raw!r} is not a number; using {EVICT_THRESHOLD_RATIO}")
        return EVICT_THRESHOLD_RATIO
    if not 0.0 < ratio < _EVICT_THRESHOLD_MAX:
        clamped = min(max(ratio, 0.05), _EVICT_THRESHOLD_MAX - 0.01)
        logger.warning(
            f"{ENV_EVICT_THRESHOLD_RATIO}={ratio} outside (0, {_EVICT_THRESHOLD_MAX}); clamped to {clamped}"
        )
        return clamped
    return ratio

# Occupancy eviction aims to reach. Strictly below the threshold so one pass
# buys several turns of headroom instead of re-triggering every turn.
EVICT_TARGET_RATIO = 0.5

# Head kept from each long string inside tool-call arguments.
TOOL_CALL_ARG_MAX_CHARS = 150

# Cap on the rendered call arguments inside a placeholder.
_MAX_CALL_SIGNATURE_CHARS = 120

_EVICTED = "Tool result evicted to free context"
_EVICTED_PREFIX = f"[{_EVICTED}"

# Anthropic packs a whole tool round into one user message as blocks of this type.
_TOOL_RESULT_BLOCK = "tool_result"


class EvictionResult(NamedTuple):
    """What Layer 1 managed to reclaim."""
    tool_results: int = 0
    tool_call_args: int = 0

    @property
    def total(self) -> int:
        return self.tool_results + self.tool_call_args


def under_pressure(context_tokens: int, context_window: int) -> bool:
    """True once the request occupies enough of the window to be worth shrinking."""
    if context_window <= 0:
        return False
    return context_tokens >= context_window * evict_threshold_ratio()


# Trailing user-turn share of the window at which reactive compact cannot help:
# Layer 2 keeps that turn intact, so an oversized single query must surface the
# provider error instead of burning an LLM summary that preserves the same text.
IRREDUCIBLE_PROMPT_RATIO = 0.9


def trailing_user_turn_start(messages: "List[Message]") -> int:
    """Index of the last real user message (not an Anthropic tool_result pack)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user" and not carries_tool_results(messages[i]):
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


def _last_batch_start(messages: "List[Message]") -> int:
    """Index where the trailing run of tool results begins.

    The pipeline runs immediately before a model call, so that trailing run is
    the batch the model has not seen yet. Evicting from it guarantees a re-run,
    and if the request is still too big without it, the answer is summarisation
    rather than throwing away the turn's own evidence.
    """
    i = len(messages)
    while i > 0 and carries_tool_results(messages[i - 1]):
        i -= 1
    return i


def _call_index(messages: "List[Message]") -> Dict[str, Tuple[Optional[str], Any]]:
    """Map tool-call id to the ``(name, arguments)`` the assistant requested.

    An Anthropic ``tool_result`` block records only ``tool_use_id``, so naming
    the evicted call means looking back at the assistant message that made it.
    """
    index: Dict[str, Tuple[Optional[str], Any]] = {}
    for msg in messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for call in msg.tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function")
            if call_id and isinstance(function, dict):
                index[call_id] = (function.get("name"), function.get("arguments"))
    return index


def _resolve_call(
    msg: "Message", block: Optional[dict], calls: Dict[str, Tuple[Optional[str], Any]]
) -> Tuple[Optional[str], Any]:
    """The ``(name, arguments)`` behind one result, whichever shape holds it."""
    if block is not None:
        return calls.get(block.get("tool_use_id") or "", (None, None))
    if msg.tool_name:
        return msg.tool_name, msg.tool_args
    return calls.get(msg.tool_call_id or "", (None, None))


def _placeholder(name: Optional[str], args: Any) -> str:
    """Name the call that produced the evicted result so it can be re-issued."""
    if not name:
        return f"[{_EVICTED}.]"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            pass
    if isinstance(args, dict):
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    elif args:
        rendered = str(args)
    else:
        rendered = ""
    if len(rendered) > _MAX_CALL_SIGNATURE_CHARS:
        rendered = rendered[:_MAX_CALL_SIGNATURE_CHARS - 3] + "..."
    return f"[{_EVICTED}: {name}({rendered}). Re-run the call if you still need it.]"


def _evictable_results(
    messages: "List[Message]", cutoff: int
) -> "Iterator[Tuple[Message, Optional[dict]]]":
    """Yield every already-seen tool result, oldest first, as ``(message, block)``.

    ``block`` is None for the OpenAI shape, where the message *is* the result;
    it is the ``tool_result`` dict for the Anthropic shape, where one message
    holds the whole round.
    """
    for msg in messages[:cutoff]:
        if msg._evicted:
            continue
        if msg.role == "tool":
            yield msg, None
            continue
        for block in tool_result_blocks(msg):
            yield msg, block


def _result_text(msg: "Message", block: Optional[dict]) -> Optional[str]:
    raw = block.get("content") if block is not None else msg.content
    if raw is None:
        return None
    return raw if isinstance(raw, str) else str(raw)


def evict_tool_results(
    messages: "List[Message]",
    *,
    context_tokens: int,
    context_window: int,
    model_id: str = "gpt-4o",
) -> int:
    """Evict the oldest tool results until the request is back under target.

    Args:
        messages:       Full message list for the current turn (mutated in place).
        context_tokens: Tokens the request currently occupies.
        context_window: Model context window. Zero disables eviction — without a
                        window there is no way to tell pressure from comfort.
        model_id:       Tokenizer selection for measuring what each eviction saves.

    Returns:
        Number of tool results whose content was replaced.
    """
    if not under_pressure(context_tokens, context_window):
        return 0

    must_save = context_tokens - int(context_window * EVICT_TARGET_RATIO)
    if must_save <= 0:
        return 0

    cutoff = _last_batch_start(messages)
    calls = _call_index(messages)
    saved = 0
    evicted = 0
    for msg, block in _evictable_results(messages, cutoff):
        text = _result_text(msg, block)
        if text is None or text.startswith(_EVICTED_PREFIX):
            continue
        if "<persisted-output>" in text:
            # Already bounded by the tool-result budget, and the path it carries
            # is the only handle on output too large to re-read into context.
            continue
        name, args = _resolve_call(msg, block, calls)
        placeholder = _placeholder(name, args)
        if len(placeholder) >= len(text):
            continue

        before = count_message_tokens(msg, model_id)
        if block is not None:
            block["content"] = placeholder
        else:
            msg.content = placeholder
        saved += before - count_message_tokens(msg, model_id)
        evicted += 1
        if _fully_evicted(msg, block):
            msg._evicted = True
        if saved >= must_save:
            break

    return evicted


def _fully_evicted(msg: "Message", block: Optional[dict]) -> bool:
    """Whether the message has nothing left worth scanning on a later pass."""
    if block is None:
        return True
    return all(
        str(b.get("content") or "").startswith(_EVICTED_PREFIX)
        for b in tool_result_blocks(msg)
    )


def shrink_tool_call_arguments(
    messages: "List[Message]",
    *,
    context_tokens: int,
    context_window: int,
    max_string_chars: int = TOOL_CALL_ARG_MAX_CHARS,
) -> int:
    """Shrink long strings inside assistant tool-call arguments (in-place).

    A ``write_file`` payload lives in the assistant message, not in a tool
    result, so eviction cannot reach it. Shrinking goes through the JSON-aware
    helper because the provider re-parses these arguments and a blunt slice
    would leave them unparseable.

    Returns:
        Number of argument strings that actually changed.
    """
    if not under_pressure(context_tokens, context_window):
        return 0

    shrunk = 0
    for msg in messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            containers = [function] if isinstance(function, dict) else []
            containers.append(tool_call)
            for container in containers:
                arguments = container.get("arguments")
                if not isinstance(arguments, str):
                    continue
                shrunken = shrink_tool_call_arguments_json(
                    arguments, max_string_chars=max_string_chars,
                )
                if shrunken != arguments:
                    container["arguments"] = shrunken
                    shrunk += 1
    return shrunk


def evict_context(
    messages: "List[Message]",
    *,
    context_tokens: int,
    context_window: int,
    model_id: str = "gpt-4o",
) -> EvictionResult:
    """Run Layer 1: reclaim context without an LLM call.

    Argument shrinking runs first because it is bounded and reaches bulk that
    eviction cannot; whatever it saves is bulk eviction then does not have to
    throw away. ``context_tokens`` is the pre-shrink measurement, so eviction
    aims slightly high — cheap, and it errs toward buying headroom rather than
    re-triggering next turn.
    """
    shrunk = shrink_tool_call_arguments(
        messages, context_tokens=context_tokens, context_window=context_window,
    )
    evicted = evict_tool_results(
        messages,
        context_tokens=context_tokens,
        context_window=context_window,
        model_id=model_id,
    )
    return EvictionResult(tool_results=evicted, tool_call_args=shrunk)
