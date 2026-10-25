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
re-issue it. It is deliberately *not* copied to a spill file: the session log
already holds the full result. When the agent can read that log
(``recoverable``: a session log plus ``search_session``), the placeholder also
quotes the tool-call id, which ``search_session(uuid=...)`` resolves to the
original output. Re-running is not an equivalent recovery: a non-idempotent
call (``execute`` reading changing state) returns different evidence, and a
call whose arguments were dropped cannot be re-run at all.

Tool *results* are not the only bulk in a transcript: a ``write_file`` or
``apply_patch`` call carries its whole payload in the assistant message's
arguments, which no amount of result eviction can reach. Those strings are
replaced with a JSON object ``{"$evicted": N}`` (not a string and not a
truncated prefix — either form is copied into the next ``send_message`` /
``write_file`` / ``execute``). The live tool round is
left intact (in-flight calls and the trailing result batch), matching the
trailing-result exclusion: that payload is what just executed. The cutoff is
by message position, not tool name — SDK ``tools=``, CLI ``--tools``, Web
extra tools, and MCP share the round with builtins. When Layer 1
cannot get the request under target, the answer is Layer 2
(``CompressionManager.auto_compact``), not eating the round in progress.

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

from agentica.compression.tool_call_args import (
    OMITTED_ARG_KEY,
    is_omitted_tool_arg,
    omitted_tool_args,
    rewrite_omitted_arg_leaves,
    shrink_tool_arg_leaves,
)
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
    """The ``(name, arguments)`` behind one result, whichever shape holds it.

    Prefer the assistant ``tool_calls`` index: Layer 1 rewrites those
    arguments in place. ``Message.tool_args`` is a snapshot from execute
    time and still holds the original payload (or nothing), so a
    placeholder built from it would invite the model to re-issue
    ``send_message`` with a body that is already gone.
    """
    if block is not None:
        return calls.get(block.get("tool_use_id") or "", (None, None))
    indexed = calls.get(msg.tool_call_id or "", (None, None))
    if indexed != (None, None):
        return indexed
    if msg.tool_name:
        return msg.tool_name, msg.tool_args
    return indexed


def _render_arg_value(value: Any) -> str:
    """Render one argument for the result placeholder, never the raw marker."""
    if is_omitted_tool_arg(value):
        if isinstance(value, dict):
            return f"<omitted chars={value[OMITTED_ARG_KEY]}>"
        return "<omitted>"
    return repr(value)


def _result_call_id(msg: "Message", block: Optional[dict]) -> Optional[str]:
    """The tool-call id a result answers, whichever shape holds it."""
    if block is not None:
        return block.get("tool_use_id") or None
    return msg.tool_call_id or None


def _placeholder(name: Optional[str], args: Any, handle: Optional[str] = None) -> str:
    """Name the call that produced the evicted result.

    Re-issue is only invited when the original arguments are still in the
    transcript. After Layer 1 drops a payload, telling the model to re-run
    ``send_message`` / ``execute`` / ``write_file`` with the marker as the
    body is how that marker reaches a peer, a shell, or the disk.

    ``handle`` is the tool-call id under which the session log keeps the full
    result; it is only passed when the model can actually read that log.
    """
    recover = f' Original output: search_session(uuid="{handle}").' if handle else ""
    if not name:
        return f"[{_EVICTED}.{recover}]"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            pass
    omitted = omitted_tool_args(args) if args is not None else []
    if isinstance(args, dict):
        rendered = ", ".join(f"{k}={_render_arg_value(v)}" for k, v in args.items())
    elif args:
        rendered = _render_arg_value(args)
    else:
        rendered = ""
    if len(rendered) > _MAX_CALL_SIGNATURE_CHARS:
        rendered = rendered[:_MAX_CALL_SIGNATURE_CHARS - 3] + "..."
    if omitted:
        return (
            f"[{_EVICTED}: {name}({rendered}). "
            f"The argument payload was dropped from context and cannot be "
            f"recovered; do not re-issue this call.{recover}]"
        )
    return (
        f"[{_EVICTED}: {name}({rendered}). "
        f"Re-run the call if you still need it.{recover}]"
    )


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
    recoverable: bool = False,
) -> int:
    """Evict the oldest tool results until the request is back under target.

    Args:
        messages:       Full message list for the current turn (mutated in place).
        context_tokens: Tokens the request currently occupies.
        context_window: Model context window. Zero disables eviction — without a
                        window there is no way to tell pressure from comfort.
        model_id:       Tokenizer selection for measuring what each eviction saves.
        recoverable:    The model can read the session log (``search_session``),
                        so each placeholder quotes the tool-call id to fetch the
                        original output with.

    Returns:
        Number of tool results whose content was replaced.
    """
    if not under_pressure(context_tokens, context_window):
        return 0

    must_save = context_tokens - int(context_window * EVICT_TARGET_RATIO)
    if must_save <= 0:
        return 0

    cutoff = live_tool_round_start(messages)
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
        placeholder = _placeholder(
            name, args, _result_call_id(msg, block) if recoverable else None,
        )
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


def upgrade_legacy_omitted_args(messages: "List[Message]") -> int:
    """Rewrite leftover string markers in every assistant call, including live.

    Those strings are already in the transcript. Leaving them is how the
    model copies a marker into ``send_message``.
    """
    changed = 0
    for msg in messages:
        if msg.role == "assistant":
            changed += _map_assistant_tool_args(msg, rewrite_omitted_arg_leaves)
    return changed


def shrink_tool_call_arguments(
    messages: "List[Message]",
    *,
    context_tokens: int,
    context_window: int,
    max_string_chars: int = TOOL_CALL_ARG_MAX_CHARS,
) -> int:
    """Omit long strings inside older assistant tool-call arguments (in-place).

    A ``write_file`` payload lives in the assistant message, not in a tool
    result, so eviction cannot reach it. The live tool round is skipped
    (``live_tool_round_start``). Older turns keep JSON valid. OpenAI stores
    the payload in ``tool_calls[].function.arguments``; Anthropic also keeps
    it in ``content`` ``tool_use.input`` — both shrink together.
    """
    if not under_pressure(context_tokens, context_window):
        return 0

    start = live_tool_round_start(messages)
    shrunk = 0
    for msg in messages[:start]:
        if msg.role != "assistant":
            continue
        shrunk += _map_assistant_tool_args(
            msg, lambda value: shrink_tool_arg_leaves(value, max_string_chars),
        )
    return shrunk


def evict_context(
    messages: "List[Message]",
    *,
    context_tokens: int,
    context_window: int,
    model_id: str = "gpt-4o",
    recoverable: bool = False,
) -> EvictionResult:
    """Run Layer 1: reclaim context without an LLM call.

    Argument shrinking runs first because it is bounded and reaches bulk that
    eviction cannot; whatever it saves is bulk eviction then does not have to
    throw away. ``context_tokens`` is the pre-shrink measurement, so eviction
    aims slightly high — cheap, and it errs toward buying headroom rather than
    re-triggering next turn.
    """
    upgrade_legacy_omitted_args(messages)
    shrunk = shrink_tool_call_arguments(
        messages, context_tokens=context_tokens, context_window=context_window,
    )
    evicted = evict_tool_results(
        messages,
        context_tokens=context_tokens,
        context_window=context_window,
        model_id=model_id,
        recoverable=recoverable,
    )
    return EvictionResult(tool_results=evicted, tool_call_args=shrunk)
