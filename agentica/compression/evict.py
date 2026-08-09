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
"""
from typing import List, NamedTuple, TYPE_CHECKING

from agentica.compression.tool_call_args import shrink_tool_call_arguments_json
from agentica.utils.tokens import count_message_tokens

if TYPE_CHECKING:
    from agentica.model.message import Message

# Occupancy at which eviction starts, as a fraction of the context window.
EVICT_THRESHOLD_RATIO = 0.7

# Occupancy eviction aims to reach. Strictly below the threshold so one pass
# buys several turns of headroom instead of re-triggering every turn.
EVICT_TARGET_RATIO = 0.5

# Head kept from each long string inside tool-call arguments.
TOOL_CALL_ARG_MAX_CHARS = 150

# Cap on the rendered call arguments inside a placeholder.
_MAX_CALL_SIGNATURE_CHARS = 120

_EVICTED = "Tool result evicted to free context"


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
    return context_tokens >= context_window * EVICT_THRESHOLD_RATIO


def _last_batch_start(messages: "List[Message]") -> int:
    """Index where the trailing run of tool results begins.

    The pipeline runs immediately before a model call, so that trailing run is
    the batch the model has not seen yet. Evicting from it guarantees a re-run,
    and if the request is still too big without it, the answer is summarisation
    rather than throwing away the turn's own evidence.
    """
    i = len(messages)
    while i > 0 and messages[i - 1].role == "tool":
        i -= 1
    return i


def _placeholder(msg: "Message") -> str:
    """Name the call that produced the evicted result so it can be re-issued."""
    if not msg.tool_name:
        return f"[{_EVICTED}.]"
    args = msg.tool_args
    if isinstance(args, dict):
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    elif args:
        rendered = str(args)
    else:
        rendered = ""
    if len(rendered) > _MAX_CALL_SIGNATURE_CHARS:
        rendered = rendered[:_MAX_CALL_SIGNATURE_CHARS - 3] + "..."
    return f"[{_EVICTED}: {msg.tool_name}({rendered}). Re-run the call if you still need it.]"


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
        Number of messages whose content was replaced.
    """
    if not under_pressure(context_tokens, context_window):
        return 0

    must_save = context_tokens - int(context_window * EVICT_TARGET_RATIO)
    if must_save <= 0:
        return 0

    cutoff = _last_batch_start(messages)
    saved = 0
    evicted = 0
    for msg in messages[:cutoff]:
        if msg.role != "tool" or msg._evicted:
            continue
        content = msg.content
        if content is None:
            continue
        content_str = content if isinstance(content, str) else str(content)
        if "<persisted-output>" in content_str:
            # Already bounded by the tool-result budget, and the path it carries
            # is the only handle on output too large to re-read into context.
            continue
        placeholder = _placeholder(msg)
        if len(placeholder) >= len(content_str):
            continue

        before = count_message_tokens(msg, model_id)
        msg.content = placeholder
        msg._evicted = True
        saved += before - count_message_tokens(msg, model_id)
        evicted += 1
        if saved >= must_save:
            break

    return evicted


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
