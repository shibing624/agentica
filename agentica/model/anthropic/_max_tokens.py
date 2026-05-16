# -*- coding: utf-8 -*-
"""
Anthropic max_tokens / output-cap resolution helpers.

Two distinct concepts that the Anthropic API confusingly conflates:

* ``max_tokens``    — OUTPUT token cap for a single response. Anthropic's
                      Messages API names it ``max_tokens`` even though it
                      only bounds the *output*. It is mandatory; the SDK
                      raises if absent.
* ``context_window``— TOTAL window (input + output). The API enforces:
                      ``input_tokens + max_tokens <= context_window``.

These helpers (ported from hermes-agent/agent/anthropic_adapter.py +
model_metadata.py) let the Claude model resolve a per-call output cap and
recover from the "max_tokens too large given prompt" error by parsing the
``available_tokens`` figure out of the API error message.
"""
import math
import re
from typing import Optional


# ── Max output token limits per Anthropic model ────────────────────────────
# Source: Anthropic docs. Hardcoding 8192 (the historical default) starves
# thinking-enabled models because thinking tokens count toward the limit.
_ANTHROPIC_OUTPUT_LIMITS = {
    # Claude 4.7
    "claude-opus-4-7":   128_000,
    # Claude 4.6
    "claude-opus-4-6":   128_000,
    "claude-sonnet-4-6":  64_000,
    # Claude 4.5
    "claude-opus-4-5":    64_000,
    "claude-sonnet-4-5":  64_000,
    "claude-haiku-4-5":   64_000,
    # Claude 4
    "claude-opus-4":      32_000,
    "claude-sonnet-4":    64_000,
    # Claude 3.7
    "claude-3-7-sonnet": 128_000,
    # Claude 3.5
    "claude-3-5-sonnet":   8_192,
    "claude-3-5-haiku":    8_192,
    # Claude 3
    "claude-3-opus":       4_096,
    "claude-3-sonnet":     4_096,
    "claude-3-haiku":      4_096,
}

# Fallback ceiling for unknown / future model IDs.
_ANTHROPIC_DEFAULT_OUTPUT_LIMIT = 128_000


def _get_anthropic_max_output(model: str) -> int:
    """Look up the max output token limit for an Anthropic model.

    Uses longest-substring matching so date-stamped model IDs
    (``claude-sonnet-4-5-20250929``) and dotted variants
    (``anthropic/claude-opus-4.6``) resolve correctly.
    """
    m = model.lower().replace(".", "-")
    best_key = ""
    best_val = _ANTHROPIC_DEFAULT_OUTPUT_LIMIT
    for key, val in _ANTHROPIC_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def _resolve_positive_anthropic_max_tokens(value) -> Optional[int]:
    """Floor ``value`` to a positive int; return None if not finite positive.

    Anthropic rejects max_tokens=0, negative, non-integer, or non-finite with
    HTTP 400. Python's ``or`` idiom catches 0 but lets -1 / 0.5 / NaN slip
    through; this helper fails locally instead.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value):
            return None
    except Exception:
        return None
    floored = int(value)
    return floored if floored > 0 else None


def resolve_anthropic_messages_max_tokens(
    requested,
    model: str,
    context_length: Optional[int] = None,
) -> int:
    """Resolve the effective output cap for an Anthropic Messages call.

    Resolution order:
      1. ``requested`` if it is a positive finite number.
      2. Model-table ceiling from ``_get_anthropic_max_output(model)``.

    If ``context_length`` is provided and the resolved cap exceeds it (small
    custom endpoints), clamp to ``max(context_length - 1, 1)``. For
    full-size models the context window (e.g. 200K) is always larger than
    the output ceiling (e.g. 128K), so the clamp is a no-op.

    NOTE: this clamping does NOT account for prompt size — if the prompt is
    large, Anthropic may still reject the request. Callers must detect
    "max_tokens too large given prompt" errors and retry with
    ``parse_available_output_tokens_from_error``.
    """
    resolved = _resolve_positive_anthropic_max_tokens(requested)
    if resolved is None:
        resolved = _get_anthropic_max_output(model)
    if resolved <= 0:
        raise ValueError(
            f"Could not resolve a positive max_tokens budget for model {model!r}; "
            f"requested={requested!r}."
        )
    if context_length and resolved > context_length:
        resolved = max(context_length - 1, 1)
    return resolved


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Detect "max_tokens too large given prompt" and return the available cap.

    Anthropic returns errors shaped like::

        "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000
         = available_tokens: 10000"

    Returns the available output tokens (e.g. 10000), or None when the error
    is NOT an output-cap error (e.g. prompt-too-long, which needs a
    different recovery path: shrink context_length + compress history).
    """
    if not error_msg:
        return None
    error_lower = error_msg.lower()

    # Must look like an output-cap error, not a prompt-length error.
    if "max_tokens" not in error_lower:
        return None
    if "available_tokens" not in error_lower and "available tokens" not in error_lower:
        return None

    patterns = [
        r"available_tokens[:\s]+(\d+)",
        r"available\s+tokens[:\s]+(\d+)",
        # fallback: trailing number after "=" in expressions like "200000 - 190000 = 10000"
        r"=\s*(\d+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            tokens = int(match.group(1))
            if tokens >= 1:
                return tokens
    return None
