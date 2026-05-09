# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Centralized state for the agentic tool loop.

Replaces scattered per-run counters (_loop_turn_count, _max_tokens_recovery_count,
_reactive_compact_done, _consecutive_all_error_turns) that were stored as mutable
attributes on the Model instance.

Created fresh at the start of each agentic loop in Runner._run_impl().
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoopState:
    """Centralized state for one agentic loop invocation.

    All counters are per-loop (not per-run). A fresh LoopState is created
    each time Runner._run_impl() enters its agentic loop.
    """

    # Turn tracking. None = no limit (main agent default).
    # Subagents set max_turns=100 as a safety net.
    turn_count: int = 0
    max_turns: Optional[int] = None

    # Max-tokens recovery (finish_reason == "length")
    max_tokens_recovery_count: int = 0
    max_tokens_recovery_limit: int = 3

    # API retry ceiling
    max_api_retry: int = 3

    # Death spiral detection
    consecutive_all_error_turns: int = 0
    death_spiral_threshold: int = 5

    # Reactive compact (one-shot per loop invocation)
    reactive_compact_done: bool = False

    # Cross-provider fallback bookkeeping for the most recent successful LLM call.
    # Set by Runner._call_with_retry; consumed by Runner._run_impl to keep
    # RunResponse.model truthful when a fallback (not the primary) actually
    # answered. Resets on every call -- this is per-call state, not per-run.
    last_used_model_id: Optional[str] = None
    # Index of the model in [primary, *fallbacks] that produced last response.
    # 0 = primary, 1+ = fallback index. -1 means no successful call yet.
    last_used_model_idx: int = -1

    # Retryable error patterns
    RETRYABLE_SUBSTRINGS: tuple = field(
        default=(
            "rate_limit", "rate limit", "429", "503", "502",
            "connection", "timeout", "overloaded",
        ),
        repr=False,
    )
    PROMPT_TOO_LONG_HINTS: tuple = field(
        default=(
            "prompt_too_long", "context_length_exceeded",
            "maximum context", "too many tokens", "413",
        ),
        repr=False,
    )
    # Provider-side content moderation. Detected as either:
    #   - normal-return finish_reason (OpenAI / Azure / OpenAI-compat providers)
    #   - exception text from providers that raise instead of returning a flag
    # When hit, switch to the next fallback model in the chain (no backoff —
    # retrying the same model is pointless because the moderator is deterministic).
    CONTENT_FILTER_FINISH_REASONS: tuple = field(
        default=("content_filter", "content-filter", "content_filtered"),
        repr=False,
    )
    CONTENT_FILTER_HINTS: tuple = field(
        default=(
            "content_filter", "content filter", "content-filter",
            "content_policy", "content policy", "responsibleaipolicyviolation",
        ),
        repr=False,
    )
