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

    # API call attempts per model in Runner-level retry/fallback handling.
    # 1 means no same-model retry; fallback can still switch models immediately.
    max_api_retry: int = 1

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

    # Retryable error patterns. Match against ``str(exc).lower()``.
    # These errors are worth retrying on the same model when max_api_retry > 1.
    # Keep the list conservative: hard outages such as connection failures,
    # service unavailable, internal server error and bad gateway should switch
    # to fallback immediately instead of burning time on same-model retries.
    RETRYABLE_SUBSTRINGS: tuple = field(
        default=(
            "rate_limit", "rate limit", "429",
            "504", "gateway timeout",
            "timeout", "overloaded",
            "remote disconnected", "remotedisconnected",
            "incomplete chunked read", "chunked encoding", "premature",
        ),
        repr=False,
    )
    # Fallback-only errors: do not retry the same model; try the next fallback
    # model immediately when a fallback chain is configured.
    FALLBACK_ONLY_SUBSTRINGS: tuple = field(
        default=(
            "connection",
            "502", "503",
            "internal server error", "bad gateway", "service unavailable",
            "temporarily unavailable",
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
