# -*- coding: utf-8 -*-
"""Tests for LoopState dataclass."""
from agentica.model.loop_state import LoopState


def test_default_values():
    state = LoopState()
    assert state.turn_count == 0
    assert state.max_tokens_recovery_count == 0
    assert state.max_tokens_recovery_limit == 3
    assert state.max_api_retry == 1
    assert state.consecutive_all_error_turns == 0
    assert state.death_spiral_threshold == 5
    assert state.reactive_compact_done is False


def test_custom_values():
    state = LoopState(
        death_spiral_threshold=10,
        max_api_retry=5,
        max_tokens_recovery_limit=1,
    )
    assert state.death_spiral_threshold == 10
    assert state.max_api_retry == 5
    assert state.max_tokens_recovery_limit == 1


def test_retryable_substrings():
    state = LoopState()
    assert "rate_limit" in state.RETRYABLE_SUBSTRINGS
    assert "429" in state.RETRYABLE_SUBSTRINGS
    assert "timeout" in state.RETRYABLE_SUBSTRINGS
    assert "connection" not in state.RETRYABLE_SUBSTRINGS
    assert "service unavailable" not in state.RETRYABLE_SUBSTRINGS
    assert "internal server error" not in state.RETRYABLE_SUBSTRINGS
    assert "bad gateway" not in state.RETRYABLE_SUBSTRINGS


def test_fallback_only_substrings():
    state = LoopState()
    assert "connection" in state.FALLBACK_ONLY_SUBSTRINGS
    assert "service unavailable" in state.FALLBACK_ONLY_SUBSTRINGS
    assert "internal server error" in state.FALLBACK_ONLY_SUBSTRINGS
    assert "bad gateway" in state.FALLBACK_ONLY_SUBSTRINGS


def test_prompt_too_long_hints():
    state = LoopState()
    assert "prompt_too_long" in state.PROMPT_TOO_LONG_HINTS
    assert "413" in state.PROMPT_TOO_LONG_HINTS
    assert "context_length_exceeded" in state.PROMPT_TOO_LONG_HINTS


def test_turn_count_increments():
    state = LoopState()
    state.turn_count += 1
    assert state.turn_count == 1
    state.turn_count += 1
    assert state.turn_count == 2


def test_max_turns_default_none():
    """Main agent: max_turns defaults to None (unlimited). Only subagents set it."""
    state = LoopState()
    assert state.max_turns is None
