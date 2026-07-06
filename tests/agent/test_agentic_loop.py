# -*- coding: utf-8 -*-
"""Tests for the agentic loop in Runner layer.

Tests the safety helpers (_response_has_tool_calls, _check_death_spiral,
_check_cost_budget, _check_stop_after_tool_call) and the integrated
agentic loop behavior driven by Runner._run_impl().

All tests mock LLM API keys per project convention.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.runner import Runner


# ---------------------------------------------------------------------------
# _response_has_tool_calls tests
# ---------------------------------------------------------------------------

class TestResponseHasToolCalls:
    def test_tool_results_present(self):
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="result"),
        ]
        assert Runner._response_has_tool_calls(messages) is True

    def test_no_tool_calls(self):
        messages = [
            Message(role="assistant", content="final answer"),
        ]
        assert Runner._response_has_tool_calls(messages) is False

    def test_empty_messages(self):
        assert Runner._response_has_tool_calls([]) is False

    def test_stop_after_tool_call_returns_false(self):
        messages = [
            Message(role="assistant", content="done", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="result"),
            Message(role="assistant", content="final", stop_after_tool_call=True),
        ]
        assert Runner._response_has_tool_calls(messages) is False

    def test_tool_messages_only(self):
        """Tool messages without preceding assistant → still detected."""
        messages = [
            Message(role="tool", content="result"),
        ]
        assert Runner._response_has_tool_calls(messages) is True


# ---------------------------------------------------------------------------
# _check_death_spiral tests
# ---------------------------------------------------------------------------

class TestCheckDeathSpiral:
    def test_increments_on_all_errors(self):
        state = LoopState(death_spiral_threshold=3)
        messages = [
            Message(role="assistant", content="calling tools"),
            Message(role="tool", content="error1", tool_call_error=True),
            Message(role="tool", content="error2", tool_call_error=True),
        ]
        assert Runner._check_death_spiral(messages, state) is False
        assert state.consecutive_all_error_turns == 1

    def test_resets_on_success(self):
        state = LoopState(death_spiral_threshold=3)
        state.consecutive_all_error_turns = 2
        messages = [
            Message(role="assistant", content="calling tools"),
            Message(role="tool", content="ok"),
        ]
        assert Runner._check_death_spiral(messages, state) is False
        assert state.consecutive_all_error_turns == 0

    def test_triggers_at_threshold(self):
        state = LoopState(death_spiral_threshold=2)
        messages = [
            Message(role="assistant", content=""),
            Message(role="tool", content="err", tool_call_error=True),
        ]
        Runner._check_death_spiral(messages, state)  # 1
        assert Runner._check_death_spiral(messages, state) is True  # 2 >= threshold

    def test_no_tool_messages(self):
        state = LoopState()
        messages = [Message(role="assistant", content="hello")]
        assert Runner._check_death_spiral(messages, state) is False
        assert state.consecutive_all_error_turns == 0


# ---------------------------------------------------------------------------
# _check_cost_budget tests
# ---------------------------------------------------------------------------

class TestCheckCostBudget:
    def test_no_budget_set(self):
        assert Runner._check_cost_budget(None, None) is None

    def test_under_budget(self):
        tracker = MagicMock(total_cost_usd=0.5)
        assert Runner._check_cost_budget(tracker, 1.0) is None

    def test_over_budget(self):
        tracker = MagicMock(total_cost_usd=1.5)
        result = Runner._check_cost_budget(tracker, 1.0)
        assert result is not None
        assert "exceeded" in result.lower()

    def test_exactly_at_budget(self):
        tracker = MagicMock(total_cost_usd=1.0)
        result = Runner._check_cost_budget(tracker, 1.0)
        assert result is not None  # >= triggers


# ---------------------------------------------------------------------------
# _check_stop_after_tool_call tests
# ---------------------------------------------------------------------------

class TestCheckStopAfterToolCall:
    def test_stop_flag_on_tool_message(self):
        messages = [
            Message(role="assistant", content="calling tools"),
            Message(role="tool", content="done", stop_after_tool_call=True),
        ]
        assert Runner._check_stop_after_tool_call(messages) is True

    def test_no_stop_flag(self):
        messages = [
            Message(role="assistant", content="calling tools"),
            Message(role="tool", content="result"),
        ]
        assert Runner._check_stop_after_tool_call(messages) is False

    def test_stop_flag_on_assistant(self):
        messages = [
            Message(role="assistant", content="done", stop_after_tool_call=True),
        ]
        assert Runner._check_stop_after_tool_call(messages) is True

    def test_empty_messages(self):
        assert Runner._check_stop_after_tool_call([]) is False
