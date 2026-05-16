# -*- coding: utf-8 -*-
"""Tests for Claude max_tokens resolution + output-cap error recovery.

Helpers are ported from hermes-agent; this test suite mirrors the contract
hermes asserts and adds Agentica-specific Claude integration checks.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentica.model.anthropic._max_tokens import (
    _get_anthropic_max_output,
    _resolve_positive_anthropic_max_tokens,
    resolve_anthropic_messages_max_tokens,
    parse_available_output_tokens_from_error,
)


class TestResolvePositive(unittest.TestCase):
    def test_positive_int_passes_through(self):
        self.assertEqual(_resolve_positive_anthropic_max_tokens(8192), 8192)

    def test_zero_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(0))

    def test_negative_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(-1))

    def test_none_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(None))

    def test_fractional_floored(self):
        self.assertEqual(_resolve_positive_anthropic_max_tokens(8192.7), 8192)

    def test_sub_one_float_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(0.5))

    def test_bool_returns_none(self):
        # Booleans are int subclass — must NOT silently become 1.
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(True))
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(False))

    def test_nan_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(_resolve_positive_anthropic_max_tokens(float("inf")))


class TestGetMaxOutput(unittest.TestCase):
    def test_known_model(self):
        self.assertEqual(_get_anthropic_max_output("claude-opus-4-6"), 128_000)
        self.assertEqual(_get_anthropic_max_output("claude-sonnet-4-6"), 64_000)
        self.assertEqual(_get_anthropic_max_output("claude-3-5-sonnet-20241022"), 8_192)

    def test_dotted_variant_normalized(self):
        # "claude-opus-4.6" should match the "claude-opus-4-6" key.
        self.assertEqual(_get_anthropic_max_output("anthropic/claude-opus-4.6"), 128_000)

    def test_longest_prefix_wins(self):
        # claude-3-5-sonnet must beat claude-3 prefix.
        self.assertEqual(_get_anthropic_max_output("claude-3-5-sonnet-latest"), 8_192)

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(_get_anthropic_max_output("future-model-9000"), 128_000)


class TestResolveMessagesMaxTokens(unittest.TestCase):
    def test_positive_requested_wins(self):
        self.assertEqual(
            resolve_anthropic_messages_max_tokens(8192, "claude-opus-4-6"),
            8192,
        )

    def test_none_falls_back_to_model_ceiling(self):
        self.assertEqual(
            resolve_anthropic_messages_max_tokens(None, "claude-opus-4-6"),
            128_000,
        )

    def test_negative_falls_back(self):
        self.assertEqual(
            resolve_anthropic_messages_max_tokens(-1, "claude-opus-4-6"),
            128_000,
        )

    def test_context_clamp_for_small_endpoint(self):
        # Custom endpoint with an 8K context window: output cap is clamped to 7999.
        result = resolve_anthropic_messages_max_tokens(
            128_000, "claude-opus-4-6", context_length=8000,
        )
        self.assertEqual(result, 7999)

    def test_context_clamp_noop_for_normal_models(self):
        # 200K window, 128K ceiling → no clamp.
        result = resolve_anthropic_messages_max_tokens(
            None, "claude-opus-4-6", context_length=200_000,
        )
        self.assertEqual(result, 128_000)


class TestParseAvailableOutput(unittest.TestCase):
    def test_anthropic_format(self):
        err = (
            "max_tokens: 32768 > context_window: 200000 - "
            "input_tokens: 190000 = available_tokens: 10000"
        )
        self.assertEqual(parse_available_output_tokens_from_error(err), 10000)

    def test_alternative_spacing(self):
        err = "max_tokens too large; available tokens 4096 remain"
        self.assertEqual(parse_available_output_tokens_from_error(err), 4096)

    def test_prompt_too_long_returns_none(self):
        # Different error class — must NOT match.
        err = "Prompt is too long: 250000 tokens > 200000 maximum"
        self.assertIsNone(parse_available_output_tokens_from_error(err))

    def test_unrelated_error_returns_none(self):
        self.assertIsNone(parse_available_output_tokens_from_error("rate limit hit"))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_available_output_tokens_from_error(""))


class TestClaudeRequestKwargs(unittest.TestCase):
    """Verify Claude.request_kwargs uses the resolver."""

    def test_default_uses_model_ceiling(self):
        from agentica.model.anthropic.claude import Claude
        model = Claude(id="claude-opus-4-6", api_key="fake")
        kwargs = model.request_kwargs
        self.assertEqual(kwargs["max_tokens"], 128_000)

    def test_user_value_passes_through(self):
        from agentica.model.anthropic.claude import Claude
        model = Claude(id="claude-opus-4-6", api_key="fake", max_tokens=4096)
        self.assertEqual(model.request_kwargs["max_tokens"], 4096)

    def test_small_context_window_clamps_output(self):
        from agentica.model.anthropic.claude import Claude
        # Custom small endpoint
        model = Claude(
            id="claude-opus-4-6", api_key="fake",
            max_tokens=64_000, context_window=8000,
        )
        self.assertEqual(model.request_kwargs["max_tokens"], 7999)


class TestMaxTokensRetry(unittest.TestCase):
    """Verify invoke() / invoke_stream() retry on max_tokens-too-large."""

    def test_invoke_retries_on_output_cap_error(self):
        from agentica.model.anthropic.claude import Claude

        err = Exception(
            "max_tokens: 32768 > context_window: 200000 - "
            "input_tokens: 195000 = available_tokens: 5000"
        )
        success_response = MagicMock(name="anthropic_response")

        model = Claude(id="claude-opus-4-6", api_key="fake")

        # Mock the client's messages.create: fail first, succeed second.
        mock_create = AsyncMock(side_effect=[err, success_response])
        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        model.client = mock_client

        # Bypass format_messages async work with a minimal stub.
        async def _fake_format(_msgs):
            return ([{"role": "user", "content": "hi"}], "sys")
        model.format_messages = _fake_format  # type: ignore[assignment]

        result = asyncio.run(model.invoke([]))

        self.assertIs(result, success_response)
        self.assertEqual(mock_create.await_count, 2)
        # Second call must use the recovered cap: 5000 - 64 = 4936.
        second_kwargs = mock_create.await_args_list[1].kwargs
        self.assertEqual(second_kwargs["max_tokens"], 4936)

    def test_invoke_does_not_retry_unrelated_error(self):
        from agentica.model.anthropic.claude import Claude

        err = Exception("rate limit exceeded")
        model = Claude(id="claude-opus-4-6", api_key="fake")

        mock_create = AsyncMock(side_effect=err)
        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        model.client = mock_client

        async def _fake_format(_msgs):
            return ([{"role": "user", "content": "hi"}], "sys")
        model.format_messages = _fake_format  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            asyncio.run(model.invoke([]))

        self.assertIn("rate limit", str(ctx.exception))
        self.assertEqual(mock_create.await_count, 1)  # no retry


if __name__ == "__main__":
    unittest.main()
