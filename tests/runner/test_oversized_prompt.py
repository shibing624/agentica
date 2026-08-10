# -*- coding: utf-8 -*-
"""Oversized single-query errors must surface the provider message, not hide
behind a useless reactive compact that preserves the same trailing user turn.
"""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.evict import is_irreducible_prompt_too_long
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat


class TestIrreduciblePromptTooLong(unittest.TestCase):
    def test_huge_trailing_user_turn_is_irreducible(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="x" * 50_000),
        ]
        self.assertTrue(
            is_irreducible_prompt_too_long(
                msgs, context_window=2_000, model_id="gpt-4o",
            )
        )

    def test_short_user_with_long_history_is_reducible(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="old " * 5_000),
            Message(role="assistant", content="old answer " * 5_000),
            Message(role="user", content="short question"),
        ]
        self.assertFalse(
            is_irreducible_prompt_too_long(
                msgs, context_window=200_000, model_id="gpt-4o",
            )
        )


class TestOversizedQuerySurfacesProviderError(unittest.TestCase):
    def test_irreducible_prompt_too_long_raises_original_without_reactive(self):
        from agentica.agent import Agent
        from agentica.runner import Runner

        model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
        model.context_window = 4_000
        agent = Agent(model=model)
        agent._run_fallback_models = []
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="填充 " * 20_000),
        ]
        provider_err = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This model's maximum context length is 4000 tokens. "
            "However, your messages resulted in 16000 tokens. "
            "Please reduce the length of the messages.\", "
            "'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}"
        )
        model.response = AsyncMock(side_effect=provider_err)
        state = LoopState()

        with patch(
            "agentica.runner.compress.CompressMixin._try_reactive_compact",
            new_callable=AsyncMock,
        ) as reactive:
            with self.assertRaises(Exception) as ctx:
                asyncio.run(
                    Runner._call_with_retry(model, messages, state, agent, stream=False)
                )
            reactive.assert_not_called()

        self.assertIn("context_length_exceeded", str(ctx.exception))
        self.assertIn("4000 tokens", str(ctx.exception))


class TestContextLengthCliFormatting(unittest.TestCase):
    def test_cli_labels_context_length_clearly(self):
        from agentica.cli.display.console import _format_agent_execution_error

        err = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This model's maximum context length is 128000 tokens. "
            "However, your messages resulted in 160400 tokens.\", "
            "'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}"
        )
        view = _format_agent_execution_error(err)
        self.assertEqual(view["summary"], "Input exceeds model context window")
        self.assertIn("128000", view["detail"])
        self.assertIn("Shorten the message", view["hint"] or "")


if __name__ == "__main__":
    unittest.main()
