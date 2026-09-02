# -*- coding: utf-8 -*-
"""Oversized single-query errors must surface the provider message, not hide
behind a useless reactive compact that preserves the same trailing user turn.
"""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.evict import is_irreducible_prompt_too_long
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse


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


TAIJI_400 = (
    "Error code: 400 - {'error': {'message': "
    "'Input tokens exceed the configured limit of 192000 tokens. "
    "Your messages resulted in 192547 tokens. "
    "Please reduce the length of the messages.', "
    "'type': 'invalid_request_error', 'param': '', "
    "'code': 'context_length_exceeded'}}"
)


class TestStreamPromptTooLongReactiveCompact(unittest.TestCase):
    """Streaming path: the 400 surfaces while CONSUMING the stream, never
    through _call_with_retry's classifier — so its reactive-compact recovery
    must be mirrored at the consumption site or the CLI (streaming by
    default) has no safety net once the context passes the model's real
    input limit (hy4-preview: catalog says 1M, deployment caps input at
    192k, so no proactive layer ever fires).
    """

    def _make_model(self) -> "OpenAIChat":
        model = OpenAIChat(id="hy4-preview", api_key="fake_openai_key")
        # models.dev catalog value for hy4-preview — 5x the deployment's
        # real 192k input limit, so proactive evict/auto-compact never fire.
        model.context_window = 1_024_000
        return model

    def test_stream_prompt_too_long_compacts_and_retries(self):
        from agentica.agent import Agent
        from agentica.model.response import ModelResponseEvent

        model = self._make_model()
        calls = {"n": 0, "msgs": []}

        async def _ok_stream(messages):
            messages.append(Message(role="assistant", content="ok"))
            yield ModelResponse(
                event=ModelResponseEvent.assistant_response.value, content="ok"
            )

        async def _fail_then_recover(messages):
            calls["n"] += 1
            calls["msgs"].append(list(messages))
            if calls["n"] == 1:
                # Raised on first __anext__: the request (and its 400) only
                # happens at consumption time, exactly like the real stream.
                raise Exception(TAIJI_400)
                yield  # pragma: no cover — marks this an async generator
            messages.append(Message(role="assistant", content="recovered"))
            yield ModelResponse(
                event=ModelResponseEvent.assistant_response.value, content="recovered"
            )

        async def _invoke(msgs):
            # Summarisation call made by reactive compact.
            return SimpleNamespace(content="summary: the user worked on the benchmark")

        model.invoke = _invoke
        agent = Agent(
            name="stream-reactive-compact-agent",
            model=model,
            add_history_to_context=True,
        )
        agent._run_fallback_models = []

        # Seed enough history that compaction visibly shrinks the request.
        model.response_stream = _ok_stream
        for turn in ("first turn", "second turn", "third turn"):
            for _ in agent.run_stream_sync(turn):
                pass

        model.response_stream = _fail_then_recover
        for _ in agent.run_stream_sync("继续"):
            pass

        self.assertEqual(calls["n"], 2, "first 400 must trigger one compact + retry")
        self.assertEqual(agent.run_response.content, "recovered")
        second_request = calls["msgs"][1]
        self.assertLess(len(second_request), len(calls["msgs"][0]))
        self.assertTrue(
            any(
                m.role == "user" and "[Context compressed]" in str(m.content)
                for m in second_request
            )
        )
        self.assertEqual(agent.run_response.context_compactions, 1)

    def test_stream_irreducible_prompt_too_long_raises_without_compact(self):
        from agentica.agent import Agent

        model = self._make_model()
        model.context_window = 192_000  # window already learned from the error

        async def _stream(messages):
            raise Exception(TAIJI_400)
            yield  # pragma: no cover

        model.response_stream = _stream
        agent = Agent(name="stream-oversized-query", model=model)
        agent._run_fallback_models = []

        with patch(
            "agentica.runner.compress.CompressMixin._try_reactive_compact",
            new_callable=AsyncMock,
        ) as reactive:
            with self.assertRaises(Exception) as ctx:
                # One user message larger than the whole window: compacting
                # history cannot help, the provider error must surface.
                for _ in agent.run_stream_sync("x " * 200_000):
                    pass
            reactive.assert_not_called()

        self.assertIn("context_length_exceeded", str(ctx.exception))
        self.assertIn("192000", str(ctx.exception))


class TestLearnContextLimitFromProviderError(unittest.TestCase):
    """The provider tells us the real input limit in the error text; agentica
    must learn it so later turns compact against the deployment's window
    instead of the (wrong) catalog value.
    """

    def test_taiji_input_limit_shape(self):
        model = OpenAIChat(id="hy4-preview", api_key="fake_openai_key")
        model.context_window = 1_024_000  # wrong models.dev catalog value
        model._learn_context_limit_from_error(TAIJI_400)
        self.assertEqual(model.context_window, 192_000)

    def test_rate_limit_shape_not_confused(self):
        model = OpenAIChat(id="hy4-preview", api_key="fake_openai_key")
        model.context_window = 1_024_000
        model._learn_context_limit_from_error(
            "Rate limit reached: 10000 requests per minute, limit of 10000"
        )
        self.assertEqual(model.context_window, 1_024_000)


if __name__ == "__main__":
    unittest.main()
