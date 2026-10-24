# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Harness resilience — provider errors that killed a whole run in production.

Every case here is taken from a real long-run session log. Each one used to end
the run with a raw provider exception instead of being retried, compacted or
routed to a fallback.
"""

import asyncio
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai

from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.runner import Runner
from agentica.runner.retry_fallback import (
    MAX_RETRY_WAIT_SECONDS,
    _is_rate_limited,
    _retry_wait_seconds,
)


def _request():
    return httpx.Request("POST", "https://example.invalid")


def _make_agent():
    from agentica.agent import Agent

    agent = Agent(name="resilience")
    agent._run_fallback_models = []
    return agent


def _fake_model(model_id):
    from agentica.model.base import Model

    model = MagicMock(spec=Model)
    model.id = model_id
    model.provider = "openai"
    model.context_window = 200000
    model.retryable_substrings = ()
    model.get_retryable_substrings = lambda defaults: tuple(defaults)
    model.has_compatible_native_checkpoint = lambda messages: False
    model.response = AsyncMock()
    return model


class TestSSEErrorEventClassification(unittest.TestCase):
    """A mid-stream ``error`` event arrives as a bare ``APIError`` with no status.

    The OpenAI SDK raises it from the SSE frame (``_streaming.py``), so
    ``status_code`` is missing and the previous status-only classification fell
    through to the substring list, which does not mention it. Production saw 18
    of these (``server_error``) end a run outright.
    """

    def _classify(self, exc):
        state = LoopState()
        return Runner._classify_api_error(
            exc, state.RETRYABLE_SUBSTRINGS, state.FALLBACK_ONLY_SUBSTRINGS,
        )

    def test_server_error_event_is_fallback_only(self):
        exc = openai.APIError(
            "The server had an error processing your request. Sorry about that!",
            request=_request(),
            body={
                "message": "The server had an error processing your request. Sorry about that!",
                "type": "server_error",
                "param": None,
                "code": None,
            },
        )
        self.assertEqual(self._classify(exc), (False, True))

    def test_rate_limit_event_is_retryable(self):
        exc = openai.APIError(
            "Rate limit reached for requests",
            request=_request(),
            body={"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"},
        )
        self.assertEqual(self._classify(exc), (True, False))

    def test_connection_error_event_is_fallback_only(self):
        exc = openai.APIError(
            "Connection error.",
            request=_request(),
            body={"message": "Connection error.", "type": "connection_error"},
        )
        self.assertEqual(self._classify(exc), (False, True))

    def test_unrelated_invalid_request_event_stays_non_retryable(self):
        exc = openai.APIError(
            "Invalid value for 'temperature'",
            request=_request(),
            body={"message": "Invalid value for 'temperature'", "type": "invalid_request_error"},
        )
        self.assertEqual(self._classify(exc), (False, False))

    def test_server_error_event_switches_to_fallback_instead_of_dying(self):
        exc = openai.APIError(
            "The server had an error processing your request.",
            request=_request(),
            body={"message": "The server had an error processing your request.", "type": "server_error"},
        )
        primary = _fake_model("primary")
        primary.response = AsyncMock(side_effect=exc)
        fallback = _fake_model("fallback")
        fallback.response = AsyncMock(return_value=ModelResponse(content="rescued", finish_reason="stop"))
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        result = asyncio.run(
            Runner._call_with_retry(primary, [Message(role="user", content="hi")], LoopState(), agent)
        )

        self.assertEqual(result.content, "rescued")
        self.assertEqual(result.used_model.id, "fallback")


class TestContextOverflowFromSSEEvent(unittest.TestCase):
    """The overflow arrives as a bare ``APIError`` whose code is not the OpenAI one.

    ``str(exc)`` is the only place the provider names the condition, and the
    message does not contain any of the configured hints. Production lost a
    3.5h run to this eight times; the request never reached reactive compact.
    """

    def test_overflow_event_reaches_reactive_compact(self):
        exc = openai.APIError(
            "Your input exceeds the context window of this model. Please adjust your input and try again.",
            request=_request(),
            body={
                "message": "Your input exceeds the context window of this model. "
                "Please adjust your input and try again.",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            },
        )
        self.assertTrue(LoopState().is_prompt_too_long(exc))

    def test_gateway_overflow_shape_is_recognized(self):
        """Measured against the live gateway: 400 / code=InvalidParameter."""
        exc = openai.BadRequestError(
            "Input exceeds the context limit (1048568 tokens). Please shorten the input.",
            response=httpx.Response(
                400,
                request=_request(),
                json={
                    "message": "Input exceeds the context limit (1048568 tokens). Please shorten the input.",
                    "type": "BadRequest",
                    "param": "",
                    "code": "InvalidParameter",
                },
            ),
            body={
                "message": "Input exceeds the context limit (1048568 tokens). Please shorten the input.",
                "type": "BadRequest",
                "param": "",
                "code": "InvalidParameter",
            },
        )
        self.assertTrue(LoopState().is_prompt_too_long(exc))


class TestRateLimitBackoff(unittest.TestCase):
    """A 429 was retried after 1-3s; the provider's own window is a minute."""

    def test_retry_after_header_sets_the_wait(self):
        exc = openai.RateLimitError(
            "TPM limit",
            response=httpx.Response(429, request=_request(), headers={"retry-after": "42"}),
            body={"message": "TPM limit", "code": "4029"},
        )
        self.assertEqual(_retry_wait_seconds(exc, attempt=0), 42.0)

    def test_retry_after_is_capped(self):
        exc = openai.RateLimitError(
            "TPM limit",
            response=httpx.Response(429, request=_request(), headers={"retry-after": "9999"}),
            body={"message": "TPM limit"},
        )
        self.assertEqual(_retry_wait_seconds(exc, attempt=0), MAX_RETRY_WAIT_SECONDS)

    def test_without_header_a_429_backs_off_further_than_one_second(self):
        exc = openai.RateLimitError(
            "TPM limit",
            response=httpx.Response(429, request=_request()),
            body={"message": "TPM limit"},
        )
        wait = _retry_wait_seconds(exc, attempt=0)
        self.assertGreaterEqual(wait, 15.0)
        self.assertTrue(_is_rate_limited(exc))

    def test_other_errors_keep_the_short_backoff(self):
        exc = httpx.ReadTimeout("slow", request=_request())
        self.assertLess(_retry_wait_seconds(exc, attempt=0), 15.0)
        self.assertFalse(_is_rate_limited(exc))


class TestConnectionErrorWithoutFallback(unittest.TestCase):
    """No fallback configured: the same model should be retried, not abandoned."""

    def test_connection_error_is_retried_on_the_same_model(self):
        primary = _fake_model("primary")
        primary.response = AsyncMock(
            side_effect=[
                openai.APIConnectionError(message="Connection error.", request=_request()),
                ModelResponse(content="recovered", finish_reason="stop"),
            ]
        )
        agent = _make_agent()
        state = LoopState(max_api_retry=2)

        with unittest.mock.patch("agentica.runner.retry_fallback.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                Runner._call_with_retry(primary, [Message(role="user", content="hi")], state, agent)
            )

        self.assertEqual(result.content, "recovered")
        self.assertEqual(result.used_model.id, "primary")
        self.assertEqual(primary.response.await_count, 2)


class TestAnthropicSSEErrorEvent(unittest.TestCase):
    """Anthropic raises a streamed ``error`` event as ``APIStatusError`` with status 200."""

    @staticmethod
    def _sse_error(error_type):
        response = httpx.Response(200, request=_request())
        body = {"type": "error", "error": {"type": error_type, "message": "m"}}
        return anthropic.AsyncAnthropic(api_key="x")._make_status_error("m", body=body, response=response)

    def _classify(self, exc):
        state = LoopState()
        return Runner._classify_api_error(exc, state.RETRYABLE_SUBSTRINGS, state.FALLBACK_ONLY_SUBSTRINGS)

    def test_overloaded_event_is_an_outage(self):
        exc = self._sse_error("overloaded_error")
        self.assertEqual(exc.status_code, 200)
        self.assertEqual(self._classify(exc), (False, True))

    def test_rate_limit_event_is_retryable_with_the_long_wait(self):
        exc = self._sse_error("rate_limit_error")
        self.assertEqual(self._classify(exc), (True, False))
        self.assertGreaterEqual(_retry_wait_seconds(exc, attempt=0), 15.0)


class TestOutageOnLastFallback(unittest.TestCase):
    """The last candidate has nowhere to switch to, so an outage is retried there."""

    def test_last_fallback_is_retried_after_a_connection_error(self):
        primary = _fake_model("primary")
        primary.response = AsyncMock(
            side_effect=openai.APIConnectionError(message="Connection error.", request=_request())
        )
        fallback = _fake_model("fallback")
        fallback.response = AsyncMock(
            side_effect=[
                openai.APIConnectionError(message="Connection error.", request=_request()),
                ModelResponse(content="rescued", finish_reason="stop"),
            ]
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        with unittest.mock.patch("agentica.runner.retry_fallback.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                Runner._call_with_retry(
                    primary, [Message(role="user", content="hi")], LoopState(max_api_retry=2), agent,
                )
            )

        self.assertEqual(result.used_model.id, "fallback")
        # The primary switched away at once; only the last candidate retried.
        self.assertEqual(primary.response.await_count, 1)
        self.assertEqual(fallback.response.await_count, 2)


class TestMidStreamErrorsUseTheSameClassifier(unittest.TestCase):
    """Errors after the first chunk bypass ``_call_with_retry``; they must not be fatal for that reason alone."""

    @staticmethod
    def _chunk(content=None, finish_reason=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, reasoning_content=None, audio=None, tool_calls=None),
            )],
            usage=None,
        )

    def _run_with_midstream_error(self, error):
        import tempfile
        from unittest.mock import patch

        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        attempts = {"n": 0}
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

        async def fake_invoke_stream(messages):
            attempts["n"] += 1
            if attempts["n"] == 1:
                yield self._chunk(content="partial")
                raise error
            yield self._chunk(content="recovered", finish_reason="stop")

        model.invoke_stream = fake_invoke_stream
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(name="t", model=model, max_api_retry=2, session_id="s-mid", session_base_dir=tmp)

            async def _drive():
                async for _ in agent.run_stream("hello"):
                    pass

            with patch("agentica.runner.loop.asyncio.sleep", new_callable=AsyncMock) as sleep:
                asyncio.run(_drive())
            return attempts["n"], agent.run_response.content, sleep

    def test_server_error_event_after_first_chunk_is_reissued(self):
        error = openai.APIError(
            "The server had an error processing your request.",
            request=_request(),
            body={"message": "The server had an error processing your request.", "type": "server_error"},
        )
        attempts, content, _ = self._run_with_midstream_error(error)
        self.assertEqual((attempts, content), (2, "recovered"))

    def test_stream_without_terminal_event_is_reissued(self):
        from agentica.model.response import StreamEndedError

        attempts, content, _ = self._run_with_midstream_error(StreamEndedError("no terminal event"))
        self.assertEqual((attempts, content), (2, "recovered"))

    def test_midstream_rate_limit_waits_the_rate_limit_backoff(self):
        error = openai.APIError(
            "Rate limit reached", request=_request(), body={"message": "Rate limit reached", "type": "tokens"},
        )
        attempts, content, sleep = self._run_with_midstream_error(error)
        self.assertEqual((attempts, content), (2, "recovered"))
        self.assertGreaterEqual(sleep.await_args.args[0], 15.0)

    def test_midstream_bad_request_is_not_reissued(self):
        error = openai.BadRequestError(
            "Invalid value for 'temperature' (timeout)",
            response=httpx.Response(400, request=_request()),
            body={"message": "Invalid value", "type": "invalid_request_error"},
        )
        with self.assertRaises(openai.BadRequestError):
            self._run_with_midstream_error(error)


if __name__ == "__main__":
    unittest.main()
