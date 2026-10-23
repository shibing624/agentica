# -*- coding: utf-8 -*-
"""
Tests for Runner._call_with_retry cross-provider fallback chain.

All tests mock LLM API keys and model calls — no real API usage.
"""
import asyncio
import unittest
import unittest.mock
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai

from agentica.model.message import Message
from agentica.model.loop_state import LoopState
from agentica.model.response import ContentFilterError, ModelResponse
from agentica.runner import ModelCallResult, Runner


def _make_agent(name="test-agent"):
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        name=name,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    )


def _fake_model(model_id, response_factory=None, raise_exc=None):
    """Build a MagicMock model with id + async response().

    response_factory: callable returning ModelResponse (called fresh per await).
    raise_exc: exception instance to raise instead.
    """
    m = MagicMock()
    m.id = model_id
    # Model declares this field; a MagicMock attribute would stand in for it.
    m._context_calibration = None

    async def _resp(messages):
        if raise_exc is not None:
            raise raise_exc
        if response_factory is not None:
            return response_factory()
        return ModelResponse(content="ok", finish_reason="stop")

    m.response = _resp
    m.response_stream = MagicMock()
    # Bridge the new Model.get_retryable_substrings hook so the runner can
    # merge SDK defaults with user-extended markers even on mocked models.
    m.get_retryable_substrings = lambda defaults: tuple(defaults)
    m.extra_retryable_substrings = None
    return m


class _CapturedObservation:
    def __init__(self):
        self.output = None
        self.updates = []
        self.metadata = {}

    def set_output(self, output):
        self.output = output

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def set_metadata(self, key, value):
        self.metadata[key] = value


def _capture_observations(calls):
    @contextmanager
    def _context(**kwargs):
        observation = _CapturedObservation()
        calls.append((kwargs, observation))
        yield observation

    return _context


class TestFallbackLangfuseGenerations(unittest.TestCase):
    def test_failed_primary_and_successful_fallback_are_child_generations(self):
        primary = _fake_model(
            "primary",
            raise_exc=httpx.ConnectError("provider unavailable"),
        )
        primary.provider = "primary-provider"
        fallback = _fake_model(
            "fallback",
            response_factory=lambda: ModelResponse(
                content="fallback answer",
                finish_reason="stop",
            ),
        )
        fallback.provider = "fallback-provider"
        agent = _make_agent()
        agent._run_fallback_models = [fallback]
        calls = []

        with unittest.mock.patch(
            "agentica.runner.retry_fallback.langfuse_observation_context",
            new=_capture_observations(calls),
        ):
            result = asyncio.run(
                Runner._call_with_retry(
                    primary,
                    [Message(role="user", content="hello")],
                    LoopState(),
                    agent,
                    stream=False,
                )
            )

        self.assertEqual(result.content, "fallback answer")
        self.assertEqual([call[0]["as_type"] for call in calls], ["generation", "generation"])
        self.assertEqual([call[0]["name"] for call in calls], ["primary", "fallback"])
        self.assertEqual(calls[0][1].updates[-1]["level"], "ERROR")
        self.assertFalse(calls[0][0]["metadata"]["is_fallback"])
        self.assertEqual(calls[1][1].output["content"], "fallback answer")
        self.assertTrue(calls[1][0]["metadata"]["is_fallback"])


class TestFallbackContentFilterFinishReason(unittest.TestCase):
    """A filtered candidate is skipped without same-model retry."""

    def test_finish_reason_switches_to_fallback_without_replaying_blocked_turn(self):
        messages = [
            Message(role="user", content="上一轮用户问题"),
            Message(role="assistant", content="上一轮正常回答"),
            Message(role="user", content="当前用户追问"),
        ]
        seen_by_fallback = []

        async def _primary_resp(messages):
            messages.append(
                Message(role="assistant", content="你好，我无法给到相关内容。")
            )
            return ModelResponse(
                content="你好，我无法给到相关内容。",
                finish_reason="content_filter",
            )

        async def _fallback_resp(messages):
            seen_by_fallback.extend(message.content for message in messages)
            return ModelResponse(content="fallback answer", finish_reason="stop")

        primary = MagicMock()
        primary.id = "primary"
        primary.response = _primary_resp
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        fallback = MagicMock()
        fallback.id = "fallback"
        fallback.response = _fallback_resp
        fallback.get_retryable_substrings = lambda defaults: tuple(defaults)
        fallback.extra_retryable_substrings = None

        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState()
        result = asyncio.run(Runner._call_with_retry(primary, messages, state, agent, stream=False))

        self.assertEqual(result.content, "fallback answer")
        self.assertEqual(
            seen_by_fallback,
            ["上一轮用户问题", "上一轮正常回答", "当前用户追问"],
        )
        self.assertEqual(
            [m.content for m in messages],
            ["上一轮用户问题", "上一轮正常回答", "当前用户追问"],
        )

class TestFallbackContentFilterException(unittest.TestCase):
    """Some providers raise instead of setting finish_reason."""

    def test_exception_switches_to_fallback_without_retrying_primary(self):
        primary = _fake_model(
            "primary",
            raise_exc=RuntimeError("ResponsibleAIPolicyViolation: blocked"),
        )
        fallback_calls = {"n": 0}

        async def _fallback_response(messages):
            fallback_calls["n"] += 1
            return ModelResponse(content="fallback answer", finish_reason="stop")

        fallback = _fake_model("fb")
        fallback.response = _fallback_response
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState()
        result = asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))
        self.assertEqual(result.content, "fallback answer")
        self.assertEqual(fallback_calls["n"], 1)


class TestFallbackRetryableExhausted(unittest.TestCase):
    """Retryable errors retry up to max_api_retry; hard outages fallback immediately."""

    def test_transport_drop_matching_both_lists_retries_same_model(self):
        """A mid-transfer drop ("peer closed connection ... incomplete chunked
        read") matches BOTH FALLBACK_ONLY ("connection") and RETRYABLE
        ("incomplete chunked read"). The explicit retryable marker must win:
        same-model backoff retry, not an immediate switch to the fallback.
        Pre-fix, the fallback-only check ran first, so those RETRYABLE
        substrings could never trigger a same-model retry.
        """
        call_count = {"n": 0}

        async def _flaky_drop(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "peer closed connection without sending complete message body "
                    "(incomplete chunked read)"
                )
            return ModelResponse(content="recovered", finish_reason="stop")

        primary = MagicMock()
        primary.id = "primary"
        primary.response = _flaky_drop
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="fallback used", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState(max_api_retry=2)
        with unittest.mock.patch("agentica.runner.retry_fallback.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                Runner._call_with_retry(primary, [], state, agent, stream=False)
            )

        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result.content, "recovered")
        self.assertEqual(state.last_used_model_id, "primary")

    def test_retries_then_falls_back_after_exhausting_local_retries(self):
        call_count = {"n": 0}

        async def _flaky_resp(messages):
            call_count["n"] += 1
            raise RuntimeError("rate_limit hit, 429")

        primary = MagicMock()
        primary.id = "primary"
        primary.response = _flaky_resp
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="recovered", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        # Tighten retry to keep test fast; skip the exponential backoff sleep.
        state = LoopState(max_api_retry=2)
        with unittest.mock.patch("agentica.runner.retry_fallback.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                Runner._call_with_retry(primary, [], state, agent, stream=False)
            )

        # primary should have been called max_api_retry times before fallback
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result.content, "recovered")

    def test_service_unavailable_falls_back_without_same_model_retry(self):
        primary_calls = {"n": 0}

        async def _503(messages):
            primary_calls["n"] += 1
            raise RuntimeError("upstream returned 503 Service Unavailable")

        primary = MagicMock()
        primary.id = "primary"
        primary.response = _503
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState(max_api_retry=3)
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )
        self.assertEqual(primary_calls["n"], 1)
        self.assertEqual(result.content, "ok")
        self.assertEqual(state.last_used_model_id, "fb")

    def test_hard_outages_fallback_without_same_model_retry(self):
        outage_messages = [
            "Connection error.",
            "internal server error",
            "502 bad gateway",
            "503 service unavailable",
        ]
        for outage_message in outage_messages:
            with self.subTest(outage_message=outage_message):
                primary_calls = {"n": 0}

                async def _outage(messages):
                    primary_calls["n"] += 1
                    raise RuntimeError(outage_message)

                primary = MagicMock()
                primary.id = "primary"
                primary.response = _outage
                primary.get_retryable_substrings = lambda defaults: tuple(defaults)
                primary.extra_retryable_substrings = None

                fallback = _fake_model(
                    "fb",
                    response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
                )
                agent = _make_agent()
                agent._run_fallback_models = [fallback]

                result = asyncio.run(
                    Runner._call_with_retry(
                        primary, [], LoopState(max_api_retry=3), agent, stream=False,
                    )
                )
                self.assertEqual(primary_calls["n"], 1)
                self.assertEqual(result.content, "ok")


class TestTypedApiErrors(unittest.TestCase):
    def test_timeout_types_work_without_message_markers(self):
        request = httpx.Request("POST", "http://test/v1/chat/completions")
        errors = [
            openai.APITimeoutError(request=request),
            httpx.ReadTimeout("", request=request),
            httpx.ConnectTimeout("", request=request),
            httpx.PoolTimeout("", request=request),
            TimeoutError(),
        ]
        for exc in errors:
            with self.subTest(exc=type(exc).__name__):
                primary = _fake_model("primary")
                primary.response = AsyncMock(side_effect=exc)
                fallback = _fake_model("fallback")
                agent = _make_agent()
                agent._run_fallback_models = [fallback]
                result = asyncio.run(Runner._call_with_retry(
                    primary, [], LoopState(max_api_retry=1), agent,
                ))
                self.assertEqual(primary.response.await_count, 1)
                self.assertEqual(result.used_model.id, "fallback")

    def test_timeout_respects_configured_runner_retry(self):
        primary = _fake_model("primary")
        primary.response = AsyncMock(side_effect=[TimeoutError(), ModelResponse(content="ok")])
        agent = _make_agent()
        agent._run_fallback_models = [_fake_model("fallback")]
        with unittest.mock.patch("agentica.runner.retry_fallback.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(Runner._call_with_retry(
                primary, [], LoopState(max_api_retry=2), agent,
            ))
        self.assertEqual(primary.response.await_count, 2)
        self.assertEqual(result.used_model.id, "primary")

    def test_connection_types_fallback_without_retry(self):
        request = httpx.Request("POST", "http://test")
        for exc in (
            openai.APIConnectionError(message="offline", request=request),
            httpx.ConnectError("offline", request=request),
            ConnectionError(),
        ):
            with self.subTest(exc=type(exc).__name__):
                primary = _fake_model("primary")
                primary.response = AsyncMock(side_effect=exc)
                agent = _make_agent()
                agent._run_fallback_models = [_fake_model("fallback")]
                result = asyncio.run(Runner._call_with_retry(
                    primary, [], LoopState(max_api_retry=3), agent,
                ))
                self.assertEqual(primary.response.await_count, 1)
                self.assertEqual(result.used_model.id, "fallback")

    def test_http_status_wins_over_misleading_message(self):
        request = httpx.Request("POST", "http://test")
        state = LoopState()
        for status, expected in (
            (400, (False, False)), (401, (False, False)), (403, (False, False)),
            (408, (True, False)), (409, (True, False)), (429, (True, False)),
            (500, (False, True)), (502, (False, True)), (503, (False, True)),
            (504, (True, False)),
        ):
            for exc in (
                openai.APIStatusError("timeout connection 503", response=httpx.Response(status, request=request), body={}),
                httpx.HTTPStatusError("timeout connection 503", request=request,
                                      response=httpx.Response(status, request=request)),
            ):
                with self.subTest(status=status, exc=type(exc).__name__):
                    self.assertEqual(Runner._classify_api_error(
                        exc, state.RETRYABLE_SUBSTRINGS, state.FALLBACK_ONLY_SUBSTRINGS,
                    ), expected)

    def test_error_after_tool_history_repair_is_classified_not_original_error(self):
        primary = _fake_model("primary")
        primary.response = AsyncMock(side_effect=[
            RuntimeError("tool message must be a response to a preceding message"),
            TimeoutError(),
        ])
        agent = _make_agent()
        agent._run_fallback_models = [_fake_model("fallback")]
        result = asyncio.run(Runner._call_with_retry(primary, [], LoopState(), agent))
        self.assertEqual(primary.response.await_count, 2)
        self.assertEqual(result.used_model.id, "fallback")

    def test_typed_auth_error_never_calls_fallback(self):
        request = httpx.Request("POST", "http://test")
        exc = openai.AuthenticationError("upstream timeout 503", response=httpx.Response(401, request=request), body={})
        primary = _fake_model("primary", raise_exc=exc)
        fallback = _fake_model("fallback")
        fallback.response = AsyncMock()
        agent = _make_agent()
        agent._run_fallback_models = [fallback]
        with self.assertRaises(openai.AuthenticationError):
            asyncio.run(Runner._call_with_retry(primary, [], LoopState(), agent))
        fallback.response.assert_not_awaited()


class TestFallbackNonRetryableImmediateRaise(unittest.TestCase):
    """Auth errors and other truly non-retryable errors should NOT trigger fallback."""

    def test_auth_error_raises_immediately_without_touching_fallback(self):
        primary = _fake_model(
            "primary",
            raise_exc=RuntimeError("401 invalid api key"),
        )
        fb_called = {"flag": False}

        async def _fb_resp(messages):
            fb_called["flag"] = True
            return ModelResponse(content="x", finish_reason="stop")

        fallback = MagicMock()
        fallback.id = "fb"
        fallback.response = _fb_resp

        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState()
        with self.assertRaises(RuntimeError) as cm:
            asyncio.run(
                Runner._call_with_retry(primary, [], state, agent, stream=False)
            )
        self.assertIn("401", str(cm.exception))
        self.assertFalse(
            fb_called["flag"],
            "Fallback must not be used for non-retryable, non-content-filter errors",
        )


class TestFallbackEmptyChain(unittest.TestCase):
    """No fallback configured surfaces policy and exhausted retry errors."""

    def test_content_filter_with_empty_chain_raises(self):
        primary = _fake_model(
            "primary",
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
        )
        agent = _make_agent()
        agent._run_fallback_models = []

        state = LoopState()
        with self.assertRaises(ContentFilterError):
            asyncio.run(
                Runner._call_with_retry(primary, [], state, agent, stream=False)
            )

    def test_normal_response_unaffected_by_chain(self):
        primary = _fake_model(
            "primary",
            response_factory=lambda: ModelResponse(content="hello", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = []

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )
        self.assertEqual(result.content, "hello")


class TestRunConfigFallbackModelsField(unittest.TestCase):
    """RunConfig.fallback_models should plumb through to agent._run_fallback_models."""

    def test_run_config_fallback_models_defaults_empty(self):
        from agentica.run.config import RunConfig
        cfg = RunConfig()
        self.assertEqual(cfg.fallback_models, [])

    def test_run_stashes_fallback_models_on_agent(self):
        """Calling run() with config.fallback_models sets agent._run_fallback_models."""
        from agentica.run.config import RunConfig
        from agentica.run.response import RunResponse, RunEvent

        agent = _make_agent()
        fb = _fake_model("fb")
        captured = {"value": None}

        # Intercept _run_impl to capture state at run time, then short-circuit
        async def _fake_impl(*a, **kw):
            captured["value"] = list(agent._run_fallback_models)
            yield RunResponse(content="", event=RunEvent.run_response.value)

        agent._runner._run_impl = _fake_impl

        agent.run_sync(message=None, config=RunConfig(fallback_models=[fb]))
        self.assertEqual(captured["value"], [fb])


class TestMaxApiRetryConfig(unittest.TestCase):
    """Agent / RunConfig max_api_retry should feed LoopState per run."""

    def test_agent_default_max_api_retry_used_by_loop_state(self):
        captured = []
        agent = _make_agent()
        agent.max_api_retry = 4

        async def _fake_call(runner, model, messages, state, agent_arg, *, stream=False):
            captured.append(state.max_api_retry)
            return ModelCallResult(
                response=ModelResponse(content="ok", finish_reason="stop"),
                used_model=model,
                used_fallback=False,
            )

        with unittest.mock.patch.object(Runner, "_call_with_retry", new=_fake_call):
            response = agent.run_sync("hi")

        self.assertEqual(response.content, "ok")
        self.assertEqual(captured, [4])

    def test_run_config_max_api_retry_overrides_agent_default(self):
        from agentica.run.config import RunConfig

        captured = []
        agent = _make_agent()
        agent.max_api_retry = 4

        async def _fake_call(runner, model, messages, state, agent_arg, *, stream=False):
            captured.append(state.max_api_retry)
            return ModelCallResult(
                response=ModelResponse(content="ok", finish_reason="stop"),
                used_model=model,
                used_fallback=False,
            )

        with unittest.mock.patch.object(Runner, "_call_with_retry", new=_fake_call):
            response = agent.run_sync("hi", config=RunConfig(max_api_retry=2))

        self.assertEqual(response.content, "ok")
        self.assertEqual(captured, [2])

    def test_invalid_max_api_retry_rejected(self):
        from agentica.run.config import RunConfig

        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.run_sync("hi", config=RunConfig(max_api_retry=0))


class TestAgentFallbackModelsField(unittest.TestCase):
    """Agent(fallback_models=...) should be the default chain when no RunConfig override."""

    def test_agent_fallback_models_defaults_empty_list(self):
        agent = _make_agent()
        self.assertEqual(agent.fallback_models, [])

    def test_agent_fallback_models_used_when_no_run_config(self):
        """Without RunConfig.fallback_models, agent.fallback_models is used."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run.response import RunResponse, RunEvent

        fb = _fake_model("agent-default-fb")
        agent = Agent(
            name="t",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
            fallback_models=[fb],
        )

        captured = {"value": None}

        async def _fake_impl(*a, **kw):
            captured["value"] = list(agent._run_fallback_models)
            yield RunResponse(content="", event=RunEvent.run_response.value)

        agent._runner._run_impl = _fake_impl
        agent.run_sync(message=None)
        self.assertEqual(captured["value"], [fb])

    def test_run_config_overrides_agent_fallback_models(self):
        """RunConfig.fallback_models takes precedence over Agent.fallback_models."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run.config import RunConfig
        from agentica.run.response import RunResponse, RunEvent

        agent_fb = _fake_model("agent-fb")
        run_fb = _fake_model("run-fb")
        agent = Agent(
            name="t",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake"),
            fallback_models=[agent_fb],
        )

        captured = {"value": None}

        async def _fake_impl(*a, **kw):
            captured["value"] = list(agent._run_fallback_models)
            yield RunResponse(content="", event=RunEvent.run_response.value)

        agent._runner._run_impl = _fake_impl
        agent.run_sync(message=None, config=RunConfig(fallback_models=[run_fb]))
        self.assertEqual(captured["value"], [run_fb])
        self.assertNotIn(agent_fb, captured["value"])


class TestPerCallNotPerRun(unittest.TestCase):
    """Fallback must be per-call: switching does NOT mutate agent.model.

    Verifies the next call starts fresh from the primary, not from the
    fallback that rescued the previous call.
    """

    def test_agent_model_unchanged_after_fallback_switch(self):
        primary = _fake_model(
            "p",
            raise_exc=RuntimeError("503 service unavailable"),
        )
        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
        )
        agent = _make_agent()
        agent.model = primary
        agent._run_fallback_models = [fallback]

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )

        self.assertEqual(result.content, "ok")
        self.assertIs(agent.model, primary, "agent.model must NOT be swapped")

    def test_subsequent_call_retries_primary(self):
        """Two sequential calls: each starts from primary."""
        primary_calls = {"n": 0}

        async def _flaky(messages):
            primary_calls["n"] += 1
            raise RuntimeError("503 service unavailable")

        primary = MagicMock()
        primary.id = "p"
        primary.response = _flaky
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
        )
        agent = _make_agent()
        agent.model = primary
        agent._run_fallback_models = [fallback]

        state1 = LoopState()
        state2 = LoopState()
        asyncio.run(Runner._call_with_retry(primary, [], state1, agent, stream=False))
        asyncio.run(Runner._call_with_retry(primary, [], state2, agent, stream=False))

        self.assertEqual(
            primary_calls["n"], 2,
            "primary must be tried again on the second call (per-call switch)",
        )


class TestLastUsedModelTruthfulness(unittest.TestCase):
    """LoopState.last_used_model_id reflects the model that actually answered."""

    def test_last_used_model_id_is_primary_on_clean_response(self):
        primary = _fake_model("p")
        agent = _make_agent()
        agent._run_fallback_models = []

        state = LoopState()
        asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        self.assertEqual(state.last_used_model_id, "p")
        self.assertEqual(state.last_used_model_idx, 0)

    def test_last_used_model_id_is_fallback_on_outage_recovery(self):
        primary = _fake_model(
            "p",
            raise_exc=RuntimeError("503 service unavailable"),
        )
        fb = _fake_model("fb")
        agent = _make_agent()
        agent._run_fallback_models = [fb]

        state = LoopState()
        asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        self.assertEqual(state.last_used_model_id, "fb")
        self.assertEqual(state.last_used_model_idx, 1)

    def test_last_used_model_id_resets_per_call(self):
        primary = _fake_model("p")
        fb = _fake_model("fb")
        agent = _make_agent()
        agent._run_fallback_models = [fb]

        state = LoopState()
        # Manually pollute state to simulate stale data; _call_with_retry must clear it.
        state.last_used_model_id = "stale-from-previous-call"
        state.last_used_model_idx = 99

        asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))
        self.assertEqual(state.last_used_model_id, "p")
        self.assertEqual(state.last_used_model_idx, 0)


class TestFallbackToolTransaction(unittest.TestCase):
    """Fallback may execute tools, but its provider-specific transcript is compacted."""

    def test_fallback_tool_transaction_compacts_replay_context(self):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run.config import RunConfig

        def add_numbers(a: int, b: int) -> int:
            return a + b

        primary = OpenAIChat(id="primary", api_key="fake_openai_key")
        fallback = OpenAIChat(id="fallback", api_key="fake_openai_key")

        async def _primary_response(messages):
            raise RuntimeError("503 service unavailable")

        fallback_calls = {"n": 0, "saw_tool_result": False}

        async def _fallback_response(messages):
            fallback_calls["n"] += 1
            if fallback_calls["n"] == 1:
                messages.append(Message(role="assistant", content="", tool_calls=[{
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": "{\"a\": 2, \"b\": 3}"},
                }]))
                return ModelResponse(content="", finish_reason="tool_calls")

            fallback_calls["saw_tool_result"] = any(
                msg.role == "tool" and msg.tool_call_id == "call_add" and msg.content == "5"
                for msg in messages
            )
            messages.append(Message(role="assistant", content="final: 5"))
            return ModelResponse(content="final: 5", finish_reason="stop")

        primary.response = _primary_response
        fallback.response = _fallback_response
        agent = Agent(
            name="fallback-tool-agent",
            model=primary,
            tools=[add_numbers],
            fallback_models=[fallback],
        )

        response = agent.run_sync(
            "compute 2+3",
            config=RunConfig(fallback_models=[fallback]),
        )

        self.assertEqual(response.content, "final: 5")
        self.assertTrue(fallback_calls["saw_tool_result"])
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.model, "fallback")
        replay_messages = response.messages or []
        self.assertFalse(any(msg.role == "tool" for msg in replay_messages))
        self.assertFalse(any(msg.tool_calls for msg in replay_messages if msg.role == "assistant"))
        self.assertTrue(response.tools)
        self.assertTrue(response.tools[0]["fallback_compacted"])
        self.assertFalse(response.tools[0]["replay"])
        self.assertEqual(primary.response, _primary_response)
        self.assertIs(agent.model, primary)

    def test_streaming_fallback_tool_transaction_compacts_replay_context(self):
        """Streaming path: fallback runs the tool-call turn; transcript compacted."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.model.response import ModelResponseEvent
        from agentica.run.config import RunConfig

        def add_numbers(a: int, b: int) -> int:
            return a + b

        primary = OpenAIChat(id="primary", api_key="fake_openai_key")
        fallback = OpenAIChat(id="fallback", api_key="fake_openai_key")

        def _primary_stream(messages):
            # Plain function so it raises at call time -> triggers fallback.
            raise RuntimeError("503 service unavailable")

        fallback_calls = {"n": 0, "saw_tool_result": False}

        async def _fallback_stream(messages):
            fallback_calls["n"] += 1
            if fallback_calls["n"] == 1:
                messages.append(Message(role="assistant", content="", tool_calls=[{
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": "{\"a\": 2, \"b\": 3}"},
                }]))
                return
                yield  # noqa: unreachable — marks this as an async generator
            fallback_calls["saw_tool_result"] = any(
                msg.role == "tool" and msg.tool_call_id == "call_add" and msg.content == "5"
                for msg in messages
            )
            messages.append(Message(role="assistant", content="final: 5"))
            yield ModelResponse(
                event=ModelResponseEvent.assistant_response.value,
                content="final: 5",
            )

        primary.response_stream = _primary_stream
        fallback.response_stream = _fallback_stream
        agent = Agent(
            name="fallback-stream-tool-agent",
            model=primary,
            tools=[add_numbers],
            fallback_models=[fallback],
        )

        for _ in agent.run_stream_sync(
            "compute 2+3",
            config=RunConfig(fallback_models=[fallback]),
        ):
            pass

        response = agent.run_response
        self.assertEqual(response.content, "final: 5")
        self.assertTrue(fallback_calls["saw_tool_result"])
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.model, "fallback")
        replay_messages = response.messages or []
        self.assertFalse(any(msg.role == "tool" for msg in replay_messages))
        self.assertFalse(any(msg.tool_calls for msg in replay_messages if msg.role == "assistant"))
        self.assertTrue(response.tools)
        self.assertTrue(response.tools[0]["fallback_compacted"])
        self.assertFalse(response.tools[0]["replay"])
        # Primary stays un-mutated; clone isolation means the patched fallback
        # object is not the one mutated during the run.
        self.assertEqual(primary.response_stream, _primary_stream)
        self.assertIs(agent.model, primary)


class TestStreamingOpenFailureFallsBack(unittest.TestCase):
    """A real ``response_stream`` is a lazy async generator: the HTTP request,
    and its connection error, happen on the first ``__anext__``, not when the
    generator is created. The fallback chain must still see that error."""

    def _agent(self, primary_stream, fallback_stream):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        primary = OpenAIChat(id="primary", api_key="fake_openai_key")
        fallback = OpenAIChat(id="fallback", api_key="fake_openai_key")
        primary.response_stream = primary_stream
        fallback.response_stream = fallback_stream
        return Agent(name="lazy-stream", model=primary, fallback_models=[fallback])

    def test_connection_error_on_first_chunk_switches_to_fallback(self):
        from agentica.model.response import ModelResponseEvent

        primary_opened = {"n": 0}

        async def primary_stream(messages):
            primary_opened["n"] += 1
            raise openai.APIConnectionError(
                message="Connection error.", request=httpx.Request("POST", "http://x")
            )
            yield  # noqa: unreachable — makes this a lazy async generator

        async def fallback_stream(messages):
            messages.append(Message(role="assistant", content="from fallback"))
            yield ModelResponse(event=ModelResponseEvent.assistant_response.value, content="from fallback")

        agent = self._agent(primary_stream, fallback_stream)
        chunks = [c.content for c in agent.run_stream_sync("hi") if c.content]

        self.assertEqual("".join(chunks), "from fallback")
        self.assertEqual(primary_opened["n"], 1)
        self.assertTrue(agent.run_response.fallback_used)
        self.assertEqual(agent.run_response.model, "fallback")

    def test_error_after_first_chunk_is_not_handed_to_fallback(self):
        """Once output is on the wire, a fallback would duplicate it."""
        from agentica.model.response import ModelResponseEvent

        fallback_opened = {"n": 0}

        async def primary_stream(messages):
            yield ModelResponse(event=ModelResponseEvent.assistant_response.value, content="partial")
            raise ValueError("decoder blew up mid-stream")

        async def fallback_stream(messages):
            fallback_opened["n"] += 1
            yield ModelResponse(event=ModelResponseEvent.assistant_response.value, content="x")

        agent = self._agent(primary_stream, fallback_stream)
        with self.assertRaises(ValueError):
            for _ in agent.run_stream_sync("hi"):
                pass
        self.assertEqual(fallback_opened["n"], 0)


class TestFallbackRecoveryEvent(unittest.TestCase):
    """Successful fallback rescue should emit a structured event for audit/observability."""

    def test_content_filter_recovery_records_its_trigger(self):
        captured = []
        primary = _fake_model(
            "primary-id",
            response_factory=lambda: ModelResponse(content=None, finish_reason="content_filter"),
        )
        fallback = _fake_model("fallback-id")
        agent = _make_agent()
        agent._run_fallback_models = [fallback]
        agent._event_callback = captured.append

        asyncio.run(Runner._call_with_retry(primary, [], LoopState(), agent, stream=False))

        event = next(item for item in captured if item.get("type") == "fallback.recovered")
        self.assertEqual(event["used_model"], "fallback-id")
        self.assertEqual(event["trigger"], "content_filter")

    def test_fallback_recovery_event_emitted_on_outage(self):
        captured = []

        primary = _fake_model(
            "primary-id",
            raise_exc=RuntimeError("503 service unavailable"),
        )
        fb = _fake_model(
            "fallback-id",
            response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
        )
        agent = _make_agent()
        agent.name = "audited"
        agent._run_fallback_models = [fb]
        agent._event_callback = captured.append

        state = LoopState()
        asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        recovery_events = [e for e in captured if e.get("type") == "fallback.recovered"]
        self.assertEqual(len(recovery_events), 1)
        evt = recovery_events[0]
        self.assertEqual(evt["primary_model"], "primary-id")
        self.assertEqual(evt["used_model"], "fallback-id")
        self.assertEqual(evt["fallback_index"], 1)
        self.assertEqual(evt["trigger"], "fallback_only")
        self.assertEqual(evt["agent_name"], "audited")

    def test_no_event_emitted_when_primary_succeeds(self):
        captured = []
        primary = _fake_model("p")
        agent = _make_agent()
        agent._event_callback = captured.append

        state = LoopState()
        asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        recovery_events = [e for e in captured if e.get("type") == "fallback.recovered"]
        self.assertEqual(len(recovery_events), 0)

    def test_recovery_log_warning_message(self):
        """Audit log line `[fallback.recovered]` must be emitted at WARNING level."""
        primary = _fake_model(
            "p", raise_exc=RuntimeError("503 service unavailable"),
        )
        fb = _fake_model("fb")
        agent = _make_agent()
        agent._run_fallback_models = [fb]

        state = LoopState()
        with unittest.mock.patch("agentica.runner.retry_fallback.logger") as mock_logger:
            asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        warning_msgs = [
            call.args[0] for call in mock_logger.warning.call_args_list
        ]
        self.assertTrue(
            any("[fallback.recovered]" in msg for msg in warning_msgs),
            f"expected [fallback.recovered] log; got {warning_msgs}",
        )


class TestContextUsageEvent(unittest.TestCase):
    """Runner exposes actual request context independently from CostTracker."""

    def test_main_request_emits_message_and_tool_context(self):
        captured = []
        primary = _fake_model("primary")
        primary.tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        primary.context_window = 128000
        agent = _make_agent()
        agent._event_callback = captured.append
        messages = [Message(role="user", content="inspect this repository " * 100)]

        asyncio.run(Runner._call_with_retry(primary, messages, LoopState(), agent, stream=False))

        event = next(e for e in captured if e["type"] == "context.usage")
        self.assertIs(event["is_main_agent"], True)
        self.assertGreater(event["context_tokens"], 100)
        self.assertEqual(event["context_window"], 128000)

    def test_event_reports_the_working_window_not_just_the_provider_limit(self):
        """The status bar takes its denominator from this event.

        With a working cap set, reporting only ``model.context_window`` makes
        the bar divide by the provider limit while the runner evicts and
        compacts against the cap — the two disagree about the same session.
        """
        captured = []
        primary = _fake_model("primary")
        primary.tools = []
        primary.context_window = 1_000_000
        agent = _make_agent()
        agent.tool_config.compact_token_limit = 512_000
        agent.tool_config.compression_manager.compact_token_limit = 512_000
        agent._event_callback = captured.append

        asyncio.run(Runner._call_with_retry(
            primary,
            [Message(role="user", content="inspect this repository " * 100)],
            LoopState(),
            agent,
            stream=False,
        ))

        event = next(e for e in captured if e["type"] == "context.usage")
        self.assertEqual(event["context_window"], 512_000)
        self.assertEqual(event["provider_window"], 1_000_000)

    def test_event_without_a_cap_keeps_both_windows_equal(self):
        captured = []
        primary = _fake_model("primary")
        primary.tools = []
        primary.context_window = 128_000
        agent = _make_agent()
        agent._event_callback = captured.append

        asyncio.run(Runner._call_with_retry(
            primary,
            [Message(role="user", content="hello")],
            LoopState(),
            agent,
            stream=False,
        ))

        event = next(e for e in captured if e["type"] == "context.usage")
        self.assertEqual(event["context_window"], 128_000)
        self.assertEqual(event["provider_window"], 128_000)

    def test_subagent_request_is_explicitly_scoped(self):
        captured = []
        primary = _fake_model("primary")
        primary.tools = []
        primary.context_window = 128000
        agent = _make_agent()
        agent._parent_run_id = "parent"
        agent._event_callback = captured.append

        asyncio.run(
            Runner._call_with_retry(
                primary,
                [Message(role="user", content="child work")],
                LoopState(),
                agent,
                stream=False,
            )
        )

        event = next(e for e in captured if e["type"] == "context.usage")
        self.assertIs(event["is_main_agent"], False)

    def test_callback_failure_does_not_abort_model_request(self):
        primary = _fake_model("primary")
        primary.tools = []
        primary.context_window = 128000
        agent = _make_agent()

        def fail_callback(_event):
            raise RuntimeError("event bus is on fire")

        agent._event_callback = fail_callback
        with self.assertLogs("agentica", level="WARNING") as logs:
            result = asyncio.run(
                Runner._call_with_retry(
                    primary,
                    [Message(role="user", content="still answer")],
                    LoopState(),
                    agent,
                    stream=False,
                )
            )

        self.assertEqual(result.content, "ok")
        self.assertTrue(any("context.usage" in line for line in logs.output))


class TestInvalidThinkingSignatureRecovery(unittest.TestCase):
    """A bad Anthropic thinking signature loses reasoning, not the tool round."""

    @staticmethod
    def _native_claude_tool_round():
        return [
            Message(role="user", content="inspect the config"),
            Message(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "check it", "signature": "bad-signature"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "config.py"},
                    },
                ],
                tool_calls=[
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"config.py"}',
                        },
                    }
                ],
                reasoning_content="check it",
                provider_data={"signature": "bad-signature"},
            ),
            Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "PORT = 8080",
                    }
                ],
            ),
        ]

    def test_retries_once_with_portable_tool_round_for_stream_and_non_stream(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                messages = self._native_claude_tool_round()
                calls = {"n": 0}
                seen_after_repair = []

                async def _respond(messages):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise RuntimeError(
                            "Anthropic Claude bad request: Invalid `signature` in `thinking` block"
                        )
                    seen_after_repair.extend(messages)
                    return ModelResponse(content="PORT is 8080", finish_reason="stop")

                async def _stream(messages):
                    response = await _respond(messages)
                    yield response

                primary = _fake_model("claude-opus-5-5")
                primary.response = _respond
                primary.response_stream = _stream
                fallback = _fake_model("fallback")
                fallback.response = AsyncMock()
                fallback.response_stream = MagicMock()
                agent = _make_agent()
                agent._run_fallback_models = [fallback]
                state = LoopState()
                # The initial user message predates this turn's tool round.
                # Rebuilding message objects must preserve that provenance so
                # flat-memory persistence keeps the assistant/tool additions.
                state.input_message_ids = {id(messages[0])}

                call = asyncio.run(
                    Runner._call_with_retry(primary, messages, state, agent, stream=stream)
                )
                if stream:
                    async def _consume():
                        return [chunk async for chunk in call.response]

                    chunks = asyncio.run(_consume())
                    self.assertEqual(chunks[-1].content, "PORT is 8080")
                else:
                    self.assertEqual(call.content, "PORT is 8080")

                self.assertEqual(calls["n"], 2)
                fallback.response.assert_not_awaited()
                self.assertTrue(state.context_collapsed)
                assistant = next(m for m in seen_after_repair if m.role == "assistant")
                self.assertEqual(assistant.content, "")
                self.assertEqual(assistant.tool_calls[0]["id"], "toolu_1")
                self.assertIsNone(assistant.reasoning_content)
                self.assertIsNone(assistant.provider_data)
                result = next(m for m in seen_after_repair if m.role == "tool")
                self.assertEqual(result.tool_call_id, "toolu_1")
                self.assertEqual(result.content, "PORT = 8080")
                repaired_user = next(m for m in seen_after_repair if m.role == "user")
                self.assertIn(id(repaired_user), state.input_message_ids)
                self.assertNotIn(id(assistant), state.input_message_ids)
                self.assertNotIn(id(result), state.input_message_ids)

    def test_bad_signature_repair_is_attempted_only_once(self):
        calls = {"n": 0}

        async def _always_bad(messages):
            calls["n"] += 1
            raise RuntimeError("messages.5.content.22: Invalid `signature` in `thinking` block")

        primary = _fake_model("claude-opus-5-5")
        primary.response = _always_bad
        agent = _make_agent()
        state = LoopState()

        with self.assertRaises(RuntimeError):
            asyncio.run(
                Runner._call_with_retry(
                    primary,
                    self._native_claude_tool_round(),
                    state,
                    agent,
                )
            )

        self.assertEqual(calls["n"], 2)

    def test_bedrock_validation_exception_thinking_cannot_be_modified_recovers(self):
        """AWS Bedrock Runtime wording (``thinking`` or ``redacted_thinking`` blocks
        in the latest assistant message cannot be modified) must trigger the same
        portable-history recovery as Anthropic Direct's ``Invalid signature``."""
        bedrock_err = (
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Anthropic Claude bad request: InvokeModelWithResponseStream: "
            "operation error Bedrock Runtime: InvokeModelWithResponseStream, https response "
            "error StatusCode: 400, RequestID: 6b0dbac3-0520-4b25-9eba-12d45a09eeba, "
            "ValidationException: ***.***.content.6: `thinking` or `redacted_thinking` "
            "blocks in the latest assistant message cannot be modified. These blocks must "
            "remain as they were in the original response.'}}"
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                messages = self._native_claude_tool_round()
                calls = {"n": 0}
                seen_after_repair = []

                async def _respond(messages):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise RuntimeError(bedrock_err)
                    seen_after_repair.extend(messages)
                    return ModelResponse(content="PORT is 8080", finish_reason="stop")

                async def _stream(messages):
                    response = await _respond(messages)
                    yield response

                primary = _fake_model("claude-opus-5-5")
                primary.response = _respond
                primary.response_stream = _stream
                agent = _make_agent()
                state = LoopState()

                call = asyncio.run(
                    Runner._call_with_retry(primary, messages, state, agent, stream=stream)
                )
                if stream:
                    async def _consume():
                        return [chunk async for chunk in call.response]

                    chunks = asyncio.run(_consume())
                    self.assertEqual(chunks[-1].content, "PORT is 8080")
                else:
                    self.assertEqual(call.content, "PORT is 8080")

                self.assertEqual(calls["n"], 2)
                assistant = next(m for m in seen_after_repair if m.role == "assistant")
                self.assertEqual(assistant.content, "")
                self.assertEqual(assistant.tool_calls[0]["id"], "toolu_1")

    def test_later_tool_round_in_same_run_can_also_recover(self):
        """A later _call_with_retry in the same LoopState that produces a new
        signed thinking block must still be allowed one recovery if rejected."""
        state = LoopState()
        agent = _make_agent()

        for round_idx in (1, 2):
            messages = self._native_claude_tool_round()
            calls = {"n": 0}

            async def _respond(messages):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError(
                        "ValidationException: ***.***.content.6: `thinking` or "
                        "`redacted_thinking` blocks in the latest assistant message "
                        "cannot be modified."
                    )
                return ModelResponse(content=f"ok-{round_idx}", finish_reason="stop")

            primary = _fake_model("claude-opus-5-5")
            primary.response = _respond
            call = asyncio.run(
                Runner._call_with_retry(primary, messages, state, agent, stream=False)
            )
            self.assertEqual(call.content, f"ok-{round_idx}")
            self.assertEqual(calls["n"], 2)


class TestToolHistorySanitizeRecovery(unittest.TestCase):
    """Cross-provider tool-call/tool-result mismatch (e.g. resuming a session
    recorded under Claude with a non-Claude model) -> strip tool artifacts
    from history and retry once on the same model, no fallback needed."""

    def _model_that_fails_once_on_tool_history(self, model_id="primary"):
        calls = {"n": 0}

        async def _resp(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': \"Messages with "
                    "role 'tool' must be a response to a preceding message "
                    "with 'tool_calls'\"}}"
                )
            return ModelResponse(content="recovered", finish_reason="stop")

        m = MagicMock()
        m.id = model_id
        m.response = _resp
        m.get_retryable_substrings = lambda defaults: tuple(defaults)
        m.extra_retryable_substrings = None
        return m, calls

    def test_strips_tool_messages_and_retries_same_model(self):
        primary, calls = self._model_that_fails_once_on_tool_history()
        messages = [
            Message(role="user", content="do X"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            ),
            Message(role="tool", content="tool result", tool_call_id="call_1"),
            Message(role="user", content="continue"),
        ]
        agent = _make_agent()

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, messages, state, agent, stream=False)
        )

        self.assertEqual(result.content, "recovered")
        self.assertEqual(calls["n"], 2)
        # In-flight messages sent to the model are cleaned of tool artifacts.
        self.assertNotIn("tool", [m.role for m in messages])
        self.assertTrue(all(not m.tool_calls for m in messages if m.role == "assistant"))
        self.assertEqual(state.tool_history_sanitized_done, True)

    def test_also_sanitizes_working_memory_runs(self):
        from agentica.run.response import RunResponse
        from agentica.memory.models import AgentRun

        primary, _ = self._model_that_fails_once_on_tool_history()
        agent = _make_agent()
        stale_run = AgentRun(
            response=RunResponse(
                content="old",
                messages=[
                    Message(role="user", content="past question"),
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[{"id": "call_9", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                    ),
                    Message(role="tool", content="past tool result", tool_call_id="call_9"),
                ],
            )
        )
        agent.working_memory.runs.append(stale_run)

        state = LoopState()
        asyncio.run(
            Runner._call_with_retry(primary, [Message(role="user", content="hi")], state, agent, stream=False)
        )

        cleaned_roles = [m.role for m in stale_run.response.messages]
        self.assertNotIn("tool", cleaned_roles)
        self.assertTrue(all(not m.tool_calls for m in stale_run.response.messages if m.role == "assistant"))

    def test_only_sanitizes_once_per_loop(self):
        """A second, unrelated tool-history-shaped error is not retried again
        (state.tool_history_sanitized_done guards against infinite loops)."""
        calls = {"n": 0}

        async def _resp(messages):
            calls["n"] += 1
            raise RuntimeError(
                "role 'tool' must be a response to a preceding message with 'tool_calls'"
            )

        primary = MagicMock()
        primary.id = "primary"
        primary.response = _resp
        primary.get_retryable_substrings = lambda defaults: tuple(defaults)
        primary.extra_retryable_substrings = None

        agent = _make_agent()
        state = LoopState()

        with self.assertRaises(RuntimeError):
            asyncio.run(
                Runner._call_with_retry(primary, [Message(role="user", content="hi")], state, agent, stream=False)
            )

        # Sanitize retried exactly once, not in an infinite loop.
        self.assertEqual(calls["n"], 2)

    def test_streaming_sanitizes_and_retries_through_real_consumption_path(self):
        """Regression test: _call_with_retry's own try/except never fires for
        stream=True — Model.response_stream() is a lazy async generator, so
        creating it raises nothing; the HTTP call (and this 400) only happens
        once the caller starts iterating it via ``async for``, which is
        *outside* _call_with_retry (in Runner._run_impl's streaming loop).
        This exercises that real consumption path end-to-end instead of unit
        -testing _call_with_retry directly, which would miss the bug."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.model.response import ModelResponseEvent

        primary = OpenAIChat(id="primary", api_key="fake_openai_key")
        calls = {"n": 0}

        async def _stream(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': \"Messages with "
                    "role 'tool' must be a response to a preceding message "
                    "with 'tool_calls'\"}}"
                )
            messages.append(Message(role="assistant", content="recovered"))
            yield ModelResponse(
                event=ModelResponseEvent.assistant_response.value,
                content="recovered",
            )

        primary.response_stream = _stream
        agent = Agent(name="tool-history-stream-agent", model=primary)

        stale_messages = [
            Message(role="user", content="do X"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            ),
            Message(role="tool", content="tool result", tool_call_id="call_1"),
            Message(role="user", content="continue"),
        ]

        for _ in agent.run_stream_sync(messages=stale_messages):
            pass

        response = agent.run_response
        self.assertEqual(response.content, "recovered")
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
