# -*- coding: utf-8 -*-
"""
Tests for Runner._call_with_retry cross-provider fallback chain.

All tests mock LLM API keys and model calls — no real API usage.
"""
import asyncio
import unittest
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

from agentica.model.loop_state import LoopState
from agentica.model.response import ModelResponse
from agentica.runner import Runner


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


class TestFallbackContentFilterFinishReason(unittest.TestCase):
    """Primary returns finish_reason=content_filter -> switch to next model."""

    def test_switches_to_fallback_when_primary_returns_content_filter(self):
        primary = _fake_model(
            "primary-gpt",
            response_factory=lambda: ModelResponse(
                content="I cannot help with that",
                finish_reason="content_filter",
            ),
        )
        fallback = _fake_model(
            "fallback-glm",
            response_factory=lambda: ModelResponse(
                content="real answer",
                finish_reason="stop",
            ),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )

        self.assertEqual(result.content, "real answer")
        self.assertEqual(result.finish_reason, "stop")

    def test_walks_full_chain_until_a_clean_finish_reason(self):
        primary = _fake_model(
            "primary",
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
        )
        fb1 = _fake_model(
            "fb1",
            response_factory=lambda: ModelResponse(content="y", finish_reason="content-filter"),
        )
        fb2 = _fake_model(
            "fb2",
            response_factory=lambda: ModelResponse(content="z", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fb1, fb2]

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )
        self.assertEqual(result.content, "z")

    def test_raises_when_entire_chain_returns_content_filter(self):
        primary = _fake_model(
            "p",
            response_factory=lambda: ModelResponse(content="a", finish_reason="content_filter"),
        )
        fb1 = _fake_model(
            "f1",
            response_factory=lambda: ModelResponse(content="b", finish_reason="content_filter"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fb1]

        state = LoopState()
        with self.assertRaises(RuntimeError) as cm:
            asyncio.run(
                Runner._call_with_retry(primary, [], state, agent, stream=False)
            )
        self.assertIn("2 model", str(cm.exception))


class TestFallbackContentFilterException(unittest.TestCase):
    """Some providers raise instead of setting finish_reason."""

    def test_switches_when_primary_raises_content_filter(self):
        primary = _fake_model(
            "primary",
            raise_exc=RuntimeError("ResponsibleAIPolicyViolation: blocked"),
        )
        fallback = _fake_model(
            "fb",
            response_factory=lambda: ModelResponse(content="ok", finish_reason="stop"),
        )
        agent = _make_agent()
        agent._run_fallback_models = [fallback]

        state = LoopState()
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )
        self.assertEqual(result.content, "ok")


class TestFallbackRetryableExhausted(unittest.TestCase):
    """5xx / timeout / 429 should retry-then-fallback."""

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

        # Tighten retry to keep test fast
        state = LoopState(max_api_retry=2)
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )

        # primary should have been called max_api_retry times before fallback
        self.assertEqual(call_count["n"], 2)
        self.assertEqual(result.content, "recovered")

    def test_503_triggers_retry_then_fallback(self):
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

        state = LoopState(max_api_retry=1)
        result = asyncio.run(
            Runner._call_with_retry(primary, [], state, agent, stream=False)
        )
        self.assertEqual(primary_calls["n"], 1)
        self.assertEqual(result.content, "ok")


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
    """No fallback configured -> behave like before (raise on content_filter / retryable)."""

    def test_content_filter_with_empty_chain_raises(self):
        primary = _fake_model(
            "primary",
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
        )
        agent = _make_agent()
        agent._run_fallback_models = []

        state = LoopState()
        with self.assertRaises(RuntimeError):
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
        from agentica.run_config import RunConfig
        cfg = RunConfig()
        self.assertEqual(cfg.fallback_models, [])

    def test_run_stashes_fallback_models_on_agent(self):
        """Calling run() with config.fallback_models sets agent._run_fallback_models."""
        from agentica.run_config import RunConfig
        from agentica.run_response import RunResponse, RunEvent

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


class TestAgentFallbackModelsField(unittest.TestCase):
    """Agent(fallback_models=...) should be the default chain when no RunConfig override."""

    def test_agent_fallback_models_defaults_empty_list(self):
        agent = _make_agent()
        self.assertEqual(agent.fallback_models, [])

    def test_agent_fallback_models_used_when_no_run_config(self):
        """Without RunConfig.fallback_models, agent.fallback_models is used."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run_response import RunResponse, RunEvent

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
        from agentica.run_config import RunConfig
        from agentica.run_response import RunResponse, RunEvent

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
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
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
            return ModelResponse(content="filtered", finish_reason="content_filter")

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

    def test_last_used_model_id_is_fallback_on_content_filter_recovery(self):
        primary = _fake_model(
            "p",
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
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


class TestFallbackRecoveryEvent(unittest.TestCase):
    """Successful fallback rescue should emit a structured event for audit/observability."""

    def test_fallback_recovery_event_emitted_on_content_filter(self):
        captured = []

        primary = _fake_model(
            "primary-id",
            response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
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
        self.assertEqual(evt["trigger"], "content_filter")
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
            "p", response_factory=lambda: ModelResponse(content="x", finish_reason="content_filter"),
        )
        fb = _fake_model("fb")
        agent = _make_agent()
        agent._run_fallback_models = [fb]

        state = LoopState()
        with unittest.mock.patch("agentica.runner.logger") as mock_logger:
            asyncio.run(Runner._call_with_retry(primary, [], state, agent, stream=False))

        warning_msgs = [
            call.args[0] for call in mock_logger.warning.call_args_list
        ]
        self.assertTrue(
            any("[fallback.recovered]" in msg for msg in warning_msgs),
            f"expected [fallback.recovered] log; got {warning_msgs}",
        )


if __name__ == "__main__":
    unittest.main()
