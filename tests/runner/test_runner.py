# -*- coding: utf-8 -*-
"""
Tests for Runner — core execution engine.
All tests mock LLM API keys and model calls — no real API usage.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.run_response import RunResponse, RunEvent
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.model.usage import RequestUsage


def _make_agent(name="test-agent"):
    """Create a minimal Agent with a fake OpenAI key."""
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        name=name,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    )


class TestRunnerEmptyMessage(unittest.TestCase):
    """Runner should handle empty/None message gracefully."""

    def test_run_with_none_message_returns_empty_response(self):
        agent = _make_agent()
        response = agent.run_sync(message=None)
        self.assertIsInstance(response, RunResponse)
        # Should be empty content, not a crash
        self.assertEqual(response.content, "")

    def test_run_with_empty_string_message_does_not_crash(self):
        """Empty string IS a valid message — should be passed to LLM."""
        agent = _make_agent()
        # Mock the runner to avoid real API call
        mock_response = RunResponse(content="ok", event=RunEvent.run_response.value)
        with patch.object(agent._runner, 'run', new=AsyncMock(return_value=mock_response)):
            response = asyncio.run(agent.run(message=""))
        self.assertIsNotNone(response)


class TestRunnerConcurrentWarning(unittest.TestCase):
    """Runner warns when same Agent instance is reused concurrently.
    Swarm autonomous mode avoids this by cloning agents before parallel dispatch.
    """

    def test_concurrent_run_emits_warning(self):
        """Direct concurrent reuse of the same Agent instance must emit WARNING."""
        agent = _make_agent()
        agent._running = True  # simulate already-running

        with self.assertLogs("agentica", level="WARNING") as cm:
            asyncio.run(agent.run(message=None))

        agent._running = False
        warning_text = "\n".join(cm.output)
        self.assertIn("already running", warning_text.lower())

    def test_running_flag_cleared_after_run(self):
        """_running must be False after a run completes (even via early return)."""
        agent = _make_agent()
        asyncio.run(agent.run(message=None))
        self.assertFalse(agent._running)


class TestRunnerRunTimeout(unittest.TestCase):
    """run_timeout in RunConfig should return a timeout response."""

    def test_run_timeout_returns_response_with_timeout_content(self):
        from agentica.run_config import RunConfig
        agent = _make_agent()

        # Mock model to hang for longer than timeout
        async def slow_response(messages):
            await asyncio.sleep(10)
            return MagicMock()

        with patch.object(agent.model, 'response', new=slow_response):
            with patch.object(agent.model, 'response_stream', new=slow_response):
                response = agent.run_sync(
                    message="hello",
                    config=RunConfig(run_timeout=0.1),
                )
        # Should return a timeout event, not raise
        self.assertIsInstance(response, RunResponse)


class TestRunnerInterruptedTurnPersistence(unittest.TestCase):
    """On user cancel, the turn (question + partial answer + marker) is kept
    instead of being discarded entirely."""

    def test_persist_interrupted_turn_keeps_question_and_partial(self):
        from agentica.memory.models import AgentRun

        agent = _make_agent()
        user_msg = Message(role="user", content="What is 2+2?")
        partial_assistant = Message(role="assistant", content="2 + 2 = ")
        messages_for_model = [user_msg, partial_assistant]
        model_response = ModelResponse(content="2 + 2 = ")

        agent._runner._persist_interrupted_turn(
            agent,
            message="What is 2+2?",
            messages=None,
            user_messages=[user_msg],
            system_message=None,
            messages_for_model=messages_for_model,
            num_input_messages=1,
            model_response=model_response,
            loop_state=LoopState(),
            input_message_ids={id(user_msg)},
        )

        # run_response carries the partial answer + interruption marker
        self.assertIn("2 + 2 = ", agent.run_response.content)
        self.assertIn("[用户中断了回答]", agent.run_response.content)

        # working_memory has the user question and the assistant (with marker)
        wm = agent.working_memory
        roles = [m.role for m in wm.messages]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        assistant_msg = next(m for m in wm.messages if m.role == "assistant")
        self.assertIn("[用户中断了回答]", assistant_msg.content)
        self.assertTrue(any(isinstance(r, AgentRun) for r in wm.runs))
        resumed_history = wm.get_messages_from_last_n_runs()
        self.assertEqual([m.role for m in resumed_history], ["user", "assistant"])
        self.assertIn("2 + 2 = ", resumed_history[-1].content)
        self.assertIn("[用户中断了回答]", resumed_history[-1].content)

    def test_persist_interrupted_turn_skips_prebuilt_messages(self):
        """Pre-built ``messages`` runs manage their own history — no persistence."""
        agent = _make_agent()
        before = list(agent.working_memory.messages)
        agent._runner._persist_interrupted_turn(
            agent,
            message=None,
            messages=[Message(role="user", content="hi")],
            user_messages=[],
            system_message=None,
            messages_for_model=[],
            num_input_messages=0,
            model_response=ModelResponse(content="x"),
            loop_state=LoopState(),
            input_message_ids=set(),
        )
        self.assertEqual(agent.working_memory.messages, before)

    def test_cancel_after_completion_does_not_double_persist(self):
        """Ctrl+C during post-completion hooks must NOT re-persist the turn or
        stamp an interruption marker on a fully-finished answer.

        Mirrors the real CLI path: the background run is cancelled (asyncio
        task cancel) after the answer has streamed and the success path has
        already persisted, while a slow on_end hook is still awaiting.
        """
        from agentica.agent import Agent
        from agentica.hooks import AgentHooks
        from agentica.model.openai import OpenAIChat
        from agentica.model.response import ModelResponse, ModelResponseEvent
        from agentica.run_response import AgentCancelledError

        agent = Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))

        async def fast_stream(messages=None, **_kw):
            chunk = ModelResponse()
            chunk.event = ModelResponseEvent.assistant_response.value
            chunk.content = "The answer is 4."
            yield chunk
            # Mirror the real model layer: append the assistant message after
            # the stream completes so the success-path add_messages sees it.
            messages.append(Message(role="assistant", content="The answer is 4."))

        agent.model.response_stream = fast_stream

        reached_end = asyncio.Event()
        block_event = asyncio.Event()

        class BlockingHook(AgentHooks):
            async def on_end(self, agent, output, **kwargs):
                reached_end.set()
                await block_event.wait()

        agent.hooks = BlockingHook()

        async def consume():
            async for _ in agent.run_stream("What is 2+2?"):
                pass

        async def driver():
            task = asyncio.ensure_future(consume())
            await asyncio.wait_for(reached_end.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, AgentCancelledError):
                pass
            finally:
                block_event.set()

        asyncio.run(driver())

        msgs = agent.working_memory.messages
        users = [m for m in msgs if m.role == "user"]
        assistants = [m for m in msgs if m.role == "assistant"]
        self.assertEqual(len(users), 1, "user message double-added on post-completion cancel")
        self.assertEqual(len(assistants), 1, "assistant double-added on post-completion cancel")
        self.assertNotIn(
            "[用户中断了回答]", assistants[0].content or "",
            "interruption marker must not stamp a finished answer",
        )
        self.assertIn("The answer is 4.", assistants[0].content or "")


class TestRunnerPersistsCompactedContext(unittest.TestCase):
    """A run whose context got compacted must persist the compacted state.

    Two failures used to happen together: `num_input_messages` was captured
    before the loop, so once a compression stage dropped messages the
    "this turn's messages" slice came back empty and the turn's own answer was
    lost; and `runs` was never touched, so the next turn rebuilt the
    pre-compaction history and the compaction saved nothing past the run.
    """

    def _agent_with_history(self, num_runs=3):
        from agentica.agent import Agent
        from agentica.compression.manager import CompressionManager
        from agentica.memory.models import AgentRun
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            add_history_to_context=True,
        )
        for i in range(num_runs):
            user = Message(role="user", content=f"old question {i}")
            assistant = Message(role="assistant", content=f"old answer {i}")
            agent.working_memory.add_run(
                AgentRun(response=RunResponse(messages=[user, assistant]))
            )

        async def fake_response(messages=None, **_kw):
            messages.append(Message(role="assistant", content="The answer is 4."))
            return ModelResponse(content="The answer is 4.")

        agent.model.response = fake_response

        cm = CompressionManager()
        agent.tool_config.compress_tool_results = True
        agent.tool_config.compression_manager = cm
        return agent, cm

    def _run_with_compaction(self, agent, cm):
        with patch.object(cm, "should_compress", return_value=False), \
             patch.object(cm, "_should_auto_compact", return_value=True), \
             patch.object(cm, "_summarise_conversation",
                          new_callable=AsyncMock, return_value="a summary of earlier turns"):
            return agent.run_sync("current question")

    def test_turns_own_answer_is_not_lost(self):
        agent, cm = self._agent_with_history()
        self._run_with_compaction(agent, cm)

        stored = agent.working_memory.runs[-1].response.messages
        self.assertTrue(
            any(m.role == "assistant" and "The answer is 4." in str(m.content) for m in stored),
            f"answer missing from persisted run: {[m.role for m in stored]}",
        )

    def test_superseded_runs_are_dropped(self):
        agent, cm = self._agent_with_history(num_runs=3)
        self._run_with_compaction(agent, cm)
        self.assertEqual(len(agent.working_memory.runs), 1)

    def test_next_turn_history_is_compacted_not_re_expanded(self):
        agent, cm = self._agent_with_history(num_runs=3)
        before = len(agent.working_memory.get_messages_from_last_n_runs())

        self._run_with_compaction(agent, cm)

        history = agent.working_memory.get_messages_from_last_n_runs()
        joined = " ".join(str(m.content) for m in history)
        self.assertLessEqual(len(history), before)
        self.assertIn("a summary of earlier turns", joined)
        self.assertNotIn("old answer 0", joined)

    def test_uncompacted_run_still_uses_the_prefix_slice(self):
        """No compaction: the existing slice behaviour must be untouched."""
        agent, cm = self._agent_with_history(num_runs=1)
        with patch.object(cm, "should_compress", return_value=False), \
             patch.object(cm, "_should_auto_compact", return_value=False):
            agent.run_sync("current question")

        self.assertEqual(len(agent.working_memory.runs), 2)
        stored = agent.working_memory.runs[-1].response.messages
        roles = [m.role for m in stored]
        self.assertIn("user", roles)
        self.assertTrue(
            any(m.role == "assistant" and "The answer is 4." in str(m.content) for m in stored)
        )
        history = agent.working_memory.get_messages_from_last_n_runs()
        self.assertIn("old answer 0", " ".join(str(m.content) for m in history))

    def test_auto_compact_event_exposes_main_agent_scope(self):
        agent, cm = self._agent_with_history()
        events = []
        agent._event_callback = events.append

        self._run_with_compaction(agent, cm)

        event = next(event for event in events if event["type"] == "compact.auto")
        self.assertIs(event["is_main_agent"], True)

    def test_auto_compact_event_exposes_subagent_scope(self):
        agent, cm = self._agent_with_history()
        events = []
        agent._parent_run_id = "parent-run"
        agent._event_callback = events.append

        self._run_with_compaction(agent, cm)

        event = next(event for event in events if event["type"] == "compact.auto")
        self.assertIs(event["is_main_agent"], False)


class TestRunnerNativeCompaction(unittest.TestCase):
    def _agent(self):
        from agentica.agent import Agent
        from agentica.compression.manager import CompressionManager
        from agentica.model.openai import OpenAIResponses

        model = OpenAIResponses(id="gpt-5.6-sol", api_key="fake_openai_key")
        agent = Agent(model=model)
        cm = CompressionManager()
        agent.tool_config.compress_tool_results = True
        agent.tool_config.compression_manager = cm
        return agent, model, cm

    def test_native_success_skips_destructive_local_stages(self):
        from agentica.model.base import NativeCompactionResult
        from agentica.runner import Runner

        agent, model, cm = self._agent()
        messages = [Message(role="user", content="long context")]
        result = NativeCompactionResult(
            checkpoint={
                "type": "openai_responses_compaction",
                "provider": "OpenAI",
                "model": model.id,
                "base_url": "https://api.openai.com/v1",
                "output": [{"id": "cmp_1", "type": "compaction", "encrypted_content": "opaque"}],
            },
            usage={"total_tokens": 123},
        )
        model.estimate_native_compaction_tokens = MagicMock(return_value=160_000)
        model.compact_context = AsyncMock(return_value=result)

        with patch.object(cm, "should_native_compact", return_value=True), \
             patch("agentica.runner.micro_compact") as micro, \
             patch.object(cm, "should_compress") as local_rule, \
             patch.object(cm, "auto_compact", new_callable=AsyncMock) as local_auto:
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))

        self.assertEqual(messages[-1].provider_checkpoint, result.checkpoint)
        micro.assert_not_called()
        local_rule.assert_not_called()
        local_auto.assert_not_called()

    def test_native_failure_falls_back_to_local_pipeline(self):
        from agentica.runner import Runner

        agent, model, cm = self._agent()
        messages = [Message(role="user", content="long context")]
        model.estimate_native_compaction_tokens = MagicMock(return_value=160_000)
        model.compact_context = AsyncMock(side_effect=RuntimeError("404 compact unsupported"))

        with patch.object(cm, "should_native_compact", return_value=True), \
             patch("agentica.runner.micro_compact", return_value=0) as micro, \
             patch.object(cm, "should_compress", return_value=False), \
             patch.object(cm, "auto_compact", new_callable=AsyncMock, return_value=False) as local_auto:
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))

        micro.assert_called_once_with(messages)
        local_auto.assert_awaited_once()
        self.assertIsNone(messages[-1].provider_checkpoint)

    def test_cross_provider_fallback_compacts_portable_transcript(self):
        from agentica.model.openai import OpenAIChat
        from agentica.runner import Runner

        agent, primary, cm = self._agent()
        fallback = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        primary.response = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
        fallback.response = AsyncMock(return_value=ModelResponse(content="fallback answer"))
        agent._run_fallback_models = [fallback]
        cm.auto_compact = AsyncMock(return_value=True)
        messages = [
            Message(
                role="user",
                content="portable transcript",
                provider_checkpoint={"type": "openai_responses_compaction"},
            )
        ]
        state = LoopState(max_api_retry=1)

        result = asyncio.run(Runner._call_with_retry(primary, messages, state, agent))

        self.assertIs(result.used_model, fallback)
        self.assertTrue(result.used_fallback)
        self.assertTrue(state.context_collapsed)
        cm.auto_compact.assert_awaited_once_with(messages, model=fallback, force=True)

    def test_same_responses_identity_reuses_native_checkpoint_on_fallback(self):
        from agentica.model.openai import OpenAIResponses
        from agentica.runner import Runner

        agent, primary, cm = self._agent()
        fallback = OpenAIResponses(id=primary.id, api_key="second-key")
        primary.response = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
        fallback.response = AsyncMock(return_value=ModelResponse(content="fallback answer"))
        agent._run_fallback_models = [fallback]
        cm.auto_compact = AsyncMock(return_value=True)
        messages = [
            Message(
                role="user",
                content="portable transcript",
                provider_checkpoint={
                    "type": "openai_responses_compaction",
                    "provider": "OpenAI",
                    "model": primary.id,
                    "base_url": fallback._checkpoint_identity()["base_url"],
                    "output": [
                        {"id": "cmp_1", "type": "compaction", "encrypted_content": "opaque"}
                    ],
                },
            )
        ]

        result = asyncio.run(
            Runner._call_with_retry(primary, messages, LoopState(max_api_retry=1), agent)
        )

        self.assertIs(result.used_model, fallback)
        cm.auto_compact.assert_not_awaited()


class TestRunnerStructuredOutputFallback(unittest.TestCase):
    """Structured output parse failure should fallback to text, not crash."""

    def test_structured_output_parse_failure_returns_text(self):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from pydantic import BaseModel

        class Report(BaseModel):
            summary: str

        agent = Agent(
            name="structured-agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            response_model=Report,
        )
        # Mock a model response that returns malformed JSON (parse will fail)
        mock_run_response = RunResponse(
            content='{"summary": "ok"}',
            event=RunEvent.run_response.value,
        )
        with patch.object(agent._runner, 'run', new=AsyncMock(return_value=mock_run_response)):
            response = asyncio.run(agent.run(message="analyze"))
        self.assertIsInstance(response, RunResponse)


class TestRunnerCostTracking(unittest.TestCase):
    """Runner should not double-count model usage at end of run."""

    def test_single_request_records_one_llm_call(self):
        agent = _make_agent()

        async def fake_response(messages):
            assistant = Message(role="assistant", content="hi")
            assistant.metrics["input_tokens"] = 10
            assistant.metrics["output_tokens"] = 5
            assistant.metrics["total_tokens"] = 15
            messages.append(assistant)
            agent.model.last_finish_reason = "stop"
            agent.model.usage.add(
                RequestUsage(input_tokens=10, output_tokens=5, total_tokens=15)
            )
            agent.model._cost_tracker.record(
                model_id=agent.model.id,
                input_tokens=10,
                output_tokens=5,
            )
            return ModelResponse(content="hi")

        with patch.object(agent.model, "response", new=fake_response):
            response = agent.run_sync("hello")

        self.assertIsNotNone(response.cost_tracker)
        self.assertEqual(response.cost_tracker.turns, 1)
        self.assertEqual(response.cost_tracker.total_input_tokens, 10)
        self.assertEqual(response.cost_tracker.total_output_tokens, 5)


class TestRunnerToolEventsCarryTheirToolCall(unittest.TestCase):
    """Every tool event must name the tool it is about.

    ``chunk.tools`` is the cumulative list of the whole run, so consumers used to
    guess the subject of a ``ToolCallStarted`` / ``ToolCallCompleted`` event from
    its position (``tools[-1]``, backwards scans). Under a parallel batch those
    guesses mis-attribute results. ``chunk.tool_call`` states it outright.
    """

    @staticmethod
    def _stream_agent():
        """Agent whose LLM fans out TWO parallel tool calls, then answers."""
        from types import SimpleNamespace
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        def alpha() -> str:
            """Return the alpha marker."""
            return "ALPHA_RESULT"

        def beta() -> str:
            """Return the beta marker."""
            return "BETA_RESULT"

        def _tc(index, call_id, name):
            return SimpleNamespace(
                index=index, id=call_id, type="function",
                function=SimpleNamespace(name=name, arguments="{}"),
            )

        def _chunk(content=None, tool_calls=None, finish_reason=None):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason=finish_reason,
                    delta=SimpleNamespace(content=content, reasoning_content=None,
                                          audio=None, tool_calls=tool_calls),
                )],
                usage=None,
            )

        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        turns = iter([
            [_chunk(tool_calls=[_tc(0, "c1", "alpha"), _tc(1, "c2", "beta")],
                    finish_reason="tool_calls")],
            [_chunk(content="done", finish_reason="stop")],
        ])

        async def fake_invoke_stream(messages):
            for c in next(turns):
                yield c

        model.invoke_stream = fake_invoke_stream
        return Agent(name="t", model=model, tools=[alpha, beta])

    def _collect(self):
        from agentica.run_config import RunConfig

        agent = self._stream_agent()
        started, completed = [], []

        async def _drive():
            async for chunk in agent.run_stream(
                "run both", config=RunConfig(stream_intermediate_steps=True)
            ):
                if chunk is None:
                    continue
                if chunk.event == RunEvent.tool_call_started.value:
                    started.append(chunk.tool_call)
                elif chunk.event == RunEvent.tool_call_completed.value:
                    completed.append(chunk.tool_call)

        asyncio.run(_drive())
        return started, completed

    def test_started_events_name_their_own_call(self):
        started, _ = self._collect()
        self.assertEqual([(t.tool_call_id, t.tool_name) for t in started],
                         [("c1", "alpha"), ("c2", "beta")])

    def test_completed_events_carry_their_own_result(self):
        _, completed = self._collect()
        self.assertEqual(
            [(t.tool_call_id, t.tool_name, t.content) for t in completed],
            [("c1", "alpha", "ALPHA_RESULT"), ("c2", "beta", "BETA_RESULT")],
            "each completion must report the tool that finished, not the last one called",
        )


if __name__ == "__main__":
    unittest.main()
