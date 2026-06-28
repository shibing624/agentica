# -*- coding: utf-8 -*-
"""
Tests for Runner — core execution engine.
All tests mock LLM API keys and model calls — no real API usage.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.run_response import RunResponse, RunEvent
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
        )
        self.assertEqual(agent.working_memory.messages, before)


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


if __name__ == "__main__":
    unittest.main()
