# -*- coding: utf-8 -*-
"""Calibrating the local token estimate with the provider's own prompt count.

Production: a 9-day gpt-6.1-sol session estimated ~640k while the gateway
billed 1,047,795 for the same request. No compression layer ever fired and the
session died on "input exceeds the context window". These tests pin that the
provider figure on the reply now moves the thresholds.
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agentica.compression.calibration import ContextCalibration
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.usage import record_context_tokens


def _reply(context_tokens: int) -> Message:
    reply = Message(role="assistant", content="ok")
    reply.metrics["context_tokens"] = context_tokens
    return reply


class TestContextCalibration(unittest.TestCase):
    def test_reply_to_the_recorded_request_sets_the_ratio(self):
        cal = ContextCalibration()
        messages = [Message(role="user", content="q")]
        cal.record_send(messages, 1000)
        messages.append(_reply(1640))

        cal.observe(messages)

        self.assertAlmostEqual(cal.ratio, 1.64)
        self.assertEqual(cal.scale(600_000), 984_000)

    def test_a_provider_count_below_the_estimate_leaves_it_alone(self):
        cal = ContextCalibration()
        messages = [Message(role="user", content="q")]
        cal.record_send(messages, 1000)
        messages.append(_reply(700))

        cal.observe(messages)

        self.assertEqual(cal.ratio, 1.0)

    def test_a_rebuilt_message_list_is_not_paired_with_the_old_estimate(self):
        cal = ContextCalibration()
        cal.record_send([Message(role="user", content="old run")], 1000)
        # Next run: new list, an unrelated history reply with its own figure.
        rebuilt = [Message(role="user", content="history"), _reply(9000)]

        cal.observe(rebuilt)

        self.assertEqual(cal.ratio, 1.0)

    def test_the_ratio_survives_until_a_newer_reply_replaces_it(self):
        cal = ContextCalibration()
        messages = [Message(role="user", content="q")]
        cal.record_send(messages, 1000)
        messages.append(_reply(2000))
        cal.observe(messages)
        # A reply with no usage (gateway omitted it) does not reset to 1.0.
        cal.record_send(messages, 1100)
        messages.append(Message(role="assistant", content="no usage"))
        cal.observe(messages)

        self.assertEqual(cal.ratio, 2.0)


class TestRecordContextTokens(unittest.TestCase):
    def test_openai_inclusive_prompt_is_taken_as_is(self):
        message = Message(role="assistant")
        record_context_tokens(message, 1000, {"cached_tokens": 900})
        self.assertEqual(message.metrics["context_tokens"], 1000)

    def test_exclusive_cache_counters_are_added_back(self):
        # OpenAI-compatible gateway fronting Claude: fresh input only, cache
        # read/creation reported beside it.
        message = Message(role="assistant")
        record_context_tokens(message, 4, {"cache_read_tokens": 90_000, "cache_creation_tokens": 500})
        self.assertEqual(message.metrics["context_tokens"], 90_504)

    def test_native_claude_counts_cache_read_and_write(self):
        from agentica.model.anthropic.claude import Claude
        from agentica.model.metrics import Metrics

        model = Claude(id="claude-opus-5-5", api_key="fake_anthropic_key")
        message = Message(role="assistant")
        metrics = Metrics()
        metrics.response_timer.start()
        metrics.response_timer.stop()
        usage = SimpleNamespace(
            input_tokens=12, output_tokens=5,
            cache_read_input_tokens=80_000, cache_creation_input_tokens=300,
        )

        model.update_usage_metrics(message, usage, metrics)

        self.assertEqual(message.metrics["context_tokens"], 80_312)


class TestRunnerUsesCalibratedCount(unittest.TestCase):
    """The Layer 2 decision reads the scaled figure, not raw tiktoken."""

    def _agent(self, window: int):
        from agentica.agent import Agent
        from agentica.compression.manager import CompressionManager
        from agentica.model.openai import OpenAIChat

        model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
        model.context_window = window
        agent = Agent(model=model)
        cm = CompressionManager()
        agent.tool_config.compression_manager = cm
        agent.tool_config.enable_evict = False
        return agent, model, cm

    def _run(self, messages, agent, model, cm):
        from agentica.runner import Runner

        with patch.object(cm, "auto_compact", new_callable=AsyncMock, return_value=False) as cut:
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))
        return cut

    def test_provider_count_pushes_an_under_threshold_estimate_into_a_cut(self):
        from agentica.utils.tokens import count_tokens

        messages = [Message(role="user", content=" ".join(f"w{i}" for i in range(3000)))]
        local = count_tokens(messages, None, "gpt-4o")
        # Locally the request sits at ~60% of the window: nothing fires.
        agent, model, cm = self._agent(window=int(local / 0.6))
        cm.fallback_claimed = True  # isolate Layer 2 from the notes nudge
        self.assertFalse(self._run(messages, agent, model, cm).await_count)

        # The provider says that request was 1.7x what we thought.
        messages.append(_reply(int(local * 1.7)))
        messages.append(Message(role="user", content="next"))
        cut = self._run(messages, agent, model, cm)

        self.assertEqual(cut.await_count, 1, "60% local x 1.7 is past the 95% cut")

    def test_without_provider_usage_the_estimate_is_unchanged(self):
        from agentica.utils.tokens import count_tokens

        messages = [Message(role="user", content=" ".join(f"w{i}" for i in range(3000)))]
        local = count_tokens(messages, None, "gpt-4o")
        agent, model, cm = self._agent(window=int(local / 0.6))
        cm.fallback_claimed = True
        self._run(messages, agent, model, cm)
        messages.append(Message(role="assistant", content="no usage reported"))
        messages.append(Message(role="user", content="next"))

        self.assertFalse(self._run(messages, agent, model, cm).await_count)


if __name__ == "__main__":
    unittest.main()
