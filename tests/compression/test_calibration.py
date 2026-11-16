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


class TestRemainingReminderSkipsARollover(unittest.TestCase):
    """The 25%-left reminder is folded into the trailing user turn; when the
    same request rolls the window over, that turn becomes the new window's
    first user row and is written back to the log, so the reminder showed up
    in the user's own bubble. A rollover's preamble already states the
    tokens left."""

    REMINDER = "tokens left in this context window."

    def _setup(self, fill_ratio: float):
        from agentica.utils.tokens import count_tokens

        question = "fix the bug"
        messages = [
            Message(role="user", content=" ".join(f"w{i}" for i in range(3000))),
            Message(role="assistant", content="ok"),
            Message(role="user", content=question),
        ]
        local = count_tokens(messages, None, "gpt-4o")
        agent, model, cm = TestRunnerUsesCalibratedCount._agent(None, window=int(local / fill_ratio))
        cm.fallback_claimed = True
        return messages, agent, model, cm, question

    def _run(self, messages, agent, model, cm):
        from agentica.runner import Runner

        seen = {}

        async def cut(msgs, **_kw):
            seen["trailing"] = msgs[-1].content
            return False

        with patch.object(cm, "auto_compact", side_effect=cut) as compact:
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))
        return compact, seen

    def test_a_request_that_rolls_over_does_not_fold_the_reminder_into_the_question(self):
        messages, agent, model, cm, question = self._setup(fill_ratio=0.97)
        compact, seen = self._run(messages, agent, model, cm)

        self.assertEqual(compact.await_count, 1, "97% of the window is past the cut")
        self.assertEqual(seen["trailing"], question)
        self.assertFalse(cm.reminder_claimed)

    def test_below_the_cut_the_reminder_is_still_folded_once(self):
        messages, agent, model, cm, question = self._setup(fill_ratio=0.85)
        compact, _ = self._run(messages, agent, model, cm)

        self.assertEqual(compact.await_count, 0)
        self.assertTrue(messages[-1].content.startswith(question + "\n\n<context_window>"))
        self.assertIn(self.REMINDER, messages[-1].content)
        self.assertTrue(cm.reminder_claimed)


if __name__ == "__main__":
    unittest.main()


class TestOneTriggerPruneOrNewWindow(unittest.TestCase):
    """One compression trigger, two answers (the dsh design).

    Below the Layer 2 line (``compact_token_limit`` or 95% of the window) the
    history is never rewritten, so the request grows as a strict prefix of the
    cached one. At the line, pruning old tool results to head + tail is applied
    only if it reaches PRUNE_TARGET_RATIO of the working window; otherwise the
    window is cut with nothing pruned first, so a crossing costs at most one
    prefix break.
    """

    def _setup(self, *, cap_fill: float, body_words: int, prose_words: int = 0):
        from agentica.agent import Agent
        from agentica.compression.manager import CompressionManager
        from agentica.model.openai import OpenAIChat
        from agentica.utils.tokens import count_tokens

        messages = [Message(role="user", content="question")]
        for i in range(30):
            messages.append(Message(role="assistant", content=" ".join(
                f"note{i}_{j}" for j in range(prose_words)), tool_calls=[{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]))
            messages.append(Message(
                role="tool", tool_call_id=f"call_{i}",
                content=" ".join(f"line{i}_{j}" for j in range(body_words)),
            ))
        messages.append(Message(role="user", content="next"))
        local = count_tokens(messages, None, "gpt-4o")

        model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
        model.context_window = local * 10  # provider window: far away
        agent = Agent(model=model)
        cm = CompressionManager()
        cm.fallback_claimed = True
        agent.tool_config.compression_manager = cm
        cm.compact_token_limit = int(local / cap_fill)
        return messages, agent, model, cm

    @staticmethod
    def _history(messages):
        return [m.content for m in messages if m.role in ("tool", "assistant")]

    def _run(self, messages, agent, model, cm):
        from agentica.runner import Runner

        with patch.object(cm, "auto_compact", new_callable=AsyncMock, return_value=False) as cut:
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))
        return cut

    def test_below_the_line_history_is_byte_identical(self):
        messages, agent, model, cm = self._setup(cap_fill=0.9, body_words=1500)
        before = self._history(messages)

        cut = self._run(messages, agent, model, cm)

        self.assertEqual(self._history(messages), before, "no tool result may be rewritten")
        self.assertFalse(cut.await_count)

    def test_at_the_line_a_prune_that_reaches_the_target_keeps_the_window(self):
        # Long results: head + tail of each frees far more than half the cap.
        messages, agent, model, cm = self._setup(cap_fill=1.05, body_words=1500)

        cut = self._run(messages, agent, model, cm)

        pruned = [m for m in messages if m.role == "tool" and "pruned from the middle" in str(m.content)]
        self.assertEqual(len(pruned), 30, "every older long result is pruned in one pass")
        self.assertFalse(cut.await_count, "the prune was enough: no new window")

    def test_at_the_line_a_prune_that_falls_short_is_not_applied(self):
        # Most of the window is prose pruning cannot touch: the prune would
        # leave the request above half the cap, so the window is cut instead
        # and history is left exactly as it was.
        messages, agent, model, cm = self._setup(cap_fill=1.05, body_words=1500, prose_words=3000)
        before = self._history(messages)

        cut = self._run(messages, agent, model, cm)

        self.assertEqual(self._history(messages), before, "no prefix break before the cut")
        self.assertEqual(cut.await_count, 1)


class TestOnePrefixBreakPerCrossing(unittest.TestCase):
    """Drive the real pipeline across a long growing session.

    Every request must be the previous one plus an appended tail, except at a
    crossing of the compaction line, where exactly one break is allowed: a
    prune that keeps the window, or a new window. With the old two-line design
    a separate 80% eviction rewrote history over and over before the cut.
    """

    def test_growth_is_append_only_between_crossings(self):
        from agentica.agent import Agent
        from agentica.compression.manager import CompressionManager
        from agentica.model.openai import OpenAIChat
        from agentica.runner import Runner
        from agentica.runner.compress import _first_prefix_break, _prefix_digests

        model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
        model.context_window = 200_000
        agent = Agent(model=model)
        cm = CompressionManager()
        cm.fallback_claimed = True
        agent.tool_config.compression_manager = cm
        cm.compact_token_limit = 60_000

        cuts = []

        async def fake_cut(messages, **_):
            cuts.append(len(messages))
            del messages[1:-1]
            return True

        messages = [Message(role="system", content="sys"), Message(role="user", content="go")]
        breaks, prunes, prev = [], 0, None
        with patch.object(cm, "auto_compact", side_effect=fake_cut):
            for i in range(80):
                messages.append(Message(role="assistant", content="", tool_calls=[{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }]))
                messages.append(Message(
                    role="tool", tool_call_id=f"c{i}",
                    content=" ".join(f"r{i}_{j}" for j in range(1200)),
                ))
                n_cuts = len(cuts)
                asyncio.run(Runner._maybe_compress_messages(messages, agent, model, LoopState()))
                digests = _prefix_digests(messages)
                if prev is not None:
                    brk = _first_prefix_break(prev, digests)
                    if brk is not None:
                        breaks.append(i)
                        pruned_now = any("pruned from the middle" in str(m.content) for m in messages)
                        self.assertTrue(len(cuts) > n_cuts or pruned_now, f"break at step {i} without a crossing")
                        prunes += len(cuts) == n_cuts
                prev = digests

        self.assertGreater(prunes, 0, "the fixture must exercise the prune path")
        self.assertGreater(len(cuts), 0, "and the new-window path")
        self.assertEqual(len(breaks), prunes + len(cuts), "one break per crossing, nothing else")
