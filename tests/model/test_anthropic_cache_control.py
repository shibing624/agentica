# -*- coding: utf-8 -*-
"""Tests for Anthropic prompt caching (cache_control injection).

Verifies that Claude model correctly injects cache_control breakpoints into:
1. System message (prepare_request_kwargs)
2. Last 3 conversation messages - system_and_3 strategy (format_messages)
3. Respects enable_cache_control=False to disable injection
4. cache_write tokens tracked in update_usage_metrics
"""
import asyncio
import unittest
from unittest.mock import MagicMock

from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message


def _make_claude(**kwargs) -> Claude:
    """Create a Claude instance without requiring a real API key."""
    defaults = dict(api_key="fake_anthropic_key")
    defaults.update(kwargs)
    return Claude(**defaults)


class TestCacheControlDefault(unittest.TestCase):
    """enable_cache_control defaults to True."""

    def test_default_enabled(self):
        model = _make_claude()
        self.assertTrue(model.enable_cache_control)

    def test_explicit_disable(self):
        model = _make_claude(enable_cache_control=False)
        self.assertFalse(model.enable_cache_control)


class TestSystemMessageCacheControl(unittest.TestCase):
    """prepare_request_kwargs wraps system message with cache_control."""

    def test_system_is_block_list_when_enabled(self):
        model = _make_claude(enable_cache_control=True)
        kwargs = model.prepare_request_kwargs("You are a helpful assistant.")
        system = kwargs["system"]

        # Should be a list of content blocks, not a plain string
        self.assertIsInstance(system, list)
        self.assertEqual(len(system), 1)

        block = system[0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "You are a helpful assistant.")
        self.assertEqual(block["cache_control"], {"type": "ephemeral"})

    def test_system_is_plain_string_when_disabled(self):
        model = _make_claude(enable_cache_control=False)
        kwargs = model.prepare_request_kwargs("You are a helpful assistant.")
        system = kwargs["system"]

        self.assertIsInstance(system, str)
        self.assertEqual(system, "You are a helpful assistant.")

    def test_empty_system_stays_plain(self):
        """Empty system message should not be wrapped even with caching enabled."""
        model = _make_claude(enable_cache_control=True)
        kwargs = model.prepare_request_kwargs("")
        # Empty string is falsy, so no wrapping
        self.assertEqual(kwargs["system"], "")

    def test_other_kwargs_preserved(self):
        """cache_control injection should not break other request kwargs."""
        model = _make_claude(enable_cache_control=True, temperature=0.5)
        kwargs = model.prepare_request_kwargs("System prompt here.")
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertIn("max_tokens", kwargs)


class TestConversationCacheControl(unittest.TestCase):
    """format_messages injects cache_control on last 3 conversation messages (system_and_3 strategy)."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_last_3_messages_get_cache_control(self):
        model = _make_claude(enable_cache_control=True)
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
        ]
        chat_msgs, system_str = self._run(model.format_messages(messages))

        # System messages extracted separately
        self.assertEqual(system_str, "System prompt")

        # Should have 3 chat messages (2 user + 1 assistant)
        self.assertEqual(len(chat_msgs), 3)

        # system_and_3: all 3 messages should have cache_control (3 <= 3)
        for msg in chat_msgs:
            content = msg["content"]
            self.assertIsInstance(content, list)
            last_block = content[-1]
            self.assertIn("cache_control", last_block)
            self.assertEqual(last_block["cache_control"], {"type": "ephemeral"})

    def test_single_user_message_gets_cache_control(self):
        """Even a single user message should get cache_control."""
        model = _make_claude(enable_cache_control=True)
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Only message"),
        ]
        chat_msgs, _ = self._run(model.format_messages(messages))
        self.assertEqual(len(chat_msgs), 1)

        last_block = chat_msgs[0]["content"][-1]
        self.assertIn("cache_control", last_block)

    def test_no_cache_control_when_disabled(self):
        model = _make_claude(enable_cache_control=False)
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Hello"),
        ]
        chat_msgs, _ = self._run(model.format_messages(messages))

        last_block = chat_msgs[0]["content"][-1]
        self.assertNotIn("cache_control", last_block)

    def test_no_crash_on_empty_messages(self):
        """Only system messages -> no chat_messages -> no crash."""
        model = _make_claude(enable_cache_control=True)
        messages = [
            Message(role="system", content="System only"),
        ]
        chat_msgs, system_str = self._run(model.format_messages(messages))
        self.assertEqual(len(chat_msgs), 0)
        self.assertEqual(system_str, "System only")

    def test_assistant_as_last_message_gets_cache_control(self):
        """Last message can be assistant (e.g. tool call round), should still get breakpoint."""
        model = _make_claude(enable_cache_control=True)
        messages = [
            Message(role="system", content="System"),
            Message(role="user", content="Do something"),
            Message(role="assistant", content="Done"),
        ]
        chat_msgs, _ = self._run(model.format_messages(messages))
        self.assertEqual(len(chat_msgs), 2)

        last_block = chat_msgs[-1]["content"][-1]
        self.assertIn("cache_control", last_block)


class TestCacheWriteTracking(unittest.TestCase):
    """update_usage_metrics passes cache_write to CostTracker."""

    def test_cache_write_recorded(self):
        model = _make_claude()
        # Mock a CostTracker
        mock_tracker = MagicMock()
        model._cost_tracker = mock_tracker

        # Build a fake Anthropic Usage object
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 200
        usage.cache_read_input_tokens = 800
        usage.cache_creation_input_tokens = 500

        from agentica.model.metrics import Metrics
        metrics = Metrics()
        metrics.response_timer.start()
        metrics.response_timer.stop()

        assistant_msg = Message(role="assistant", content="response")
        model.update_usage_metrics(assistant_msg, usage, metrics)

        # CostTracker.record should be called with both cache_read and cache_write
        mock_tracker.record.assert_called_once()
        call_kwargs = mock_tracker.record.call_args
        # Could be positional or keyword args
        if call_kwargs.kwargs:
            self.assertEqual(call_kwargs.kwargs["cache_read_tokens"], 800)
            self.assertEqual(call_kwargs.kwargs["cache_write_tokens"], 500)
        else:
            # positional: model_id, input, output, cache_read, cache_write
            args = call_kwargs.args
            self.assertEqual(args[3], 800)  # cache_read_tokens
            self.assertEqual(args[4], 500)  # cache_write_tokens

    def test_cache_zero_when_no_cache(self):
        """When no caching occurs, cache tokens should be 0."""
        model = _make_claude()
        mock_tracker = MagicMock()
        model._cost_tracker = mock_tracker

        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 100
        usage.cache_read_input_tokens = None
        usage.cache_creation_input_tokens = None

        from agentica.model.metrics import Metrics
        metrics = Metrics()
        metrics.response_timer.start()
        metrics.response_timer.stop()

        assistant_msg = Message(role="assistant", content="response")
        model.update_usage_metrics(assistant_msg, usage, metrics)

        mock_tracker.record.assert_called_once()
        call_kwargs = mock_tracker.record.call_args
        if call_kwargs.kwargs:
            self.assertEqual(call_kwargs.kwargs["cache_read_tokens"], 0)
            self.assertEqual(call_kwargs.kwargs["cache_write_tokens"], 0)


class TestEndToEndRequestShape(unittest.TestCase):
    """Verify the full request shape matches Anthropic API expectations."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_full_multi_turn_request(self):
        """Simulate a 3-turn conversation and verify both system and message cache_control."""
        model = _make_claude(enable_cache_control=True)
        messages = [
            Message(role="system", content="You are a coding assistant with deep knowledge of Python."),
            Message(role="user", content="What is a decorator?"),
            Message(role="assistant", content="A decorator is a function that wraps another function."),
            Message(role="user", content="Show me an example."),
        ]

        chat_msgs, system_str = self._run(model.format_messages(messages))
        request_kwargs = model.prepare_request_kwargs(system_str)

        # System: block list with cache_control
        system_blocks = request_kwargs["system"]
        self.assertIsInstance(system_blocks, list)
        self.assertEqual(system_blocks[0]["cache_control"], {"type": "ephemeral"})

        # Messages: 3 items (user, assistant, user)
        self.assertEqual(len(chat_msgs), 3)

        # system_and_3 strategy: all 3 messages get cache_control (3 <= 3)
        for msg in chat_msgs:
            content = msg["content"]
            last_block = content[-1] if isinstance(content, list) else content
            self.assertIn("cache_control", last_block)


if __name__ == "__main__":
    unittest.main()
