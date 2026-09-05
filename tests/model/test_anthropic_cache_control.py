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

import pytest

from agentica.model.anthropic.claude import Claude
from agentica.model import cache_routing as routing
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

    def test_marker_split_when_enabled(self):
        """VOLATILE_SYSTEM_MARKER splits stable/volatile; breakpoint on stable head only."""
        from agentica.model.message import VOLATILE_SYSTEM_MARKER

        model = _make_claude(enable_cache_control=True)
        system = f"STABLE PREFIX\n{VOLATILE_SYSTEM_MARKER}\nvolatile memory"
        blocks = model.prepare_request_kwargs(system)["system"]

        self.assertIsInstance(blocks, list)
        self.assertEqual(len(blocks), 2)
        # Stable head carries the breakpoint; volatile tail does not.
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocks[1])
        # The marker must not leak into either block.
        for block in blocks:
            self.assertNotIn(VOLATILE_SYSTEM_MARKER, block["text"])
        self.assertEqual(blocks[0]["text"], "STABLE PREFIX")
        self.assertEqual(blocks[1]["text"], "volatile memory")

    def test_marker_stripped_when_disabled(self):
        """Caching off: marker is still stripped so it never reaches the model."""
        from agentica.model.message import VOLATILE_SYSTEM_MARKER

        model = _make_claude(enable_cache_control=False)
        system = f"STABLE PREFIX\n{VOLATILE_SYSTEM_MARKER}\nvolatile memory"
        result = model.prepare_request_kwargs(system)["system"]

        self.assertIsInstance(result, str)
        self.assertNotIn(VOLATILE_SYSTEM_MARKER, result)
        self.assertEqual(result, "STABLE PREFIX\n\nvolatile memory")

    def test_no_marker_single_block(self):
        """Without a marker the whole system is one cached block (no regression)."""
        model = _make_claude(enable_cache_control=True)
        blocks = model.prepare_request_kwargs("Just a plain system prompt.")["system"]
        self.assertIsInstance(blocks, list)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})


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

        details = model.usage.input_tokens_details
        self.assertEqual(details.cached_tokens, 800)
        self.assertEqual(details.cache_read_tokens, 800)
        self.assertEqual(details.cache_creation_tokens, 500)

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


class TestStickyRoutingHeader(unittest.TestCase):
    """cache_control_session_header pins requests to one proxy backend.

    Load-balancing proxies fan out across upstreams unless a sticky header is
    sent; unrouted requests show a much higher rate of schema-validation 400s.
    """

    def _headers(self, model: Claude) -> dict:
        client = model.get_client()
        return {k.lower(): v for k, v in (client.default_headers or {}).items()}

    def test_off_by_default(self):
        model = _make_claude()
        model.session_id = "sess-1"
        self.assertNotIn("x-session-id", self._headers(model))

    def test_session_id_is_used_when_set(self):
        model = _make_claude(cache_control_session_header="X-Session-Id")
        model.session_id = "sess-abc-123"
        self.assertEqual(self._headers(model).get("x-session-id"), "sess-abc-123")

    def test_falls_back_to_persistent_id_without_session(self):
        """Bare SDK use: no session, but a stable key still beats none."""
        model = _make_claude(cache_control_session_header="X-Session-Id",
                              base_url="https://proxy.example/anthropic")
        self.assertIsNone(model.session_id)
        sid = self._headers(model).get("x-session-id")
        self.assertTrue(sid and sid.startswith("agentica-cache-"))

    def test_id_is_stable_across_calls(self):
        model = _make_claude(cache_control_session_header="X-Session-Id",
                              base_url="https://proxy.example/anthropic")
        first = self._headers(model).get("x-session-id")
        second = self._headers(model).get("x-session-id")
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_explicit_default_headers_win(self):
        """A same-named entry in default_headers is deliberate and not clobbered."""
        model = _make_claude(
            cache_control_session_header="X-Session-Id",
            default_headers={"X-Session-Id": "pinned-by-user"},
        )
        model.session_id = "sess-abc-123"
        self.assertEqual(self._headers(model).get("x-session-id"), "pinned-by-user")

    def test_header_survives_client_rebuild(self):
        """The CLI rebuilds the client per event loop; the id must stick across it."""
        model = _make_claude(cache_control_session_header="X-Session-Id",
                              base_url="https://proxy.example/anthropic")

        async def build():
            return model.get_client()

        c1 = asyncio.run(build())
        c2 = asyncio.run(build())  # fresh loop -> fresh client
        self.assertIsNot(c1, c2)
        self.assertEqual(
            {k.lower(): v for k, v in c1.default_headers.items()}.get("x-session-id"),
            {k.lower(): v for k, v in c2.default_headers.items()}.get("x-session-id"),
        )

    def test_session_id_arriving_late_is_not_frozen_out(self):
        """Regression: the id must be resolved per client build, not memoized.

        session_id is None until Agent.update_model() runs. A client built
        before that used to cache the fallback id, and every later rebuild
        inherited it — the real session never reached the wire.
        """
        model = _make_claude(cache_control_session_header="X-Session-Id",
                              base_url="https://proxy.example/anthropic")

        async def build():
            return model.get_client()

        first = self._headers(model).get("x-session-id")
        self.assertTrue(first.startswith("agentica-cache-"))

        model.client = None  # force a rebuild, as a loop change would
        model.session_id = "sess-arrives-later"
        self.assertEqual(self._headers(model).get("x-session-id"), "sess-arrives-later")


if __name__ == "__main__":
    unittest.main()
