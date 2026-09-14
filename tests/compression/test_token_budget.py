# -*- coding: utf-8 -*-
"""Token-budget fragments stay off the frozen system prefix."""
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.token_budget import (
    REMINDER_REMAINING_RATIO,
    WINDOW_CONTINUATION_MARK,
    full_window_text,
    is_context_window_message,
    remaining_text,
    reminder_threshold,
    tokens_remaining,
    window_message,
)
from agentica.agent.config import ToolConfig
from agentica.compression.manager import (
    auto_compact_threshold,
    compact_token_limit_of,
    working_context_window,
)
from agentica.model.message import Message


class TestWorkingBudgetIsOneSource(unittest.TestCase):
    """Every surface that reports or decides compression reads the cap here.

    The bug this pins: the runner evicted against ``min(cap, window)`` while
    the status bar divided by ``model.context_window`` alone, so one session
    read as "over budget, evicting" in one place and "61%, healthy" in the
    other. Two derivations of the same policy value is what allowed that.
    """

    def test_cap_below_the_window_is_the_working_window(self):
        self.assertEqual(working_context_window(1_000_000, 512_000), 512_000)

    def test_cap_above_the_window_cannot_raise_it(self):
        self.assertEqual(working_context_window(128_000, 2_000_000), 128_000)

    def test_no_cap_leaves_the_provider_window(self):
        self.assertEqual(working_context_window(128_000, None), 128_000)
        self.assertEqual(working_context_window(128_000, 0), 128_000)

    def test_unknown_window_yields_zero_not_a_wrong_budget(self):
        self.assertEqual(working_context_window(0, 512_000), 0)

    def test_layer2_threshold_uses_the_cap_as_an_absolute_budget(self):
        """A cap is a working budget, not 95% of one: 512k fires at 512k."""
        self.assertEqual(auto_compact_threshold(1_000_000, 512_000), 512_000)
        self.assertEqual(auto_compact_threshold(1_000_000, None), 950_000)

    def test_limit_is_read_from_the_tool_config_field(self):
        self.assertEqual(compact_token_limit_of(ToolConfig(compact_token_limit=512_000)), 512_000)
        self.assertIsNone(compact_token_limit_of(ToolConfig()))
        self.assertIsNone(compact_token_limit_of(ToolConfig(compact_token_limit=0)))


class TestTokenBudget(unittest.TestCase):
    def test_full_window_is_a_user_fragment(self):
        msg = window_message(1, 12_000, "/tmp/s.notes.md")
        self.assertEqual(msg.role, "user")
        self.assertTrue(is_context_window_message(msg))
        self.assertIn("Current context window 1.", msg.content)
        self.assertIn("12000 tokens left", msg.content)
        self.assertIn("/tmp/s.notes.md", msg.content)
        self.assertIn("search_session", msg.content)

    def test_reminder_threshold_is_a_quarter_of_the_working_window(self):
        self.assertEqual(reminder_threshold(1000), int(1000 * REMINDER_REMAINING_RATIO))
        self.assertEqual(tokens_remaining(800, 1000), 200)
        self.assertEqual(tokens_remaining(1200, 1000), 0)

    def test_remaining_text_does_not_open_a_new_window_id(self):
        text = remaining_text(250, "/tmp/s.notes.md")
        self.assertIn("250 tokens left", text)
        self.assertNotIn("Current context window", text)
        self.assertIn("/tmp/s.notes.md", text)

    def test_system_messages_are_never_tagged_as_budget(self):
        self.assertFalse(
            is_context_window_message(Message(role="system", content=full_window_text(1, 1)))
        )
        self.assertFalse(is_context_window_message(Message(role="user", content="hello")))

    def test_window_fragments_do_not_explain_a_missing_summary(self):
        """A fresh window reports facts, not the absence of a digest.

        The cut no longer runs a summarizer, so neither fragment may tell the
        model a summary was skipped — that reads as a loss it should look for.
        """
        for text in (full_window_text(2, 512_000), full_window_text(2, 512_000, "/tmp/s.notes.md")):
            self.assertNotIn("summar", text.lower())
        self.assertNotIn("summar", WINDOW_CONTINUATION_MARK.lower())


if __name__ == "__main__":
    unittest.main()
