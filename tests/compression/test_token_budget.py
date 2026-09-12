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
from agentica.model.message import Message


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
