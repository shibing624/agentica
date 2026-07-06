# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for think scrubber + retryable error classification.
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.think_scrubber import (
    scrub_reasoning, contains_reasoning_leak, sanitize_assistant_content_for_history,
)
from agentica.agent.history_filter import apply_history_pipeline
from agentica.agent.config import HistoryConfig
from agentica.model.message import Message


class TestScrubReasoning(unittest.TestCase):
    def test_strips_think_block(self):
        text = "<think>let me ponder this</think>The answer is 42."
        self.assertEqual(scrub_reasoning(text), "The answer is 42.")

    def test_strips_thinking_block_multiline(self):
        text = "<thinking>\nstep 1\nstep 2\n</thinking>\n\nFinal output"
        self.assertEqual(scrub_reasoning(text), "Final output")

    def test_strips_reasoning_and_scratchpad(self):
        self.assertEqual(scrub_reasoning("<reasoning>x</reasoning>done"), "done")
        self.assertEqual(scrub_reasoning("<scratchpad>y</scratchpad>done"), "done")

    def test_no_tags_is_noop(self):
        text = "Just a normal answer with no tags."
        self.assertIs(scrub_reasoning(text), text)

    def test_unterminated_tag_left_alone(self):
        # No closing tag — could be truncated real content, don't strip.
        text = "<think>partial reasoning without close"
        self.assertEqual(scrub_reasoning(text), text)

    def test_contains_reasoning_leak(self):
        self.assertTrue(contains_reasoning_leak("<think>a</think>b"))
        self.assertFalse(contains_reasoning_leak("plain text"))
        self.assertFalse(contains_reasoning_leak("<think>no close"))

    def test_sanitize_list_content(self):
        content = [{"type": "text", "text": "<think>hmm</think>visible"}]
        out = sanitize_assistant_content_for_history(content)
        self.assertEqual(out[0]["text"], "visible")

    def test_case_insensitive(self):
        self.assertEqual(scrub_reasoning("<THINK>x</THINK>ok"), "ok")


class TestHistoryPipelineScrub(unittest.TestCase):
    def test_history_scrubs_assistant_reasoning(self):
        history = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="<think>secret plan</think>Hello!"),
        ]
        out = apply_history_pipeline(history, HistoryConfig(), None)
        assistant = [m for m in out if m.role == "assistant"][0]
        self.assertEqual(assistant.content, "Hello!")
        self.assertNotIn("secret plan", assistant.content)

    def test_history_scrub_disabled(self):
        history = [Message(role="assistant", content="<think>x</think>Hi")]
        out = apply_history_pipeline(history, HistoryConfig(scrub_reasoning=False), None)
        self.assertIn("<think>", out[0].content)

    def test_user_content_untouched(self):
        history = [Message(role="user", content="<think>user wrote this</think>keep")]
        out = apply_history_pipeline(history, HistoryConfig(), None)
        self.assertIn("<think>", out[0].content)


if __name__ == "__main__":
    unittest.main()
