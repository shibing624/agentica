# -*- coding: utf-8 -*-
"""Wire format of history messages on the OpenAI-compatible chat path.

``reasoning_content`` is a response-only field: providers accept it on input
but ignore it (Taiji/hy4 measured: 215KB of reasoning_content changed
prompt_tokens by 0; DeepSeek documents the same). Sending it back on every
history message is pure request-body dead weight — 85K tokens of it on a long
reasoning-model session — so the chat wire format must strip it. Persistence
and replay keep it via ``to_replay_dict``.
"""
import unittest

from agentica.model.message import Message
from agentica.model.openai import OpenAIChat


class TestReasoningContentNotSentOnWire(unittest.TestCase):
    def test_format_message_strips_reasoning_content(self):
        model = OpenAIChat(id="hy4-preview", api_key="fake")
        msg = Message(
            role="assistant",
            content="answer",
            reasoning_content="private chain of thought " * 100,
        )
        wire = model.format_message(msg)
        self.assertNotIn("reasoning_content", wire)
        self.assertEqual(wire["content"], "answer")

    def test_format_message_keeps_reasoning_out_of_tool_round_history(self):
        model = OpenAIChat(id="hy4-preview", api_key="fake")
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a1", reasoning_content="thinking"),
            Message(role="user", content="next"),
        ]
        wire = [model.format_message(m) for m in msgs]
        self.assertTrue(all("reasoning_content" not in d for d in wire))

    def test_replay_dict_still_carries_reasoning_for_persistence(self):
        msg = Message(role="assistant", content="a", reasoning_content="kept")
        self.assertEqual(msg.to_replay_dict()["reasoning_content"], "kept")


if __name__ == "__main__":
    unittest.main()
