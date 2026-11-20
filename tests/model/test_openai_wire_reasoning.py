# -*- coding: utf-8 -*-
"""Wire format of history messages on the OpenAI-compatible chat path.

``reasoning_content`` goes back as the model returned it. DeepSeek thinking
mode with tools concatenates every earlier turn's reasoning into the context;
with it stripped, the first request of each new user turn re-rendered the
previous tool loop and lost the cache from there on (measured on
api.deepseek.com, deepseek-flash: new-turn reuse 0.83 stripped vs 0.99 kept;
0.18 on a 97K agent session). Endpoints that ignore the field on input
(Taiji/hy4) pay request bytes, not tokens.
"""
import asyncio
import unittest
from unittest import mock

from agentica.model.message import Message
from agentica.model.openai import OpenAIChat


class TestReasoningContentNotSentOnWire(unittest.TestCase):
    def test_format_message_replays_reasoning_content_verbatim(self):
        model = OpenAIChat(id="hy4-preview", api_key="fake")
        reasoning = "private chain of thought " * 100
        msg = Message(
            role="assistant",
            content="answer",
            reasoning_content=reasoning,
        )
        wire = model.format_message(msg)
        self.assertEqual(wire["reasoning_content"], reasoning)
        self.assertEqual(wire["content"], "answer")

    def test_a_finished_tool_loop_is_sent_the_same_way_in_the_next_turn(self):
        """The new turn must start with the exact bytes the last loop request sent."""
        model = OpenAIChat(id="deepseek-flash", api_key="fake")
        call = {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="", reasoning_content="read it first", tool_calls=[call]),
            Message(role="tool", tool_call_id="t1", content="file body"),
        ]
        in_loop = model.format_messages(msgs)
        next_turn = model.format_messages(
            msgs + [
                Message(role="assistant", content="done", reasoning_content="summarise"),
                Message(role="user", content="next"),
            ]
        )
        self.assertEqual(next_turn[: len(in_loop)], in_loop)
        self.assertEqual(next_turn[1]["reasoning_content"], "read it first")
        self.assertEqual(next_turn[3]["reasoning_content"], "summarise")

    def test_replay_dict_still_carries_reasoning_for_persistence(self):
        msg = Message(role="assistant", content="a", reasoning_content="kept")
        self.assertEqual(msg.to_replay_dict()["reasoning_content"], "kept")

    def test_reasoning_only_assistant_turn_is_omitted_from_request(self):
        """Gateway 4000 ``message has no content``: a truncated thinking turn.

        The output budget can be spent entirely on ``reasoning_content``,
        which the wire strips. Measured on a gateway: ``{role: assistant,
        content: null}`` and ``content: ""`` both 400, while dropping the
        turn (user,user or tool,user) is accepted. The turn carries nothing
        sendable, so it is omitted. A tool-call turn with null content stays.
        """
        model = OpenAIChat(id="claude-opus-5-5", api_key="fake")
        tool_call = {"id": "t", "type": "function", "function": {"name": "execute", "arguments": "{}"}}
        history = [
            Message(role="user", content="read the code"),
            Message(role="assistant", content=None, tool_calls=[tool_call]),
            Message(role="tool", tool_call_id="t", content="ok"),
            Message(role="assistant", content=None, reasoning_content="x" * 1000),
            Message(role="assistant", content="", reasoning_content="y"),
            Message(role="user", content="Continue."),
        ]
        create = mock.AsyncMock(return_value=object())
        client = mock.Mock()
        client.chat.completions.create = create
        with mock.patch.object(model, "get_client", return_value=client):
            asyncio.run(model.invoke(history))

        sent = create.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "tool", "user"])
        self.assertIsNone(sent[1]["content"])
        # The call is sent as stored, plus the placeholder signature a wire
        # that validates signatures (Gemini) needs on a call issued elsewhere.
        call = sent[1]["tool_calls"][0]
        self.assertEqual(
            {k: call[k] for k in ("id", "type", "function")},
            {"id": "t", "type": "function", "function": {"name": "execute", "arguments": "{}"}},
        )
        self.assertEqual(
            call["extra_content"], {"google": {"thought_signature": "skip_thought_signature_validator"}}
        )
        self.assertTrue(all("reasoning_content" not in m for m in sent))
        # History itself is untouched: persistence still sees the turn.
        self.assertEqual(len(history), 6)


if __name__ == "__main__":
    unittest.main()
