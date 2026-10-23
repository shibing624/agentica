# -*- coding: utf-8 -*-
"""A reply that never fits in max_tokens is reported, not returned as complete.

Measured on real providers (opus-5-5 / glm-5.3 / deepseek-v4.1-flash,
max_tokens=1500, 15 long-answer prompts): 5 of 15 runs ended on a hop with no
visible text, so ``content`` came back "" with ``is_complete=True``, once after
an earlier hop had written 5056 characters.

Hops are not joined into one answer: a plain join showed a seam (verbatim
re-emitted tail, or "(continued)" meta text) at 6 of the 11 continuation
boundaries collected.
"""
import asyncio
import unittest
from unittest import mock

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from agentica import Agent
from agentica.model.openai import OpenAIChat
from agentica.run.response import RunBreakReason

CEILING = "agentica.model.openai.chat.get_model_max_output_tokens"


def _completion(text, finish, tool_calls=None):
    return ChatCompletion(
        id="c", object="chat.completion", created=0, model="m",
        choices=[Choice(index=0, finish_reason=finish,
                        message=ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=2000, total_tokens=2010),
    )


async def _stream(text, finish):
    chunks = []
    if text:
        chunks.append({"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None})
    chunks.append({"index": 0, "delta": {}, "finish_reason": finish})
    for choice in chunks:
        yield ChatCompletionChunk.model_validate(
            {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}
        )
    yield ChatCompletionChunk.model_validate(
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2000, "total_tokens": 2010}}
    )


def _run(replies, stream):
    """Run a real Agent whose provider returns ``replies`` = [(text, finish[, tool_calls]), ...]."""
    with mock.patch(CEILING, return_value=None):
        model = OpenAIChat(id="m", api_key="k", max_tokens=2000)
    script = iter(replies)

    async def create(**kwargs):
        text, finish, *tool_calls = next(script)
        if kwargs.get("stream"):
            return _stream(text, finish)
        return _completion(text, finish, tool_calls[0] if tool_calls else None)

    client = mock.Mock()
    client.chat.completions.create = create
    agent = Agent(model=model)

    async def go():
        with mock.patch.object(OpenAIChat, "get_client", return_value=client):
            if stream:
                async for _ in agent.run_stream("q"):
                    pass
                return agent.run_response
            return await agent.run("q")

    return asyncio.run(go())


class TestOutputTruncated(unittest.TestCase):
    def test_finished_after_continuation_is_complete(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                r = _run([("part one ", "length"), ("part two.", "stop")], stream)
                self.assertTrue(r.is_complete)
                self.assertIsNone(r.break_reason)
                self.assertEqual(r.content, "part two.")

    def test_still_cut_off_after_all_recoveries_is_reported(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                r = _run([("a", "length"), ("b", "length"), ("c", "length"), ("d", "length")], stream)
                self.assertFalse(r.is_complete)
                self.assertEqual(r.break_reason, RunBreakReason.OUTPUT_TRUNCATED.value)
                self.assertIn("max_tokens", r.break_message)
                self.assertEqual(r.content, "d")

    def test_empty_final_hop_returns_the_last_text_written(self):
        """Real shape: deepseek-v4.1-flash code_py, segments [0, 0, 5056, 0] chars."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                r = _run([("", "length"), ("", "length"), ("WRITTEN", "length"), ("", "length")], stream)
                self.assertEqual(r.break_reason, RunBreakReason.OUTPUT_TRUNCATED.value)
                self.assertEqual(r.content, "WRITTEN")

    def test_segments_are_not_concatenated(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                r = _run([("first half ", "length"), ("restated first half, then more", "length"),
                          ("x", "length"), ("y", "length")], stream)
                self.assertEqual(r.content, "y")

    def test_text_before_a_tool_round_is_not_the_answer(self):
        """An aside written next to a tool call belongs to that round, not to the reply."""
        call = [{"id": "t1", "type": "function",
                 "function": {"name": "search_session", "arguments": '{"query": "x"}'}}]
        r = _run([("let me look that up", "tool_calls", call),
                  ("", "length"), ("", "length"), ("", "length"), ("", "length")], stream=False)
        self.assertEqual(r.break_reason, RunBreakReason.OUTPUT_TRUNCATED.value)
        self.assertEqual(r.content, "")

    def test_chain_takes_the_last_segment_not_a_concatenation(self):
        """A restating hop must not be glued onto the hop it restates.

        Boundary 8 of the collected fixtures is exactly this: deepseek-v4.1-flash
        list_40 hop 2 re-emits item 31 in new words after hop 0 ended mid-item-31.
        Concatenating the chain would print item 31 twice.
        """
        from agentica.model.message import Message
        from agentica.runner.loop import LoopMixin

        hop0 = Message(role="assistant", content="...end of item 30.\n\n**31. Look at saturation first.**\nConnection")
        hop2 = Message(role="assistant", content="**31. Look at saturation first.**\nCheck CPU and memory first.\n")
        hop3 = Message(role="assistant", content=None)
        messages = [Message(role="user", content="q"), hop0, hop2, hop3]
        picked = LoopMixin._last_visible_continuation(messages)
        self.assertEqual(picked, hop2.content)
        self.assertNotIn("end of item 30", picked)
        self.assertEqual(picked.count("31. Look at saturation"), 1)


if __name__ == "__main__":
    unittest.main()
