# -*- coding: utf-8 -*-
"""A content-less 200 reply is retried, not taken as the end of the turn.

Seen from an LLM proxy: 5 of 494 requests in one session came back HTTP 200
with no text, no reasoning, no tool call and no exception. No retry layer
fired, the loop treated it as a deliberate stop, and a multi-hour task ended
mid-way (twice right after a ``wait`` on a finished training job returned).
"""
import asyncio
import unittest
from unittest import mock

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from agentica import Agent
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ContentFilterError
from agentica.run.response import RunBreakReason
from agentica.runner import Runner

CEILING = "agentica.model.openai.chat.get_model_max_output_tokens"
CALL = [{"id": "t1", "type": "function", "function": {"name": "search_session", "arguments": '{"query": "x"}'}}]
EMPTY = (None, "stop")


def _completion(text, finish, tool_calls=None):
    return ChatCompletion(
        id="c", object="chat.completion", created=0, model="m",
        choices=[Choice(index=0, finish_reason=finish,
                        message=ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=0 if not text else 5, total_tokens=15),
    )


async def _stream(text, finish, tool_calls=None):
    chunks = []
    if text:
        chunks.append({"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None})
    if tool_calls:
        delta_calls = [{"index": i, **c} for i, c in enumerate(tool_calls)]
        chunks.append({"index": 0, "delta": {"role": "assistant", "tool_calls": delta_calls}, "finish_reason": None})
    chunks.append({"index": 0, "delta": {}, "finish_reason": finish})
    for choice in chunks:
        yield ChatCompletionChunk.model_validate(
            {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [choice]}
        )
    # The failing proxy still sent a usage block (input reported as 0).
    yield ChatCompletionChunk.model_validate(
        {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    )


def _run(replies, stream, sent=None, *, fallback=False, used_models=None):
    """Run a real Agent against a scripted provider; returns (run_response, sent requests)."""
    with mock.patch(CEILING, return_value=None):
        model = OpenAIChat(id="m", api_key="k", max_tokens=2000)
    script = iter(replies)
    sent = [] if sent is None else sent

    async def create(**kwargs):
        sent.append(kwargs["messages"])
        if used_models is not None:
            used_models.append(kwargs["model"])
        text, finish, *tool_calls = next(script)
        tool_calls = tool_calls[0] if tool_calls else None
        if kwargs.get("stream"):
            return _stream(text, finish, tool_calls)
        return _completion(text, finish, tool_calls)

    client = mock.Mock()
    client.chat.completions.create = create
    fallback_models = (
        [OpenAIChat(id="fallback", api_key="k", max_tokens=2000)]
        if fallback
        else []
    )
    agent = Agent(model=model, fallback_models=fallback_models)

    async def go():
        with mock.patch.object(OpenAIChat, "get_client", return_value=client):
            if stream:
                async for _ in agent.run_stream("q"):
                    pass
                return agent.run_response
            return await agent.run("q")

    return asyncio.run(go()), sent


class TestEmptyResponseEndToEnd(unittest.TestCase):
    def test_content_filter_switches_to_fallback_without_retrying_primary(self):
        for stream in (False, True):
            for filtered_text in (None, "filtered partial"):
                with self.subTest(stream=stream, filtered_text=filtered_text):
                    sent = []
                    used_models = []
                    response, _ = _run(
                        [(filtered_text, "content_filter"), ("fallback answer", "stop")],
                        stream,
                        sent,
                        fallback=True,
                        used_models=used_models,
                    )
                    self.assertEqual(used_models, ["m", "fallback"])
                    self.assertEqual(len(sent), 2)
                    self.assertEqual(response.content, "fallback answer")
                    self.assertTrue(response.fallback_used)

    def test_content_filter_is_reported_without_retry(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                sent = []
                with self.assertRaises(ContentFilterError):
                    _run([(None, "content_filter"), ("must not be requested", "stop")], stream, sent)
                self.assertEqual(len(sent), 1)

    def test_all_filtered_candidates_are_tried_once(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                sent = []
                used_models = []
                with self.assertRaises(ContentFilterError):
                    _run(
                        [(None, "content_filter"), (None, "content_filter")],
                        stream,
                        sent,
                        fallback=True,
                        used_models=used_models,
                    )
                self.assertEqual(used_models, ["m", "fallback"])
                self.assertEqual(len(sent), 2)

    def test_partial_content_filter_is_not_accepted_as_complete(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                sent = []
                with self.assertRaises(ContentFilterError):
                    _run([("filtered partial", "content_filter"), ("must not be requested", "stop")], stream, sent)
                self.assertEqual(len(sent), 1)

    def test_empty_reply_is_retried_with_the_same_request(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                r, sent = _run([EMPTY, ("done.", "stop")], stream)
                self.assertEqual(len(sent), 2)
                self.assertTrue(r.is_complete)
                self.assertEqual(r.content, "done.")
                # The empty turn is not replayed: the retry is the failed request.
                self.assertEqual(sent[1], sent[0])

    def test_empty_reply_after_a_tool_round_does_not_end_the_task(self):
        """The field shape: a tool result comes back, the next call is empty."""
        for stream in (False, True):
            with self.subTest(stream=stream):
                r, sent = _run([(None, "tool_calls", CALL), EMPTY, ("done.", "stop")], stream)
                self.assertEqual(len(sent), 3)
                self.assertEqual(sent[2], sent[1])
                self.assertTrue(r.is_complete)
                self.assertEqual(r.content, "done.")

    def test_empty_every_time_is_reported_after_the_retry_limit(self):
        limit = LoopState().empty_response_retry_limit
        for stream in (False, True):
            with self.subTest(stream=stream):
                r, sent = _run([EMPTY] * (limit + 1) + [("never reached", "stop")], stream)
                self.assertEqual(len(sent), limit + 1)
                self.assertFalse(r.is_complete)
                self.assertEqual(r.break_reason, RunBreakReason.EMPTY_RESPONSE.value)
                self.assertEqual(r.break_message, f"No reply {limit + 1} times. Type continue or 继续.")

    def test_limit_counts_consecutive_empties_only(self):
        """A long run survives isolated empties spread across many calls."""
        limit = LoopState().empty_response_retry_limit
        replies = [EMPTY] * limit + [(None, "tool_calls", CALL)] + [EMPTY] * limit + [("done.", "stop")]
        r, sent = _run(replies, stream=False)
        self.assertEqual(len(sent), len(replies))
        self.assertTrue(r.is_complete)
        self.assertEqual(r.content, "done.")

    def test_any_visible_reply_ends_the_turn_in_one_request(self):
        """The costly side: a too-loose emptiness check retries every final answer."""
        for text in ("a", "  x  ", "."):
            for stream in (False, True):
                with self.subTest(text=text, stream=stream):
                    r, sent = _run([(text, "stop"), ("must not be requested", "stop")], stream)
                    self.assertEqual(len(sent), 1)
                    self.assertTrue(r.is_complete)

    def test_whitespace_only_reply_counts_as_empty(self):
        """The chat wire drops it too, so it carries nothing to the user or the model."""
        r, sent = _run([(" \n", "stop"), ("done.", "stop")], stream=False)
        self.assertEqual(len(sent), 2)
        self.assertEqual(r.content, "done.")


class TestEmptyReplyPredicate(unittest.TestCase):
    def _post(self, reply, finish="stop", state=None):
        model = OpenAIChat(id="gpt-4o-mini", api_key="k")
        model.last_finish_reason = finish
        state = state or LoopState()
        return Runner._loop_post_response([Message(role="user", content="q"), reply], model, state, False), state

    def test_reasoning_or_thinking_only_is_a_reply(self):
        for reply in (
            Message(role="assistant", reasoning_content="thought it through"),
            Message(role="assistant", thinking="thought it through"),
            Message(role="assistant", redacted_thinking="opaque"),
        ):
            with self.subTest(reply=reply):
                cont, state = self._post(reply)
                self.assertFalse(cont)
                self.assertEqual(state.empty_response_retry_count, 0)
                self.assertFalse(state.empty_response)

    def test_length_finish_keeps_its_own_recovery(self):
        cont, state = self._post(Message(role="assistant", content=None), finish="length")
        self.assertTrue(cont)
        self.assertEqual(state.max_tokens_recovery_count, 1)
        self.assertEqual(state.empty_response_retry_count, 0)

    def test_no_assistant_message_is_not_retried(self):
        model = OpenAIChat(id="gpt-4o-mini", api_key="k")
        model.last_finish_reason = "stop"
        self.assertFalse(Runner._loop_post_response([], model, LoopState(), had_tool_calls=False))


if __name__ == "__main__":
    unittest.main()
