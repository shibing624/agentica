# -*- coding: utf-8 -*-
"""Regression tests for OpenAIChat.build_tool_calls streaming reassembly.

Standard OpenAI streams number tool_call deltas with a stable ``index`` so
fragments can be reassembled positionally. OpenAI-compatible proxies that
translate Anthropic Claude tool_use into the OpenAI wire format
may omit ``index`` (send ``None``) and rely on ``id`` to delimit calls. The
original positional-only logic raised ``TypeError`` on a ``None`` index,
dropping every tool call and silently degrading the turn to plain text — the
root of the "agent stalls mid-turn" symptom.

These tests lock down the defensive reassembly (positional path + id-based
fallback) and fail against the old ``build_tool_calls``.
"""

import unittest

from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall as TC,
    ChoiceDeltaToolCallFunction as FN,
)

from agentica.model.openai.chat import OpenAIChat
from agentica.model.metrics import Metrics


def _mk(index=None, id=None, type=None, name=None, args=None):
    """Build a delta tool-call fragment, bypassing pydantic validation so we
    can simulate loose proxy output (e.g. index=None)."""
    return TC.model_construct(
        index=index,
        id=id,
        type=type,
        function=FN.model_construct(name=name, arguments=args),
    )


class TestBuildToolCalls(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIChat(id="claude-opus-4-8", api_key="fake")

    def test_index_none_does_not_crash(self):
        """Regression: index=None used to raise TypeError and drop the call."""
        out = self.model.build_tool_calls([_mk(id="toolu_1", type="function", name="add", args='{"a":21}')])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "toolu_1")
        self.assertEqual(out[0]["function"]["name"], "add")
        self.assertEqual(out[0]["function"]["arguments"], '{"a":21}')

    def test_standard_positional_split(self):
        """Standard OpenAI path: name in first chunk, args in the next (same index)."""
        out = self.model.build_tool_calls([_mk(0, "t1", "function", "add", ""), _mk(0, None, None, None, '{"a":21}')])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["arguments"], '{"a":21}')

    def test_indexless_two_calls_delimited_by_id(self):
        out = self.model.build_tool_calls(
            [
                _mk(id="t1", type="function", name="add", args='{"a":1}'),
                _mk(id="t2", type="function", name="mul", args='{"b":2}'),
            ]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["id"], "t1")
        self.assertEqual(out[1]["id"], "t2")
        self.assertEqual(out[1]["function"]["name"], "mul")

    def test_indexless_idless_continuation(self):
        """id on first fragment, then id-less argument fragments accumulate."""
        out = self.model.build_tool_calls(
            [
                _mk(id="t1", type="function", name="add", args=""),
                _mk(name=None, args='{"a":'),
                _mk(name=None, args="21}"),
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["arguments"], '{"a":21}')

    def test_sparse_positional_indices_drop_empty_slots(self):
        """index 0 and 2 present, 1 absent — no empty placeholder leaks out."""
        out = self.model.build_tool_calls([_mk(0, "t0", "function", "a", "{}"), _mk(2, "t2", "function", "c", "{}")])
        self.assertEqual(len(out), 2)
        self.assertEqual([tc["id"] for tc in out], ["t0", "t2"])

    def test_empty_input(self):
        self.assertEqual(self.model.build_tool_calls([]), [])

    def test_index_zero_is_positional_not_fallback(self):
        """index=0 is a valid int and must use the positional path."""
        out = self.model.build_tool_calls([_mk(0, "t1", "function", "add", '{"a":1}')])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "t1")


class TestGeminiToolCallsSurviveParsing(unittest.TestCase):
    """Gemini (Venus chat wire) tool calls, as measured on gemini-3.7-flash.

    Main before this fix: a two-step tool task streamed 400 "Function call is
    missing a thought_signature" on the second request (the signature was
    dropped while assembling deltas), and non-streamed it never ran a tool
    ("Could not find function to call": ``type`` came back null).
    """

    SIG = "AY89a18-real-signature"

    def setUp(self):
        self.model = OpenAIChat(id="gemini-3.7-flash", api_key="fake")

    def test_streamed_signature_is_kept_on_the_call(self):
        delta = TC.model_validate({
            "index": 0, "id": "g1", "type": None,
            "function": {"name": "lookup", "arguments": '{"s":"a"}'},
            "thought_signature": self.SIG,
            "extra_content": {"google": {"thought_signature": self.SIG}},
        })
        call = self.model.build_tool_calls([delta])[0]
        self.assertEqual(call["extra_content"], {"google": {"thought_signature": self.SIG}})
        self.assertEqual(call["type"], "function")

    def test_the_real_signature_goes_back_on_the_wire_not_the_placeholder(self):
        from agentica.model.message import Message

        delta = TC.model_validate({
            "index": 0, "id": "g1", "type": None,
            "function": {"name": "lookup", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": self.SIG}},
        })
        call = self.model.build_tool_calls([delta])[0]
        wire = self.model.format_messages([
            Message(role="user", content="q"),
            Message(role="assistant", content="", tool_calls=[call]),
            Message(role="tool", tool_call_id="g1", content="r"),
        ])
        self.assertEqual(wire[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"], self.SIG)

    def test_non_streamed_null_type_is_stored_as_a_function_call(self):
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

        from agentica.tools.base import Function, get_function_call_for_tool_call

        gemini_call = ChatCompletionMessageToolCall.model_construct(
            id="g1", type=None, function=FN.model_construct(name="lookup", arguments='{"s":"a"}')
        )
        message = self.model.create_assistant_message(
            ChatCompletionMessage.model_construct(role="assistant", content=None, tool_calls=[gemini_call]),
            Metrics(),
            None,
        )
        call = message.tool_calls[0]
        self.assertEqual(call["type"], "function")
        resolved = get_function_call_for_tool_call(call, {"lookup": Function(name="lookup")})
        self.assertIsNotNone(resolved, "the runner dispatches on type == 'function'")


if __name__ == "__main__":
    unittest.main()
