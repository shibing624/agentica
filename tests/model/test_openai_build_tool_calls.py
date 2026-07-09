# -*- coding: utf-8 -*-
"""Regression tests for OpenAIChat.build_tool_calls streaming reassembly.

Standard OpenAI streams number tool_call deltas with a stable ``index`` so
fragments can be reassembled positionally. OpenAI-compatible proxies that
translate Anthropic Claude tool_use into the OpenAI wire format (e.g. Venus)
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


if __name__ == "__main__":
    unittest.main()
