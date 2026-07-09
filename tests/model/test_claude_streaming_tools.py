# -*- coding: utf-8 -*-
"""Regression tests for Claude streaming tool-call parsing + message formatting.

These lock down two bugs that made Claude appear to "stop mid-turn" (no tool
ever fired) when served through the native /v1/messages endpoint:

1. The high-level anthropic ``messages.stream()`` helper yields *parsed* event
   subtypes (``ParsedContentBlockStopEvent`` / ``ParsedMessageStopEvent``) that
   do NOT inherit from the base ``ContentBlockStopEvent`` / ``MessageStopEvent``.
   The old ``isinstance(delta, ContentBlockStopEvent)`` check silently dropped
   every tool_use block (text still streamed, tools never ran).

2. ``format_messages`` passed OpenAI-style ``role="tool"`` messages straight
   through, producing an invalid Anthropic role (only "user"/"assistant" allowed)
   and — since this class emits its own ``role="user"`` tool_result — a
   duplicate ``tool_result`` for the same id.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message


class _AsyncStreamFromEvents:
    """Minimal async-iterator wrapping a fixed list of stream events."""

    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _StreamMgr:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return _AsyncStreamFromEvents(self._events)

    async def __aexit__(self, *_a):
        return None


class TestClaudeStreamingToolCalls(unittest.TestCase):
    """The streaming parser must collect tool_use from the SDK's Parsed* events."""

    def _run_stream(self, model, events):
        def _stream(**_kwargs):
            return _StreamMgr(events)

        mock_client = MagicMock()
        mock_client.messages.stream = _stream
        model.client = mock_client

        async def _fake_format(_msgs):
            return ([{"role": "user", "content": "hi"}], "sys")

        model.format_messages = _fake_format  # type: ignore[assignment]

        collected_messages = []

        async def _drain():
            msgs = collected_messages
            async for _c in model.response_stream(msgs):
                pass
            return msgs

        return asyncio.run(_drain())

    def test_parsed_content_block_stop_captures_tool_use(self):
        """A ParsedContentBlockStopEvent carrying a ToolUseBlock must yield a tool call."""
        from anthropic.types import ToolUseBlock
        from anthropic.lib.streaming._types import (
            ParsedContentBlockStopEvent,
            ParsedMessageStopEvent,
        )

        tool_block = ToolUseBlock(type="tool_use", id="toolu_1", name="add", input={"a": 21, "b": 21})
        # model_construct bypasses strict pydantic validation so we can build a
        # lightweight stand-in for the SDK's parsed events without a full
        # ParsedMessage graph. The parser only reads .type/.content_block/.message.
        stop_event = ParsedContentBlockStopEvent.model_construct(
            type="content_block_stop", index=0, content_block=tool_block
        )
        msg = MagicMock()
        msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        msg.stop_reason = "tool_use"
        msg_stop = ParsedMessageStopEvent.model_construct(type="message_stop", message=msg)

        model = Claude(id="claude-opus-4-6", api_key="fake")
        model.run_tools = True
        # Register a matching function so handle_stream_tool_calls doesn't error.
        model.functions = {}

        messages = self._run_stream(model, [stop_event, msg_stop])

        # The assistant message the stream appended must carry the tool call.
        assistant = next(m for m in messages if m.role == "assistant")
        self.assertIsNotNone(assistant.tool_calls)
        self.assertEqual(len(assistant.tool_calls), 1)
        self.assertEqual(assistant.tool_calls[0]["function"]["name"], "add")
        self.assertEqual(assistant.tool_calls[0]["id"], "toolu_1")

    def test_parsed_message_stop_sets_finish_reason(self):
        """ParsedMessageStopEvent must set last_finish_reason (max_tokens -> length)."""
        from anthropic.lib.streaming._types import ParsedMessageStopEvent

        msg = MagicMock()
        msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        msg.stop_reason = "max_tokens"
        msg_stop = ParsedMessageStopEvent.model_construct(type="message_stop", message=msg)

        model = Claude(id="claude-opus-4-6", api_key="fake")
        self._run_stream(model, [msg_stop])

        self.assertEqual(model.last_finish_reason, "length")


class TestClaudeFormatMessagesToolRole(unittest.TestCase):
    """role='tool' messages must be dropped (this class emits its own user tool_result)."""

    def test_tool_role_message_is_skipped(self):
        model = Claude(id="claude-opus-4-6", api_key="fake")
        model.enable_cache_control = False

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Compute 21+21 with add tool"),
            Message(
                role="assistant",
                content=[
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 21, "b": 21}},
                ],
                tool_calls=[{"type": "function", "function": {"name": "add", "arguments": "{}"}, "id": "t1"}],
            ),
            # OpenAI-style leftover tool message — must NOT reach the API.
            Message(role="tool", tool_call_id="t1", content="42"),
            # The Anthropic-formatted tool result (role="user") — this one stays.
            Message(role="user", content=[{"type": "tool_result", "tool_use_id": "t1", "content": "42"}]),
        ]

        chat_messages, system = asyncio.run(model.format_messages(messages))

        # No message may carry an invalid role.
        roles = [m["role"] for m in chat_messages]
        for r in roles:
            self.assertIn(r, ("user", "assistant"), f"invalid role {r!r} sent to Anthropic")

        # Exactly one tool_result block for id t1 (no duplicate).
        tool_result_ids = [
            block.get("tool_use_id")
            for m in chat_messages
            if isinstance(m["content"], list)
            for block in m["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertEqual(tool_result_ids.count("t1"), 1)
        self.assertEqual(system, "You are helpful.")

    def test_assistant_at_index_one_not_misrouted_to_system(self):
        """An assistant message at idx 1 must stay a chat message, not become system."""
        model = Claude(id="claude-opus-4-6", api_key="fake")
        model.enable_cache_control = False

        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
            Message(role="user", content="continue"),
        ]

        chat_messages, system = asyncio.run(model.format_messages(messages))

        roles = [m["role"] for m in chat_messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(system, "")


class TestClaudeCacheControlBudget(unittest.TestCase):
    """cache_control breakpoints must never exceed Anthropic's max of 4.

    Regression: format_messages added a breakpoint to the last 3 messages
    every turn. Because ``messages`` is reused across turns (and tool_result
    blocks may already carry cache_control), breakpoints accumulated and the
    API rejected the request with 'A maximum of 4 blocks with cache_control
    may be provided. Found 5.' The fix strips existing breakpoints and caps
    message breakpoints at 3 (system takes the 4th).
    """

    @staticmethod
    def _count_msg_cache_control(chat_messages):
        n = 0
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        n += 1
        return n

    def test_message_breakpoints_capped_at_three(self):
        model = Claude(id="claude-opus-4-8", api_key="fake")
        model.enable_cache_control = True
        messages = [Message(role="user", content=f"m{i}") for i in range(8)]
        chat_messages, _ = asyncio.run(model.format_messages(messages))
        # 3 on messages + 1 on system (added in prepare_request_kwargs) = 4.
        self.assertLessEqual(self._count_msg_cache_control(chat_messages), 3)

    def test_reused_message_list_does_not_accumulate(self):
        """Formatting the same growing list across turns stays within budget."""
        model = Claude(id="claude-opus-4-8", api_key="fake")
        model.enable_cache_control = True
        messages = [Message(role="user", content="q1"), Message(role="assistant", content="a1")]

        for turn in range(4):
            messages.append(Message(role="user", content=f"q{turn + 2}"))
            messages.append(Message(role="assistant", content=f"a{turn + 2}"))
            chat_messages, _ = asyncio.run(model.format_messages(messages))
            self.assertLessEqual(
                self._count_msg_cache_control(chat_messages),
                3,
                f"turn {turn}: message breakpoints exceeded 3",
            )

    def test_preexisting_breakpoints_are_stripped(self):
        """A message already carrying cache_control must not stack a 2nd block."""
        model = Claude(id="claude-opus-4-8", api_key="fake")
        model.enable_cache_control = True
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "r",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "more", "cache_control": {"type": "ephemeral"}},
                ],
            ),
        ]
        chat_messages, _ = asyncio.run(model.format_messages(messages))
        self.assertLessEqual(self._count_msg_cache_control(chat_messages), 3)

    def test_disabled_cache_control_adds_nothing(self):
        model = Claude(id="claude-opus-4-8", api_key="fake")
        model.enable_cache_control = False
        messages = [Message(role="user", content=f"m{i}") for i in range(5)]
        chat_messages, _ = asyncio.run(model.format_messages(messages))
        self.assertEqual(self._count_msg_cache_control(chat_messages), 0)


if __name__ == "__main__":
    unittest.main()
