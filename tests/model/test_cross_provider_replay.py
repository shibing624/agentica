# -*- coding: utf-8 -*-
"""Replaying a transcript on a provider that did not write it.

SessionLog stores every provider's tool rounds in one shape: an assistant
message carrying ``tool_calls`` answered by ``role="tool"`` results. That is
also the shape OpenAI-wire models produce live. Claude's adapter converts it
to ``tool_use`` / ``tool_result`` blocks, so a resumed or switched session keeps
its tool rounds on every provider; ``to_portable_history`` rewrites Claude's
native block shape into the shared one on a model switch.
"""
import asyncio
import json
import unittest

from agentica.agent.history_filter import make_history_portable, to_portable_history
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
from agentica.model.openai.responses import OpenAIResponses
from agentica.run.response import RunResponse


def _replayed_history() -> list:
    """History as ``SessionLog.load()`` hands it back after a tool round."""
    return [
        Message(role="user", content="read config.py"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "config.py"}'},
                }
            ],
        ),
        Message(role="tool", tool_call_id="call_1", content="PORT = 8080"),
        Message(role="assistant", content="The port is 8080."),
        Message(role="user", content="what did I ask before?"),
    ]


def _claude_native_round() -> list:
    """A live Claude tool round, as the adapter leaves it in memory."""
    return [
        Message(role="user", content="read it"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "plan", "signature": "claude-sig"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
            tool_calls=[
                {
                    "id": "tu_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
            reasoning_content="plan",
            provider_data={"signature": "claude-sig"},
        ),
        Message(
            role="user",
            content=[{"type": "tool_result", "tool_use_id": "tu_1", "content": "PORT = 8080"}],
        ),
        Message(role="assistant", content="The port is 8080."),
    ]


class TestAnthropicFormatsReplayedHistory(unittest.TestCase):
    def setUp(self):
        self.model = Claude(api_key="fake_anthropic_key")

    def _format(self, messages):
        return asyncio.run(self.model.format_messages(messages))

    def test_no_empty_text_block_reaches_the_api(self):
        chat_messages, _ = self._format(_replayed_history())
        for message in chat_messages:
            for block in message["content"]:
                if block.get("type") == "text":
                    self.assertTrue(block["text"].strip(), f"empty text block in {message}")

    def test_cache_control_never_lands_on_an_empty_block(self):
        chat_messages, _ = self._format(_replayed_history())
        cached = [
            block
            for message in chat_messages
            for block in message["content"]
            if "cache_control" in block
        ]
        self.assertTrue(cached)
        for block in cached:
            self.assertNotEqual(block.get("text"), "")

    def test_the_persisted_tool_round_replays_as_blocks(self):
        chat_messages, _ = self._format(_replayed_history())
        roles = [m["role"] for m in chat_messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])
        assistant = chat_messages[1]["content"]
        self.assertEqual(
            [(b["type"], b.get("id"), b.get("name")) for b in assistant],
            [("tool_use", "call_1", "read_file")],
        )
        self.assertEqual(assistant[0]["input"], {"path": "config.py"})
        result = chat_messages[2]["content"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "call_1")
        self.assertEqual(result[0]["content"], "PORT = 8080")

    def test_tool_call_with_text_keeps_text_first(self):
        messages = [
            Message(role="user", content="check the port"),
            Message(
                role="assistant",
                content="Reading config.",
                tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                ],
            ),
            Message(role="tool", tool_call_id="c1", content="PORT = 8080"),
        ]
        chat_messages, _ = self._format(messages)
        self.assertEqual(
            [b["type"] for b in chat_messages[1]["content"]], ["text", "tool_use"]
        )

    def test_arguments_that_are_not_json_do_not_break_the_turn(self):
        """A truncated tool call already failed; its result says so."""
        messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": '{"pa'}}
                ],
            ),
            Message(role="tool", tool_call_id="c1", content="Error: interrupted", tool_call_error=True),
        ]
        chat_messages, _ = self._format(messages)
        self.assertEqual(chat_messages[1]["content"][0]["input"], {})
        self.assertTrue(chat_messages[2]["content"][0]["is_error"])

    def test_a_user_turn_with_only_images_is_kept(self):
        model = Claude(api_key="fake_anthropic_key", supports_images=True)

        async def _fake_image(image):
            return {"type": "image", "source": {"type": "base64", "data": image}}

        model.add_image = _fake_image
        messages = [Message(role="user", content="", images=["abc"])]
        chat_messages, _ = asyncio.run(model.format_messages(messages))
        self.assertEqual(len(chat_messages), 1)
        self.assertEqual(chat_messages[0]["content"][0]["type"], "image")

    def test_native_tool_round_is_left_alone(self):
        """A live Anthropic turn keeps its tool_use / tool_result blocks."""
        messages = [
            Message(role="user", content="read config.py"),
            Message(
                role="assistant",
                content=[{"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {}}],
                tool_calls=[{"id": "tu_1", "type": "function", "function": {"name": "read_file"}}],
            ),
            Message(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": "tu_1", "content": "PORT = 8080"}],
            ),
        ]
        chat_messages, _ = self._format(messages)
        self.assertEqual(len(chat_messages), 3)
        self.assertEqual(chat_messages[1]["content"][0]["type"], "tool_use")
        self.assertEqual(chat_messages[2]["content"][0]["type"], "tool_result")

    def test_native_round_plus_repaired_placeholder_makes_one_user_turn(self):
        """A partial native round padded by the shared repair code is one turn."""
        messages = [
            Message(role="user", content="two reads"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "a", "type": "function", "function": {"name": "read_file"}},
                    {"id": "b", "type": "function", "function": {"name": "read_file"}},
                ],
            ),
            Message(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": "a", "content": "A ok"}],
            ),
            Message(role="tool", tool_call_id="b", content="B ok"),
        ]
        chat_messages, _ = self._format(messages)
        self.assertEqual([m["role"] for m in chat_messages], ["user", "assistant", "user"])
        results = chat_messages[2]["content"]
        self.assertEqual([b["tool_use_id"] for b in results], ["a", "b"])
        self.assertEqual([b["content"] for b in results], ["A ok", "B ok"])

    def test_the_user_turn_after_a_converted_round_joins_it(self):
        chat_messages, _ = self._format(
            _replayed_history()[:3] + [Message(role="user", content="and now?")]
        )
        self.assertEqual([m["role"] for m in chat_messages], ["user", "assistant", "user"])
        self.assertEqual(
            [b["type"] for b in chat_messages[2]["content"]], ["tool_result", "text"]
        )

    def test_duplicate_tool_rows_for_an_answered_round_are_dropped(self):
        messages = _replayed_history()
        messages.insert(3, Message(role="tool", tool_call_id="call_1", content="PORT = 8080"))
        chat_messages, _ = self._format(messages)
        results = [b for m in chat_messages for b in m["content"] if b.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)


class TestReplayedToolHistoryCapability(unittest.TestCase):
    def test_no_provider_needs_a_replay_capability_flag(self):
        """Every adapter now replays the shared shape, so the flag is gone."""
        for model in (
            OpenAIChat(api_key="fake_openai_key"),
            OpenAIResponses(api_key="fake_openai_key"),
            Claude(api_key="fake_anthropic_key"),
        ):
            self.assertFalse(hasattr(model, "supports_replayed_tool_history"), type(model))


class TestModelSwitchPortableHistory(unittest.TestCase):
    """/model keeps tool rounds too; only what is bound to the old model goes."""

    def test_claude_native_round_crosses_to_openai_with_tools(self):
        out = to_portable_history(_claude_native_round())
        model = OpenAIChat(api_key="fake_openai_key")
        wires = [model.format_message(m) for m in out]
        self.assertEqual(
            [w["role"] for w in wires],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(wires[1]["content"], "checking")
        self.assertEqual(wires[1]["tool_calls"][0]["id"], "tu_1")
        self.assertEqual(wires[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(wires[1]["tool_calls"][0]["function"]["arguments"]), {"path": "a.py"})
        self.assertEqual(wires[2]["tool_call_id"], "tu_1")
        self.assertEqual(wires[2]["content"], "PORT = 8080")
        for wire in wires:
            self.assertNotIn("tool_use", str(wire.get("content")))
            self.assertNotIn("reasoning_content", wire)

    def test_openai_round_crosses_to_claude_with_tools(self):
        out = to_portable_history(_replayed_history())
        model = Claude(api_key="fake_anthropic_key")
        chat_messages, _ = asyncio.run(model.format_messages(out))
        types = [b["type"] for m in chat_messages for b in m["content"]]
        self.assertIn("tool_use", types)
        self.assertIn("tool_result", types)

    def test_provider_bound_state_does_not_travel(self):
        out = to_portable_history(_claude_native_round())
        for m in out:
            self.assertIsNone(m.reasoning_content)
            self.assertIsNone(m.provider_data)
            self.assertIsNone(m.provider_checkpoint)
            self.assertIsNone(m.thinking)

    def test_conversion_is_idempotent(self):
        once = to_portable_history(_replayed_history())
        twice = to_portable_history(once)
        self.assertEqual(
            [(m.role, m.content, m.tool_call_id) for m in twice],
            [(m.role, m.content, m.tool_call_id) for m in once],
        )

    def test_both_stores_of_a_working_memory_are_converted(self):
        history = _claude_native_round()
        memory = WorkingMemory()
        memory.add_messages(history)
        memory.add_run(AgentRun(response=RunResponse(messages=list(history))))
        make_history_portable(memory)
        for messages in (memory.messages, memory.runs[0].response.messages):
            self.assertFalse(
                any(isinstance(m.content, list) for m in messages), "block content left behind"
            )
            self.assertTrue(any(m.tool_calls for m in messages), "the tool round was lost")
        self.assertFalse(any(m.role == "system" for m in memory.runs[0].response.messages))


if __name__ == "__main__":
    unittest.main()
