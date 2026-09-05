# -*- coding: utf-8 -*-
"""Replaying a persisted transcript on a provider that did not write it.

SessionLog always stores tool rounds in the OpenAI wire shape (an assistant
message carrying ``tool_calls`` plus ``role="tool"`` results), whatever provider
produced them. Anthropic's /v1/messages cannot accept that shape, and the
tool-call assistant message is persisted with empty text — which used to reach
the API as ``{"type": "text", "text": ""}`` and 400 with
"cache_control cannot be set for empty text blocks".
"""
import asyncio
import unittest

from agentica.agent.history_filter import strip_all_tool_artifacts, strip_tool_artifacts_from_memory
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
from agentica.run_response import RunResponse


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

    def test_text_turns_survive_and_tool_round_does_not(self):
        chat_messages, _ = self._format(_replayed_history())
        texts = [
            block["text"]
            for message in chat_messages
            for block in message["content"]
            if block.get("type") == "text"
        ]
        self.assertEqual(texts, ["read config.py", "The port is 8080.", "what did I ask before?"])

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


class TestReplayedToolHistoryCapability(unittest.TestCase):
    def test_openai_replays_the_persisted_shape(self):
        self.assertTrue(OpenAIChat(api_key="fake_openai_key").supports_replayed_tool_history)

    def test_anthropic_does_not(self):
        self.assertFalse(Claude(api_key="fake_anthropic_key").supports_replayed_tool_history)

    def test_openai_chat_capability_is_the_wire_format_not_the_model_id(self):
        """A Claude id on OpenAIChat still speaks the OpenAI wire. Portable
        history is the /model sanitizer's job, not a name check."""
        self.assertTrue(
            OpenAIChat(id="claude-opus-5", api_key="fake_openai_key").supports_replayed_tool_history
        )


class TestModelSwitchPortableHistory(unittest.TestCase):
    """Either direction of /model keeps Q&A text and drops thinking + tools."""

    def test_claude_history_formats_on_openai_after_strip(self):
        history = [
            Message(role="user", content="read it"),
            Message(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "plan", "signature": "claude-sig"},
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {}},
                ],
                tool_calls=[{"id": "tu_1", "type": "function", "function": {"name": "read_file"}}],
                reasoning_content="plan",
            ),
            Message(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": "tu_1", "content": "PORT = 8080"}],
            ),
            Message(role="assistant", content="The port is 8080."),
        ]
        out = strip_all_tool_artifacts(history)
        model = OpenAIChat(api_key="fake_openai_key")
        wires = [model.format_message(m) for m in out]
        self.assertEqual(
            [(w["role"], w["content"]) for w in wires],
            [("user", "read it"), ("assistant", "checking"), ("assistant", "The port is 8080.")],
        )
        for wire in wires:
            self.assertNotIn("tool_calls", wire)
            self.assertNotIn("reasoning_content", wire)
            self.assertIsInstance(wire["content"], str)

    def test_openai_thinking_history_formats_on_claude_after_strip(self):
        history = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "gpt thoughts", "signature": "not-claude"},
                    {"type": "text", "text": "hello"},
                ],
                reasoning_content="gpt thoughts",
            ),
            Message(role="assistant", content="done", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "glob"}},
            ]),
            Message(role="tool", tool_call_id="c1", content="files"),
        ]
        out = strip_all_tool_artifacts(history)
        model = Claude(api_key="fake_anthropic_key")
        chat_messages, _ = asyncio.run(model.format_messages(out))
        types = [block.get("type") for msg in chat_messages for block in msg["content"]]
        self.assertNotIn("thinking", types)
        self.assertNotIn("tool_use", types)
        self.assertNotIn("tool_result", types)
        texts = [
            block["text"]
            for msg in chat_messages
            for block in msg["content"]
            if block.get("type") == "text"
        ]
        self.assertEqual(texts, ["hi", "hello", "done"])


class TestStripToolArtifactsFromMemory(unittest.TestCase):
    def _memory(self) -> WorkingMemory:
        history = _replayed_history()
        memory = WorkingMemory()
        memory.add_messages(history)
        memory.add_run(AgentRun(response=RunResponse(messages=list(history))))
        return memory

    def test_both_stores_keep_only_text_turns(self):
        memory = self._memory()
        strip_tool_artifacts_from_memory(memory)

        from agentica.agent.history_filter import ELIDED_TOOLS_MARK

        for messages in (memory.messages, memory.runs[0].response.messages):
            roles = [(m.role, m.content) for m in messages]
            self.assertEqual(
                roles[:3],
                [
                    ("user", "read config.py"),
                    ("assistant", "The port is 8080."),
                    ("user", "what did I ask before?"),
                ],
            )
            self.assertEqual(roles[-1][0], "assistant")
            self.assertIn(ELIDED_TOOLS_MARK, roles[-1][1])
            self.assertNotIn("PORT = 8080", roles[-1][1])


if __name__ == "__main__":
    unittest.main()
