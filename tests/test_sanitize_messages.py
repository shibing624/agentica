# -*- coding: utf-8 -*-
"""
Tests for Model.sanitize_messages() - ensures tool_call message sequences are valid.
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica.model.base import Model
from agentica.model.message import Message


class TestSanitizeMessages(unittest.TestCase):
    """Test Model.sanitize_messages fixes broken tool_call sequences."""

    def test_valid_messages_unchanged(self):
        """Normal conversation messages should not be modified."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = Model.sanitize_messages(messages)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].role, "system")
        self.assertEqual(result[1].role, "user")
        self.assertEqual(result[2].role, "assistant")

    def test_valid_tool_call_sequence_unchanged(self):
        """A complete tool_call + tool response sequence should not be modified."""
        messages = [
            Message(role="user", content="What's the weather?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}}
                ],
            ),
            Message(role="tool", tool_call_id="call_1", content='{"temp": 72}'),
            Message(role="assistant", content="It's 72°F in NYC."),
        ]
        result = Model.sanitize_messages(messages)
        self.assertEqual(len(result), 4)

    def test_missing_single_tool_response_fixed(self):
        """A missing tool response should be filled with a placeholder."""
        messages = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "get_skill_info", "arguments": '{"name":"test"}'}}
                ],
            ),
            Message(role="user", content="Continue"),
        ]
        result = Model.sanitize_messages(messages)
        self.assertEqual(len(result), 4)
        # The inserted placeholder should be at index 2
        placeholder = result[2]
        self.assertEqual(placeholder.role, "tool")
        self.assertEqual(placeholder.tool_call_id, "call_1")
        self.assertIn("get_skill_info", placeholder.content)
        self.assertIn("Error", placeholder.content)

    def test_missing_multiple_tool_responses_fixed(self):
        """Multiple missing tool responses should all be filled."""
        messages = [
            Message(role="user", content="Do stuff"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            ),
            Message(role="assistant", content="Done"),
        ]
        result = Model.sanitize_messages(messages)
        # Should have 2 placeholder tool messages inserted
        self.assertEqual(len(result), 5)
        tool_msgs = [m for m in result if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        tool_call_ids = {m.tool_call_id for m in tool_msgs}
        self.assertEqual(tool_call_ids, {"call_1", "call_2"})

    def test_partial_tool_responses_fixed(self):
        """If only some tool responses exist, fill the missing ones."""
        messages = [
            Message(role="user", content="Do stuff"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": "tool_c", "arguments": "{}"}},
                ],
            ),
            Message(role="tool", tool_call_id="call_1", content="result_a"),
            Message(role="tool", tool_call_id="call_3", content="result_c"),
            Message(role="assistant", content="All done"),
        ]
        result = Model.sanitize_messages(messages)
        # call_2 is missing, should be inserted
        tool_msgs = [m for m in result if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 3)
        tool_call_ids = {m.tool_call_id for m in tool_msgs}
        self.assertEqual(tool_call_ids, {"call_1", "call_2", "call_3"})

    def test_multiple_tool_call_rounds(self):
        """Multiple rounds of tool calls should each be validated."""
        messages = [
            Message(role="user", content="Step 1"),
            # Round 1: complete
            Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}}],
            ),
            Message(role="tool", tool_call_id="call_1", content="ok"),
            # Round 2: missing
            Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}}],
            ),
            Message(role="user", content="Continue"),
        ]
        result = Model.sanitize_messages(messages)
        # call_2 should have a placeholder inserted
        self.assertEqual(len(result), 6)
        tool_msgs = [m for m in result if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 2)

    def test_no_tool_calls_no_change(self):
        """Assistant messages without tool_calls should not be affected."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="Bye"),
            Message(role="assistant", content="Goodbye!"),
        ]
        result = Model.sanitize_messages(messages)
        self.assertEqual(len(result), 4)

    def test_empty_messages(self):
        """Empty message list should work without error."""
        result = Model.sanitize_messages([])
        self.assertEqual(len(result), 0)

    def test_tool_call_at_end_of_messages(self):
        """Tool call at the very end with no response should be fixed."""
        messages = [
            Message(role="user", content="Do something"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"test"}'}}
                ],
            ),
        ]
        result = Model.sanitize_messages(messages)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[2].role, "tool")
        self.assertEqual(result[2].tool_call_id, "call_1")

    def test_inplace_modification(self):
        """sanitize_messages should modify the list in-place and return it."""
        messages = [
            Message(role="user", content="Test"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
            ),
        ]
        result = Model.sanitize_messages(messages)
        self.assertIs(result, messages)
        self.assertEqual(len(messages), 3)


if __name__ == "__main__":
    unittest.main()
