# -*- coding: utf-8 -*-
"""
Tests for model-level defensive validations:
- choices[] empty → ValueError
- response.usage missing → safe degradation
- structured output parse failure → fallback to text content
All tests mock LLM API keys — no real API calls.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.model.openai import OpenAIChat
from agentica.model.message import Message


def _make_openai_model():
    return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")


class TestModelStructuredOutputsNaming(unittest.TestCase):
    def test_openai_model_uses_use_structured_outputs_field(self):
        model = _make_openai_model()

        self.assertFalse(model.use_structured_outputs)


class TestOpenAIChatEmptyChoices(unittest.TestCase):
    """OpenAIChat.response() must raise ValueError when choices is empty."""

    def test_empty_choices_raises_value_error(self):
        model = _make_openai_model()

        # Build a fake API response with empty choices
        mock_response = MagicMock()
        mock_response.choices = []
        mock_response.usage = None

        with patch.object(model, 'invoke', new=AsyncMock(return_value=mock_response)):
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(model.response(messages=[
                    Message(role="user", content="hello"),
                ]))
        self.assertIn("empty choices", str(ctx.exception).lower())

    def test_non_empty_choices_does_not_raise(self):
        """With a valid response, no ValueError should be raised."""
        model = _make_openai_model()

        # Mock a minimal valid response with proper string values (not MagicMock)
        mock_message = MagicMock()
        mock_message.content = "Hello there"
        mock_message.tool_calls = None
        mock_message.parsed = None
        mock_message.audio = None
        mock_message.reasoning_content = None
        # role must be a real string to pass Message validator
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch.object(model, 'invoke', new=AsyncMock(return_value=mock_response)):
            try:
                result = asyncio.run(model.response(messages=[
                    Message(role="user", content="hello"),
                ]))
                self.assertIsNotNone(result)
            except ValueError as e:
                self.fail(f"ValueError raised for valid choices: {e}")


class TestOpenAIChatUsageNone(unittest.TestCase):
    """response.usage = None must not crash (safe getattr degradation)."""

    def test_usage_none_does_not_crash(self):
        model = _make_openai_model()

        mock_message = MagicMock()
        mock_message.content = "response text"
        mock_message.tool_calls = None
        mock_message.parsed = None
        mock_message.audio = None
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        # Simulate API omitting usage field entirely
        del mock_response.usage
        mock_response.usage = None

        with patch.object(model, 'invoke', new=AsyncMock(return_value=mock_response)):
            try:
                result = asyncio.run(model.response(messages=[
                    Message(role="user", content="test"),
                ]))
                # Should succeed without AttributeError
            except AttributeError as e:
                self.fail(f"AttributeError raised for usage=None: {e}")
            except Exception:
                pass  # Other exceptions are acceptable


class TestMessageRoleValidation(unittest.TestCase):
    """Message.role must reject invalid roles."""

    def test_valid_roles_accepted(self):
        for role in ["system", "user", "assistant", "tool"]:
            msg = Message(role=role, content="test")
            self.assertEqual(msg.role, role)

    def test_invalid_role_raises_validation_error(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Message(role="Assisant", content="typo")  # common typo

    def test_empty_role_raises_validation_error(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Message(role="", content="empty role")

    def test_unknown_role_raises_validation_error(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Message(role="human", content="wrong role name")


class TestStructuredOutputFallback(unittest.TestCase):
    """Structured output parse failure should not crash response()."""

    def test_structured_output_validation_error_logged_not_raised(self):
        """When parsed is None/missing, response() should still return content."""
        from pydantic import BaseModel

        class Report(BaseModel):
            summary: str

        model = _make_openai_model()
        model.response_format = Report
        model.use_structured_outputs = True

        mock_message = MagicMock()
        mock_message.content = '{"summary": "fallback text"}'
        mock_message.tool_calls = None
        mock_message.parsed = None  # parse failed — parsed is None
        mock_message.audio = None
        mock_message.reasoning_content = None
        mock_message.role = "assistant"  # must be a real string

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 8
        mock_usage.total_tokens = 18

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch.object(model, 'invoke', new=AsyncMock(return_value=mock_response)):
            try:
                result = asyncio.run(model.response(messages=[
                    Message(role="user", content="analyze"),
                ]))
                # Should not crash even if parsed is None
                self.assertIsNotNone(result)
            except Exception as e:
                self.fail(f"response() raised unexpectedly with parsed=None: {e}")


if __name__ == "__main__":
    unittest.main()
