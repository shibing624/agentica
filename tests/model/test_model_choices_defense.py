# -*- coding: utf-8 -*-
"""
Tests for the unified `choices` empty-defense path.

Goal: every Model subclass that touches `response.choices[0]` MUST go through
`require_first_choice(...)`, which raises a clear ValueError when providers
return an empty `choices` list (rate-limit, content filter, transient API).
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock LLM API keys so model construction does not 401 in CI
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agentica.model.base import require_first_choice
from agentica.model.message import Message


class TestRequireFirstChoiceHelper(unittest.TestCase):
    def test_returns_first_choice_when_present(self):
        first = SimpleNamespace(message="hi", finish_reason="stop")
        response = SimpleNamespace(choices=[first, SimpleNamespace()])
        self.assertIs(require_first_choice(response, context="unit"), first)

    def test_raises_value_error_on_empty_choices(self):
        response = SimpleNamespace(choices=[])
        with self.assertRaises(ValueError) as cm:
            require_first_choice(response, context="LiteLLMChat 'gpt-x'")
        self.assertIn("empty choices", str(cm.exception))
        self.assertIn("LiteLLMChat 'gpt-x'", str(cm.exception))


try:
    from litellm import completion as _litellm_completion  # noqa: F401
    _LITELLM_OK = True
except Exception:
    _LITELLM_OK = False


@unittest.skipUnless(_LITELLM_OK, "litellm not installed")
class TestLiteLLMChatEmptyChoices(unittest.IsolatedAsyncioTestCase):
    async def test_response_raises_value_error_on_empty_choices(self):
        from agentica.model.litellm.chat import LiteLLMChat

        model = LiteLLMChat(id="gpt-test")
        empty_response = SimpleNamespace(choices=[], usage=None)

        with patch.object(model, "invoke", new=AsyncMock(return_value=empty_response)):
            with self.assertRaises(ValueError) as cm:
                await model.response(messages=[Message(role="user", content="hi")])
        self.assertIn("empty choices", str(cm.exception))


class TestOpenAIAudioEmptyChoices(unittest.TestCase):
    def test_answer_question_raises_value_error_on_empty_choices(self):
        import tempfile

        from agentica.model.openai.audio import OpenAIAudioModel

        model = OpenAIAudioModel()
        empty_completion = SimpleNamespace(choices=[])

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00\x00")
            audio_path = f.name

        try:
            with patch.object(
                model._client.chat.completions,
                "create",
                return_value=empty_completion,
            ):
                with self.assertRaises(ValueError) as cm:
                    model.audio_question_answering(audio_path, "what is in this audio?")
            self.assertIn("empty choices", str(cm.exception))
        finally:
            os.unlink(audio_path)


if __name__ == "__main__":
    unittest.main()
