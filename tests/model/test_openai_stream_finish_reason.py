# -*- coding: utf-8 -*-
"""OpenAIChat.response_stream finish_reason capture."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.model.message import Message


class TestOpenAIStreamFinishReason:
    """Test that OpenAIChat.response_stream correctly captures finish_reason."""

    def _make_openai_chat(self):
        from agentica.model.openai import OpenAIChat
        return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

    def test_finish_reason_captured_from_last_chunk(self):
        """stream_finish_reason should be captured from the chunk where finish_reason is not None."""
        model = self._make_openai_chat()

        # Build mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.audio = None
        chunk1.choices[0].delta.tool_calls = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "stop"
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.audio = None
        chunk2.choices[0].delta.tool_calls = None
        chunk2.usage = None

        async def mock_invoke_stream(messages):
            yield chunk1
            yield chunk2

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]
        collected = []

        async def run():
            async for resp in model.response_stream(messages=messages):
                collected.append(resp)

        asyncio.run(run())
        assert model.last_finish_reason == "stop"

    def test_finish_reason_length_captured(self):
        """When output is truncated, finish_reason should be 'length'."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = "length"
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "partial output..."
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.audio = None
        chunk.choices[0].delta.tool_calls = None
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason == "length"

    def test_finish_reason_none_when_no_choices(self):
        """When stream has no choices, finish_reason should remain None."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason is None
