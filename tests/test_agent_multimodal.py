# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for multimodal input (images, audio, videos) through Agent.run().
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunResponse


def _mock_resp(content="OK"):
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.created_at = None
    return resp


# ===========================================================================
# TestImageInput
# ===========================================================================


class TestImageInput:
    """Test images parameter through Agent.run()."""

    @pytest.mark.asyncio
    async def test_run_with_single_image_url(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("I see an image")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = await agent.run(
                "Describe this image",
                images=["https://example.com/image.png"],
            )
            assert isinstance(resp, RunResponse)
            assert resp.content == "I see an image"

    @pytest.mark.asyncio
    async def test_run_with_multiple_images(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Two images")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = await agent.run(
                "Compare these images",
                images=[
                    "https://example.com/img1.png",
                    "https://example.com/img2.png",
                ],
            )
            assert resp.content == "Two images"

    @pytest.mark.asyncio
    async def test_run_with_base64_image(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Base64 image")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            # Base64-encoded 1x1 white pixel PNG
            b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
            resp = await agent.run(
                "What is this?",
                images=[b64],
            )
            assert isinstance(resp, RunResponse)

    def test_run_sync_with_images(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Image OK")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = agent.run_sync(
                "Describe",
                images=["https://example.com/img.png"],
            )
            assert resp.content == "Image OK"


# ===========================================================================
# TestAudioInput
# ===========================================================================


class TestAudioInput:
    """Test audio parameter through Agent.run()."""

    @pytest.mark.asyncio
    async def test_run_with_audio_parameter(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Audio processed")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = await agent.run(
                "Transcribe this",
                audio={"data": "base64_audio_data", "format": "wav"},
            )
            assert isinstance(resp, RunResponse)


# ===========================================================================
# TestVideoInput
# ===========================================================================


class TestVideoInput:
    """Test videos parameter through Agent.run()."""

    @pytest.mark.asyncio
    async def test_run_with_video_parameter(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Video processed")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = await agent.run(
                "Describe video",
                videos=["https://example.com/video.mp4"],
            )
            assert isinstance(resp, RunResponse)


# ===========================================================================
# TestMultimodalCombined
# ===========================================================================


class TestMultimodalCombined:
    """Test combining text with images, audio, video."""

    @pytest.mark.asyncio
    async def test_text_and_image_combined(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Combined")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            resp = await agent.run(
                "What is this?",
                images=["https://example.com/img.png"],
            )
            assert resp.content == "Combined"

    @pytest.mark.asyncio
    async def test_multimodal_in_stream_mode(self):
        async def mock_stream(messages, **kwargs):
            yield ModelResponse(content="Streaming multimodal", event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            chunks = []
            async for chunk in agent.run_stream(
                "Describe",
                images=["https://example.com/img.png"],
            ):
                chunks.append(chunk)
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_multimodal_history_multi_turn(self):
        """Multi-turn with images — second call should not fail."""
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Turn 1")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
            await agent.run("Turn 1", images=["https://example.com/img.png"])

        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_resp("Turn 2")):
            resp = await agent.run("Turn 2 without images")
            assert resp.content == "Turn 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
