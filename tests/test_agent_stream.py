# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for streaming output — RunEvent state machine, content accumulation, sync adapter.

Uses real OpenAIChat with patched response_stream to test the full streaming pipeline.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunResponse, RunEvent


# ===========================================================================
# TestStreamContent
# ===========================================================================


class TestStreamContent:
    """Tests for streaming content accumulation."""

    @pytest.mark.asyncio
    async def test_stream_yields_multiple_chunks(self):
        async def mock_stream(messages, **kwargs):
            for c in ["Hello", " ", "World"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            chunks = []
            async for chunk in agent.run_stream("Hi"):
                chunks.append(chunk)
            assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_stream_content_accumulates_correctly(self):
        parts = ["Hello", " beautiful", " world"]

        async def mock_stream(messages, **kwargs):
            for p in parts:
                yield ModelResponse(content=p, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            collected = []
            async for chunk in agent.run_stream("Hi"):
                if chunk.content:
                    collected.append(chunk.content)
            assert "Hello" in collected
            assert " beautiful" in collected
            assert " world" in collected

    @pytest.mark.asyncio
    async def test_stream_empty_chunks_handled(self):
        async def mock_stream(messages, **kwargs):
            for c in ["A", "", "B"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            chunks = []
            async for chunk in agent.run_stream("Hi"):
                chunks.append(chunk)
            # Should not crash regardless of empty content
            assert len(chunks) >= 1


# ===========================================================================
# TestStreamEvents
# ===========================================================================


class TestStreamEvents:
    """Tests for RunEvent in streaming output."""

    @pytest.mark.asyncio
    async def test_stream_chunks_have_run_response_event(self):
        async def mock_stream(messages, **kwargs):
            yield ModelResponse(content="Hello", event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            async for chunk in agent.run_stream("Hi"):
                if chunk.content:
                    assert chunk.event == RunEvent.run_response.value

    @pytest.mark.asyncio
    async def test_stream_intermediate_steps_emits_extra_events(self):
        async def mock_stream(messages, **kwargs):
            yield ModelResponse(content="Hello", event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            events = []
            async for chunk in agent.run_stream("Hi", stream_intermediate_steps=True):
                events.append(chunk.event)
            # With intermediate steps, should have RunStarted at minimum
            assert RunEvent.run_started.value in events

    @pytest.mark.asyncio
    async def test_stream_reasoning_content(self):
        """Model returning reasoning_content should produce reasoning chunks."""

        async def mock_stream(messages, **kwargs):
            yield ModelResponse(
                reasoning_content="Thinking...",
                event=ModelResponseEvent.assistant_response.value,
            )
            yield ModelResponse(
                content="Final answer",
                event=ModelResponseEvent.assistant_response.value,
            )

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            reasoning = []
            content = []
            async for chunk in agent.run_stream("Hi"):
                if chunk.reasoning_content:
                    reasoning.append(chunk.reasoning_content)
                if chunk.content:
                    content.append(chunk.content)
            assert len(reasoning) >= 1
            assert len(content) >= 1


# ===========================================================================
# TestStreamSync
# ===========================================================================


class TestStreamSync:
    """Tests for run_stream_sync() — synchronous streaming adapter."""

    def test_stream_sync_yields_same_content(self):
        parts = ["X", "Y", "Z"]

        async def mock_stream(messages, **kwargs):
            for p in parts:
                yield ModelResponse(content=p, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            collected = []
            for chunk in agent.run_stream_sync("Hi"):
                if chunk.content:
                    collected.append(chunk.content)
            assert "X" in collected
            assert "Y" in collected
            assert "Z" in collected

    def test_stream_sync_usable_in_for_loop(self):
        async def mock_stream(messages, **kwargs):
            for c in ["A", "B"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            count = 0
            for _ in agent.run_stream_sync("Hi"):
                count += 1
            assert count >= 1

    def test_stream_sync_not_coroutine(self):
        agent = Agent(name="A")
        assert not asyncio.iscoroutinefunction(agent.run_stream_sync)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
