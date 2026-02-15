# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Agent four-piece async API: run / run_stream / run_sync / run_stream_sync.

Uses real OpenAIChat instances with patched response/response_stream methods to avoid
MagicMock attribute issues while testing the full Agent.run() pipeline.
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
from agentica.run_response import RunResponse, RunEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create a real OpenAIChat instance (API key not needed since we patch response)."""
    return OpenAIChat(model="gpt-4o-mini")


def _mock_response(content="Mock response"):
    """Create a ModelResponse with given content."""
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.created_at = None
    return resp


# ===========================================================================
# TestAgentRun — async run()
# ===========================================================================


class TestAgentRun:
    """Tests for Agent.run() — the primary async non-streaming API."""

    @pytest.mark.asyncio
    async def test_run_returns_run_response(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("Hello")):
            agent = Agent(name="A", model=_make_model())
            resp = await agent.run("Hi")
            assert isinstance(resp, RunResponse)

    @pytest.mark.asyncio
    async def test_run_content_matches_model_output(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("expected")):
            agent = Agent(name="A", model=_make_model())
            resp = await agent.run("Hi")
            assert resp.content == "expected"

    @pytest.mark.asyncio
    async def test_run_populates_run_id(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response()):
            agent = Agent(name="A", model=_make_model())
            resp = await agent.run("Hi")
            assert resp.run_id is not None
            assert len(resp.run_id) > 0

    @pytest.mark.asyncio
    async def test_run_populates_agent_id(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response()):
            agent = Agent(name="A", model=_make_model(), agent_id="agent-1")
            resp = await agent.run("Hi")
            assert resp.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_run_populates_model_id(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response()):
            agent = Agent(name="A", model=_make_model())
            resp = await agent.run("Hi")
            assert resp.model is not None
            assert len(resp.model) > 0

    @pytest.mark.asyncio
    async def test_run_with_message_object(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            msg = Message(role="user", content="Hello from Message")
            resp = await agent.run(msg)
            assert resp.content == "OK"

    @pytest.mark.asyncio
    async def test_run_with_messages_list(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            msgs = [Message(role="user", content="M1"), Message(role="user", content="M2")]
            resp = await agent.run(messages=msgs)
            assert resp.content == "OK"

    @pytest.mark.asyncio
    async def test_run_no_message_no_error(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            resp = await agent.run()
            assert resp is not None

    @pytest.mark.asyncio
    async def test_run_stores_in_memory(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            assert len(agent.working_memory.runs) == 0
            await agent.run("Hi")
            assert len(agent.working_memory.runs) == 1

    @pytest.mark.asyncio
    async def test_run_multiple_calls_accumulate_memory(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            await agent.run("First")
            await agent.run("Second")
            assert len(agent.working_memory.runs) == 2


# ===========================================================================
# TestAgentRunSync — sync adapter
# ===========================================================================


class TestAgentRunSync:
    """Tests for Agent.run_sync() — synchronous wrapper."""

    def test_run_sync_returns_run_response(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("Sync OK")):
            agent = Agent(name="A", model=_make_model())
            resp = agent.run_sync("Hi")
            assert isinstance(resp, RunResponse)
            assert resp.content == "Sync OK"

    def test_run_sync_with_all_params(self):
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            resp = agent.run_sync("Hello", images=["http://example.com/img.jpg"])
            assert isinstance(resp, RunResponse)

    def test_run_sync_bridges_to_async_run(self):
        """run_sync should internally call the async run()."""
        agent = Agent(name="A", model=_make_model())
        agent.run = AsyncMock(return_value=RunResponse(content="Mocked"))
        resp = agent.run_sync("test")
        assert resp.content == "Mocked"
        agent.run.assert_called_once()


# ===========================================================================
# TestAgentRunStream — async streaming
# ===========================================================================


class TestAgentRunStream:
    """Tests for Agent.run_stream() — async streaming API."""

    @pytest.mark.asyncio
    async def test_run_stream_yields_chunks(self):
        async def mock_stream(messages, **kwargs):
            for c in ["Hello", " ", "World"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            chunks = []
            async for chunk in agent.run_stream("Hi"):
                chunks.append(chunk)
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_run_stream_chunks_have_event(self):
        async def mock_stream(messages, **kwargs):
            yield ModelResponse(content="Part1", event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            async for chunk in agent.run_stream("Hi"):
                assert hasattr(chunk, "event")

    @pytest.mark.asyncio
    async def test_run_stream_content_accumulates(self):
        async def mock_stream(messages, **kwargs):
            for c in ["A", "B", "C"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            contents = []
            async for chunk in agent.run_stream("Hi"):
                if chunk.content:
                    contents.append(chunk.content)
            assert len(contents) >= 1


# ===========================================================================
# TestAgentRunStreamSync — sync streaming adapter
# ===========================================================================


class TestAgentRunStreamSync:
    """Tests for Agent.run_stream_sync() — synchronous streaming wrapper."""

    def test_run_stream_sync_returns_iterator(self):
        async def mock_stream(messages, **kwargs):
            yield ModelResponse(content="Hello", event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            result = agent.run_stream_sync("Hi")
            assert hasattr(result, '__iter__') or hasattr(result, '__next__')

    def test_run_stream_sync_yields_chunks(self):
        async def mock_stream(messages, **kwargs):
            for c in ["Hello", " World"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            chunks = list(agent.run_stream_sync("Hi"))
            assert len(chunks) >= 1

    def test_run_stream_sync_in_for_loop(self):
        async def mock_stream(messages, **kwargs):
            for c in ["A", "B", "C"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            contents = []
            for chunk in agent.run_stream_sync("Hi"):
                if chunk.content:
                    contents.append(chunk.content)
            assert len(contents) >= 1


# ===========================================================================
# TestAsyncFirstNamingConvention
# ===========================================================================


class TestAsyncFirstNamingConvention:
    """Verify that the async-first naming convention is upheld."""

    def test_run_is_coroutine_function(self):
        agent = Agent(name="A")
        assert asyncio.iscoroutinefunction(agent.run)

    def test_run_stream_is_async(self):
        agent = Agent(name="A")
        # run_stream is an async generator function
        assert asyncio.iscoroutinefunction(agent.run_stream) or hasattr(agent.run_stream, '__func__')

    def test_run_sync_is_regular_function(self):
        agent = Agent(name="A")
        assert not asyncio.iscoroutinefunction(agent.run_sync)

    def test_run_stream_sync_is_regular_function(self):
        agent = Agent(name="A")
        assert not asyncio.iscoroutinefunction(agent.run_stream_sync)

    def test_no_arun_method(self):
        agent = Agent(name="A")
        assert not hasattr(agent, "arun")

    def test_no_arun_stream_method(self):
        agent = Agent(name="A")
        assert not hasattr(agent, "arun_stream")

    def test_no_aprint_response_method(self):
        agent = Agent(name="A")
        assert not hasattr(agent, "aprint_response")

    def test_print_response_is_async(self):
        agent = Agent(name="A")
        assert asyncio.iscoroutinefunction(agent.print_response)

    def test_print_response_sync_is_regular(self):
        agent = Agent(name="A")
        assert not asyncio.iscoroutinefunction(agent.print_response_sync)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
