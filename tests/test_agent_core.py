# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Agent four-piece async API: run / run_stream / run_sync / run_stream_sync.

Uses real OpenAIChat instances with patched response/response_stream methods to avoid
MagicMock attribute issues while testing the full Agent.run() pipeline.
"""
import asyncio
import inspect
import sys
import os
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import (
    Agent,
    AgentDefinition,
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentSafetyConfig,
)
from agentica.hooks import RunHooks
from agentica.model.openai import OpenAIChat
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunResponse, RunEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create a real OpenAIChat instance with fake API key (response is patched)."""
    return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")


def _mock_response(content="Mock response"):
    """Create a ModelResponse with given content."""
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.finish_reason = None
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
    async def test_run_passes_user_and_session_to_langfuse_trace(self):
        calls = []

        @contextmanager
        def fake_langfuse_trace_context(**kwargs):
            calls.append(kwargs)

            class FakeTrace:
                def set_output(self, _output):
                    pass

                def set_metadata(self, _key, _value):
                    pass

            yield FakeTrace()

        with (
            patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response()),
            patch("agentica.runner.langfuse_trace_context", new=fake_langfuse_trace_context),
        ):
            agent = Agent(
                name="A",
                model=_make_model(),
                user_id="user-1",
                session_id="session-1",
            )
            await agent.run("Hi")

        assert calls[0]["user_id"] == "user-1"
        assert calls[0]["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_run_messages_passes_input_to_langfuse_trace(self):
        calls = []

        @contextmanager
        def fake_langfuse_trace_context(**kwargs):
            calls.append(kwargs)

            class FakeTrace:
                def set_output(self, _output):
                    pass

                def set_metadata(self, _key, _value):
                    pass

            yield FakeTrace()

        with (
            patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response()),
            patch("agentica.runner.langfuse_trace_context", new=fake_langfuse_trace_context),
        ):
            agent = Agent(name="A", model=_make_model())
            await agent.run(messages=[Message(role="user", content="M1")])

        assert calls[0]["input_data"] == [{"role": "user", "content": "M1"}]

    @pytest.mark.asyncio
    async def test_run_hooks_create_langfuse_spans(self):
        span_calls = []

        @contextmanager
        def fake_langfuse_span_context(**kwargs):
            span_call = {"kwargs": kwargs, "updates": []}
            span_calls.append(span_call)

            class FakeSpan:
                def update(self, **update_kwargs):
                    span_call["updates"].append(update_kwargs)

            yield FakeSpan()

        class InlineHooks(RunHooks):
            async def on_agent_start(self, agent, **kwargs):
                pass

            async def on_user_prompt(self, agent, message, **kwargs):
                return f"{message} modified"

            async def on_agent_end(self, agent, output, **kwargs):
                pass

        with (
            patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")),
            patch("agentica.runner.langfuse_span_context", new=fake_langfuse_span_context),
        ):
            agent = Agent(name="A", model=_make_model())
            await agent.run("Hi", hooks=InlineHooks())

        names = [call["kwargs"]["name"] for call in span_calls]
        assert "hook.run.on_agent_start" in names
        assert "hook.run.on_user_prompt" in names
        assert "hook.run.on_agent_end" in names
        prompt_span = next(
            call for call in span_calls
            if call["kwargs"]["name"] == "hook.run.on_user_prompt"
        )
        assert prompt_span["kwargs"]["input_data"] == "Hi"
        assert prompt_span["updates"] == [{"output": "Hi modified"}]

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
    async def test_run_messages_mode_skips_history_and_memory_write(self):
        calls = []

        async def fake_response(*args, **kwargs):
            calls.append(list(kwargs["messages"]))
            return _mock_response("OK")

        with patch.object(OpenAIChat, 'response', new=fake_response):
            agent = Agent(name="A", model=_make_model(), add_history_to_context=True)
            await agent.run("stateful history")
            await agent.run(messages=[Message(role="user", content="stateless turn")])

        latest_contents = [m.content for m in calls[-1]]
        assert "stateful history" not in latest_contents
        assert "stateless turn" in latest_contents
        assert len(agent.working_memory.runs) == 1

    @pytest.mark.asyncio
    async def test_run_message_and_messages_are_mutually_exclusive(self):
        agent = Agent(name="A", model=_make_model())
        with pytest.raises(ValueError, match="mutually exclusive"):
            await agent.run("Hi", messages=[Message(role="user", content="M1")])

    @pytest.mark.asyncio
    async def test_run_messages_mode_rejects_top_level_media(self):
        agent = Agent(name="A", model=_make_model())
        with pytest.raises(ValueError, match="audio/images/videos"):
            await agent.run(
                messages=[Message(role="user", content="M1")],
                images=["https://example.com/image.png"],
            )

    @pytest.mark.asyncio
    async def test_run_rejects_removed_add_messages(self):
        agent = Agent(name="A", model=_make_model())
        with pytest.raises(TypeError, match="add_messages was removed"):
            await agent.run("Hi", add_messages=[{"role": "system", "content": "x"}])

    @pytest.mark.asyncio
    async def test_run_rejects_unknown_kwarg_with_hint(self):
        agent = Agent(name="A", model=_make_model())
        with pytest.raises(TypeError, match="Did you mean 'timeout'"):
            await agent.run("Hi", timoeut=5)

    @pytest.mark.asyncio
    async def test_run_rejects_unknown_kwarg_without_hint(self):
        agent = Agent(name="A", model=_make_model())
        with pytest.raises(TypeError, match="Unknown run\\(\\) keyword argument 'totally_bogus'"):
            await agent.run("Hi", totally_bogus=1)

    def test_run_entrypoint_signatures_match(self):
        expected = [
            "self", "message", "messages", "audio", "images", "videos",
            "timeout", "hooks", "config", "kwargs",
        ]
        for method_name in ("run", "run_stream", "run_sync", "run_stream_sync"):
            signature = inspect.signature(getattr(Agent, method_name))
            assert list(signature.parameters) == expected

    @pytest.mark.asyncio
    async def test_run_accepts_inline_hooks(self):
        called = []

        class InlineHooks(RunHooks):
            async def on_agent_start(self, agent, **kwargs):
                called.append(agent.name)

        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, return_value=_mock_response("OK")):
            agent = Agent(name="A", model=_make_model())
            await agent.run("Hi", hooks=InlineHooks())

        assert called == ["A"]

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
        """run_sync should internally call the runner's async run()."""
        agent = Agent(name="A", model=_make_model())
        agent._runner.run = AsyncMock(return_value=RunResponse(content="Mocked"))
        resp = agent.run_sync("test")
        assert resp.content == "Mocked"
        agent._runner.run.assert_called_once()


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

    def test_run_stream_sync_cancels_agent_on_early_break(self):
        """Abandoning the iterator must cancel the background agent run so it
        stops calling tools/LLMs with nobody listening (no silent token burn)."""
        async def mock_stream(messages, **kwargs):
            for c in ["A", "B", "C"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            with patch.object(agent, 'cancel') as mock_cancel:
                for _chunk in agent.run_stream_sync("Hi"):
                    break  # abandon the stream after the first chunk
            mock_cancel.assert_called_once()

    def test_run_stream_sync_no_cancel_on_full_consumption(self):
        """Fully draining the iterator is a normal completion — do not cancel."""
        async def mock_stream(messages, **kwargs):
            for c in ["A", "B", "C"]:
                yield ModelResponse(content=c, event=ModelResponseEvent.assistant_response.value)

        with patch.object(OpenAIChat, 'response_stream', side_effect=mock_stream):
            agent = Agent(name="A", model=_make_model())
            with patch.object(agent, 'cancel') as mock_cancel:
                list(agent.run_stream_sync("Hi"))
            mock_cancel.assert_not_called()


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


class TestAgentFromParts:
    def test_agent_retains_user_id_without_workspace(self):
        agent = Agent(name="A", model=_make_model(), user_id="user-1")
        assert agent.user_id == "user-1"

    def test_update_model_propagates_trace_identity_to_model(self):
        model = _make_model()
        agent = Agent(
            name="A",
            model=model,
            user_id="user-1",
            session_id="session-1",
        )
        agent.update_model()

        assert model.user_id == "user-1"
        assert model.session_id == "session-1"
        assert model.agent_name == "A"

    def test_from_parts_groups_constructor_surface(self):
        agent = Agent.from_parts(
            definition=AgentDefinition(
                name="Planner",
                model=_make_model(),
                instructions="Plan carefully",
            ),
            execution=AgentExecutionConfig(
                add_history_to_context=True,
                max_api_retry=2,
                session_id="session-1",
            ),
            memory=AgentMemoryConfig(
                enable_long_term_memory=True,
                enable_experience_capture=True,
                context={"mode": "planning"},
            ),
            safety=AgentSafetyConfig(
                input_guardrails=["input-check"],
                output_guardrails=["output-check"],
            ),
        )

        assert agent.name == "Planner"
        assert agent.instructions == "Plan carefully"
        assert agent.add_history_to_context is True
        assert agent.max_api_retry == 2
        assert agent.session_id == "session-1"
        assert agent.enable_long_term_memory is True
        assert agent.enable_experience_capture is True
        assert agent.context == {"mode": "planning"}
        assert agent.input_guardrails == ["input-check"]
        assert agent.output_guardrails == ["output-check"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
