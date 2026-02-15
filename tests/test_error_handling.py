# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for error handling and recovery paths.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run_response import RunResponse
from agentica.tools.base import Function, FunctionCall, StopAgentRun, RetryAgentRun, ToolCallException


# ===========================================================================
# TestToolErrors
# ===========================================================================


class TestToolErrors:
    """Tests for tool exception handling."""

    @pytest.mark.asyncio
    async def test_value_error_captured(self):
        def bad_tool() -> str:
            """Bad."""
            raise ValueError("bad value")

        f = Function.from_callable(bad_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")
        success = await fc.execute()
        assert success is False
        assert fc.error is not None
        assert "bad value" in fc.error

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self):
        async def bad_async_tool() -> str:
            """Bad async."""
            raise RuntimeError("runtime failure")

        f = Function.from_callable(bad_async_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")
        success = await fc.execute()
        assert success is False
        assert "runtime failure" in fc.error

    @pytest.mark.asyncio
    async def test_stop_agent_run_with_user_message(self):
        async def stop_tool() -> str:
            """Stop."""
            raise StopAgentRun("Stop now", user_message="Please stop")

        f = Function.from_callable(stop_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")

        with pytest.raises(StopAgentRun) as exc_info:
            await fc.execute()
        assert exc_info.value.user_message == "Please stop"
        assert exc_info.value.stop_execution is True

    @pytest.mark.asyncio
    async def test_retry_agent_run_with_messages(self):
        async def retry_tool() -> str:
            """Retry."""
            raise RetryAgentRun(
                "Retry",
                agent_message="Try again with different params",
            )

        f = Function.from_callable(retry_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")

        with pytest.raises(RetryAgentRun) as exc_info:
            await fc.execute()
        assert exc_info.value.agent_message == "Try again with different params"

    @pytest.mark.asyncio
    async def test_tool_call_exception_attributes(self):
        exc = ToolCallException(
            "error",
            user_message="user msg",
            agent_message="agent msg",
            stop_execution=True,
        )
        assert exc.user_message == "user msg"
        assert exc.agent_message == "agent msg"
        assert exc.stop_execution is True


# ===========================================================================
# TestModelErrors
# ===========================================================================


class TestModelErrors:
    """Tests for model-level error handling."""

    @pytest.mark.asyncio
    async def test_model_response_exception_handled(self):
        """Agent should handle model exception gracefully."""
        with patch.object(OpenAIChat, 'response', new_callable=AsyncMock, side_effect=Exception("API error")):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            with pytest.raises(Exception, match="API error"):
                await agent.run("Hi")

    @pytest.mark.asyncio
    async def test_model_stream_exception_handled(self):
        """Agent should handle stream exception."""
        async def bad_stream(messages, **kwargs):
            raise Exception("Stream error")
            yield  # noqa – make it an async generator

        with patch.object(OpenAIChat, 'response_stream', side_effect=bad_stream):
            agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
            with pytest.raises(Exception, match="Stream error"):
                async for _ in agent.run_stream("Hi"):
                    pass


# ===========================================================================
# TestCancellation
# ===========================================================================


class TestCancellation:
    """Tests for agent cancellation."""

    def test_agent_has_cancel_method(self):
        agent = Agent(name="A")
        assert hasattr(agent, "cancel")

    def test_agent_cancel_sets_flag(self):
        agent = Agent(name="A")
        if hasattr(agent, "_cancelled"):
            agent.cancel()
            assert agent._cancelled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
