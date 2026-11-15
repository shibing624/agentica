# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Integration tests for Agent + Model + Tool call round-trip.
Uses mocked LLM but real tool execution.
"""
import asyncio
import time
import weakref
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run.response import RunResponse, RunEvent
from agentica.agent.hooks import RunHooks
from agentica.tools.base import Function, FunctionCall, Tool, StopAgentRun, RetryAgentRun
from agentica.utils.hook_recorder import HookRecorder


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def sync_add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


async def async_search(query: str) -> str:
    """Search for something."""
    await asyncio.sleep(0.01)
    return f"Found: {query}"


async def async_slow_a(x: int = 0) -> str:
    """Slow tool A."""
    await asyncio.sleep(0.1)
    return f"a:{x}"


async def async_slow_b(x: int = 0) -> str:
    """Slow tool B."""
    await asyncio.sleep(0.1)
    return f"b:{x}"


async def async_slow_c(x: int = 0) -> str:
    """Slow tool C."""
    await asyncio.sleep(0.1)
    return f"c:{x}"


def failing_tool_sync() -> str:
    """Always fails."""
    raise ValueError("sync failure")


async def failing_tool_async() -> str:
    """Always fails async."""
    raise RuntimeError("async failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_with_tool_call(tool_calls_response, final_response_content="Final answer"):
    """Create a model that first returns tool_calls, then returns text.

    tool_calls_response: ModelResponse with tool_calls field
    """
    model = MagicMock(spec=Model)
    model.id = "tool-model"
    model.name = "ToolModel"
    model.provider = "mock"
    model.tools = []
    model.functions = {}
    model.function_call_stack = None
    model.run_tools = True
    model.tool_call_limit = None
    model.system_prompt = None
    model.instructions = None
    model.use_structured_outputs = None
    model.supports_structured_outputs = False
    model.context_window = 128000
    model.metrics = {}
    model.response_format = None
    model.session_id = None
    model.user_id = None
    model.agent_name = None
    model.get_tools_for_api = MagicMock(return_value=[])
    model.add_tool = MagicMock()
    model.sanitize_messages = Model.sanitize_messages
    model.to_dict = MagicMock(return_value={"id": "tool-model"})
    model.deactivate_function_calls = MagicMock()

    # First call returns tool_calls, second call returns final text
    model.response = AsyncMock(
        side_effect=[
            tool_calls_response,
            ModelResponse(content=final_response_content),
        ]
    )
    return model


# ===========================================================================
# TestSingleToolCall
# ===========================================================================


class TestSingleToolCall:
    """Test single sync/async tool execution through FunctionCall."""

    @pytest.mark.asyncio
    async def test_sync_tool_executes(self):
        f = Function.from_callable(sync_add)
        fc = FunctionCall(function=f, arguments={"a": 3, "b": 4}, call_id="c1")
        success = await fc.execute()
        assert success is True
        assert fc.result == "7"
        assert fc.error is None

    @pytest.mark.asyncio
    async def test_async_tool_executes(self):
        f = Function.from_callable(async_search)
        fc = FunctionCall(function=f, arguments={"query": "test"}, call_id="c1")
        success = await fc.execute()
        assert success is True
        assert "Found: test" in fc.result

    @pytest.mark.asyncio
    async def test_tool_failure_sets_error(self):
        f = Function.from_callable(failing_tool_sync)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")
        success = await fc.execute()
        assert success is False
        assert fc.error is not None
        assert "sync failure" in fc.error

    @pytest.mark.asyncio
    async def test_async_tool_failure_sets_error(self):
        f = Function.from_callable(failing_tool_async)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")
        success = await fc.execute()
        assert success is False
        assert "async failure" in fc.error


# ===========================================================================
# TestParallelToolExecution
# ===========================================================================


class TestParallelToolExecution:
    """Test parallel tool execution via Model.run_function_calls()."""

    def _make_model(self):
        from agentica.model.openai import OpenAIChat
        m = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        m.metrics = {}
        m.function_call_stack = None
        m.tool_call_limit = None
        return m

    @pytest.mark.asyncio
    async def test_parallel_tools_run_concurrently(self):
        """Three concurrency-safe calls overlap; the schedule, not a clock.

        Asserting wall-clock time ("three 0.1s sleeps finish in <0.25s") makes
        this fail on a loaded machine without the schedule being wrong. Count
        how many bodies are in flight at once instead: peak 3 can only happen
        when the executor gathered them.
        """
        model = self._make_model()
        in_flight = 0
        peak = 0

        async def tool_a(x: int) -> str:
            """Wait briefly and report."""
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"a{x}"

        async def tool_b(x: int) -> str:
            """Wait briefly and report."""
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"b{x}"

        async def tool_c(x: int) -> str:
            """Wait briefly and report."""
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"c{x}"

        fcs = []
        for i, fn in enumerate((tool_a, tool_b, tool_c), 1):
            function = Function.from_callable(fn)
            function.concurrency_safe = True  # from_callable does not infer it
            fcs.append(FunctionCall(function=function, arguments={"x": i}, call_id=f"c{i}"))
        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass
        assert len(results) == 3
        assert peak == 3, f"peak concurrency {peak} — the batch did not run in parallel"

    @pytest.mark.asyncio
    async def test_tool_run_hooks_record_to_agent_metadata(self):
        model = self._make_model()

        class AgentStub:
            agent_id = "agent-1"
            name = "Agent"
            run_id = "run-1"
            tool_input_guardrails = []
            tool_output_guardrails = []
            context = None
            _run_hooks = None
            _cancelled = False
            approve = None
            _session_log = None

            def __init__(self):
                self._hook_recorder = HookRecorder()
                self._run_hooks = None

        class ToolAuditHooks(RunHooks):
            async def on_tool_start(self, agent, tool_name="", tool_call_id="", tool_args=None, **kwargs):
                pass

            async def on_tool_end(
                self, agent, tool_name="", tool_call_id="", tool_args=None,
                result=None, is_error=False, elapsed=0.0, **kwargs
            ):
                pass

        agent = AgentStub()
        agent._run_hooks = ToolAuditHooks()
        model._agent_ref = weakref.ref(agent)
        function_call = FunctionCall(
            function=Function.from_callable(sync_add),
            arguments={"a": 1, "b": 2},
            call_id="c1",
        )
        results = []

        async for _ in model.run_function_calls([function_call], results):
            pass

        records = agent._hook_recorder.export()
        observed = {
            (item["hook_class"], item["method"], item.get("meta", {}).get("tool_name"))
            for item in records
        }
        assert ("ToolAuditHooks", "on_tool_start", "sync_add") in observed
        assert ("ToolAuditHooks", "on_tool_end", "sync_add") in observed
        assert all(item["ok"] for item in records)

    @pytest.mark.asyncio
    async def test_mixed_sync_async_parallel(self):
        """Mix of sync and async tools should all execute."""
        model = self._make_model()
        fcs = [
            FunctionCall(function=Function.from_callable(sync_add), arguments={"a": 1, "b": 2}, call_id="c1"),
            FunctionCall(function=Function.from_callable(async_search), arguments={"query": "test"}, call_id="c2"),
        ]
        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass
        assert len(results) == 2
        assert fcs[0].result == "3"
        assert "Found: test" in fcs[1].result

    @pytest.mark.asyncio
    async def test_partial_failure_others_succeed(self):
        """One failing tool should not prevent others from succeeding."""
        model = self._make_model()
        fcs = [
            FunctionCall(function=Function.from_callable(sync_add), arguments={"a": 1, "b": 2}, call_id="c1"),
            FunctionCall(function=Function.from_callable(failing_tool_sync), arguments={}, call_id="c2"),
            FunctionCall(function=Function.from_callable(async_search), arguments={"query": "ok"}, call_id="c3"),
        ]
        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass
        assert fcs[0].result == "3"
        assert fcs[1].error is not None
        assert "Found: ok" in fcs[2].result


# ===========================================================================
# TestToolHooks
# ===========================================================================


class TestToolHooks:
    """Test pre/post hooks on tool execution."""

    @pytest.mark.asyncio
    async def test_async_pre_hook_called(self):
        called = []

        async def pre_hook():
            called.append("pre")

        f = Function.from_callable(sync_add)
        f.pre_hook = pre_hook
        fc = FunctionCall(function=f, arguments={"a": 1, "b": 2}, call_id="c1")
        await fc.execute()
        assert "pre" in called

    @pytest.mark.asyncio
    async def test_async_post_hook_called(self):
        called = []

        async def post_hook():
            called.append("post")

        f = Function.from_callable(sync_add)
        f.post_hook = post_hook
        fc = FunctionCall(function=f, arguments={"a": 1, "b": 2}, call_id="c1")
        await fc.execute()
        assert "post" in called

    @pytest.mark.asyncio
    async def test_sync_hook_with_async_tool(self):
        called = []

        def sync_hook():
            called.append("sync_pre")

        f = Function.from_callable(async_search)
        f.pre_hook = sync_hook
        fc = FunctionCall(function=f, arguments={"query": "test"}, call_id="c1")
        await fc.execute()
        assert "sync_pre" in called

    @pytest.mark.asyncio
    async def test_both_hooks_called_in_order(self):
        order = []

        async def pre():
            order.append("pre")

        async def post():
            order.append("post")

        f = Function.from_callable(sync_add)
        f.pre_hook = pre
        f.post_hook = post
        fc = FunctionCall(function=f, arguments={"a": 0, "b": 0}, call_id="c1")
        await fc.execute()
        assert order == ["pre", "post"]


# ===========================================================================
# TestFlowControlExceptions
# ===========================================================================


class TestFlowControlExceptions:
    """Test StopAgentRun and RetryAgentRun exceptions."""

    @pytest.mark.asyncio
    async def test_stop_agent_run_propagates(self):
        async def stop_tool() -> str:
            """Stops."""
            raise StopAgentRun("Stopping", user_message="Please stop")

        f = Function.from_callable(stop_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")

        with pytest.raises(StopAgentRun):
            await fc.execute()

    @pytest.mark.asyncio
    async def test_retry_agent_run_propagates(self):
        async def retry_tool() -> str:
            """Retries."""
            raise RetryAgentRun("Retry", agent_message="Try again")

        f = Function.from_callable(retry_tool)
        fc = FunctionCall(function=f, arguments={}, call_id="c1")

        with pytest.raises(RetryAgentRun):
            await fc.execute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
