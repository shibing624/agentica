# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for Model base class — interface, parallel tool execution, add_tool.
"""
import asyncio
import inspect
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.tools.base import Function, FunctionCall, Tool, ToolCallException, StopAgentRun


# ---------------------------------------------------------------------------
# TestModelInterface — Async-only abstract methods
# ---------------------------------------------------------------------------


class TestModelInterface:
    """Verify Model base class exposes only async abstract methods."""

    def test_invoke_is_coroutine(self):
        assert asyncio.iscoroutinefunction(Model.invoke)

    def test_invoke_stream_is_coroutine(self):
        assert asyncio.iscoroutinefunction(Model.invoke_stream)

    def test_response_is_coroutine(self):
        assert asyncio.iscoroutinefunction(Model.response)

    def test_response_stream_is_coroutine(self):
        assert asyncio.iscoroutinefunction(Model.response_stream)

    def test_no_sync_response_method(self):
        """There should be no synchronous 'aresponse' or sync 'response' pair."""
        assert not hasattr(Model, "aresponse")
        assert not hasattr(Model, "ainvoke")
        assert not hasattr(Model, "ainvoke_stream")
        assert not hasattr(Model, "aresponse_stream")

    def test_run_function_calls_is_async(self):
        assert asyncio.iscoroutinefunction(Model.run_function_calls) or inspect.isasyncgenfunction(Model.run_function_calls)


# ---------------------------------------------------------------------------
# TestModelAddTool
# ---------------------------------------------------------------------------


class TestModelAddTool:
    """Tests for Model.add_tool() with various input types."""

    def _make_model(self):
        from agentica.model.openai import OpenAIChat
        m = OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key")
        m.tools = None
        m.functions = None
        return m

    def test_add_callable_tool(self):
        model = self._make_model()
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"
        model.add_tool(greet)
        assert model.functions is not None
        assert "greet" in model.functions

    def test_add_tool_class(self):
        model = self._make_model()
        tool = Tool(name="test_tool")
        def sample(x: int) -> int:
            """Double."""
            return x * 2
        tool.register(sample)
        model.add_tool(tool)
        assert "sample" in model.functions

    def test_add_function_object(self):
        model = self._make_model()
        func = Function.from_callable(lambda x: x)
        func.name = "identity"
        func.description = "Identity function"
        model.add_tool(func)
        assert model.functions is not None
        assert "identity" in model.functions

    def test_add_dict_tool(self):
        model = self._make_model()
        raw_schema = {"type": "function", "function": {"name": "raw", "parameters": {}}}
        model.add_tool(raw_schema)
        assert model.tools is not None
        assert raw_schema in model.tools

    def test_duplicate_tool_deduplicated(self):
        model = self._make_model()
        def my_func(x: int) -> int:
            """Func."""
            return x
        model.add_tool(my_func)
        model.add_tool(my_func)
        assert len(model.functions) == 1

    def test_get_tools_for_api_format(self):
        model = self._make_model()
        def greet(name: str) -> str:
            """Greet."""
            return name
        model.add_tool(greet)
        tools_api = model.get_tools_for_api()
        assert tools_api is not None
        assert len(tools_api) >= 1
        first = tools_api[0]
        assert first.get("type") == "function"
        assert "function" in first


# ---------------------------------------------------------------------------
# TestRunFunctionCalls — Parallel tool execution
# ---------------------------------------------------------------------------


class TestRunFunctionCalls:
    """Tests for Model.run_function_calls() using asyncio.TaskGroup."""

    def _make_model_instance(self):
        from agentica.model.openai import OpenAIChat
        m = OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key")
        m.metrics = {}
        m.function_call_stack = None
        m.tool_call_limit = None
        return m

    def _make_fc(self, func, arguments=None, call_id="call_1"):
        f = Function.from_callable(func)
        fc = FunctionCall(function=f, arguments=arguments or {}, call_id=call_id)
        return fc

    @pytest.mark.asyncio
    async def test_single_tool_execution(self):
        model = self._make_model_instance()
        def add(a: int, b: int) -> str:
            """Add."""
            return str(a + b)
        fc = self._make_fc(add, {"a": 1, "b": 2}, "c1")
        results = []
        async for resp in model.run_function_calls([fc], results):
            pass
        assert fc.result == "3"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_serial(self):
        """N tools each sleeping 0.1s should complete in ≈0.1s (parallel), not N*0.1s."""
        model = self._make_model_instance()
        n = 5

        async def slow_tool(x: int = 0) -> str:
            """Slow tool."""
            await asyncio.sleep(0.1)
            return str(x)

        fcs = [self._make_fc(slow_tool, {"x": i}, f"c{i}") for i in range(n)]
        results = []
        start = time.monotonic()
        async for _ in model.run_function_calls(fcs, results):
            pass
        elapsed = time.monotonic() - start

        # Parallel: should be ~0.1s. Serial would be ~0.5s.
        assert elapsed < 0.3, f"Parallel execution took too long: {elapsed:.2f}s (expected < 0.3s)"
        assert len(results) == n

    @pytest.mark.asyncio
    async def test_parallel_execution_preserves_order(self):
        """Results should be in the same order as the input function calls."""
        model = self._make_model_instance()

        async def ordered_tool(idx: int = 0) -> str:
            """Ordered tool."""
            await asyncio.sleep(0.05 - idx * 0.01)  # Earlier tools finish later
            return str(idx)

        fcs = [self._make_fc(ordered_tool, {"idx": i}, f"c{i}") for i in range(3)]
        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass
        # Results should be in input order, not completion order
        assert results[0].tool_call_id == "c0"
        assert results[1].tool_call_id == "c1"
        assert results[2].tool_call_id == "c2"

    @pytest.mark.asyncio
    async def test_tool_started_events_emitted_first(self):
        """All tool_call_started events should be emitted before any completed events."""
        model = self._make_model_instance()

        async def tool(x: int = 0) -> str:
            """Tool."""
            return str(x)

        fcs = [self._make_fc(tool, {"x": i}, f"c{i}") for i in range(3)]
        events = []
        async for resp in model.run_function_calls(fcs, []):
            events.append(resp.event)

        started_indices = [i for i, e in enumerate(events) if e == ModelResponseEvent.tool_call_started.value]
        completed_indices = [i for i, e in enumerate(events) if e == ModelResponseEvent.tool_call_completed.value]

        # All started events should come before any completed event
        if started_indices and completed_indices:
            assert max(started_indices) < min(completed_indices)

    @pytest.mark.asyncio
    async def test_tool_exception_isolated(self):
        """One tool failing should not prevent other tools from completing."""
        model = self._make_model_instance()

        async def good_tool(x: int = 0) -> str:
            """Good."""
            return "ok"

        async def bad_tool() -> str:
            """Bad."""
            raise ValueError("fail")

        fc_good = self._make_fc(good_tool, {"x": 1}, "c_good")
        fc_bad = self._make_fc(bad_tool, {}, "c_bad")

        results = []
        async for _ in model.run_function_calls([fc_good, fc_bad], results):
            pass

        assert len(results) == 2
        # Good tool should have succeeded
        assert fc_good.result == "ok"
        # Bad tool should have error captured
        assert fc_bad.error is not None

    @pytest.mark.asyncio
    async def test_function_call_stack_tracked(self):
        model = self._make_model_instance()

        def tool(x: int = 0) -> str:
            """Tool."""
            return str(x)

        fc = self._make_fc(tool, {"x": 1}, "c1")
        async for _ in model.run_function_calls([fc], []):
            pass

        assert model.function_call_stack is not None
        assert len(model.function_call_stack) == 1

    @pytest.mark.asyncio
    async def test_tool_call_limit_respected(self):
        model = self._make_model_instance()
        model.tool_call_limit = 2

        def tool(x: int = 0) -> str:
            """Tool."""
            return str(x)

        fcs = [self._make_fc(tool, {"x": i}, f"c{i}") for i in range(5)]
        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass

        # Should have processed at most 2 due to tool_call_limit
        assert len(model.function_call_stack) <= 2

    @pytest.mark.asyncio
    async def test_metrics_recorded(self):
        model = self._make_model_instance()

        def tool(x: int = 0) -> str:
            """Tool."""
            return str(x)

        fc = self._make_fc(tool, {"x": 1}, "c1")
        async for _ in model.run_function_calls([fc], []):
            pass

        assert "tool_call_times" in model.metrics
        assert "tool" in model.metrics["tool_call_times"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
