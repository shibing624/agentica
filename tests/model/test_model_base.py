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
        m = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
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

    def test_skips_unavailable_function(self):
        model = self._make_model()

        def gated_tool() -> str:
            """Conditionally available."""
            return "ok"

        func = Function.from_callable(gated_tool)
        func.available_when = lambda: False

        model.add_tool(func)

        assert model.functions is not None
        assert "gated_tool" in model.functions
        assert model.tools == []


# ---------------------------------------------------------------------------
# TestRunFunctionCalls — Parallel tool execution
# ---------------------------------------------------------------------------


class TestRunFunctionCalls:
    """Tests for Model.run_function_calls() using asyncio.TaskGroup."""

    def _make_model_instance(self):
        from agentica.model.openai import OpenAIChat
        m = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
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
    async def test_tool_display_metadata_reaches_completion_event_not_model_content(self):
        from agentica.tools.helpers import ToolDisplayOutput

        model = self._make_model_instance()

        def write_tool() -> str:
            """Return a write result with presentation metadata."""
            return ToolDisplayOutput(
                "updated",
                {"files": [{"path": "a.py", "action": "update", "before": "a", "after": "b"}]},
            )

        fc = self._make_fc(write_tool, call_id="write-1")
        results = []
        events = []
        async for response in model.run_function_calls([fc], results):
            events.append(response)

        completed = next(
            response for response in events
            if response.event == ModelResponseEvent.tool_call_completed.value
        )
        assert completed.tool_call["tool_display_meta"] == {
            "files": [{"path": "a.py", "action": "update", "before": "a", "after": "b"}]
        }
        assert results[0].content == "updated"
        assert results[0].to_model_dict()["content"] == "updated"
        assert "tool_display_meta" not in results[0].to_model_dict()

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_serial(self):
        """N concurrency_safe tools each sleeping 0.1s should complete in ≈0.1s (parallel), not N*0.1s."""
        model = self._make_model_instance()
        n = 5

        async def slow_tool(x: int = 0) -> str:
            """Slow tool."""
            await asyncio.sleep(0.1)
            return str(x)

        fcs = [self._make_fc(slow_tool, {"x": i}, f"c{i}") for i in range(n)]
        # Mark as concurrency_safe so the new split-execution path runs them in parallel.
        for fc in fcs:
            fc.function.concurrency_safe = True
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
    async def test_a_cancelled_run_does_not_start_parallel_tools(self):
        """Parallel tools used to skip the cancellation check entirely, so
        Ctrl+C still fired off every read — and, since ``task`` became
        concurrency_safe, every subagent — in the pending batch."""
        import weakref

        model = self._make_model_instance()
        started = []

        async def read_tool(x: int = 0) -> str:
            """Read."""
            started.append(x)
            return str(x)

        class _CancelledAgent:
            agent_id = "a"
            name = "a"
            run_id = "r"
            tool_input_guardrails = []
            tool_output_guardrails = []
            context = None
            _run_hooks = None
            _cancelled = True
            approve = None
            _session_log = None

        agent = _CancelledAgent()
        model._agent_ref = weakref.ref(agent)
        fcs = [self._make_fc(read_tool, {"x": i}, f"c{i}") for i in range(3)]
        for fc in fcs:
            fc.function.concurrency_safe = True

        results = []
        async for _ in model.run_function_calls(fcs, results):
            pass

        assert started == [], "cancelled run still launched parallel tools"
        assert all("cancelled by user" in str(m.content).lower() for m in results)

    @pytest.mark.asyncio
    async def test_a_cancelled_run_still_runs_tools_that_cannot_be_interrupted(self):
        """``interrupt_behavior="block"`` means the tool cannot be torn down
        cleanly, so the parallel branch must honour it the way the serial one
        always has."""
        import weakref

        model = self._make_model_instance()
        started = []

        async def blocking_tool(x: int = 0) -> str:
            """Blocking."""
            started.append(x)
            return str(x)

        class _CancelledAgent:
            agent_id = "a"
            name = "a"
            run_id = "r"
            tool_input_guardrails = []
            tool_output_guardrails = []
            context = None
            _run_hooks = None
            _cancelled = True
            approve = None
            _session_log = None

        agent = _CancelledAgent()
        model._agent_ref = weakref.ref(agent)
        fc = self._make_fc(blocking_tool, {"x": 1}, "c0")
        fc.function.concurrency_safe = True
        fc.function.interrupt_behavior = "block"

        async for _ in model.run_function_calls([fc], []):
            pass

        assert started == [1]

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
        assert len(model.function_call_stack) <= 6

    @pytest.mark.asyncio
    async def test_manages_own_timeout_skips_outer_timeout_wrapper(self):
        model = self._make_model_instance()

        async def slow_tool() -> str:
            """Slow tool."""
            await asyncio.sleep(0.05)
            return "ok"

        fc = self._make_fc(slow_tool, {}, "c_timeout")
        fc.function.timeout = 0
        fc.function.manages_own_timeout = True

        results = []
        async for _ in model.run_function_calls([fc], results):
            pass

        assert fc.result == "ok"
        assert fc.error is None
        assert len(results) == 1

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

    @pytest.mark.asyncio
    async def test_concurrent_function_calls_ignore_shared_stack_reset(self):
        model = self._make_model_instance()

        async def slow_tool(x: int = 0) -> str:
            """Slow tool."""
            await asyncio.sleep(0.05)
            return str(x)

        async def run_one(x: int):
            fc = self._make_fc(slow_tool, {"x": x}, f"c{x}")
            results = []
            async for _ in model.run_function_calls([fc], results):
                pass
            return fc.result, len(results)

        first = asyncio.create_task(run_one(1))
        await asyncio.sleep(0.01)
        model.function_call_stack = None
        model._failed_call_counts = None
        second = asyncio.create_task(run_one(2))

        assert await asyncio.gather(first, second) == [("1", 1), ("2", 1)]

    @pytest.mark.asyncio
    async def test_tool_choice_is_task_local(self):
        async def read_choice(choice: str, delay: float) -> str:
            token = Model.begin_run_state()
            try:
                model = self._make_model_instance()
                model.set_tool_choice(choice)
                await asyncio.sleep(delay)
                return model.get_tool_choice()
            finally:
                Model.reset_run_state(token)

        assert await asyncio.gather(
            read_choice("none", 0.03),
            read_choice("auto", 0.01),
        ) == ["none", "auto"]


# ---------------------------------------------------------------------------
# TestCloseClient — per-turn async HTTP client teardown
# ---------------------------------------------------------------------------


class TestCloseClient:
    """close_client() must actually close the cached async SDK client.

    Regression: it only looked for ``aclose()``, but the openai / anthropic
    async clients expose an async ``close()`` instead, so teardown silently
    no-op'd and the httpx pool was finalized later on a dead loop
    ("RuntimeError: Event loop is closed").
    """

    def _make_model_instance(self):
        from agentica.model.openai import OpenAIChat
        return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

    @pytest.mark.asyncio
    async def test_closes_client_exposing_only_async_close(self):
        model = self._make_model_instance()
        client = MagicMock(spec=["close"])
        client.close = AsyncMock()
        model.client = client

        await model.close_client()

        client.close.assert_awaited_once()
        assert model.client is None

    @pytest.mark.asyncio
    async def test_prefers_aclose_when_available(self):
        model = self._make_model_instance()
        client = MagicMock(spec=["aclose"])
        client.aclose = AsyncMock()
        model.client = client

        await model.close_client()

        client.aclose.assert_awaited_once()
        assert model.client is None

    @pytest.mark.asyncio
    async def test_supports_sync_close(self):
        """A plain (non-awaitable) close() must still be invoked."""
        model = self._make_model_instance()
        client = MagicMock(spec=["close"])
        model.client = client

        await model.close_client()

        client.close.assert_called_once()
        assert model.client is None

    @pytest.mark.asyncio
    async def test_no_client_is_noop(self):
        model = self._make_model_instance()
        model.client = None
        await model.close_client()  # must not raise
        assert model.client is None

    @pytest.mark.asyncio
    async def test_real_openai_client_is_closeable(self):
        """Pin the real SDK surface: the built client must expose a closer.

        This is the check that would have caught the original bug — the mock
        tests above pass either way.
        """
        model = self._make_model_instance()
        client = model.get_client()
        assert getattr(client, "aclose", None) or getattr(client, "close", None)

        await model.close_client()
        assert model.client is None

    @pytest.mark.asyncio
    async def test_close_failure_does_not_propagate(self):
        model = self._make_model_instance()
        client = MagicMock(spec=["close"])
        client.close = AsyncMock(side_effect=RuntimeError("Event loop is closed"))
        model.client = client

        await model.close_client()  # teardown is best-effort

        assert model.client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
