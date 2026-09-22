# -*- coding: utf-8 -*-
"""Tool events leave the Runner on streaming and non-streaming executions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentica.model.base import Model
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run.events import RunEventType
from agentica.runner import Runner
from agentica.runner.loop import _tool_preview
from agentica.tools.base import Function, FunctionCall


class _Model:
    async def run_function_calls(self, **_kwargs):
        function_call = SimpleNamespace(
            function=SimpleNamespace(name="write_file"),
            call_id="call-1",
            arguments={
                "file_path": "/tmp/result.txt",
                "content": "secret body that must not leave the process",
            },
        )
        callback = _kwargs["tool_event_callback"]
        callback("started", function_call, None, None, None)
        yield ModelResponse(
            event=ModelResponseEvent.tool_call_started.value,
            tool_call={
                "tool_call_id": "call-1",
                "tool_name": "write_file",
                "tool_args": {
                    "file_path": "/tmp/result.txt",
                    "content": "secret body that must not leave the process",
                },
            },
        )
        callback(
            "completed",
            function_call,
            False,
            1.25,
            "Error: https://example.test/run?token=plain-secret",
        )
        yield ModelResponse(
            event=ModelResponseEvent.tool_call_completed.value,
            tool_call={
                "tool_call_id": "call-1",
                "tool_name": "write_file",
                "tool_args": {
                    "file_path": "/tmp/result.txt",
                    "content": "secret body that must not leave the process",
                },
                "content": "Error: https://example.test/run?token=plain-secret",
                "tool_call_error": True,
                "metrics": {"time": 1.25},
            },
        )


class _ExecutionModel(Model):
    @property
    def request_kwargs(self):
        return {}

    async def invoke(self, messages):
        raise NotImplementedError

    async def invoke_stream(self, messages):
        raise NotImplementedError

    async def response(self, messages):
        raise NotImplementedError

    async def response_stream(self, messages):
        raise NotImplementedError


def test_tool_preview_redacts_external_credentials():
    preview = _tool_preview(
        {"command": "curl 'https://example.test/run?token=plain-secret'"}
    )
    assert "plain-secret" not in preview
    assert "token=***" in preview


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_shared_executor_emits_bounded_tool_events(stream):
    runner = Runner(SimpleNamespace())
    seen = []
    runner._emit_event = lambda event, payload: seen.append((event, payload))

    chunks = [
        chunk
        async for chunk in runner._execute_tool_calls(
            [],
            [],
            SimpleNamespace(),
            _Model(),
            stream=stream,
        )
    ]

    assert len(chunks) == 2
    assert [event for event, _ in seen] == [
        RunEventType.tool_started,
        RunEventType.tool_completed,
    ]
    started = seen[0][1]
    completed = seen[1][1]
    assert started["preview"] == "/tmp/result.txt"
    assert "secret body" not in str(seen)
    assert completed["ok"] is False
    assert "plain-secret" not in completed["error"]
    assert "token=***" in completed["error"]
    assert completed["duration_seconds"] == 1.25


@pytest.mark.asyncio
async def test_serial_tool_events_follow_the_real_execution_boundaries():
    trace = []

    async def first() -> str:
        trace.append("body:first")
        return "one"

    async def second() -> str:
        trace.append("body:second")
        return "two"

    calls = [
        FunctionCall(function=Function.from_callable(first), call_id="first"),
        FunctionCall(function=Function.from_callable(second), call_id="second"),
    ]
    runner = Runner(SimpleNamespace())
    runner._emit_event = lambda event, payload: trace.append(
        f"{event.value}:{payload['tool_call_id']}"
    )
    async for _ in runner._execute_tool_calls(
        calls, [], SimpleNamespace(), _ExecutionModel(id="test")
    ):
        pass

    assert trace == [
        "tool.started:first",
        "body:first",
        "tool.completed:first",
        "tool.started:second",
        "body:second",
        "tool.completed:second",
    ]


@pytest.mark.asyncio
async def test_parallel_tool_completion_is_emitted_as_each_call_finishes():
    trace = []

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "slow"

    async def fast() -> str:
        await asyncio.sleep(0)
        return "fast"

    slow_function = Function.from_callable(slow)
    fast_function = Function.from_callable(fast)
    slow_function.concurrency_safe = True
    fast_function.concurrency_safe = True
    calls = [
        FunctionCall(function=slow_function, call_id="slow"),
        FunctionCall(function=fast_function, call_id="fast"),
    ]
    runner = Runner(SimpleNamespace())
    runner._emit_event = lambda event, payload: trace.append(
        f"{event.value}:{payload['tool_call_id']}"
    )
    async for _ in runner._execute_tool_calls(
        calls, [], SimpleNamespace(), _ExecutionModel(id="test")
    ):
        pass

    assert trace.index("tool.completed:fast") < trace.index("tool.completed:slow")


@pytest.mark.asyncio
async def test_cancelled_running_tool_still_emits_completion():
    trace = []

    async def cancelled() -> str:
        raise asyncio.CancelledError

    function = Function.from_callable(cancelled)
    function.concurrency_safe = True
    call = FunctionCall(function=function, call_id="cancelled")
    runner = Runner(SimpleNamespace())
    runner._emit_event = lambda event, payload: trace.append(
        f"{event.value}:{payload['tool_call_id']}"
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in runner._execute_tool_calls(
            [call], [], SimpleNamespace(), _ExecutionModel(id="test")
        ):
            pass

    assert trace == [
        "tool.started:cancelled",
        "tool.completed:cancelled",
    ]
