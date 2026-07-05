# -*- coding: utf-8 -*-
"""Tests for agentica.subagent registry execution helpers."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentica.subagent import (
    SubagentConfig,
    SubagentRegistry,
    SubagentType,
    _CUSTOM_SUBAGENT_CONFIGS,
)
from agentica.tools.base import Function, Tool


class FakeToolConfig:
    def __init__(self, tool_call_limit=None):
        self.tool_call_limit = tool_call_limit


class RecordingAgent:
    """Minimal Agent stand-in.

    The new ``SubagentRegistry.spawn`` drives the child via ``run_stream`` and
    yields a single ``RunResponse``-shaped chunk to capture the final content.
    The shim also exposes ``model.usage.merge`` so the registry's usage
    aggregation step is exercised without instantiating a real model.
    """

    last_init_kwargs = None
    run_delay = 0.0

    def __init__(self, **kwargs):
        RecordingAgent.last_init_kwargs = kwargs
        self.name = kwargs.get("name", "child")
        # Preserve the cloned model passed by ``SubagentRegistry.spawn`` so the
        # registry's ``parent.usage.merge(child.usage)`` step gets a real
        # ``Usage`` instance.
        self.model = kwargs.get("model")

    async def run_stream(self, task, config=None):
        await asyncio.sleep(self.run_delay)
        yield SimpleNamespace(event="RunResponse", content=f"done:{task}", tools=None)


class _FakeModel:
    """Stand-in for ``Model`` so ``copy.copy`` clones cleanly during tests."""

    def __init__(self):
        self.tools = None
        self.functions = None
        self.function_call_stack = None
        self.tool_choice = None
        self.metrics = {}
        from agentica.model.usage import Usage
        self.usage = Usage()


def _make_parent_agent():
    working_memory = SimpleNamespace(summary=SimpleNamespace(summary="parent summary"))
    return SimpleNamespace(
        name="parent",
        agent_id="parent-agent-id",
        instructions=None,
        model=_FakeModel(),
        tools=[
            Function(name="read_file", entrypoint=lambda: None),
            Function(name="write_file", entrypoint=lambda: None),
            Function(name="task", entrypoint=lambda: None),
        ],
        workspace="workspace-ref",
        knowledge="knowledge-ref",
        working_memory=working_memory,
        context={},
        _event_callback=None,
        # arch_v5.md Phase 0 lineage fields read by SubagentRegistry.spawn():
        # subagent.spawn() expects parent_agent to expose Agent's RunContext
        # surface (see agentica/agent/base.py). None mirrors the real default
        # when no run is in flight.
        run_context=None,
        run_id=None,
    )


def _make_toolkit():
    def read_file():
        return None

    def write_file():
        return None

    def task():
        return None

    toolkit = Tool(name="file_tools")
    toolkit.register(read_file)
    toolkit.register(write_file)
    toolkit.register(task)
    return toolkit


@pytest.fixture(autouse=True)
def reset_subagent_registry():
    SubagentRegistry._instance = None
    _CUSTOM_SUBAGENT_CONFIGS.clear()
    RecordingAgent.last_init_kwargs = None
    RecordingAgent.run_delay = 0.0
    yield
    SubagentRegistry._instance = None
    _CUSTOM_SUBAGENT_CONFIGS.clear()


def test_spawn_applies_config_to_child_agent():
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["reviewer"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="reviewer",
        description="reviewer",
        system_prompt="system prompt",
        allowed_tools=["read_file", "task"],
        denied_tools=["task"],
        tool_call_limit=7,
        can_spawn_subagents=False,
        inherit_workspace=True,
        inherit_knowledge=True,
        inherit_context=True,
        timeout=5,
    )

    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", RecordingAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="review this", agent_type="reviewer")
        )

    assert result["status"] == "completed"
    init_kwargs = RecordingAgent.last_init_kwargs
    assert init_kwargs is not None
    assert "parent summary" in init_kwargs["instructions"]
    assert [tool.name for tool in init_kwargs["tools"]] == ["read_file"]
    assert init_kwargs["workspace"] == "workspace-ref"
    assert init_kwargs["knowledge"] == "knowledge-ref"
    assert init_kwargs["tool_config"].tool_call_limit == 7
    assert init_kwargs["context"]["_subagent_depth"] == 1
    assert init_kwargs["context"]["_can_spawn_subagents"] is False


def test_spawn_filters_toolkit_functions_by_allowed_and_denied_lists():
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["reviewer"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="reviewer",
        description="reviewer",
        system_prompt="system prompt",
        allowed_tools=["read_file", "task"],
        denied_tools=["task"],
    )
    parent = _make_parent_agent()
    parent.tools = [_make_toolkit()]

    with patch("agentica.agent.Agent", RecordingAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="review this", agent_type="reviewer")
        )

    assert result["status"] == "completed"
    init_kwargs = RecordingAgent.last_init_kwargs
    assert init_kwargs is not None
    assert len(init_kwargs["tools"]) == 1
    toolkit = init_kwargs["tools"][0]
    assert isinstance(toolkit, Tool)
    assert list(toolkit.functions.keys()) == ["read_file"]


def test_spawn_honors_timeout():
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["slow"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="slow",
        description="slow",
        system_prompt="system prompt",
        timeout=0.01,
    )
    parent = _make_parent_agent()
    RecordingAgent.run_delay = 0.05

    with patch("agentica.agent.Agent", RecordingAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="slow task", agent_type="slow")
        )

    assert result["status"] == "timeout"
    assert "timed out" in result["error"].lower()
    # Even with no partial output produced, timeout must include the
    # standard payload keys so callers can render the result uniformly.
    assert result["partial"] is True
    assert "content" in result
    assert "tool_calls_summary" in result


class _CumulativeToolStreamAgent:
    """Mimics ``Agent.run_stream`` producing the cumulative ``chunk.tools`` list
    that ``Runner`` actually emits — every ToolCall* event includes ALL tool
    calls so far in the run, not just the newly-affected one. The registry is
    expected to dedupe by ``tool_call_id``.
    """

    last_init_kwargs = None

    def __init__(self, **kwargs):
        _CumulativeToolStreamAgent.last_init_kwargs = kwargs
        self.name = kwargs.get("name", "child")
        self.model = kwargs.get("model")

    async def run_stream(self, task, config=None):
        # Simulate two tool calls (read_file + ls), each going through
        # started -> completed, with the cumulative list growing each chunk.
        t1 = {"id": "call_1", "tool_name": "read_file",
              "tool_args": {"file_path": "a.py"}}
        t2 = {"id": "call_2", "tool_name": "ls",
              "tool_args": {"directory": "."}}
        # call_1 started
        yield SimpleNamespace(event="ToolCallStarted", content=None, tools=[dict(t1)])
        # call_1 completed (cumulative list still has call_1, now with content)
        t1_done = {**t1, "content": "file content"}
        yield SimpleNamespace(event="ToolCallCompleted", content=None, tools=[t1_done])
        # call_2 started — chunk.tools now has both
        yield SimpleNamespace(event="ToolCallStarted", content=None,
                              tools=[t1_done, dict(t2)])
        # call_2 completed — chunk.tools still has both, both with content
        t2_done = {**t2, "content": "listing"}
        yield SimpleNamespace(event="ToolCallCompleted", content=None,
                              tools=[t1_done, t2_done])
        yield SimpleNamespace(event="RunResponse", content=f"done:{task}", tools=None)


def test_spawn_dedupes_subagent_tool_events_by_call_id():
    """Regression: cumulative chunk.tools must not cause duplicate
    subagent.tool_started/completed events or inflate tool_count."""
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["coder"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="coder",
        description="coder",
        system_prompt="prompt",
    )
    parent = _make_parent_agent()
    received = []
    parent._event_callback = received.append

    with patch("agentica.agent.Agent", _CumulativeToolStreamAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="do work", agent_type="coder")
        )

    assert result["status"] == "completed"
    assert result["tool_count"] == 2, (
        f"Expected exactly 2 tool calls, got {result['tool_count']}. "
        "Cumulative chunk.tools must be deduped by tool_call_id."
    )
    started = [e for e in received if e["type"] == "subagent.tool_started"]
    completed = [e for e in received if e["type"] == "subagent.tool_completed"]
    assert len(started) == 2
    assert len(completed) == 2
    assert [e["tool_name"] for e in started] == ["read_file", "ls"]
    assert [e["tool_name"] for e in completed] == ["read_file", "ls"]


def test_spawn_batch_returns_error_for_invalid_spec_and_keeps_order():
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["reviewer"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="reviewer",
        description="reviewer",
        system_prompt="system prompt",
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", RecordingAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        results = asyncio.run(
            registry.spawn_batch(
                parent_agent=parent,
                tasks=[
                    {"type": "reviewer"},
                    {"task": "valid task", "type": "reviewer"},
                ],
            )
        )

    assert len(results) == 2
    assert results[0]["status"] == "error"
    assert "task" in results[0]["error"].lower()
    assert results[1]["status"] == "completed"
    assert results[1]["content"] == "done:valid task"


# ---------------------------------------------------------------------------
# Regression tests for the "task tool keeps failing" bug family.
#
# Root cause (pre-fix): ``_run_child_streaming`` accumulated ``final_content``
# into a local variable, so when ``asyncio.wait_for`` cancelled it on timeout
# (or when the user Ctrl+C'd, or an exception bubbled), every byte of
# streamed content and every completed tool call was silently discarded.
# The parent Agent then saw ``content=""`` and reported "task failed".
#
# Fix: spawn() injects a mutable ``partial_sink`` dict which _run_child_streaming
# mirrors on every chunk. On timeout / exception / cancel, spawn() reads the
# sink and returns partial output with a truthful ``status`` and ``partial=True``.
# ---------------------------------------------------------------------------


class _PartialThenHangAgent(RecordingAgent):
    """Yields two content chunks + one completed tool call, then hangs forever.

    Simulates a real subagent that did meaningful work before a timeout fires.
    """

    async def run_stream(self, task, config=None):
        yield SimpleNamespace(event="RunResponse", content="first-half. ", tools=None)
        yield SimpleNamespace(
            event="ToolCallStarted", content=None,
            tools=[{"tool_call_id": "tc-1", "tool_name": "read_file",
                    "tool_args": {"file_path": "a.py"}}],
        )
        yield SimpleNamespace(
            event="ToolCallCompleted", content=None,
            tools=[{"tool_call_id": "tc-1", "tool_name": "read_file",
                    "tool_args": {"file_path": "a.py"}, "content": "content of a.py"}],
        )
        yield SimpleNamespace(event="RunResponse", content="second-half.", tools=None)
        # Hang so wait_for cancels us mid-flight.
        await asyncio.sleep(10)


def test_spawn_timeout_preserves_partial_content_and_tool_calls():
    """Bug: pre-fix, timeout returned content='' and threw away tool calls."""
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["slowish"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="slowish",
        description="slow",
        system_prompt="s",
        timeout=1,  # cut short after 1s
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _PartialThenHangAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="do slow work", agent_type="slowish")
        )

    assert result["status"] == "timeout"
    assert result["partial"] is True
    # Both content chunks must survive the timeout.
    assert "first-half" in result["content"]
    assert "second-half" in result["content"]
    # Note in the header must be present so the parent LLM knows why.
    assert "timed out" in result["content"].lower()
    # Completed tool call must be preserved.
    assert result["tool_count"] == 1
    assert result["tool_calls_summary"][0]["name"] == "read_file"


class _MaxTurnsBreakAgent(RecordingAgent):
    """Simulates Runner reaching max_turns: returns content + break_reason set."""

    async def run_stream(self, task, config=None):
        yield SimpleNamespace(event="RunResponse", content="partial answer before limit", tools=None)
        # Runner sets ``self.run_response.break_reason = "MAX_TURNS"`` when the
        # turn budget is exhausted.
        self.run_response = SimpleNamespace(break_reason="MAX_TURNS")


def test_spawn_max_turns_returns_truncated_status_with_content():
    """Bug: pre-fix, max_turns silently returned ``status=completed`` with the
    partial content — indistinguishable from a real answer."""
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["maxturner"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="maxturner",
        description="mt",
        system_prompt="s",
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _MaxTurnsBreakAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="huge task", agent_type="maxturner")
        )

    assert result["status"] == "max_turns"
    assert result["partial"] is True
    assert "partial answer before limit" in result["content"]
    assert "max_turns" in result["content"].lower() or "limit" in result["content"].lower()


class _ToolCallLimitBreakAgent(RecordingAgent):
    async def run_stream(self, task, config=None):
        yield SimpleNamespace(event="RunResponse", content="work in progress", tools=None)
        self.run_response = SimpleNamespace(break_reason="TOOL_CALL_LIMIT")


def test_spawn_tool_call_limit_returns_dedicated_status():
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["tcl"] = SubagentConfig(
        type=SubagentType.CUSTOM, name="tcl", description="tcl", system_prompt="s",
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _ToolCallLimitBreakAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="t", agent_type="tcl")
        )

    assert result["status"] == "tool_call_limit"
    assert result["partial"] is True
    assert "work in progress" in result["content"]


class _PartialThenRaiseAgent(RecordingAgent):
    async def run_stream(self, task, config=None):
        yield SimpleNamespace(event="RunResponse", content="got this far ", tools=None)
        raise RuntimeError("model API blew up")


def test_spawn_exception_returns_partial_not_empty():
    """Bug: pre-fix, any mid-stream exception returned content=''."""
    registry = SubagentRegistry()
    _CUSTOM_SUBAGENT_CONFIGS["boomer"] = SubagentConfig(
        type=SubagentType.CUSTOM, name="boomer", description="b", system_prompt="s",
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _PartialThenRaiseAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(parent_agent=parent, task="t", agent_type="boomer")
        )

    assert result["status"] == "error"
    assert result["partial"] is True
    assert "got this far" in result["content"]
    assert "model API blew up" in result["error"]


def test_default_timeout_is_generous():
    """Guard against regression: default timeout must be big enough that real
    code/research tasks don't hit it just from network latency + tool loops."""
    cfg = SubagentConfig(
        type=SubagentType.CUSTOM, name="x", description="x", system_prompt="s",
    )
    # 30 min. If someone lowers this again, this test fails loudly.
    assert cfg.timeout >= 1800


def test_builtin_task_tool_surfaces_partial_on_timeout():
    """BuiltinTaskTool must forward partial content when the subagent hits a
    budget limit — not just drop it with a bare error string."""
    import json
    from agentica.tools.builtin_task_tool import BuiltinTaskTool

    tool = BuiltinTaskTool()

    class _FakeRegistry:
        async def spawn(self, **kwargs):
            return {
                "status": "timeout",
                "error": "Subagent timed out after 1 seconds",
                "agent_type": "code",
                "subagent_name": "code",
                "run_id": "r-1",
                "content": "[timed out]\n\npartial work done",
                "tool_calls_summary": [{"name": "read_file", "info": "a.py"}],
                "tool_count": 1,
                "elapsed_seconds": 1.0,
                "partial": True,
            }

    parent = SimpleNamespace(
        _event_callback=None,
        run_context=None,
        run_id=None,
    )
    tool._parent_agent = parent

    with patch("agentica.subagent.SubagentRegistry", return_value=_FakeRegistry()):
        raw = asyncio.run(tool.task(description="d", subagent_type="code"))
    payload = json.loads(raw)
    assert payload["success"] is False
    assert payload["status"] == "timeout"
    assert payload["partial"] is True
    assert "partial work done" in payload["result"]
    assert payload["tool_calls_summary"][0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# ReAct feedback loop: the parent Agent must be able to READ the failure
# reason from a task result and RETRY with adjusted params (timeout / prompt /
# resume_from_run_id) — without having to restart the task from zero.
# ---------------------------------------------------------------------------


def test_spawn_timeout_override_wins_over_config_default():
    """Parent Agent must be able to pass ``timeout_override`` to extend the
    budget for a specific retry, without re-registering the subagent type."""
    _CUSTOM_SUBAGENT_CONFIGS["overridable"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="overridable",
        description="o",
        system_prompt="s",
        timeout=1,  # tiny default
    )
    seen_timeout = {}

    class _RecordingAgent(RecordingAgent):
        async def run_stream(self, task, config=None):
            # Grab whatever ``timeout`` the registry ended up applying by
            # sleeping just under it and finishing normally.
            yield SimpleNamespace(event="RunResponse", content="ok", tools=None)

    registry = SubagentRegistry()
    parent = _make_parent_agent()

    # Without override -> uses config's timeout=1. Verify by tracking.
    orig_wait_for = asyncio.wait_for

    async def _spy_wait_for(coro, timeout):
        seen_timeout["value"] = timeout
        return await orig_wait_for(coro, timeout)

    with patch("agentica.agent.Agent", _RecordingAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ), patch("agentica.subagent.asyncio.wait_for", side_effect=_spy_wait_for):
        asyncio.run(
            registry.spawn(
                parent_agent=parent, task="t", agent_type="overridable",
                timeout_override=42,
            )
        )
    assert seen_timeout["value"] == 42, (
        "spawn should have used the override, not the SubagentConfig default"
    )


def test_spawn_resume_stitches_previous_partial_into_task():
    """Parent Agent passes ``resume_from_run_id`` from a prior partial run;
    spawn() must prepend the previous partial output to the new task so the
    subagent continues instead of restarting."""
    _CUSTOM_SUBAGENT_CONFIGS["resumable"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="resumable",
        description="r",
        system_prompt="s",
    )
    seen_task_text = {}

    class _CaptureTaskAgent(RecordingAgent):
        async def run_stream(self, task, config=None):
            seen_task_text["value"] = task
            yield SimpleNamespace(event="RunResponse", content="done", tools=None)

    registry = SubagentRegistry()
    # Seed a fake prior run in the singleton registry with partial output.
    from agentica.subagent import SubagentRun
    from datetime import datetime
    prior_run_id = "prior-run-id-xyz"
    registry._runs[prior_run_id] = SubagentRun(
        run_id=prior_run_id,
        subagent_type=SubagentType.CUSTOM,
        parent_agent_id="parent-1",
        task_label="original task",
        task_description="original task",
        started_at=datetime.now(),
        status="timeout",
        result="I already analyzed files A.py and B.py. Findings so far: ...",
    )
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _CaptureTaskAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(
                parent_agent=parent,
                task="Continue: now analyze C.py.",
                agent_type="resumable",
                resume_from_run_id=prior_run_id,
            )
        )

    assert result["status"] == "completed"
    # The subagent must have received the stitched prompt.
    stitched = seen_task_text["value"]
    assert "[RESUME]" in stitched
    assert "A.py and B.py" in stitched, "previous partial output must be injected"
    assert "Continue: now analyze C.py." in stitched, "new task must still be present"
    assert prior_run_id in stitched


def test_spawn_resume_unknown_run_id_returns_error():
    _CUSTOM_SUBAGENT_CONFIGS["resumable2"] = SubagentConfig(
        type=SubagentType.CUSTOM, name="resumable2", description="r", system_prompt="s",
    )
    registry = SubagentRegistry()
    parent = _make_parent_agent()

    result = asyncio.run(
        registry.spawn(
            parent_agent=parent, task="t", agent_type="resumable2",
            resume_from_run_id="does-not-exist",
        )
    )
    assert result["status"] == "error"
    assert "does-not-exist" in result["error"]


def test_partial_payload_carries_next_action_hint_and_run_id():
    """The ``next_action`` string is the single most important field for the
    ReAct loop — it tells the parent Agent literally which arguments to pass
    on retry. Without it, the model has to guess."""
    _CUSTOM_SUBAGENT_CONFIGS["hinter"] = SubagentConfig(
        type=SubagentType.CUSTOM,
        name="hinter",
        description="h",
        system_prompt="s",
    )

    class _HangAgent(RecordingAgent):
        async def run_stream(self, task, config=None):
            yield SimpleNamespace(event="RunResponse", content="progress made", tools=None)
            await asyncio.sleep(5)

    registry = SubagentRegistry()
    parent = _make_parent_agent()

    with patch("agentica.agent.Agent", _HangAgent), patch(
        "agentica.agent.config.ToolConfig", FakeToolConfig
    ):
        result = asyncio.run(
            registry.spawn(
                parent_agent=parent, task="t", agent_type="hinter",
                timeout_override=1,
            )
        )

    assert result["status"] == "timeout"
    hint = result.get("next_action") or ""
    assert result["run_id"] in hint, "hint must reference the run_id to resume"
    assert "resume_from_run_id" in hint
    assert "timeout" in hint
    # Larger timeout suggestion must be strictly greater than the failed one.
    assert "2" in hint  # 1s * 2 = 2s
