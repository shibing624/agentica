# -*- coding: utf-8 -*-
"""Tests for dsh-inspired prefix-cache invariants and cross-wire cache alignment."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import openai
import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from agentica import Agent, OpenAIChat, RunConfig
from agentica.agent.config import PromptConfig, ToolConfig
from agentica.compression.manager import CompressionManager
from agentica.memory.session_log import SessionLog
from agentica.memory.trace import analyze_entries
from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message, VOLATILE_SYSTEM_MARKER
from agentica.model.metrics import Metrics
from agentica.model.openai.responses import OpenAIResponses
from agentica.runner.persist import PersistMixin
from agentica.tools.base import Tool
from agentica.workspace import Workspace


def _fake_chat_completion(text: str = "ok") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        created=1,
        model="test-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        usage=CompletionUsage(prompt_tokens=100, completion_tokens=5, total_tokens=105),
    )


class _SpyOpenAIChat(OpenAIChat):
    def __init__(self, **kwargs: Any):
        super().__init__(api_key="fake_openai_key", **kwargs)
        self.captured: List[tuple[List[Dict[str, Any]], Dict[str, Any]]] = []

    async def invoke(self, messages: List[Message]) -> ChatCompletion:
        formatted = self.format_messages(messages)
        formatted, kw = self._apply_cache_control(formatted, self.request_kwargs)
        self.captured.append((formatted, kw))
        return _fake_chat_completion("ack")


# ── 1. OpenAIChat auto cache_control + sticky routing when cache_control=False ─


def test_openai_chat_auto_enables_cache_control_for_claude_models():
    claude_proxy = OpenAIChat(id="claude-haiku-5-5", api_key="fake_key")
    assert claude_proxy.enable_cache_control is True

    explicit_off = OpenAIChat(id="claude-haiku-5-5", api_key="fake_key", enable_cache_control=False)
    assert explicit_off.enable_cache_control is False

    deepseek = OpenAIChat(id="DeepSeek-V4.1-Flash", api_key="fake_key")
    assert deepseek.enable_cache_control is False


def test_sticky_header_still_applied_when_cache_control_is_off():
    model = OpenAIChat(
        id="DeepSeek-V4.1-Flash",
        api_key="fake_key",
        enable_cache_control=False,
        cache_control_session_header="X-Session-Id",
    )
    model.session_id = "sess-ds-1"
    msgs = [{"role": "system", "content": f"head\n{VOLATILE_SYSTEM_MARKER}\ntail"}, {"role": "user", "content": "hi"}]
    out, kw = model._apply_cache_control(msgs, {})
    assert VOLATILE_SYSTEM_MARKER not in out[0]["content"]
    assert isinstance(out[0]["content"], str)
    assert kw["extra_headers"]["X-Session-Id"] == "sess-ds-1"


def test_venus_proxy_gets_default_sticky_session_header():
    model = OpenAIChat(
        id="claude-haiku-5-5",
        api_key="fake_key",
        base_url="http://v2.open.venus.oa.com/llmproxy/v1",
    )
    model.session_id = "sess-venus-42"
    _, kw = model._apply_cache_control([{"role": "user", "content": "hi"}], {})
    assert kw["extra_headers"]["Venus-Session-Id"] == "sess-venus-42"


def test_unsupported_prompt_cache_key_falls_back_and_remembers():
    model = OpenAIChat(id="strict-model", api_key="fake_key", base_url="https://strict.example/v1")
    model.session_id = "sess-1"
    calls: List[Dict[str, Any]] = []

    async def fake_create(kwargs: Dict[str, Any]) -> str:
        calls.append(dict(kwargs))
        if "prompt_cache_key" in kwargs:
            req = httpx.Request("POST", "https://strict.example/v1/chat/completions")
            resp = httpx.Response(
                400,
                request=req,
                json={"error": {"message": "Unknown parameter", "code": "unsupported_parameter", "param": "prompt_cache_key"}},
            )
            raise openai.BadRequestError("Unknown parameter", response=resp, body=resp.json())
        return "ok"

    try:
        res1 = asyncio.run(model._create_with_auto_cap(fake_create, model.request_kwargs))
        assert res1 == "ok"
        assert len(calls) == 2
        assert "prompt_cache_key" in calls[0]
        assert "prompt_cache_key" not in calls[1]

        # Second call on the same deployment skips prompt_cache_key immediately.
        calls.clear()
        res2 = asyncio.run(model._create_with_auto_cap(fake_create, model.request_kwargs))
        assert res2 == "ok"
        assert len(calls) == 1
        assert "prompt_cache_key" not in calls[0]
    finally:
        OpenAIChat._UNSUPPORTED_PROMPT_CACHE_KEY_DEPLOYMENTS.discard(model._output_cap_deployment_key())


# ── 2. OpenAIResponses prompt_cache_key + sticky header + marker strip ────────


def test_openai_responses_carries_prompt_cache_key_sticky_header_and_strips_marker():
    model = OpenAIResponses(
        id="gpt-5.4-mini",
        api_key="fake_key",
        base_url="https://v2.open.venus.woa.com/llmproxy/v1",
        reasoning="low",
    )
    model.session_id = "sess-resp-9"
    assert model.enable_cache_control is False

    kw = model.request_kwargs
    assert kw["prompt_cache_key"] == "sess-resp-9"
    assert kw["extra_headers"]["Venus-Session-Id"] == "sess-resp-9"

    formatted = model.format_messages([
        Message(role="system", content=f"S-HEAD\n{VOLATILE_SYSTEM_MARKER}\nS-TAIL"),
        Message(role="user", content="hello"),
    ])
    assert VOLATILE_SYSTEM_MARKER not in formatted[0]["content"]
    assert formatted[0]["content"] == "S-HEAD\n\nS-TAIL"


# ── 3. Native Anthropic Claude: tools breakpoint, 4-block budget, metrics, stream fallback ──


def _count_claude_breakpoints(chat_msgs: List[Dict[str, Any]], request_kwargs: Dict[str, Any]) -> int:
    total = 0
    for t in request_kwargs.get("tools") or []:
        if isinstance(t, dict) and "cache_control" in t:
            total += 1
    sys_val = request_kwargs.get("system")
    if isinstance(sys_val, list):
        total += sum(1 for b in sys_val if isinstance(b, dict) and "cache_control" in b)
    for m in chat_msgs:
        content = m.get("content")
        if isinstance(content, list):
            total += sum(1 for b in content if isinstance(b, dict) and "cache_control" in b)
    return total


def test_claude_tags_last_tool_and_caps_total_breakpoints_at_four():
    def alpha(x: str) -> str:
        """Tool alpha."""
        return x

    def beta(y: int) -> str:
        """Tool beta."""
        return str(y)

    model = Claude(id="claude-haiku-5-5", api_key="fake_anthropic_key")
    model.add_tool(alpha)
    model.add_tool(beta)

    msgs = [
        Message(role="system", content=f"Stable head\n{VOLATILE_SYSTEM_MARKER}\nVolatile tail"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u2"),
        Message(role="assistant", content="a2"),
        Message(role="user", content="u3"),
    ]
    chat_msgs, sys_str = asyncio.run(model.format_messages(msgs))
    kw = model.prepare_request_kwargs(sys_str)

    assert "cache_control" not in kw["tools"][0]
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert _count_claude_breakpoints(chat_msgs, kw) == 4


def test_claude_in_history_system_message_does_not_mutate_top_level_system():
    model = Claude(id="claude-haiku-5-5", api_key="fake_anthropic_key")
    msgs = [
        Message(role="system", content="Immutable node 0"),
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="system", content="Mid-session policy addition"),
        Message(role="user", content="second question"),
    ]
    chat_msgs, sys_str = asyncio.run(model.format_messages(msgs))
    assert sys_str == "Immutable node 0"
    assert [m["role"] for m in chat_msgs] == ["user", "assistant", "user"]
    assert any(
        isinstance(b, dict) and b.get("text") == "<system-reminder>\nMid-session policy addition\n</system-reminder>"
        for m in chat_msgs
        for b in m["content"]
    )


def test_claude_update_usage_metrics_populates_message_prompt_details(tmp_path):
    model = Claude(id="claude-haiku-5-5", api_key="fake_anthropic_key")
    usage = SimpleNamespace(
        input_tokens=4,
        output_tokens=50,
        cache_read_input_tokens=15000,
        cache_creation_input_tokens=500,
    )
    metrics = Metrics()
    asst = Message(role="assistant", content="answer")
    model.update_usage_metrics(asst, usage, metrics)

    assert asst.metrics["prompt_tokens"] == 15504
    assert asst.metrics["prompt_tokens_details"] == {
        "cached_tokens": 15000,
        "cache_read_tokens": 15000,
        "cache_creation_tokens": 500,
    }

    slog = SessionLog("s-claude-metrics", base_dir=str(tmp_path))
    dummy_agent = SimpleNamespace(_session_log=slog)
    PersistMixin._trace_token_usage(dummy_agent, [asst])
    events = [e for e in slog.iter_raw_entries() if e.get("type") == "event" and e.get("name") == "token_usage"]
    assert events[0]["request"] == {
        "input": 4,
        "cache_read": 15000,
        "cache_write": 500,
        "output": 50,
        "total": 15554,
    }


def test_claude_invoke_falls_back_to_stream_when_sdk_requires_streaming():
    model = Claude(id="claude-haiku-5-5", api_key="fake_anthropic_key")
    expected_msg = MagicMock(name="AnthropicMessage")

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get_final_message(self):
            return expected_msg

    fake_messages = MagicMock()
    fake_messages.create.side_effect = ValueError(
        "Streaming is required for operations that may take longer than 10 minutes."
    )
    fake_messages.stream.return_value = _FakeStreamCtx()
    model.client = MagicMock(messages=fake_messages)

    result = asyncio.run(model.invoke([Message(role="user", content="hi")]))
    assert result is expected_msg
    fake_messages.stream.assert_called_once()


# ── 4. dsh-inspired session prefix pinning, in-history reminders, tool order & resume ──


def test_system_prompt_stays_byte_pinned_across_date_and_channel_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path))
    spy = _SpyOpenAIChat(id="DeepSeek-V4.1-Flash")
    agent = Agent(
        name="CacheAgent",
        model=spy,
        add_history_to_context=True,
        prompt_config=PromptConfig(add_datetime_to_instructions=True),
        session_id="sess-pin-1",
        session_base_dir=str(tmp_path / "sessions"),
    )

    with patch("agentica.agent.prompts.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 11, 14, 2, 0, 0)
        asyncio.run(agent.run("Turn 1 question"))

        # Next day + switch to feishu IM channel: messages[0] MUST NOT change!
        mock_dt.now.return_value = datetime(2026, 11, 15, 9, 0, 0)
        asyncio.run(agent.run("Turn 2 question", config=RunConfig(channel="feishu")))

        # Same day + same feishu channel on Turn 3: no duplicate reminder emitted!
        asyncio.run(agent.run("Turn 3 question", config=RunConfig(channel="feishu")))

    m1, _ = spy.captured[0]
    m2, _ = spy.captured[1]
    m3, _ = spy.captured[2]

    assert m1[0] == m2[0] == m3[0], "messages[0] system prompt mutated mid-window!"
    assert "2026-11-14" in m2[0]["content"]
    # Turn 2 user message carries the in-history <system-reminder>
    assert "<system-reminder>" in m2[-1]["content"]
    assert "2026-11-15" in m2[-1]["content"]
    assert "feishu" in m2[-1]["content"]
    # Turn 3 user message has NO duplicate <system-reminder> since state did not change again
    assert "<system-reminder>" not in m3[-1]["content"]
    assert m3[-1]["content"] == "Turn 3 question"


def test_mid_session_tool_addition_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path))

    class _DynamicToolkit(Tool):
        def __init__(self):
            super().__init__(name="dyn")
            self.register(self.first_tool)
            self.register(self.second_tool)

        def first_tool(self, a: str) -> str:
            """First."""
            return a

        def second_tool(self, b: str) -> str:
            """Second."""
            return b

    def prepended_later_tool(c: str) -> str:
        """Added mid-session at the front of agent.tools."""
        return c

    spy = _SpyOpenAIChat(id="DeepSeek-V4.1-Flash")
    tk = _DynamicToolkit()
    agent = Agent(
        model=spy,
        tools=[tk],
        add_history_to_context=True,
        tool_config=ToolConfig(auto_load_mcp=False),
    )

    asyncio.run(agent.run("step 1"))
    names_turn1 = [t["function"]["name"] for t in spy.captured[0][1]["tools"]]

    # Caller prepends a new tool at index 0 of agent.tools before turn 2:
    # _stabilize_model_tool_order must keep existing tools in place and append the new one last.
    agent.tools = [prepended_later_tool, *agent.tools]
    asyncio.run(agent.run("step 2"))
    names_turn2 = [t["function"]["name"] for t in spy.captured[1][1]["tools"]]

    assert names_turn2[: len(names_turn1)] == names_turn1
    assert names_turn2[-1] == "prepended_later_tool"


def test_session_resume_recovers_exact_system_prompt_tools_and_wire_user_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path))
    ws_path = tmp_path / "ws"
    ws1 = Workspace(str(ws_path))
    ws1.initialize()
    (ws1.path / "MEMORY.md").write_text("# Memory\n- Rule v1\n", encoding="utf-8")

    sess_dir = tmp_path / "sessions"
    spy1 = _SpyOpenAIChat(id="DeepSeek-V4.1-Flash")
    a1 = Agent(
        name="ResumeAgent",
        model=spy1,
        workspace=ws1,
        enable_long_term_memory=True,
        add_history_to_context=True,
        work_dir=str(Path(__file__).resolve().parents[2]),
        prompt_config=PromptConfig(add_datetime_to_instructions=True, add_git_snapshot=True),
        tool_config=ToolConfig(auto_load_mcp=False),
        session_id="sess-resume-exact",
        session_base_dir=str(sess_dir),
    )
    asyncio.run(a1.run("initial task"))
    asyncio.run(a1.run("follow up"))

    # Disk memory changes before a brand-new Agent process resumes the session:
    (ws1.path / "MEMORY.md").write_text("# Memory\n- Rule v2 changed on disk!\n", encoding="utf-8")
    ws2 = Workspace(str(ws_path))
    spy2 = _SpyOpenAIChat(id="DeepSeek-V4.1-Flash")
    a2 = Agent(
        name="ResumeAgent",
        model=spy2,
        workspace=ws2,
        enable_long_term_memory=True,
        add_history_to_context=True,
        work_dir=str(Path(__file__).resolve().parents[2]),
        prompt_config=PromptConfig(add_datetime_to_instructions=True, add_git_snapshot=True),
        tool_config=ToolConfig(auto_load_mcp=False),
        session_id="sess-resume-exact",
        session_base_dir=str(sess_dir),
    )
    asyncio.run(a2.run("resumed turn"))

    turn2_msgs, turn2_kw = spy1.captured[1]
    resumed_msgs, resumed_kw = spy2.captured[0]

    # Tools schema and every historical message (system + turn 1 with <git_snapshot> + turn 2) match byte-for-byte!
    assert resumed_kw["tools"] == turn2_kw["tools"]
    assert resumed_msgs[: len(turn2_msgs)] == turn2_msgs


def test_auto_compact_resets_window_pins_so_new_window_rebaselines(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path))
    spy = _SpyOpenAIChat(id="DeepSeek-V4.1-Flash")
    cm = CompressionManager()
    agent = Agent(
        model=spy,
        add_history_to_context=True,
        prompt_config=PromptConfig(add_datetime_to_instructions=True),
        tool_config=ToolConfig(compression_manager=cm, auto_load_mcp=False),
    )

    with patch("agentica.agent.prompts.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 11, 14, 1, 0, 0)
        asyncio.run(agent.run("turn 1"))
        assert "2026-11-14" in (agent._pinned_system_prompt or "")

        # Force a Layer 2 window cut; pins must clear so the new window re-baselines node 0.
        agent.update_model()
        msgs = [
            Message(role="system", content=agent._pinned_system_prompt),
            Message(role="user", content="turn 1"),
            Message(role="assistant", content="ack"),
            Message(role="user", content="turn 2"),
        ]
        compacted = asyncio.run(cm.auto_compact(msgs, model=spy, force=True))
        assert compacted is True
        assert agent._pinned_system_prompt is None
        agent.working_memory.collapse_runs(msgs)

        mock_dt.now.return_value = datetime(2026, 11, 16, 1, 0, 0)
        asyncio.run(agent.run("turn after compact"))
        assert "2026-11-16" in spy.captured[-1][0][0]["content"]


def test_trace_records_prefix_reuse_ratio_and_detects_cache_cliffs(tmp_path):
    slog = SessionLog("s-cliff-trace", base_dir=str(tmp_path))
    dummy_agent = SimpleNamespace(_session_log=slog)

    # Step 1: cold start 4000 tokens
    slog.append_event("request_begin")
    m1 = Message(role="assistant", content="a1", metrics={"prompt_tokens": 4000, "output_tokens": 100})
    PersistMixin._trace_token_usage(dummy_agent, [m1])
    slog.append_event("request_end", status="completed")

    # Step 2: warm hit (3968 / 4000 = 0.992, no cliff)
    slog.append_event("request_begin")
    m2 = Message(
        role="assistant",
        content="a2",
        metrics={"prompt_tokens": 4500, "output_tokens": 80, "prompt_tokens_details": {"cached_tokens": 3968}},
    )
    PersistMixin._trace_token_usage(dummy_agent, [m2])
    slog.append_event("request_end", status="completed")

    # Step 3: cliff (512 / 4500 = 0.1138 < 0.9, cliff=True)
    slog.append_event("request_begin")
    m3 = Message(
        role="assistant",
        content="a3",
        metrics={"prompt_tokens": 4800, "output_tokens": 60, "prompt_tokens_details": {"cached_tokens": 512}},
    )
    PersistMixin._trace_token_usage(dummy_agent, [m3])
    slog.append_event("request_end", status="completed")

    analysis = analyze_entries(slog.iter_raw_entries())
    totals = analysis["totals"]
    assert totals["prefixReuseSteps"] == 2
    assert totals["prefixReuseHits"] == 1
    assert totals["cacheCliffs"] == 1
    assert slog.trajectory_stats()["cache_cliffs"] == 1
