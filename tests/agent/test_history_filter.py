# -*- coding: utf-8 -*-
"""Tests for history filtering pipeline (HistoryConfig + history_filter callable)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent.config import HistoryConfig
from agentica.agent.history_filter import apply_history_pipeline
from agentica.model.message import Message


def _user(content: str) -> Message:
    return Message(role="user", content=content)


def _assistant(content: str = None, tool_calls=None) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def _tool(call_id: str, name: str, content: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, tool_name=name, content=content)


def _make_history_with_tools() -> list:
    """user → assistant(tool_calls=[search,calc]) → tool(search) → tool(calc) → assistant("done")."""
    return [
        _user("query"),
        _assistant(
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "calc", "arguments": "{}"}},
            ]
        ),
        _tool("c1", "web_search", "huge search dump..." * 100),
        _tool("c2", "calc", "42"),
        _assistant("done"),
    ]


def test_no_config_no_filter_returns_input_copy():
    history = _make_history_with_tools()
    out = apply_history_pipeline(history, config=None, user_filter=None)
    assert out == history


def test_excluded_tools_drops_tool_message_and_strips_paired_tool_call():
    history = _make_history_with_tools()
    out = apply_history_pipeline(
        history,
        config=HistoryConfig(excluded_tools=["web_search"]),
        user_filter=None,
    )
    assert not any(m.role == "tool" and m.tool_name == "web_search" for m in out)
    assert any(m.role == "tool" and m.tool_name == "calc" for m in out)
    assistant_with_calls = next(m for m in out if m.role == "assistant" and m.tool_calls)
    assert [tc["id"] for tc in assistant_with_calls.tool_calls] == ["c2"]


def test_excluded_tools_glob_pattern():
    history = _make_history_with_tools()
    out = apply_history_pipeline(
        history,
        config=HistoryConfig(excluded_tools=["web_*", "fetch_*"]),
        user_filter=None,
    )
    assert not any(m.role == "tool" and m.tool_name == "web_search" for m in out)


def test_excluded_tools_drops_assistant_message_when_all_calls_removed():
    history = [
        _user("q"),
        _assistant(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]),
        _tool("c1", "search", "..."),
        _assistant("final"),
    ]
    out = apply_history_pipeline(
        history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None
    )
    # Assistant turn was pure tool_calls and got fully dropped.
    assert not any(m.role == "assistant" and not m.content for m in out)
    assert any(m.role == "assistant" and m.content == "final" for m in out)


def test_assistant_max_chars_truncates_long_content():
    history = [
        _user("q"),
        _assistant("x" * 500),
    ]
    out = apply_history_pipeline(
        history, config=HistoryConfig(assistant_max_chars=100), user_filter=None
    )
    truncated = next(m for m in out if m.role == "assistant")
    assert len(truncated.content) == 103  # 100 + "..."
    assert truncated.content.endswith("...")


def test_assistant_max_chars_does_not_touch_short_content():
    history = [_assistant("short")]
    out = apply_history_pipeline(
        history, config=HistoryConfig(assistant_max_chars=100), user_filter=None
    )
    assert out[0].content == "short"


def test_user_filter_runs_after_config_rules():
    history = _make_history_with_tools()
    seen_in_filter = []

    def my_filter(msgs):
        seen_in_filter.extend(msgs)
        return [m for m in msgs if m.role != "user"]

    out = apply_history_pipeline(
        history,
        config=HistoryConfig(excluded_tools=["web_search"]),
        user_filter=my_filter,
    )
    # user_filter saw the history POST-config (web_search already gone)
    assert not any(m.role == "tool" and m.tool_name == "web_search" for m in seen_in_filter)
    assert not any(m.role == "user" for m in out)


def test_consistency_fix_strips_orphan_tool_calls_after_user_filter():
    history = _make_history_with_tools()

    def aggressive_filter(msgs):
        # Drop ALL tool messages but leave assistant.tool_calls untouched (sloppy filter).
        return [m for m in msgs if m.role != "tool"]

    out = apply_history_pipeline(history, config=None, user_filter=aggressive_filter)
    for m in out:
        if m.role == "assistant":
            assert not m.tool_calls, f"orphan tool_calls survived: {m.tool_calls}"


def test_does_not_mutate_input_messages():
    history = _make_history_with_tools()
    original_assistant = history[1]
    original_call_ids = [tc["id"] for tc in original_assistant.tool_calls]

    apply_history_pipeline(
        history,
        config=HistoryConfig(excluded_tools=["web_search"], assistant_max_chars=10),
        user_filter=None,
    )
    assert [tc["id"] for tc in original_assistant.tool_calls] == original_call_ids
    assert history[2].content.startswith("huge search dump")


def test_empty_excluded_tools_is_noop():
    history = _make_history_with_tools()
    out = apply_history_pipeline(history, config=HistoryConfig(excluded_tools=[]), user_filter=None)
    assert [m.role for m in out] == [m.role for m in history]
    assert [m.tool_name for m in out if m.role == "tool"] == ["web_search", "calc"]


def test_excluded_tools_preserves_assistant_content_when_partial_drop():
    """Assistant has content + single tool_call; tool excluded -> content kept, tool_calls=None."""
    history = [
        _user("q"),
        _assistant(
            content="thinking out loud...",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        ),
        _tool("c1", "search", "..."),
        _assistant("final"),
    ]
    out = apply_history_pipeline(
        history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None
    )
    target = next(m for m in out if m.role == "assistant" and m.content == "thinking out loud...")
    assert target.tool_calls is None


def test_multimodal_assistant_content_with_partial_tool_call_drop():
    """Assistant.content can be a list (multimodal). Must not crash on .strip()."""
    multimodal = [{"type": "text", "text": "see image"}, {"type": "image_url", "image_url": "..."}]
    history = [
        _user("q"),
        _assistant(
            content=multimodal,
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        ),
        _tool("c1", "search", "..."),
    ]
    out = apply_history_pipeline(
        history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None
    )
    target = next(m for m in out if m.role == "assistant")
    assert target.content == multimodal
    assert target.tool_calls is None


def test_user_message_strip_via_callable():
    """User-defined callable can do anything: e.g. strip a prefix from user messages."""
    history = [_user("用纯文本回复 你好"), _assistant("hi")]

    def strip_prefix(msgs):
        out = []
        for m in msgs:
            if m.role == "user" and isinstance(m.content, str):
                m = m.model_copy(update={"content": m.content.removeprefix("用纯文本回复 ")})
            out.append(m)
        return out

    out = apply_history_pipeline(history, config=None, user_filter=strip_prefix)
    assert out[0].content == "你好"
