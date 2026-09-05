# -*- coding: utf-8 -*-
"""Tests for history filtering pipeline (HistoryConfig + history_filter callable)."""

import os

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.agent.config import HistoryConfig
from agentica.agent.history_filter import (
    ELIDED_TOOLS_MARK,
    apply_history_pipeline,
    elided_tools_notice,
    strip_all_tool_artifacts,
    strip_tool_artifacts_from_memory,
    summarize_elided_writes,
)
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
    out = apply_history_pipeline(history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None)
    # Assistant turn was pure tool_calls and got fully dropped.
    assert not any(m.role == "assistant" and not m.content for m in out)
    assert any(m.role == "assistant" and m.content == "final" for m in out)


def test_assistant_max_chars_truncates_long_content():
    history = [
        _user("q"),
        _assistant("x" * 500),
    ]
    out = apply_history_pipeline(history, config=HistoryConfig(assistant_max_chars=100), user_filter=None)
    truncated = next(m for m in out if m.role == "assistant")
    assert len(truncated.content) == 103  # 100 + "..."
    assert truncated.content.endswith("...")


def test_assistant_max_chars_does_not_touch_short_content():
    history = [_assistant("short")]
    out = apply_history_pipeline(history, config=HistoryConfig(assistant_max_chars=100), user_filter=None)
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
    out = apply_history_pipeline(history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None)
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
    out = apply_history_pipeline(history, config=HistoryConfig(excluded_tools=["search"]), user_filter=None)
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


# --- strip_all_tool_artifacts: cross-provider /model switch sanitizer ---


def test_strip_openai_style_tool_artifacts():
    """OpenAI wire format: role='tool' dropped, assistant.tool_calls -> text only."""
    history = [
        _user("hello"),
        _assistant("ok", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "glob"}}]),
        _tool("c1", "glob", "file1"),
        _assistant("done"),
    ]
    out = strip_all_tool_artifacts(history, drop_system=True)
    assert [(m.role, m.content) for m in out] == [
        ("user", "hello"),
        ("assistant", "ok"),
        ("assistant", "done"),
    ]
    assert all(m.role != "tool" for m in out)
    assert all(not m.tool_calls for m in out)


def test_strip_anthropic_style_list_content_tool_artifacts():
    """Anthropic wire format: list content blocks (tool_use/tool_result) stripped.

    Regression for the /model cross-provider switch 400:
    'unexpected tool_use_id found in tool_result blocks'.
    """
    history = [
        Message(role="system", content="sys"),
        _user("run a tool"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_01", "name": "glob", "input": {}},
            ],
            tool_calls=[{"id": "toolu_01", "type": "function", "function": {"name": "glob"}}],
        ),
        Message(
            role="user",
            content=[{"type": "tool_result", "tool_use_id": "toolu_01", "content": "file1 file2"}],
        ),
        _assistant("I see two files."),
    ]
    out = strip_all_tool_artifacts(history, drop_system=True)
    assert [(m.role, m.content) for m in out] == [
        ("user", "run a tool"),
        ("assistant", "Let me check."),
        ("assistant", "I see two files."),
    ]
    # No tool_use / tool_result blocks may survive in any content list.
    for m in out:
        if isinstance(m.content, list):
            assert not any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m.content)


def test_strip_pure_tool_call_assistant_is_dropped():
    """An assistant turn that is only a tool_use block (no text) is removed entirely."""
    history = [
        Message(
            role="assistant",
            content=[{"type": "tool_use", "id": "x", "name": "f", "input": {}}],
            tool_calls=[{"id": "x"}],
        ),
    ]
    assert strip_all_tool_artifacts(history) == []


def test_strip_thinking_blocks_without_tool_calls():
    """GPT leftover thinking+signature must not survive a /model switch onto Claude."""
    history = [
        _user("hi"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "gpt thoughts", "signature": "not-claude"},
                {"type": "text", "text": "hello"},
            ],
        ),
    ]
    out = strip_all_tool_artifacts(history)
    assert [(m.role, m.content) for m in out] == [("user", "hi"), ("assistant", "hello")]


def test_strip_thinking_only_assistant_is_dropped():
    history = [
        Message(
            role="assistant",
            content=[{"type": "thinking", "thinking": "x", "signature": "s"}],
        ),
    ]
    assert strip_all_tool_artifacts(history) == []


def test_strip_drops_reasoning_content_on_plain_assistant():
    history = [
        _user("q"),
        Message(role="assistant", content="answer", reasoning_content="secret chain"),
    ]
    out = strip_all_tool_artifacts(history)
    assert [(m.role, m.content) for m in out] == [("user", "q"), ("assistant", "answer")]
    assert out[1].reasoning_content is None


def test_strip_flattens_text_block_lists():
    history = [
        _user("q"),
        Message(role="assistant", content=[{"type": "text", "text": "hello"}]),
    ]
    out = strip_all_tool_artifacts(history)
    assert out[1].content == "hello"
    assert isinstance(out[1].content, str)


def test_strip_keeps_system_when_not_dropping():
    history = [Message(role="system", content="sys"), _user("hi")]
    out = strip_all_tool_artifacts(history, drop_system=False)
    assert [(m.role, m.content) for m in out] == [("system", "sys"), ("user", "hi")]


def test_strip_keeps_images_carried_in_content_blocks():
    """An Anthropic-shaped user turn can carry the image inside ``content``.

    Only ``Message.images`` was preserved, so a switch onto a model that reads
    the block form dropped the picture and answered a question it could not
    see. Both shapes have to travel.
    """
    image_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
    }
    history = [
        Message(role="user", content=[{"type": "text", "text": "what is this?"}, image_block]),
        Message(role="assistant", content="a cat"),
    ]
    out = strip_all_tool_artifacts(history)
    assert len(out) == 2
    assert out[0].role == "user"
    assert out[0].content == "what is this?"
    assert image_block in (out[0].images or [])


def test_strip_does_not_duplicate_images_from_both_shapes():
    """A session that fills ``Message.images`` and the content block must not
    get the same image twice, which some providers reject."""
    image_block = {"type": "image", "source": {"type": "base64", "data": "AAA"}}
    history = [Message(role="user", content=[image_block], images=[image_block])]
    out = strip_all_tool_artifacts(history)
    assert len(out[0].images) == 1


def test_summarize_elided_writes_lists_apply_patch_not_execute_body():
    history = [
        _assistant(
            tool_calls=[
                {
                    "id": "w1",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": (
                            '{"patch": "*** Update File: docs/foo.md\\n'
                            '@@\\n-a\\n+b\\n"}'
                        ),
                    },
                },
                {
                    "id": "e1",
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "arguments": '{"command": "cat ~/.agentica/config.yaml"}',
                    },
                },
            ]
        ),
        _tool("w1", "apply_patch", "Successfully applied patch to 1 file (+1 -1)"),
        _tool("e1", "execute", "api_key: SECRET-DO-NOT-LEAK\n"),
    ]
    lines = summarize_elided_writes(history)
    assert any("apply_patch" in line and "docs/foo.md" in line for line in lines)
    assert any("Successfully applied" in line for line in lines)
    assert not any("SECRET" in line or "execute" in line for line in lines)


def test_elided_notice_warns_even_without_writes():
    text = elided_tools_notice([_user("hi"), _assistant("I wrote docs/x.md")])
    assert ELIDED_TOOLS_MARK in text
    assert "not proof a file exists" in text
    assert "Writes that actually ran" not in text


def test_strip_from_memory_appends_one_assistant_notice():
    from agentica.memory.models import AgentRun
    from agentica.memory.working import WorkingMemory
    from agentica.run_response import RunResponse

    history = [
        _user("edit it"),
        _assistant(
            tool_calls=[{
                "id": "w1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"file_path": "docs/foo.md", "content": "x"}',
                },
            }]
        ),
        _tool("w1", "write_file", "Wrote docs/foo.md"),
        _assistant("写好了：docs/foo.md（193 行）"),
    ]
    memory = WorkingMemory()
    memory.add_messages(history)
    memory.add_run(AgentRun(response=RunResponse(messages=list(history))))
    strip_tool_artifacts_from_memory(memory)

    for messages in (memory.messages, memory.runs[0].response.messages):
        assert all(m.role != "tool" for m in messages)
        assert all(not m.tool_calls for m in messages)
        notice = messages[-1]
        assert notice.role == "assistant"
        assert ELIDED_TOOLS_MARK in notice.content
        assert "write_file docs/foo.md" in notice.content
        assert "Wrote docs/foo.md" in notice.content
        assert messages[-2].content == "写好了：docs/foo.md（193 行）"


def test_strip_keeps_image_only_user_turn():
    """No text at all: the image is the whole question, so the turn survives."""
    image_block = {"type": "image", "source": {"type": "base64", "data": "AAA"}}
    history = [Message(role="user", content=[image_block])]
    out = strip_all_tool_artifacts(history)
    assert len(out) == 1
    assert out[0].images == [image_block]
