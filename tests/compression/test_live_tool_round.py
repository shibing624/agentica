# -*- coding: utf-8 -*-
"""The live tool round is untouchable.

Layer 1 runs before the next LLM call. The round the model has not seen yet
(or is still executing) is the assistant that issued the trailing results —
or a trailing assistant with tool_calls and no results yet. Evicting those
results or shrinking those arguments is how the agent loop eats its own
evidence: write_file/apply_patch write ``...[truncated]`` to disk, execute
runs a cut command, grep/glob lose the match set they just produced.

This file pins the round, not a tool-name allowlist. Builtin file/shell
tools and user-defined tools (SDK ``tools=``, CLI ``--tools``, Web extra
tools, MCP) occupy the same round and must all survive ``evict_context``.
"""
import json

import pytest

from agentica.compression.evict import evict_context, live_tool_round_start
from agentica.compression.tool_call_args import omitted_tool_arg
from agentica.model.message import Message

WINDOW = 10_000
# Well past the 0.8 trigger, and large enough that result eviction has work.
PRESSURE = dict(context_tokens=1_000_000, context_window=WINDOW)


def _result_body(tag: str, words: int = 200) -> str:
    return " ".join(f"{tag}-token{i}" for i in range(words))


def _call(name: str, args: dict, call_id: str, result: str) -> list:
    return [
        Message(role="assistant", tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }]),
        Message(role="tool", tool_call_id=call_id, tool_name=name, content=result),
    ]


def _arg(messages, assistant_index: int, key: str):
    raw = messages[assistant_index].tool_calls[0]["function"]["arguments"]
    return json.loads(raw)[key]


def _tool_content(messages, tool_index: int) -> str:
    return messages[tool_index].content


# (tool name, argument dict with a long string, which key is the payload)
LIVE_TOOLS = [
    ("write_file", {"file_path": "a.py", "content": "W" * 300}, "content"),
    ("apply_patch", {"patch": "*** Begin Patch\n" + "P" * 300}, "patch"),
    ("execute", {"command": "python3 -c " + "E" * 300}, "command"),
    ("glob", {"pattern": "src/" + "G" * 300}, "pattern"),
    ("grep", {"pattern": "TODO" + "R" * 300, "path": "."}, "pattern"),
    # User-defined tools share the live round. Names are not builtins so an
    # allowlist of write_file/execute/grep cannot silently drop them.
    ("query_orders", {"customer_id": "c1", "query": "Q" * 300}, "query"),  # SDK tools=
    ("search_bocha", {"query": "B" * 300}, "query"),  # CLI --tools
    ("ticket_reply", {"text": "T" * 300}, "text"),  # Web / gateway extra
]


def _conversation(name, args, key, *, with_older=True):
    """One user turn: optional older round of the same tool, then the live round."""
    live_args = dict(args)
    live_args[key] = args[key][:-1] + "L"  # distinct from the older payload
    msgs = [Message(role="user", content=f"use {name}")]
    if with_older:
        msgs += _call(name, args, f"{name}_old", _result_body(f"{name}-old"))
    msgs += _call(name, live_args, f"{name}_live", _result_body(f"{name}-live"))
    return msgs, live_args[key], args[key] if with_older else None


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_evict_context_does_not_touch_live_round_arguments(name, args, key):
    msgs, live_payload, _ = _conversation(name, args, key)
    live_assistant = len(msgs) - 2

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, live_assistant, key) == live_payload
    assert "...[truncated]" not in json.dumps(msgs[live_assistant].tool_calls)


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_evict_context_does_not_touch_live_round_results(name, args, key):
    msgs, _, _ = _conversation(name, args, key)
    live_tool = len(msgs) - 1
    before = _tool_content(msgs, live_tool)

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert _tool_content(msgs, live_tool) == before
    assert not before.startswith("[Tool result evicted")
    assert msgs[live_tool]._evicted is not True


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_in_flight_call_arguments_are_not_shrunk(name, args, key):
    """Assistant is last: tools have not returned. Compress must not rewrite the call."""
    payload = args[key]
    msgs = [
        Message(role="user", content=f"use {name}"),
        *_call("glob", {"pattern": "x" * 300}, "prior", _result_body("prior")),
        Message(role="assistant", tool_calls=[{
            "id": f"{name}_flying",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }]),
    ]
    assert live_tool_round_start(msgs) == len(msgs) - 1

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, -1, key) == payload
    assert "...[truncated]" not in json.dumps(msgs[-1].tool_calls)


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_older_round_in_the_same_user_turn_may_shrink_args(name, args, key):
    """Reclaiming older payloads is Layer 1's job. The live round stays whole."""
    msgs, live_payload, old_payload = _conversation(name, args, key)
    old_assistant = 1
    live_assistant = len(msgs) - 2

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, live_assistant, key) == live_payload
    assert _arg(msgs, old_assistant, key) == omitted_tool_arg(len(old_payload))
    assert "...[truncated]" not in _arg(msgs, old_assistant, key)


def test_parallel_live_batch_of_mixed_tools_survives():
    """A round of N+1 parallel calls used to lose the first result. Names must not matter."""
    long = "X" * 300
    assistant = Message(role="assistant", tool_calls=[
        {"id": "c_grep", "type": "function", "function": {
            "name": "grep", "arguments": json.dumps({"pattern": long, "path": "."}),
        }},
        {"id": "c_glob", "type": "function", "function": {
            "name": "glob", "arguments": json.dumps({"pattern": long}),
        }},
        {"id": "c_exec", "type": "function", "function": {
            "name": "execute", "arguments": json.dumps({"command": long}),
        }},
        {"id": "c_write", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"file_path": "a.txt", "content": long}),
        }},
        {"id": "c_patch", "type": "function", "function": {
            "name": "apply_patch", "arguments": json.dumps({"patch": long}),
        }},
        {"id": "c_orders", "type": "function", "function": {
            "name": "query_orders", "arguments": json.dumps({"query": long}),
        }},
        {"id": "c_bocha", "type": "function", "function": {
            "name": "search_bocha", "arguments": json.dumps({"query": long}),
        }},
        {"id": "c_ticket", "type": "function", "function": {
            "name": "ticket_reply", "arguments": json.dumps({"text": long}),
        }},
    ])
    results = [
        Message(role="tool", tool_call_id="c_grep", tool_name="grep", content=_result_body("grep")),
        Message(role="tool", tool_call_id="c_glob", tool_name="glob", content=_result_body("glob")),
        Message(role="tool", tool_call_id="c_exec", tool_name="execute", content=_result_body("exec")),
        Message(role="tool", tool_call_id="c_write", tool_name="write_file", content="created a.txt"),
        Message(role="tool", tool_call_id="c_patch", tool_name="apply_patch", content="patched"),
        Message(role="tool", tool_call_id="c_orders", tool_name="query_orders", content=_result_body("orders")),
        Message(role="tool", tool_call_id="c_bocha", tool_name="search_bocha", content=_result_body("bocha")),
        Message(role="tool", tool_call_id="c_ticket", tool_name="ticket_reply", content=_result_body("ticket")),
    ]
    msgs = [
        Message(role="user", content="go"),
        *_call("grep", {"pattern": "old" * 100}, "old", _result_body("old")),
        assistant,
        *results,
    ]
    args_before = json.dumps(assistant.tool_calls)
    result_before = [m.content for m in results]

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert json.dumps(assistant.tool_calls) == args_before
    assert [m.content for m in results] == result_before
    assert all(not m._evicted for m in results)


def test_live_tool_round_start_includes_the_issuing_assistant():
    msgs = [
        Message(role="user", content="go"),
        *_call("grep", {"pattern": "a"}, "old", "old-hit"),
        *_call("execute", {"command": "pwd"}, "live", " /tmp"),
    ]
    assert live_tool_round_start(msgs) == 3  # live assistant
    assert msgs[3].role == "assistant"
    assert msgs[4].role == "tool"


def test_anthropic_packed_live_round_is_not_evicted():
    """Claude packs the whole round into one user message of tool_result blocks."""
    long = "C" * 300
    live_result = _result_body("claude-live")
    old_result = _result_body("claude-old")
    old_assistant = Message(
        role="assistant",
        content=[{"type": "tool_use", "id": "old", "name": "grep",
                  "input": {"pattern": long}}],
        tool_calls=[{"id": "old", "type": "function",
                     "function": {"name": "grep", "arguments": json.dumps({"pattern": long})}}],
    )
    live_assistant = Message(
        role="assistant",
        content=[{"type": "tool_use", "id": "live", "name": "ticket_reply",
                  "input": {"text": long}}],
        tool_calls=[{"id": "live", "type": "function",
                     "function": {"name": "ticket_reply", "arguments": json.dumps({"text": long})}}],
    )
    msgs = [
        Message(role="user", content="go"),
        old_assistant,
        Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": "old", "content": old_result},
        ]),
        live_assistant,
        Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": "live", "content": live_result},
        ]),
    ]

    evict_context(msgs, model_id="gpt-4o", **PRESSURE)

    assert live_assistant.tool_calls[0]["function"]["arguments"] == json.dumps({"text": long})
    assert live_assistant.content[0]["input"]["text"] == long
    assert msgs[-1].content[0]["content"] == live_result
