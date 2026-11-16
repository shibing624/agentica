# -*- coding: utf-8 -*-
"""The live tool round is untouchable.

Layer 1 runs before the next LLM call, at the compaction line. The round the model has not seen yet
(or is still executing) is the assistant that issued the trailing results —
or a trailing assistant with tool_calls and no results yet. Pruning those
results is how the agent loop eats its own
evidence: write_file/apply_patch write ``...[truncated]`` to disk, execute
runs a cut command, grep/glob lose the match set they just produced.

This file pins the round, not a tool-name allowlist. Builtin file/shell
tools and user-defined tools (SDK ``tools=``, CLI ``--tools``, Web extra
tools, MCP) occupy the same round and must all survive ``prune_tool_results``.
"""
import json

import pytest

from agentica.compression.evict import live_tool_round_start, prune_tool_results, trailing_user_turn_start
from agentica.model.message import Message

# Unconditional: these tests pin *what* may be pruned, not whether it pays.
PRESSURE = dict(context_tokens=1_000_000, target_tokens=None)


def _result_body(tag: str, words: int = 600) -> str:
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
def test_prune_does_not_touch_live_round_arguments(name, args, key):
    msgs, live_payload, _ = _conversation(name, args, key)
    live_assistant = len(msgs) - 2

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, live_assistant, key) == live_payload
    assert "...[truncated]" not in json.dumps(msgs[live_assistant].tool_calls)


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_prune_does_not_touch_live_round_results(name, args, key):
    msgs, _, _ = _conversation(name, args, key)
    live_tool = len(msgs) - 1
    before = _tool_content(msgs, live_tool)

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _tool_content(msgs, live_tool) == before
    assert "pruned from the middle" not in before


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

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, -1, key) == payload
    assert "...[truncated]" not in json.dumps(msgs[-1].tool_calls)


@pytest.mark.parametrize("name,args,key", LIVE_TOOLS)
def test_older_round_arguments_are_never_rewritten(name, args, key):
    """Layer 1 evicts the older round's result, not its arguments."""
    msgs, live_payload, old_payload = _conversation(name, args, key)
    old_assistant = 1
    live_assistant = len(msgs) - 2

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, live_assistant, key) == live_payload
    assert _arg(msgs, old_assistant, key) == old_payload
    assert "pruned from the middle" in msgs[old_assistant + 1].content


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

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert json.dumps(assistant.tool_calls) == args_before
    assert [m.content for m in results] == result_before


def test_live_tool_round_start_includes_the_issuing_assistant():
    msgs = [
        Message(role="user", content="go"),
        *_call("grep", {"pattern": "a"}, "old", "old-hit"),
        *_call("execute", {"command": "pwd"}, "live", " /tmp"),
    ]
    assert live_tool_round_start(msgs) == 3  # live assistant
    assert msgs[3].role == "assistant"
    assert msgs[4].role == "tool"


def test_anthropic_packed_live_round_is_not_pruned():
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

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert live_assistant.tool_calls[0]["function"]["arguments"] == json.dumps({"text": long})
    assert live_assistant.content[0]["input"]["text"] == long
    assert msgs[-1].content[0]["content"] == live_result


def _injected(content: str) -> Message:
    """A mid-turn steering / peer message, as the runner appends it."""
    message = Message(role="user", content=content)
    message._injected = True
    return message


def test_injected_guidance_does_not_unprotect_the_in_flight_call():
    """Guidance arriving mid-round must not hand Layer 1 the running call.

    ``_inject_steering`` / ``_inject_peer_messages`` append a user message when
    there is no trailing ``role="tool"`` result to fold into — i.e. exactly
    while a call is in flight. That append used to become the "tail", so the
    cutoff moved above the assistant and its own unread payload was shrunk to
    the placeholder. Observed live in older sessions: an in-flight string tool
    argument became ``<evicted-tool-arg chars=1885>``.
    """
    long = "W" * 300
    msgs = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[{
            "id": "flying", "type": "function",
            "function": {"name": "write_file",
                         "arguments": json.dumps({"file_path": "a.py", "content": long})},
        }]),
        _injected("[User guidance received while you were working]\nuse tabs"),
    ]

    assert live_tool_round_start(msgs) == 1  # the in-flight assistant

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, 1, "content") == long


def test_injected_message_does_not_unprotect_a_packed_round():
    """Same hole on the Anthropic shape: results live inside a user message.

    There is no trailing ``role="tool"`` for the injection to fold into, so it
    appends — and the round it sits after is still the one that just ran.
    """
    long = "C" * 300
    live_result = _result_body("claude-live")
    live_assistant = Message(
        role="assistant",
        content=[{"type": "tool_use", "id": "live", "name": "write_note",
                  "input": {"message": long}}],
        tool_calls=[{"id": "live", "type": "function",
                     "function": {"name": "write_note", "arguments": json.dumps({"message": long})}}],
    )
    packed = Message(role="user", content=[
        {"type": "tool_result", "tool_use_id": "live", "content": live_result},
    ])
    msgs = [
        Message(role="user", content="go"),
        live_assistant,
        packed,
        _injected("[Message from another agent session 'p'] hi"),
    ]

    assert live_tool_round_start(msgs) == 1

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert live_assistant.tool_calls[0]["function"]["arguments"] == json.dumps({"message": long})
    assert live_assistant.content[0]["input"]["message"] == long
    assert packed.content[0]["content"] == live_result


def test_an_injected_message_alone_creates_no_protected_round():
    """With nothing but injected text there is no round, and nothing to lose.

    The boundary falls after the injected tail; either way nothing is pruned,
    because pruning only rewrites tool results — never a user message.
    """
    msgs = [
        Message(role="user", content="go"),
        _injected("[Message from another agent session 'p'] hi"),
    ]
    before = [m.content for m in msgs]

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert [m.content for m in msgs] == before

def test_an_ordinary_user_message_still_ends_the_live_round():
    """Only marked injections are skipped. A real next question is a new turn."""
    long = "R" * 300
    msgs = [
        Message(role="user", content="go"),
        *_call("write_file", {"file_path": "a.py", "content": long}, "done", "wrote a.py"),
        Message(role="user", content="now do something else"),
    ]

    # Nothing is protected: the round above already returned, and a real next
    # question means it is history now.
    assert live_tool_round_start(msgs) == len(msgs)

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, 1, "content") == long


def test_the_todo_reminder_does_not_unprotect_the_batch_it_follows():
    """The post-tool hook appends its reminder right after a result batch.

    That batch is the one the model has not been called on yet, so it must stay
    live. Unmarked, the reminder became the tail and its own results were
    pruned before the model ever read them.
    """
    long = "T" * 300
    live_result = _result_body("live")
    live_assistant = Message(role="assistant", tool_calls=[{
        "id": "live", "type": "function",
        "function": {"name": "write_file",
                     "arguments": json.dumps({"file_path": "a.py", "content": long})},
    }])
    live_tool = Message(role="tool", tool_call_id="live", tool_name="write_file", content=live_result)
    reminder = Message(role="user", content="[Todo Reminder] The write_todos tool ...")
    reminder._injected = True
    msgs = [
        Message(role="user", content="go"),
        live_assistant,
        live_tool,
        reminder,
    ]

    assert live_tool_round_start(msgs) == 1

    prune_tool_results(msgs, model_id="gpt-4o", **PRESSURE)

    assert _arg(msgs, 1, "content") == long
    assert live_tool.content == live_result


def test_a_mid_turn_injection_is_not_where_the_turn_starts():
    """A compaction at the line keeps the trailing turn. The reminder, a steer
    and a peer message are user messages inside that turn: taken as its start,
    the question and the tool round above them were dropped from the window."""
    for tag in ("[Todo Reminder] ...", "[User guidance received while you were working]\nx"):
        injected = Message(role="user", content=tag)
        injected._injected = True
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="earlier"),
            Message(role="assistant", content="answer"),
            Message(role="user", content="do the work"),
            Message(role="assistant", content="working"),
            injected,
        ]
        assert trailing_user_turn_start(msgs) == 3, tag
    # Only injections: no real turn to keep.
    assert trailing_user_turn_start([injected]) == 1
