# -*- coding: utf-8 -*-
"""Trace analysis invariants — sequential scan, not timestamp slicing."""
import json

from agentica.memory.session_log import SessionLog
from agentica.memory.trace import (
    analyze_entries, call_body, format_trace_text, last_completed_round,
    search_calls, search_conversation,
)


def _e(ts, **fields):
    row = {"timestamp": ts, **fields}
    return row


def test_events_are_not_replayed_into_model_context(tmp_path):
    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    log.append("user", "hello")
    log.append_event("request_begin")
    log.append_event("text")
    log.append_event("request_end", status="completed")
    log.append("assistant", "hi")
    messages = log.load()
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"]
    raw = list(log.iter_raw_entries())
    assert any(e.get("type") == "event" and e.get("name") == "request_begin" for e in raw)


def test_seal_skips_events_when_finding_the_conversation_tail(tmp_path):
    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    log.append("user", "q")
    log.append("assistant", "a")
    log.append_event("request_end", status="completed")
    assert log.seal_incomplete_turn() is None


def test_trace_prelude_is_written_once_per_content(tmp_path):
    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    args = dict(
        model="gpt-4o-mini", provider="OpenAI", context_window=128000,
        tools=["read_file"], system_prompt="be terse",
    )
    assert log.append_trace_prelude(**args) is True
    assert log.append_trace_prelude(**args) is False
    names = [e.get("name") for e in log.iter_raw_entries()]
    assert names == ["session_meta", "tool_list_ready", "system_prompt"]
    prompt = next(e for e in log.iter_raw_entries() if e.get("name") == "system_prompt")
    assert "tokens" in prompt
    assert "chars" not in prompt
    assert isinstance(prompt["tokens"], int)
    assert prompt["tokens"] > 0
    # A mid-session change (profile switch, skills reload) must be recorded, or
    # the header would describe requests it no longer matches.
    assert log.append_trace_prelude(**{**args, "system_prompt": "be verbose"}) is True
    assert [e.get("name") for e in log.iter_raw_entries()].count("system_prompt") == 2


def test_trace_prelude_is_never_replayed_into_the_model(tmp_path):
    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    log.append_trace_prelude(
        model="m", provider="p", context_window=1, tools=["t"], system_prompt="secret",
    )
    log.append("user", "hi")
    assert [m["role"] for m in log.load()] == ["user"]


def test_serial_model_segments_start_at_previous_event():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="hi"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="thinking"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:04.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert out["hasTimeline"] is True
    segs = out["modelSegments"]
    assert [s["kind"] for s in segs] == ["thinking", "text"]
    assert segs[0]["startTs"] == "2026-01-01T00:00:01.000Z"
    assert segs[0]["endTs"] == "2026-01-01T00:00:02.000Z"
    assert segs[1]["startTs"] == "2026-01-01T00:00:02.000Z"
    assert segs[0]["taskIndex"] == 0
    assert segs[0]["key"]


def test_user_messages_occupy_no_model_bar():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="hi"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
    ]
    kinds = [s["kind"] for s in analyze_entries(entries)["modelSegments"]]
    assert kinds == ["text"]


def test_tool_approval_wait_is_subtracted_from_llm_ms():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="edit"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="apply_patch"),
        _e("2026-01-01T00:00:12.000Z", type="event", name="approval_decision", tool_call_id="c1", decision="allow"),
        _e("2026-01-01T00:00:13.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:14.000Z", type="tool", tool_call_id="c1", content="ok"),
    ]
    out = analyze_entries(entries)
    req = out["requests"][0]
    assert req["durationMs"] == 12000
    assert req["approvalWaitMs"] == 10000
    assert req["activeMs"] == 2000
    assert out["tasks"][0]["llmMs"] == 2000
    span = out["toolSpans"][0]
    assert span["callTs"] == "2026-01-01T00:00:02.000Z"
    assert span["approvalTs"] == "2026-01-01T00:00:12.000Z"
    assert span["outputTs"] == "2026-01-01T00:00:14.000Z"


def test_approval_decision_row_includes_request_wait_and_similar_grant():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="rm"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="execute"),
        _e(
            "2026-01-01T00:00:05.000Z", type="event", name="approval_decision",
            tool_call_id="c1", decision="allow_prefix", tool="execute",
            wait_s=2.8, mode="ask",
            question="Allow running the following command?",
            preview="rm -f /tmp/a.ini", similar_label="rm -f",
            options=["allow", "allow_prefix", "deny"],
            arguments={"command": "rm -f /tmp/a.ini"},
            grant={
                "scope": "similar", "persisted": True,
                "kind": "command", "command_class": "rm -f",
            },
        ),
        _e("2026-01-01T00:00:06.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:07.000Z", type="tool", tool_call_id="c1", content="ok"),
    ]
    out = analyze_entries(entries)
    assert out["requests"][0]["approvalWaitMs"] == 2800
    row = next(e for e in out["rounds"][0]["entries"] if e["kind"] == "approval_decision")
    assert "allow_prefix execute 2.80s similar=rm -f" == row["summary"]
    assert "Allow running the following command?" in row["detail"]
    assert "command_class" in row["detail"]
    assert "rm -f" in row["detail"]


def test_plain_text_turn_then_next_user_starts_a_new_task():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="one"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T01:00:00.000Z", type="user", content="two"),
        _e("2026-01-01T01:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T01:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T01:00:03.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert [t["taskIndex"] for t in out["tasks"]] == [0, 1]
    assert out["modelSegments"][0]["taskIndex"] == 0
    assert out["modelSegments"][1]["taskIndex"] == 1


def test_tool_call_continues_the_same_task_on_the_next_request():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="do it"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="read_file"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:04.000Z", type="tool", tool_call_id="c1", content="data"),
        _e("2026-01-01T00:00:05.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert len(out["tasks"]) == 1
    assert out["modelSegments"][0]["taskIndex"] == 0
    assert out["modelSegments"][1]["taskIndex"] == 0


def test_compact_boundary_is_its_own_task_and_does_not_fold_the_next_user_turn():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="one"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="read_file"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:04.000Z", type="compact_boundary", summary="sum"),
        _e("2026-01-01T00:00:05.000Z", type="user", content="two"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:08.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    kinds = [bool(t.get("compaction")) for t in out["tasks"]]
    assert True in kinds
    user_tasks = [t for t in out["tasks"] if not t.get("compaction")]
    assert len(user_tasks) == 2


def test_context_snapshot_is_last_request_not_a_sum():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="do"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="read_file"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="token_usage", request={"cache_read": 10, "cache_write": 2, "output": 5, "total": 17}),
        _e("2026-01-01T00:00:04.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:05.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="token_usage", request={"cache_read": 80, "cache_write": 0, "output": 20, "total": 100}),
        _e("2026-01-01T00:00:08.000Z", type="event", name="request_end", status="completed"),
    ]
    t = analyze_entries(entries)["tasks"][0]
    assert t["tokens"]["output"] == 25
    assert t["tokens"]["cacheRead"] == 90
    assert t["context"]["cacheRead"] == 80
    assert t["context"]["output"] == 20


def test_elapsed_ms_sums_tasks_not_file_span():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="one"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T10:00:00.000Z", type="user", content="two"),
        _e("2026-01-01T10:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T10:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T10:00:03.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert out["elapsedMs"] < 60_000
    assert out["elapsedMs"] > 0


def test_same_millisecond_rows_stay_on_the_task_the_scan_assigned():
    ts = "2026-01-01T00:00:03.000Z"
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="one"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e(ts, type="event", name="request_end", status="completed"),
        _e(ts, type="compact_boundary", summary="sum"),
        _e(ts, type="user", content="two"),
        _e(ts, type="event", name="request_begin"),
        _e("2026-01-01T00:00:04.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:05.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert out["messageTask"][3] == 0
    compact_idx = next(i for i, e in enumerate(entries) if e.get("type") == "compact_boundary")
    compact_task = next(t for t in out["tasks"] if t["taskIndex"] == out["messageTask"][compact_idx])
    assert compact_task.get("compaction") is True
    user2 = next(i for i, e in enumerate(entries) if e.get("type") == "user" and e.get("content") == "two")
    assert out["messageTask"][user2] != out["messageTask"][compact_idx]


def test_steering_user_does_not_start_a_new_task():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="do"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="read_file"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:04.000Z", type="user", content="__RELAYED__ keep going"),
        _e("2026-01-01T00:00:05.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert len([t for t in out["tasks"] if not t.get("compaction")]) == 1


def test_explicit_steer_flag_stays_in_round_and_renders_clean_steering_row():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="do"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="read_file"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:03.500Z", type="tool", tool_call_id="c1", tool_name="read_file", content="ok"),
        _e("2026-01-01T00:00:04.000Z", type="user", steer=True, content="优化展示效果"),
        _e("2026-01-01T00:00:05.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:07.100Z", type="assistant", content="done"),
    ]
    out = analyze_entries(entries)
    rounds = [r for r in out["rounds"] if not r.get("compaction")]
    assert len(rounds) == 1
    steer_calls = [c for c in rounds[0]["calls"] if c["kind"] == "steering"]
    assert len(steer_calls) == 1
    assert steer_calls[0]["summary"] == "优化展示效果"


def test_old_session_without_events_has_no_timeline():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="hi"),
        _e("2026-01-01T00:00:01.000Z", type="assistant", content="hello"),
    ]
    out = analyze_entries(entries)
    assert out["hasTimeline"] is False
    assert out["modelSegments"] == []
    assert out["toolSpans"] == []


class TestRoundView:
    """The per-round view the Trace page draws: attribution, bodies, phases."""

    def _turn(self):
        """One tool-using turn, in the order the runner writes it."""
        return [
            _e("2026-01-01T00:00:00.000Z", type="event", name="session_meta",
               model="gpt-4o-mini", provider="OpenAI", context_window=128000, tool_count=2),
            _e("2026-01-01T00:00:00.001Z", type="event", name="tool_list_ready",
               tools=["read_file", "execute"], count=2),
            _e("2026-01-01T00:00:00.002Z", type="event", name="system_prompt",
               content="You are agentica. Be terse.", chars=27),
            _e("2026-01-01T00:00:00.100Z", type="user", content="how many py files"),
            _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
            _e("2026-01-01T00:00:03.000Z", type="event", name="thinking"),
            _e("2026-01-01T00:00:04.000Z", type="event", name="tool_call",
               tool_call_id="c1", tool_name="execute"),
            _e("2026-01-01T00:00:04.100Z", type="event", name="token_usage",
               request={"input": 100, "cache_read": 900, "cache_write": 0, "output": 20, "total": 1020}),
            _e("2026-01-01T00:00:04.200Z", type="event", name="request_end", status="completed"),
            _e("2026-01-01T00:00:04.300Z", type="assistant", content="", reasoning_content="count them",
               tool_calls=[{"id": "c1", "type": "function",
                            "function": {"name": "execute", "arguments": '{"command": "ls"}'}}]),
            _e("2026-01-01T00:00:05.000Z", type="tool", tool_call_id="c1", tool_name="execute",
               content="670", metrics={"time": 0.25}),
            _e("2026-01-01T00:00:05.100Z", type="event", name="request_begin"),
            _e("2026-01-01T00:00:07.000Z", type="event", name="text"),
            _e("2026-01-01T00:00:07.100Z", type="event", name="token_usage",
               request={"input": 50, "cache_read": 1000, "cache_write": 0, "output": 30, "total": 1080}),
            _e("2026-01-01T00:00:07.200Z", type="event", name="request_end", status="completed"),
            _e("2026-01-01T00:00:07.300Z", type="assistant", content="670 files."),
        ]

    def test_the_question_leads_the_round_it_caused(self):
        out = analyze_entries(self._turn())
        assert len(out["rounds"]) == 1
        rd = out["rounds"][0]
        assert rd["title"] == "how many py files"
        kinds = [e["kind"] for e in rd["entries"]]
        # The prelude configured this round, so it is shown inside it, ahead of
        # the prompt — not orphaned outside every round.
        assert kinds[:4] == ["session_meta", "tool_list_ready", "system_prompt", "user"]

    def test_system_prompt_and_tool_table_are_expandable(self):
        rd = analyze_entries(self._turn())["rounds"][0]
        prompt = next(e for e in rd["entries"] if e["kind"] == "system_prompt")
        assert prompt["detail"] == "You are agentica. Be terse."
        assert prompt["summary"].endswith(" tokens")
        assert "chars" not in prompt["summary"]
        assert analyze_entries(self._turn())["meta"]["systemPromptTokens"] > 0
        tools = next(e for e in rd["entries"] if e["kind"] == "tool_list_ready")
        assert tools["detail"].splitlines() == ["read_file", "execute"]
        assert analyze_entries(self._turn())["meta"]["tools"] == ["read_file", "execute"]

    def test_markers_are_paired_with_the_bodies_written_after_them(self):
        rd = analyze_entries(self._turn())["rounds"][0]
        thinking = next(e for e in rd["entries"] if e["kind"] == "thinking")
        assert thinking["detail"] == "count them"
        text = next(e for e in rd["entries"] if e["kind"] == "text")
        assert text["detail"] == "670 files."
        # A body claimed by a marker must not also appear as its own row.
        assert [e["kind"] for e in rd["entries"]].count("assistant") == 0

    def test_tool_call_carries_arguments_and_the_result_carries_output(self):
        rd = analyze_entries(self._turn())["rounds"][0]
        call = next(e for e in rd["entries"] if e["kind"] == "tool_call")
        assert call["toolName"] == "execute"
        assert '"command": "ls"' in call["detail"]
        result = next(e for e in rd["entries"] if e["kind"] == "tool_result")
        assert result["detail"] == "670"
        assert result["isError"] is False
        assert result["durationMs"] == 250

    def test_task_rows_carry_structured_subagent_fields(self):
        report = "x" * 50000  # past DETAIL_CHAR_CAP: the JSON in `detail` is cut
        result = {
            "success": True, "subagent_type": "explore", "subagent_name": "Explore Agent",
            "result": report, "execution_time": 12.5, "tool_count": 2,
            "tool_calls_summary": [{"name": "grep", "info": "pattern: foo"}, {"name": "read_file", "info": "a.py"}],
        }
        entries = [
            _e("2026-01-01T00:00:00.100Z", type="user", content="find foo"),
            _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
            _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="t1", tool_name="task"),
            _e("2026-01-01T00:00:02.100Z", type="event", name="request_end", status="completed"),
            _e("2026-01-01T00:00:02.200Z", type="assistant", content="",
               tool_calls=[{"id": "t1", "type": "function", "function": {
                   "name": "task", "arguments": json.dumps({"description": "Find foo\nin repo", "subagent_type": "explore"})}}]),
            _e("2026-01-01T00:00:15.000Z", type="tool", tool_call_id="t1", tool_name="task",
               content=json.dumps(result)),
        ]
        rd = analyze_entries(entries)["rounds"][0]
        call = next(e for e in rd["entries"] if e["kind"] == "tool_call")
        assert call["task"] == {"description": "Find foo\nin repo", "subagentType": "explore"}
        assert call["summary"] == "task Find foo in repo"
        out = next(e for e in rd["entries"] if e["kind"] == "tool_result")
        task = out["task"]
        assert task["success"] is True and task["status"] == "completed"
        assert task["subagentName"] == "Explore Agent"
        assert task["executionMs"] == 12500 and task["toolCount"] == 2
        assert [s["name"] for s in task["steps"]] == ["grep", "read_file"]
        assert task["result"].startswith("xxx")
        assert out["summary"].startswith("Explore Agent · completed · ")

    def test_task_rows_surface_resolved_model_and_tier_in_calls_and_body(self):
        result = {
            "success": True,
            "subagent_type": "code",
            "subagent_name": "Code Agent",
            "model": "openai/claude-opus-5-5",
            "model_tier": "main",
            "result": "analyzed module",
            "execution_time": 4.2,
            "tool_count": 1,
            "tool_calls_summary": [{"name": "read_file", "info": "a.py"}],
        }
        entries = [
            _e("2026-01-01T00:00:00.100Z", type="user", content="trace module"),
            _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
            _e(
                "2026-01-01T00:00:02.000Z",
                type="event",
                name="tool_call",
                tool_call_id="t1",
                tool_name="task",
                model="openai/claude-opus-5-5",
            ),
            _e("2026-01-01T00:00:02.100Z", type="event", name="request_end", status="completed"),
            _e(
                "2026-01-01T00:00:02.200Z",
                type="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": json.dumps(
                                {"description": "Trace deep module", "subagent_type": "code", "model_tier": "main"}
                            ),
                        },
                    }
                ],
            ),
            _e("2026-01-01T00:00:06.200Z", type="tool", tool_call_id="t1", tool_name="task", content=json.dumps(result)),
        ]
        rd = analyze_entries(entries)["rounds"][0]
        tool_call_row = next(c for c in rd["calls"] if c["kind"] == "tool")
        assert tool_call_row["toolModel"] == "openai/claude-opus-5-5"
        assert tool_call_row["toolModelTier"] == "main"
        assert tool_call_row["body"]["taskResult"]["model"] == "openai/claude-opus-5-5"
        assert tool_call_row["body"]["taskResult"]["modelTier"] == "main"
        model_call_row = next(c for c in rd["calls"] if c["kind"] == "model")
        assert model_call_row["body"]["toolCalls"][0]["toolModel"] == "openai/claude-opus-5-5"
        assert model_call_row["body"]["toolCalls"][0]["toolModelTier"] == "main"

    def test_failed_task_result_shows_its_error(self):
        entries = [
            _e("2026-01-01T00:00:00.100Z", type="user", content="go"),
            _e("2026-01-01T00:00:01.000Z", type="tool", tool_call_id="t1", tool_name="task",
               content=json.dumps({"success": False, "status": "timeout", "error": "timed out after 300s", "run_id": "r9"})),
        ]
        out = next(e for e in analyze_entries(entries)["rounds"][0]["entries"] if e["kind"] == "tool_result")
        assert out["task"]["status"] == "timeout" and out["task"]["runId"] == "r9"
        assert "timed out after 300s" in out["summary"]

    def test_phases_split_the_round_and_never_exceed_it(self):
        rd = analyze_entries(self._turn())["rounds"][0]
        phases = rd["phases"]
        assert phases["thinking"] == 2000
        assert phases["toolArgs"] == 1000
        # From the call being issued to its result landing on disk.
        assert phases["toolExec"] == 1000
        assert phases["text"] == 1900
        assert sum(phases.values()) <= rd["durationMs"] + phases["other"]
        assert phases["other"] >= 0

    def test_totals_report_the_session_at_a_glance(self):
        out = analyze_entries(self._turn())
        totals = out["totals"]
        assert totals["rounds"] == 1
        assert totals["requests"] == 2
        assert totals["toolCalls"] == 1
        assert totals["toolOk"] == 1
        assert totals["toolErrors"] == 0
        assert totals["tokens"]["input"] == 150
        assert totals["tokens"]["cacheRead"] == 1900
        assert totals["tokens"]["cacheWrite"] == 0
        assert totals["tokens"]["output"] == 50
        assert totals["tokens"]["prompt"] == 2050
        assert totals["tokens"]["cacheHitPercent"] == 92.7
        assert totals["costUsd"] > 0
        assert totals["tps"] > 0
        rd = out["rounds"][0]
        assert rd["tokens"]["prompt"] == 2050
        assert rd["tokens"]["cacheHitPercent"] == 92.7

    def test_totals_carry_tool_time_and_mean_first_token_latency(self):
        turn = self._turn()
        # Two answered requests: first tokens at 0.8s and 1.6s.
        turn[9]["metrics"] = {"time_to_first_token": 0.8}
        turn[-1]["metrics"] = {"time_to_first_token": 1.6}
        totals = analyze_entries(turn)["totals"]
        assert totals["ttftMs"] == 1200
        # The tool's own run time (metrics.time), not the gap around it.
        assert totals["toolMs"] == 250
        assert analyze_entries(self._turn())["totals"]["ttftMs"] is None

    def test_cost_is_none_without_a_known_model(self):
        entries = [e for e in self._turn() if e.get("name") != "session_meta"]
        entries = [e for e in entries if e.get("type") != "assistant" or "content" not in e]
        out = analyze_entries([e for e in entries if e.get("type") != "assistant"])
        assert out["meta"]["model"] is None
        assert out["totals"]["costUsd"] is None

    def test_token_usage_without_an_input_bucket_is_derived_not_zero(self):
        entries = [
            _e("2026-01-01T00:00:00.000Z", type="user", content="q"),
            _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
            _e("2026-01-01T00:00:02.000Z", type="event", name="token_usage",
               request={"cache_read": 900, "cache_write": 0, "output": 20, "total": 1020}),
            _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        ]
        tok = analyze_entries(entries)["rounds"][0]["tokens"]
        assert tok["input"] == 100
        assert tok["prompt"] == 1000
        assert tok["cacheHitPercent"] == 90.0

    def test_old_session_without_events_still_shows_its_messages(self):
        entries = [
            _e("2026-01-01T00:00:00.000Z", type="user", content="hi"),
            _e("2026-01-01T00:00:01.000Z", type="assistant", content="hello"),
        ]
        out = analyze_entries(entries)
        rd = out["rounds"][0]
        assert [e["kind"] for e in rd["entries"]] == ["user", "assistant"]
        assert rd["entries"][1]["detail"] == "hello"

    def test_a_failed_tool_result_is_flagged(self):
        entries = self._turn()
        for e in entries:
            if e.get("type") == "tool":
                e["is_error"] = True
        out = analyze_entries(entries)
        result = next(e for e in out["rounds"][0]["entries"] if e["kind"] == "tool_result")
        assert result["isError"] is True
        assert out["totals"]["toolErrors"] == 1
        assert out["totals"]["toolOk"] == 0

    def test_calls_fold_request_markers_into_model_calls(self):
        calls = analyze_entries(self._turn())["rounds"][0]["calls"]
        # The markers are not rows; the prelude lives in `meta`.
        assert [c["kind"] for c in calls] == ["system_prompt", "user", "model", "tool", "model"]
        first, tool, second = calls[2], calls[3], calls[4]
        assert first["durationMs"] == 3200
        assert first["tokens"]["prompt"] == 1000 and first["tokens"]["output"] == 20
        assert first["tokens"]["cacheHitPercent"] == 90.0
        assert first["toolCalls"] == 1
        assert first["body"]["thinking"] == "count them"
        assert len(first["body"]["toolCalls"]) == 1
        assert first["body"]["toolCalls"][0]["toolName"] == "execute"
        assert first["body"]["toolCalls"][0]["resultSummary"] == "670"
        assert second["tokens"]["output"] == 30
        assert second["body"]["text"] == "670 files."
        assert second["summary"] == "670 files."
        assert tool["toolName"] == "execute" and tool["durationMs"] == 250
        assert tool["summary"] == 'execute {"command": "ls"}'
        assert tool["resultSummary"] == "670"
        assert '"command": "ls"' in tool["body"]["args"]
        assert tool["body"]["result"] == "670"

    def test_tool_call_carries_its_approval_wait_and_decision(self):
        entries = [
            _e("2026-01-01T00:00:00.000Z", type="user", content="rm"),
            _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
            _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="execute"),
            _e("2026-01-01T00:00:05.000Z", type="event", name="approval_decision",
               tool_call_id="c1", decision="allow", tool="execute", wait_s=3.0),
            _e("2026-01-01T00:00:06.000Z", type="event", name="request_end", status="completed"),
            _e("2026-01-01T00:00:07.000Z", type="tool", tool_call_id="c1", content="ok"),
        ]
        tool = next(c for c in analyze_entries(entries)["rounds"][0]["calls"] if c["kind"] == "tool")
        assert tool["waitMs"] == 3000 and tool["decision"] == "allow"
        assert '"decision": "allow"' in tool["body"]["approval"]

    def test_lite_drops_every_body_and_call_body_returns_one(self):
        full = analyze_entries(self._turn())
        lite = analyze_entries(self._turn(), lite=True)
        assert "modelSegments" not in lite and "entries" not in lite["rounds"][0]
        assert all("body" not in c for c in lite["rounds"][0]["calls"])
        assert lite["totals"] == full["totals"]
        tool = next(c for c in lite["rounds"][0]["calls"] if c["kind"] == "tool")
        assert call_body(self._turn(), tool["index"])["result"] == "670"
        assert call_body(self._turn(), 9999) is None

    def test_search_matches_bodies_not_just_summaries(self):
        calls = analyze_entries(self._turn())["rounds"][0]["calls"]
        tool = next(c for c in calls if c["kind"] == "tool")
        model = next(c for c in calls if c["kind"] == "model")
        # "670" is only in the tool's result body and the final reply.
        hits = search_calls(self._turn(), "670")
        assert tool["index"] in hits
        assert model["index"] not in hits  # the first request only thought
        # Thinking is searched too, case-insensitively.
        assert search_calls(self._turn(), "COUNT THEM") == [model["index"]]
        assert search_calls(self._turn(), "   ") == []


class TestAnalysisCache:
    """One scan of an unchanged log serves the page, each call body and search."""

    def _log(self, tmp_path, n=3):
        log = SessionLog(session_id="cached", base_dir=str(tmp_path))
        log.append("user", "count the files")
        for i in range(n):
            log.append_event("request_begin")
            log.append_event("tool_call", tool_call_id=f"c{i}", tool_name="execute", stop_reason="completed",
                             arguments=json.dumps({"command": f"ls {i}"}))
            log.append_event("request_end", status="completed")
            log.append("tool", f"out-{i}", tool_call_id=f"c{i}")
        log.append("assistant", "done")
        return log

    def _count_scans(self, monkeypatch):
        import agentica.memory.trace as trace
        real = trace.analyze_entries
        calls = []

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        monkeypatch.setattr(trace, "analyze_entries", counting)
        return calls

    def test_lookups_after_the_first_do_not_rescan_the_file(self, tmp_path, monkeypatch):
        from agentica.memory import session_log as sl
        sl._analysis_cache.clear()
        log = self._log(tmp_path)
        scans = self._count_scans(monkeypatch)
        lite = log.analyze(lite=True)
        indices = [c["index"] for r in lite["rounds"] for c in r["calls"] if c["kind"] == "tool"]
        assert len(indices) == 3
        for i in indices:
            assert log.trace_call_body(i)["result"].startswith("out-")
        assert log.trace_search("out-1") != []
        assert len(scans) == 1

    def test_a_log_that_grew_is_read_again(self, tmp_path, monkeypatch):
        from agentica.memory import session_log as sl
        sl._analysis_cache.clear()
        log = self._log(tmp_path, n=1)
        scans = self._count_scans(monkeypatch)
        assert log.analyze(lite=True)["totals"]["toolCalls"] == 1
        log.append_event("request_begin")
        log.append_event("tool_call", tool_call_id="late", tool_name="execute", stop_reason="completed")
        log.append_event("request_end", status="completed")
        log.append("tool", "out-late", tool_call_id="late")
        assert log.analyze(lite=True)["totals"]["toolCalls"] == 2
        assert len(scans) == 2

    def test_stamping_session_fields_does_not_leak_into_the_cache(self, tmp_path):
        from agentica.memory import session_log as sl
        sl._analysis_cache.clear()
        log = self._log(tmp_path)
        first = log.analyze(lite=True)
        first["meta"]["cwd"] = "/changed"
        first["rounds"][0]["calls"].clear()
        again = log.analyze(lite=True)
        assert again["meta"]["cwd"] != "/changed"
        assert again["rounds"][0]["calls"], "the cached analysis must not be mutated by a caller"
        assert log.trace_call_body(next(c["index"] for c in again["rounds"][0]["calls"] if c["kind"] == "tool"))

    def test_lite_matches_the_direct_lite_scan_and_can_leave_calls_out(self, tmp_path):
        from agentica.memory import session_log as sl
        sl._analysis_cache.clear()
        log = self._log(tmp_path)
        direct = analyze_entries(log.iter_raw_entries(), lite=True)
        served = log.analyze(lite=True)
        assert served["rounds"] == direct["rounds"]
        assert served["totals"] == direct["totals"]
        assert set(direct) - set(served) == set()
        slim = log.analyze(lite=True, calls=False)
        assert all("calls" not in r for r in slim["rounds"])
        assert slim["rounds"][0]["tokens"] == direct["rounds"][0]["tokens"]

    def test_cache_is_bounded(self, tmp_path):
        from agentica.memory import session_log as sl
        sl._analysis_cache.clear()
        for k in range(sl._ANALYSIS_CACHE_MAX_ENTRIES + 3):
            SessionLog(session_id=f"s{k}", base_dir=str(tmp_path)).append("user", f"q{k}")
            SessionLog(session_id=f"s{k}", base_dir=str(tmp_path)).analyze(lite=True)
        assert len(sl._analysis_cache) <= sl._ANALYSIS_CACHE_MAX_ENTRIES


def test_interrupt_compensation_tool_call_stays_off_the_lane():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="do"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e(
            "2026-01-01T00:00:02.000Z",
            type="event",
            name="tool_call",
            tool_call_id="c1",
            tool_name="execute",
            stop_reason="cancelled",
        ),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
    ]
    out = analyze_entries(entries)
    assert out["toolSpans"] == []
    assert out["modelSegments"][0]["kind"] == "tool_call"


def test_an_event_can_carry_the_moment_it_happened(tmp_path):
    """A streamed request only knows afterwards when reasoning stopped, so the
    row has to be able to say so. Stamping it at write time collapsed the whole
    request into one instant and the timeline drew a single bar."""
    import time

    from agentica.memory.session_log import iso_timestamp

    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    then = time.time() - 30
    log.append_event("thinking", timestamp=iso_timestamp(then))
    (row,) = [e for e in log.iter_raw_entries() if e.get("name") == "thinking"]
    assert row["timestamp"] == iso_timestamp(then)
    assert row["timestamp"] < iso_timestamp()


def test_phase_marks_are_written_in_the_order_they_happened(tmp_path):
    """The analyzer chains segments — one mark's timestamp is the next one's
    start — so a fixed thinking → text order would run the chain backwards for
    a model that answers before it calls a tool, and a negative segment draws as
    a flat edge."""
    import time

    from agentica.model.message import Message
    from agentica.runner.persist import PersistMixin

    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    agent = type("A", (), {"_session_log": log})()
    now = time.time()
    messages = [Message(
        role="assistant", content="answer", reasoning_content="hmm",
        tool_calls=[{"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}],
    )]
    # Reasoning ran first, the reply came later, the tool call closes the stream.
    PersistMixin._trace_request_segments(
        agent, messages, {"thinking": now - 4, "text": now - 1},
    )
    marks = [e["name"] for e in log.iter_raw_entries()]
    assert marks == ["thinking", "text", "tool_call"]
    stamps = [e["timestamp"] for e in log.iter_raw_entries()]
    assert stamps == sorted(stamps)


def test_a_streamed_turn_shows_thinking_and_reply_as_separate_phases(tmp_path):
    """End to end over the analyzer: the phases a round reports must add up to
    what the model actually spent, not to one lump."""
    import time

    from agentica.memory.session_log import iso_timestamp
    from agentica.memory.trace import analyze_entries

    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    t0 = time.time()
    log.append("user", "hi")
    log.append_event("request_begin", timestamp=iso_timestamp(t0))
    log.append_event("thinking", timestamp=iso_timestamp(t0 + 2))
    log.append_event("text", timestamp=iso_timestamp(t0 + 5))
    log.append_event("request_end", status="completed", timestamp=iso_timestamp(t0 + 5))

    (round0,) = analyze_entries(list(log.iter_raw_entries()))["rounds"]
    assert round0["phases"]["thinking"] == 2000
    assert round0["phases"]["text"] == 3000


def test_last_completed_round_tps_uses_llm_ms_not_gateway_wall():
    """Chat used to divide output by chat_stream wall time (~4s → 2.8 tok/s).

    Trace divides by request_begin→end minus approval wait (1.90s → 5.3 tok/s).
    Footer and View trace must share this round.
    """
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="hi"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:01.100Z", type="event", name="token_usage",
           request={"input": 8900, "cache_read": 4400, "cache_write": 0, "output": 10, "total": 13310}),
        _e("2026-01-01T00:00:02.900Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:02.900Z", type="assistant", content="hi"),
    ]
    rd = last_completed_round(entries)
    assert rd is not None
    assert rd["durationMs"] == 1900
    assert rd["llmMs"] == 1900
    assert rd["tokens"]["output"] == 10
    tps = rd["tps"]
    assert abs(tps - (10 / 1.9)) < 1e-9
    assert f"{tps:.1f}" == "5.3"
    # The wrong denominator (gateway wall ~3.57s) is what printed 2.8 tok/s.
    assert f"{10 / 3.57:.1f}" == "2.8"


def test_session_log_analyze_and_export(tmp_path):
    log = SessionLog(session_id="s-export", base_dir=str(tmp_path))
    log.append("user", "hello")
    log.append_event("request_begin")
    log.append_event("text")
    log.append_event(
        "token_usage",
        request={"input": 10, "cache_read": 0, "cache_write": 0, "output": 4, "total": 14},
    )
    log.append_event("request_end", status="completed")
    log.append("assistant", "hi")
    analysis = log.analyze()
    assert analysis["session_id"] == "s-export"
    assert analysis["file"]["path"] == str(log.path)
    assert analysis["file"]["sizeBytes"] > 0
    assert analysis["hasTimeline"] is True
    text = log.format_trace()
    assert "Totals:" in text
    assert "hello" in text
    assert " B)" not in text
    dest = tmp_path / "copy.jsonl"
    out = log.export(dest)
    assert out == dest
    assert dest.read_text(encoding="utf-8") == log.path.read_text(encoding="utf-8")
    analysis_path = log.export(tmp_path / "s.trace.json", kind="analysis")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "s-export"
    round_text = log.format_trace(round_n=1)
    assert "Round 1:" in round_text
    assert "[user]" in round_text


def test_format_trace_duration_has_no_space():
    text = format_trace_text(
        {
            "session_id": "s",
            "file": {},
            "meta": {},
            "totals": {"rounds": 1, "tokens": {}, "costUsd": 0, "toolCalls": 0, "elapsedMs": 905_000},
            "rounds": [{"title": "q", "durationMs": 3_905_000, "tokens": {}, "costUsd": 0, "toolCalls": 0}],
        }
    )
    assert "15m5s" in text
    assert "15m 5s" not in text
    assert "1h5m5s" in text


def test_format_trace_uses_human_file_size():
    text = format_trace_text(
        {
            "session_id": "s",
            "file": {"name": "s.jsonl", "path": "/tmp/s.jsonl", "sizeBytes": 4_141_579},
            "meta": {},
            "totals": {
                "rounds": 0,
                "tokens": {},
                "costUsd": 0,
                "toolCalls": 0,
                "toolErrors": 0,
                "elapsedMs": 0,
            },
            "rounds": [],
        }
    )
    assert "3.9MB" in text
    assert "4,141,579" not in text


def test_automated_turn_is_its_own_kind_and_not_a_user_title():
    """A background job's report opens a round, but nobody typed it."""
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="run the tests"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:03.100Z", type="assistant", content="started"),
        _e("2026-01-01T00:05:00.000Z", type="user", automated=True,
           content="[Background terminal #1 (term_1) finished: exit 0 after 5m]\nCommand: pytest"),
        _e("2026-01-01T00:05:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:05:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:05:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:05:03.100Z", type="assistant", content="all green"),
    ]
    first, second = analyze_entries(entries)["rounds"]
    assert first["title"] == "run the tests" and first["automated"] is False
    assert second["automated"] is True
    assert second["title"].startswith("[Background terminal #1")
    assert [c["kind"] for c in second["calls"]][:2] == ["automated", "model"]


def test_search_conversation_matches_queries_and_answers_with_turn_numbers():
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="how do I start sglang?"),
        _e("2026-01-01T00:00:01.000Z", type="assistant", content="Run it on gpu 29.206.1.234."),
        _e("2026-01-01T00:00:02.000Z", type="user", content="what about vllm?"),
        _e("2026-01-01T00:00:03.000Z", type="assistant", content="Use gpu4 at 28.8.205.124."),
    ]
    q_hits = search_conversation(entries, "sglang")
    assert len(q_hits) == 1 and q_hits[0]["turn"] == 1 and q_hits[0]["role"] == "user"
    a_hits = search_conversation(entries, "28.8.205.124")
    assert len(a_hits) == 1 and a_hits[0]["turn"] == 2 and a_hits[0]["role"] == "assistant"


def test_window_preamble_with_relayed_mention_is_not_classified_as_steering_or_round_title():
    from agentica.memory.trace import chat_turn_page

    preamble = (
        "<context_window>\nCurrent context window 1.\n</context_window>\n\n"
        '<session_notes path="/tmp/s.notes.md">\n- Handled `__RELAYED__` and `steer: true`\n</session_notes>\n\n'
        "<dropped_span>\n- assistant: fixed\n</dropped_span>\n\n"
        "New context window started. Continue from session notes and search_session."
    )
    entries = [
        _e("2026-01-01T00:00:00.000Z", type="user", content="first turn"),
        _e("2026-01-01T00:00:01.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:02.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:03.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:03.100Z", type="assistant", content="first reply"),
        _e("2026-01-01T00:00:04.000Z", type="compact_boundary", trigger="manual", summary=""),
        _e("2026-01-01T00:00:04.100Z", type="user", content=preamble),
        _e("2026-01-01T00:00:05.000Z", type="user", content="second turn after compact"),
        _e("2026-01-01T00:00:06.000Z", type="event", name="request_begin"),
        _e("2026-01-01T00:00:07.000Z", type="event", name="text"),
        _e("2026-01-01T00:00:08.000Z", type="event", name="request_end", status="completed"),
        _e("2026-01-01T00:00:08.100Z", type="assistant", content="second reply"),
    ]
    out = analyze_entries(entries)
    non_compact = [r for r in out["rounds"] if not r.get("compaction")]
    assert [r["title"] for r in non_compact] == ["first turn", "second turn after compact"]
    for r in out["rounds"]:
        assert not any(c["kind"] == "steering" for c in r["calls"])

    page = chat_turn_page(entries, turns=1)
    assert page["older_turns"] == 1
    assert page["events"][0]["type"] == "compact_boundary"
    assert page["events"][1]["content"] == preamble
    assert page["events"][2]["content"] == "second turn after compact"


class TestChatTurnPageIncludeTurn:
    """A search hit deep in a long log is reached in one page, not N/6 of them."""

    N = 40

    def _entries(self):
        out = []
        for i in range(1, self.N + 1):
            out.append(_e(f"2026-01-01T00:{i:02d}:00.000Z", type="user", content=f"q{i}"))
            out.append(_e(f"2026-01-01T00:{i:02d}:01.000Z", type="event", name="request_begin"))
            out.append(_e(f"2026-01-01T00:{i:02d}:02.000Z", type="assistant", content=f"a{i}"))
        return out

    @staticmethod
    def _user_turns(page):
        return [int(e["content"][1:]) for e in page["events"] if e["type"] == "user"]

    def test_without_it_the_page_is_the_last_few_turns(self):
        from agentica.memory.trace import chat_turn_page
        page = chat_turn_page(self._entries(), 6)
        assert self._user_turns(page) == list(range(35, 41)) and page["older_turns"] == 34

    def test_one_read_reaches_the_turn_with_one_turn_of_context(self):
        from agentica.memory.trace import chat_turn_page
        page = chat_turn_page(self._entries(), 6, include_turn=10)
        turns = self._user_turns(page)
        assert turns[0] == 9 and turns[-1] == 40, "from the turn before the hit through the end"
        # What the web client checks to stop paging: the hit is no longer "older".
        assert page["older_turns"] < 10 and page["older_turns"] == 8
        assert page["has_more"] is True and page["next_before"] > 0

    def test_first_turn_reaches_the_start_of_the_log(self):
        from agentica.memory.trace import chat_turn_page
        for t in (1, 2):
            page = chat_turn_page(self._entries(), 6, include_turn=t)
            assert self._user_turns(page)[0] == 1 and page["older_turns"] == 0 and page["has_more"] is False

    def test_a_turn_already_in_the_window_changes_nothing(self):
        from agentica.memory.trace import chat_turn_page
        plain = chat_turn_page(self._entries(), 6)
        assert chat_turn_page(self._entries(), 6, include_turn=38) == plain
        assert chat_turn_page(self._entries(), 6, include_turn=999) == plain  # beyond the log

    def test_pages_from_a_cursor_only_reach_back_from_that_cursor(self):
        from agentica.memory.trace import chat_turn_page
        first = chat_turn_page(self._entries(), 6)
        page = chat_turn_page(self._entries(), 6, before=first["next_before"], include_turn=3)
        turns = self._user_turns(page)
        assert turns[0] == 2 and turns[-1] == 34, "stops where the loaded window begins"
        assert page["older_turns"] == 1
