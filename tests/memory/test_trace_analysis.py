# -*- coding: utf-8 -*-
"""Trace analysis invariants — sequential scan, not timestamp slicing."""
from agentica.memory.session_log import SessionLog
from agentica.memory.trace import analyze_entries


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
        _e("2026-01-01T00:00:02.000Z", type="event", name="tool_call", tool_call_id="c1", tool_name="edit_file"),
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
        assert totals["tokens"] == {"input": 150, "cacheRead": 1900, "cacheWrite": 0, "output": 50}
        assert totals["costUsd"] > 0
        assert totals["tps"] > 0

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
        assert analyze_entries(entries)["rounds"][0]["tokens"]["input"] == 100

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
