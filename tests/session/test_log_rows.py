# -*- coding: utf-8 -*-
"""What a session-log user row means — pinned once, for every reader.

The shapes below are the ones real logs hold: the current continuation line,
the older wording, nested preambles from back-to-back cuts, the remaining-tokens
fragment folded after a question, a steer, a relayed row, a job report.
"""
import tempfile

from agentica.session.rows import (
    RELAYED_MARKER,
    STEER_GUIDANCE_PREFIX,
    classify_user,
    image_marker,
    row_payload,
    split_image_markers,
    strip_image_markers,
    tool_error_of,
    split_window_preamble,
    strip_window_preamble,
)
from agentica.session.log import SessionLog
from agentica.session.trace import chat_turn_page, with_row_views

CONT = "New context window started. Continue from session notes and search_session."
LEGACY_CONT = "New context window started without a conversation summary. Use search_session."


def _window(n=2):
    return f"<context_window>\nCurrent context window {n}.\n</context_window>"


def _notes(body="- goal: ship", path="/h/p/s.notes.md"):
    return f'<session_notes path="{path}">\n{body}\n</session_notes>'


def _row(content, **extra):
    return {"type": "user", "uuid": "u1", "content": content, **extra}


class TestClassifyUser:
    def test_plain_row_is_a_turn(self):
        v = classify_user(_row("hi"))
        assert (v.automated, v.steer, v.text, v.window) == (False, False, "hi", None)
        assert v.starts_turn and v.opens_round

    def test_preamble_with_question_keeps_only_the_question(self):
        content = "\n\n".join([_window(), _notes(), "<dropped_span>\n- user: x\n</dropped_span>", CONT, "现在怎么办？"])
        v = classify_user(_row(content))
        assert v.text == "现在怎么办？"
        assert v.window.notes == "- goal: ship"
        assert v.window.notes_path == "/h/p/s.notes.md"
        assert v.starts_turn

    def test_older_continuation_wording_does_not_leak_into_the_question(self):
        # Seen in real logs: this wording used to survive in search / Web.
        content = "\n\n".join([_window(), LEGACY_CONT, "排查一下"])
        assert classify_user(_row(content)).text == "排查一下"

    def test_nested_preambles_are_all_peeled(self):
        content = f"{_window(3)}\n\n{_window(2)}\n\n现在怎么办？"
        assert classify_user(_row(content)).text == "现在怎么办？"

    def test_remaining_tokens_fragment_after_the_question_is_dropped(self):
        tail = "<context_window>\nYou have 13572 tokens left in this context window.\n</context_window>"
        v = classify_user(_row(f"{_window(1)}\n\n换个模型后多了东西\n\n{tail}"))
        assert v.text == "换个模型后多了东西"

    def test_bare_preamble_is_chrome_not_a_turn(self):
        v = classify_user(_row(f"{_window()}\n\n{_notes()}\n\n{CONT}"))
        assert v.is_pure_window and v.text == ""
        assert not v.starts_turn and not v.opens_round

    def test_bare_window_without_notes(self):
        v = classify_user(_row(_window()))
        assert v.is_pure_window and v.window.notes == ""

    def test_notes_that_quote_closing_tags_stay_in_the_notes(self):
        body = "- parsed `</session_notes>`, `__RELAYED__` and `[User guidance received while you were working]`"
        v = classify_user(_row(f"{_window()}\n\n{_notes(body)}\n\n{CONT}\n\nq"))
        assert v.text == "q"
        assert "__RELAYED__" in v.window.notes
        assert not v.steer

    def test_steer_flag_and_prefix_both_mark_a_steer(self):
        assert classify_user(_row("用默认即可", steer=True)).steer
        folded = classify_user(_row(f"{STEER_GUIDANCE_PREFIX}\n用默认即可"))
        assert folded.steer and folded.text == "用默认即可"
        relayed = classify_user(_row(f"{RELAYED_MARKER}\n来自别处"))
        assert relayed.steer and relayed.text == "来自别处"
        for v in (folded, relayed):
            assert not v.starts_turn and not v.opens_round

    def test_automated_and_steer_are_independent(self):
        report = classify_user(_row("[Background terminal #1 failed]", automated=True))
        assert report.automated and not report.steer
        assert not report.starts_turn
        assert report.opens_round, "a job report is answered by a model round"
        peer = classify_user(_row("hello from a peer", automated=True, steer=True))
        assert peer.automated and peer.steer
        assert not peer.opens_round, "a mid-run peer message joins the running round"

    def test_automated_text_is_shown_as_logged(self):
        content = f"{STEER_GUIDANCE_PREFIX}\nquoted in a report"
        assert classify_user(_row(content, automated=True)).text == content

    def test_preamble_then_steer_prefix(self):
        v = classify_user(_row(f"{_window()}\n\n{STEER_GUIDANCE_PREFIX}\n再加一条"))
        assert v.steer and v.text == "再加一条" and v.window is not None
        assert not v.is_pure_window

    def test_non_string_content_reads_as_empty(self):
        assert classify_user(_row(None)).text == ""


class TestPreambleHelpers:
    def test_strip_leaves_a_plain_row_alone(self):
        assert strip_window_preamble("plain") == "plain"
        assert split_window_preamble("plain") is None

    def test_a_question_that_merely_mentions_the_tag_is_not_a_preamble(self):
        text = "为什么 <context_window> 会出现在回答里？"
        assert split_window_preamble(text) is None


class TestWire:
    def test_payload_is_the_view(self):
        payload = row_payload(_row(f"{_window()}\n\n{_notes()}\n\n{CONT}\n\nq"))
        assert payload == {
            "automated": False,
            "steer": False,
            "text": "q",
            "images": [],
            "window": {"notes": "- goal: ship", "notes_path": "/h/p/s.notes.md"},
        }

    def test_other_rows_carry_no_payload(self):
        assert row_payload({"type": "assistant", "content": "x"}) is None

    def test_with_row_views_marks_only_user_and_tool_rows_and_leaves_the_log_rows_alone(self):
        entries = [_row("hi"), {"type": "assistant", "content": "yo"}, {"type": "tool", "content": "ok"}]
        out = with_row_views(entries)
        assert out[0]["row"]["text"] == "hi"
        assert "row" not in out[1]
        assert out[2]["row"] == {"error": False, "text": "ok", "steers": []}
        assert "row" not in entries[0]

    def test_turn_numbers_count_from_the_start_of_the_log(self):
        entries = [
            _row("q1"),
            _row("steer", steer=True),
            _row("report", automated=True),
            _row(f"{_window()}\n\n{_notes()}\n\n{CONT}"),
            _row("q2"),
        ]
        out = with_row_views(entries, turns_before=4)
        assert [e["row"].get("turn") for e in out] == [5, None, None, None, 6]


IMG = "a" * 64 + ".png"


class TestImagesAndTools:
    def test_image_markers_split_into_prose_and_names(self):
        content = "看下这个" + image_marker([IMG, "b" * 64 + ".jpg"]) + "\n\n[图片内容]\n一张猫"
        assert split_image_markers(content) == ("看下这个", [IMG, "b" * 64 + ".jpg"])
        assert strip_image_markers(image_marker([IMG]).lstrip()) == ""
        assert strip_image_markers("提到 [图片: x] 的文字") == "提到 [图片: x] 的文字"

    def test_user_payload_carries_images_not_markers(self):
        payload = row_payload(_row("hi" + image_marker([IMG])))
        assert (payload["text"], payload["images"]) == ("hi", [IMG])

    def test_automated_rows_are_shown_as_logged(self):
        content = "report" + image_marker([IMG])
        payload = row_payload(_row(content, automated=True))
        assert (payload["text"], payload["images"]) == (content, [])

    def test_tool_error_reads_either_spelling(self):
        assert tool_error_of({"is_error": True}) is True
        assert tool_error_of({"tool_call_error": True}) is True
        assert tool_error_of({"is_error": False, "tool_call_error": True}) is True
        assert tool_error_of({}) is False

    def test_tool_payload_splits_folded_steers(self):
        content = f"out\n\n{STEER_GUIDANCE_PREFIX}\nfoo\n\n{STEER_GUIDANCE_PREFIX}\nbar"
        payload = row_payload({"type": "tool", "content": content, "tool_call_error": True})
        assert payload == {"error": True, "text": "out", "steers": ["foo", "bar"]}

    def test_tool_payload_without_steers_keeps_output_whole(self):
        payload = row_payload({"type": "tool_audit", "content": "a\n\nb"})
        assert payload == {"error": False, "text": "a\n\nb", "steers": []}

    def test_chat_page_serves_row_on_every_user_event(self):
        entries = [
            _row("old", uuid="a"),
            {"type": "assistant", "content": "ok"},
            {"type": "compact_boundary", "trigger": "auto"},
            _row(f"{_window()}\n\n{CONT}\n\nnext", uuid="b"),
            _row("btw", uuid="c", steer=True),
        ]
        page = chat_turn_page(entries, turns=10)
        users = [e for e in page["events"] if e["type"] == "user"]
        assert [e["row"]["text"] for e in users] == ["old", "next", "btw"]
        assert [e["row"]["steer"] for e in users] == [False, False, True]
        assert users[1]["row"]["window"] is not None


class TestEveryReaderAgrees:
    """The same log, read by each consumer, gives the same turns."""

    def _log(self, tmp):
        log = SessionLog("s", base_dir=tmp)
        log.append("user", f"{_window(1)}\n\n{LEGACY_CONT}")  # bare preamble row
        log.append("user", "第一问")
        log.append("assistant", "答一")
        log.append("user", "[Background terminal #1 failed]", automated=True)
        log.append("assistant", "收到")
        log.append("user", "插话", steer=True)
        log.append("user", f"{_window(2)}\n\n{CONT}\n\n第二问")
        return log

    def test_counts_titles_fork_points_and_search_see_the_same_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp)
            assert log.first_user_message() == "第一问"
            assert log.trajectory_stats()["turns"] == 2
            assert [m["content"] for m in log.list_user_messages()] == [
                "第二问", "[Background terminal #1 failed]", "第一问",
            ]
            # The model-facing index keeps what the model read (a report, a
            # steer), flagged — but never the bare preamble or its wording.
            hits = log.list_user_questions()
            assert [h["snippet"] for h in hits] == ["第二问", "插话", "[Background terminal #1 failed]", "第一问"]
            assert [bool(h.get("automated")) for h in hits] == [False, False, True, False]
            assert log.search_entries("New context window") == []
            page = chat_turn_page(log.iter_raw_entries(), turns=1)
            assert page["older_turns"] == 1

    def test_legacy_serialized_anthropic_tool_result_user_row_is_ignored_and_unpacked_on_load(self):
        import asyncio
        import json

        from agentica.model.anthropic.claude import Claude
        from agentica.model.message import Message
        from agentica.session.rows import mark_window_replays

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("legacy-bad", base_dir=tmp)
            log.append("user", "查下数据")
            calls = [{"id": "toolu_bad_1", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}]
            log.append(
                "assistant", "",
                tool_calls=calls,
                provider_data={"anthropic_model": "claude-sonnet-4-5"},
            )
            log.append("tool", "ok-result", tool_call_id="toolu_bad_1", tool_name="fetch_url")
            log.append_compact_boundary("", trigger="auto")
            log.append("user", f"{_window(1)}\n\n{CONT}\n\n查下数据", window_replay=True)
            legacy_asst = json.dumps([
                {"type": "thinking", "thinking": "t", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_bad_1", "name": "fetch_url", "input": {}},
            ])
            log.append("assistant", legacy_asst, tool_calls=calls, window_replay=True)
            # Historical bug: tool_result container written as a plain user JSON row.
            legacy_user_tr = json.dumps([
                {"type": "tool_result", "tool_use_id": "toolu_bad_1", "content": "ok-result", "is_error": True},
            ])
            log.append("user", legacy_user_tr)
            log.append("assistant", "完成")

            raw = list(log.iter_raw_entries())
            marked = mark_window_replays(raw)
            assert marked[-2]["type"] == "tool"
            assert marked[-2]["tool_call_id"] == "toolu_bad_1"
            assert marked[-2]["window_replay"] is True
            assert log.trajectory_stats()["turns"] == 1
            assert [m["content"] for m in log.list_user_messages()] == ["查下数据"]
            assert log.search_entries("toolu_bad_1") == []

            page = chat_turn_page(raw, turns=10)["events"]
            user_texts = [e["row"]["text"] for e in page if e["type"] == "user" and e["row"]["text"]]
            assert user_texts == ["查下数据"]
            assert [e["tool_call_id"] for e in page if e["type"] == "tool"] == ["toolu_bad_1"]

            replayed = log.load()
            assert [m["role"] for m in replayed] == ["user", "assistant", "tool", "assistant"]
            assert replayed[1]["content"] == ""
            assert replayed[1]["provider_data"]["anthropic_model"] == "claude-sonnet-4-5"
            assert len(replayed[1]["provider_data"]["anthropic_blocks"]) == 2
            assert replayed[2]["tool_call_id"] == "toolu_bad_1"
            assert replayed[2]["tool_name"] == "fetch_url"
            assert replayed[2]["tool_call_error"] is True
            assert replayed[2]["content"] == "ok-result"

            # Origin check prevents replaying one model's thinking signature to another model.
            same_fmt, _ = asyncio.run(
                Claude(id="claude-sonnet-4-5", api_key="k").format_messages([Message(**m) for m in replayed])
            )
            diff_fmt, _ = asyncio.run(
                Claude(id="claude-opus-4-5", api_key="k").format_messages([Message(**m) for m in replayed])
            )
            assert [b["type"] for b in same_fmt[1]["content"]] == ["thinking", "tool_use"]
            assert [b["type"] for b in diff_fmt[1]["content"]] == ["tool_use"]

    def test_legacy_unknown_origin_assistant_drops_thinking_signature(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("legacy-unknown-model", base_dir=tmp)
            log.append("user", "q")
            calls = [{"id": "toolu_u1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            log.append("assistant", "", tool_calls=calls)
            log.append("tool", "ok", tool_call_id="toolu_u1", tool_name="read_file")
            log.append_compact_boundary("", trigger="auto")
            log.append("user", f"{_window(1)}\n\n{CONT}\n\nq", window_replay=True)
            legacy_asst = json.dumps([
                {"type": "thinking", "thinking": "t", "signature": "sig_unknown"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_u1", "name": "read_file", "input": {}},
            ])
            log.append("assistant", legacy_asst, tool_calls=calls, window_replay=True)
            log.append("user", json.dumps([
                {"type": "tool_result", "tool_use_id": "toolu_u1", "content": "ok"},
            ]))

            replayed = log.load()
            assert replayed[1]["content"] == "checking"
            assert "provider_data" not in replayed[1]
            assert replayed[1]["tool_calls"][0]["id"] == "toolu_u1"

    def test_real_user_message_with_tool_result_json_is_kept(self):
        import json

        pasted_json = json.dumps([
            {"type": "tool_result", "tool_use_id": "toolu_pasted_1", "content": "example"}
        ])
        with tempfile.TemporaryDirectory() as tmp:
            # Case 1: no compact_boundary at all.
            plain = SessionLog("plain-json-user", base_dir=tmp)
            plain.append("user", pasted_json)
            plain.append("assistant", "got it")
            assert [m["content"] for m in plain.list_user_messages()] == [pasted_json]
            assert plain.trajectory_stats()["turns"] == 1
            assert [m["role"] for m in plain.load()] == ["user", "assistant"]
            page_plain = chat_turn_page(plain.iter_raw_entries(), turns=10)["events"]
            assert [e["row"]["text"] for e in page_plain if e["type"] == "user"] == [pasted_json]

            # Case 2: after compact_boundary, user pastes it in a new turn (no pending tool_calls).
            compacted = SessionLog("compacted-json-user", base_dir=tmp)
            compacted.append("user", "first")
            compacted.append("assistant", "ans1")
            compacted.append_compact_boundary("", trigger="auto")
            compacted.append("user", f"{_window(1)}\n\n{CONT}")
            compacted.append("user", pasted_json)
            compacted.append("assistant", "ans2")
            replayed_compacted = compacted.load()
            assert [m["role"] for m in replayed_compacted] == ["user", "assistant"]
            assert replayed_compacted[0]["content"] == pasted_json
            assert [m["content"] for m in compacted.list_user_messages()] == [pasted_json, "first"]
            page_compacted = chat_turn_page(compacted.iter_raw_entries(), turns=10)["events"]
            assert [e["row"]["text"] for e in page_compacted if e["type"] == "user" and e["row"]["text"]] == [
                "first", pasted_json,
            ]

    def test_legacy_unique_tool_result_after_double_compact_stays_visible_in_chat(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("double-compact-legacy", base_dir=tmp)
            log.append("user", "查下")
            call1 = [{"id": "toolu_c1", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}]
            log.append("assistant", "", tool_calls=call1)
            # Boundary 1 happened before toolu_c1 was flushed as a "tool" row;
            # old append_post_compact_messages left the only copy as a legacy user row.
            log.append_compact_boundary("", trigger="auto")
            log.append("user", f"{_window(1)}\n\n{CONT}\n\n查下", window_replay=True)
            log.append("assistant", "", tool_calls=call1, window_replay=True)
            log.append("user", json.dumps([
                {"type": "tool_result", "tool_use_id": "toolu_c1", "content": "only-copy"},
            ]))
            # Boundary 2 in the same turn replays both again.
            log.append_compact_boundary("", trigger="auto")
            log.append("user", f"{_window(2)}\n\n{CONT}\n\n查下", window_replay=True)
            log.append("assistant", "", tool_calls=call1, window_replay=True)
            log.append("user", json.dumps([
                {"type": "tool_result", "tool_use_id": "toolu_c1", "content": "only-copy"},
            ]))
            log.append("assistant", "done")

            page = chat_turn_page(log.iter_raw_entries(), turns=10)["events"]
            tool_events = [e for e in page if e["type"] == "tool"]
            assert len(tool_events) == 1
            assert tool_events[0]["tool_call_id"] == "toolu_c1"
            assert tool_events[0]["row"]["text"] == "only-copy"
            assert [e["row"]["text"] for e in page if e["type"] == "user" and e["row"]["text"]] == ["查下"]
            replayed = log.load()
            assert [m["role"] for m in replayed] == ["user", "assistant", "tool", "assistant"]
            assert replayed[2]["content"] == "only-copy"
