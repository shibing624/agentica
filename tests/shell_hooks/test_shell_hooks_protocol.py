# -*- coding: utf-8 -*-
"""The hook wire: payload documents out, replies in.

Every case here is a "would silently do the wrong thing" case: a reply that
cannot be understood must be no decision, never an allow.
"""

from __future__ import annotations

import json

from agentica.shell_hooks.protocol import build_payload, parse_reply


class TestPayload:
    def test_the_event_name_is_in_the_json_not_argv(self):
        doc = build_payload("needs.approval", session_id="s1", work_dir="/w")
        assert doc["hook_event_name"] == "needs.approval"

    def test_optional_fields_are_omitted_not_null(self):
        doc = build_payload("run.started", session_id="s1")
        assert "question" not in doc
        assert "options" not in doc
        assert "tool_call_id" not in doc

    def test_an_approval_carries_its_correlation_id_and_options(self):
        doc = build_payload(
            "needs.approval",
            session_id="s1",
            work_dir="/w",
            extra={
                "tool_name": "execute",
                "tool_call_id": "call_1",
                "options": ["allow", "deny"],
                "question": "run it?",
                "preview": "rm -rf build",
            },
        )
        assert doc["tool_call_id"] == "call_1"
        assert doc["options"] == ["allow", "deny"]

    def test_a_question_carries_no_decision_vocabulary(self):
        """The reply to needs.input is free text; offering the four approval
        words would invite a consumer to answer with one."""
        doc = build_payload("needs.input", session_id="s1", extra={"question": "which?"},
                            options=("date-fns",))
        assert "decision" not in doc
        assert doc["options"] == ["date-fns"]

    def test_prompt_is_clipped(self):
        doc = build_payload("needs.approval", session_id="s1", prompt="x" * 900)
        assert len(doc["prompt"]) < 900

    def test_cwd_defaults_to_the_process_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        doc = build_payload("run.started", session_id="s1")
        assert doc["cwd"] == str(tmp_path)

    def test_session_id_and_process_identity_are_always_present(self):
        doc = build_payload("run.started")
        assert doc["session_id"].startswith("process-")
        assert doc["transport"]["ppid"] > 0
        assert doc["transport"]["cwd"] == doc["cwd"]

    def test_transport_tolerates_stdin_without_fileno(self, monkeypatch):
        import agentica.notify.transport as transport_mod

        monkeypatch.setattr(transport_mod.sys, "stdin", object())
        doc = build_payload("run.started", work_dir="/tmp")
        assert "tty" not in doc["transport"]


class TestReply:
    def test_a_decision(self):
        assert parse_reply('{"decision": "allow"}', "needs.approval") == {"decision": "allow"}

    def test_an_answer(self):
        assert parse_reply('{"answer": "date-fns"}', "needs.input") == {"answer": "date-fns"}

    def test_all_four_decision_words(self):
        for word in ("allow", "allow_prefix", "deny", "deny_prefix"):
            assert parse_reply(json.dumps({"decision": word}), "needs.approval") == {
                "decision": word
            }

    def test_an_unknown_decision_word_is_no_decision(self):
        assert parse_reply('{"decision": "sure"}', "needs.approval") is None

    def test_an_approval_body_is_not_an_answer(self):
        """``{decision: allow}`` is not a reply to a question, and must not be
        read as the string "allow"."""
        assert parse_reply('{"decision": "allow"}', "needs.input") is None

    def test_a_non_approval_body_is_not_a_decision(self):
        assert parse_reply('{"answer": "yes"}', "needs.approval") is None

    def test_blank_or_missing_is_no_decision(self):
        for text in ("", "   ", None, "not json", "{}", "[]", '{"decision": null}'):
            assert parse_reply(text, "needs.approval") is None

    def test_an_empty_answer_is_not_an_answer(self):
        assert parse_reply('{"answer": "   "}', "needs.input") is None

    def test_trailing_noise_is_tolerated_when_a_document_is_present(self):
        assert parse_reply('{"decision": "deny"}\n', "needs.approval") == {"decision": "deny"}

    def test_request_id_must_match(self):
        reply = '{"request_id":"r1","decision":"allow"}'
        assert parse_reply(reply, "needs.approval", request_id="r1") == {
            "decision": "allow"
        }
        assert parse_reply(reply, "needs.approval", request_id="r2") is None
        assert parse_reply('{"decision":"allow"}', "needs.approval", request_id="r1") is None
