# -*- coding: utf-8 -*-
"""Tests for SessionLog — CC-style append-only JSONL with UUID chain and compact boundaries."""

import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agentica.memory.session_log import SessionLog, assert_trajectory_equivalent
from agentica.model.message import Message
from agentica.runner import Runner


class _FakeModel:
    functions = {}


class _FakeRunResponse:
    def __init__(self, messages, tools):
        self.messages = messages
        self.tools = tools


class _FakeAgent:
    """Minimal stand-in so we can call Runner._persist_assistant_tool_calls
    without booting a real Agent / Model (no API key needed)."""

    def __init__(self, log, messages, tools):
        self._session_log = log
        self.model = _FakeModel()
        self.run_response = _FakeRunResponse(messages, tools)


@pytest.fixture
def tmp_dir():
    """Create a temp directory for session logs, cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


class RecordingIndex:
    def __init__(self):
        self.calls = []

    def index_message(self, session_id, role, content, timestamp=None):
        self.calls.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
        )


class TestSessionLogBasic:
    """Core append + load tests."""

    def test_append_and_load_messages(self, tmp_dir):
        log = SessionLog("s1", base_dir=tmp_dir)
        log.append("user", "hello")
        log.append("assistant", "hi there")
        log.append("user", "how are you?")

        messages = log.load()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hi there"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "how are you?"

    def test_load_empty_log(self, tmp_dir):
        log = SessionLog("empty", base_dir=tmp_dir)
        assert log.load() == []
        assert log.exists() is False

    def test_load_nonexistent_session(self, tmp_dir):
        log = SessionLog("nonexistent", base_dir=tmp_dir)
        assert log.load() == []

    def test_entry_count(self, tmp_dir):
        log = SessionLog("s2", base_dir=tmp_dir)
        assert log.entry_count() == 0
        log.append("user", "a")
        log.append("assistant", "b")
        assert log.entry_count() == 2

    def test_exists(self, tmp_dir):
        log = SessionLog("s3", base_dir=tmp_dir)
        assert log.exists() is False
        log.append("user", "test")
        assert log.exists() is True

    def test_append_tool_result(self, tmp_dir):
        log = SessionLog("s4", base_dir=tmp_dir)
        log.append("user", "run ls")
        log.append("tool", "file1.py\nfile2.py", tool_name="execute", tool_call_id="call-1")
        log.append("assistant", "found 2 files")

        messages = log.load()
        assert len(messages) == 3
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_error"] is False
        assert "is_error" not in messages[1]
        assert "file1.py" in messages[1]["content"]

    def test_load_preserves_assistant_replay_metadata(self, tmp_dir):
        log = SessionLog("replay", base_dir=tmp_dir)
        log.append(
            "assistant",
            "answer",
            reasoning_content="reason",
            finish_reason="stop",
            model="m1",
            usage={"input_tokens": 10, "output_tokens": 2},
        )

        messages = log.load()

        assert messages[0]["reasoning_content"] == "reason"
        assert messages[0]["finish_reason"] == "stop"
        assert messages[0]["model"] == "m1"
        assert messages[0]["usage"] == {"input_tokens": 10, "output_tokens": 2}

    def test_assistant_tool_calls_round_trip_no_orphan_tool(self, tmp_dir):
        """Resume-400 regression: assistant(tool_calls) must be persisted before
        its tool result so the replay is a valid assistant->tool pair, not an
        orphaned tool message (which 400s on OpenAI-compatible providers).
        """
        log = SessionLog("resume-400", base_dir=tmp_dir)
        log.append("user", "read the file")
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ]
        log.append("assistant", "", tool_calls=tool_calls)
        log.append("tool", "file content", tool_name="read_file", tool_call_id="call_1")
        log.append("assistant", "done", model="m1")

        messages = log.load()

        # tool_calls survive the round trip
        assistant_tc = messages[1]
        assert assistant_tc["role"] == "assistant"
        assert assistant_tc["tool_calls"] == tool_calls

        # every tool message is preceded (within the turn) by an assistant
        # carrying tool_calls -> no orphan.
        for i, m in enumerate(messages):
            if m["role"] != "tool":
                continue
            prev_tc = None
            for j in range(i - 1, -1, -1):
                if messages[j]["role"] == "assistant":
                    prev_tc = messages[j].get("tool_calls")
                    break
                if messages[j]["role"] == "user":
                    break
            assert prev_tc, f"orphaned tool message at index {i}"

    def test_multi_round_tool_interleaving_preserved(self, tmp_dir):
        """Resume-400 regression (multi-round agentic turn).

        A single turn that issues tool calls across several assistant rounds
        (e.g. ``read_file`` then ``grep``) must be persisted as an interleaved
        ``assistant(tool_calls) -> tool`` sequence for EVERY round. The earlier
        implementation grouped all assistant tool-calls before all tool results,
        which produced ``assistant(tc_A), assistant(tc_B), tool_A, tool_B`` and
        re-broke resume on OpenAI-compatible providers for any multi-round turn.
        This test drives ``Runner._persist_assistant_tool_calls`` directly and
        asserts each tool result immediately follows its requesting assistant.
        """
        log = SessionLog("multi-round", base_dir=tmp_dir)
        agent = _FakeAgent(
            log,
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[{"id": "A", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
                    provider_data={"object": "response", "output": [{"type": "function_call"}]},
                ),
                Message(role="tool", tool_call_id="A", content="file content", tool_name="read_file"),
                Message(
                    role="assistant",
                    tool_calls=[{"id": "B", "type": "function", "function": {"name": "grep", "arguments": "{}"}}],
                ),
                Message(role="tool", tool_call_id="B", content="matches", tool_name="grep"),
                Message(role="assistant", content="done"),
            ],
            tools=[
                {"tool_call_id": "A", "tool_name": "read_file", "content": "file content", "replay": True},
                {"tool_call_id": "B", "tool_name": "grep", "content": "matches", "replay": True},
            ],
        )
        Runner._persist_assistant_tool_calls(agent)

        messages = log.load()
        # Expect strict interleaving: assistant(A), tool(A), assistant(B), tool(B).
        roles = [m["role"] for m in messages]
        assert roles == ["assistant", "tool", "assistant", "tool"], roles
        assert messages[0]["tool_calls"][0]["id"] == "A"
        assert messages[0]["provider_data"]["object"] == "response"
        assert messages[1].get("tool_call_id") == "A"
        assert messages[2]["tool_calls"][0]["id"] == "B"
        assert messages[3].get("tool_call_id") == "B"

        # Invariant: every tool message is immediately preceded by an assistant
        # whose tool_calls contains that tool's id.
        for i, m in enumerate(messages):
            if m["role"] != "tool":
                continue
            prev = messages[i - 1]
            assert prev["role"] == "assistant"
            assert m["tool_call_id"] in [t["id"] for t in prev.get("tool_calls", [])]

    def test_provider_checkpoint_round_trip_and_out_of_band_attachment(self, tmp_dir):
        checkpoint = {
            "type": "openai_responses_compaction",
            "provider": "OpenAI",
            "model": "gpt-5.6-sol",
            "base_url": "https://api.openai.com/v1",
            "output": [{"id": "cmp_1", "type": "compaction", "encrypted_content": "opaque"}],
        }
        log = SessionLog("provider-checkpoint", base_dir=tmp_dir)
        log.append("user", "first", provider_checkpoint=checkpoint)
        log.append("assistant", "answer")

        loaded = log.load()
        assert loaded[0]["provider_checkpoint"] == checkpoint

        log.append_provider_checkpoint(checkpoint)
        loaded = log.load()
        assert loaded[-1]["role"] == "assistant"
        assert loaded[-1]["provider_checkpoint"] == checkpoint

    def test_sidecar_name_and_archived_coexist(self, tmp_dir):
        log = SessionLog("meta", base_dir=tmp_dir)
        log.append("user", "hello")

        SessionLog.rename_session("meta", "Important Chat", base_dir=tmp_dir)
        SessionLog.archive_session("meta", True, base_dir=tmp_dir)

        sessions = SessionLog.list_sessions(base_dir=tmp_dir)
        assert sessions[0]["name"] == "Important Chat"
        assert sessions[0]["archived"] is True

        SessionLog.archive_session("meta", False, base_dir=tmp_dir)
        sessions = SessionLog.list_sessions(base_dir=tmp_dir)
        assert sessions[0]["name"] == "Important Chat"
        assert sessions[0]["archived"] is False


class TestUUIDChain:
    """Verify UUID + parent_uuid chain (CC's core design)."""

    def test_uuid_chain_integrity(self, tmp_dir):
        """Each entry should have uuid, parent_uuid chains to previous."""
        log = SessionLog("chain", base_dir=tmp_dir)
        log.append("user", "msg1")
        log.append("assistant", "msg2")
        log.append("user", "msg3")

        with open(log.path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # First entry: parent_uuid is None
        assert entries[0]["parent_uuid"] is None
        assert entries[0]["uuid"] is not None

        # Subsequent entries chain to previous
        assert entries[1]["parent_uuid"] == entries[0]["uuid"]
        assert entries[2]["parent_uuid"] == entries[1]["uuid"]

        # All uuids are unique
        uuids = [e["uuid"] for e in entries]
        assert len(set(uuids)) == 3

    def test_compact_boundary_breaks_chain(self, tmp_dir):
        """Compact boundary should have parent_uuid=null (breaks chain)."""
        log = SessionLog("chain-break", base_dir=tmp_dir)
        log.append("user", "before")
        log.append_compact_boundary("summary")
        log.append("user", "after")

        with open(log.path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # Boundary: parent_uuid is null
        assert entries[1]["type"] == "compact_boundary"
        assert entries[1]["parent_uuid"] is None

        # Entry after boundary chains to boundary uuid
        assert entries[2]["parent_uuid"] == entries[1]["uuid"]

    def test_append_returns_uuid(self, tmp_dir):
        log = SessionLog("ret-uuid", base_dir=tmp_dir)
        u1 = log.append("user", "hello")
        u2 = log.append("assistant", "hi")
        assert u1 != u2
        assert len(u1) == 36  # UUID format


class TestCompactBoundary:
    """Compact boundary = resume checkpoint."""

    def test_resume_from_compact_boundary(self, tmp_dir):
        """Messages before boundary should be replaced by summary."""
        log = SessionLog("compact1", base_dir=tmp_dir)

        log.append("user", "old message 1")
        log.append("assistant", "old response 1")
        log.append("user", "old message 2")
        log.append("assistant", "old response 2")

        log.append_compact_boundary("User asked 2 questions, assistant answered both.")

        log.append("user", "new question")
        log.append("assistant", "new answer")

        messages = log.load()

        # Should NOT contain old messages
        assert not any("old message" in m["content"] for m in messages)

        # Should contain: resumed summary + new messages
        assert len(messages) == 4  # summary(user) + ack(assistant) + user + assistant
        assert "[Resumed session" in messages[0]["content"]
        assert "User asked 2 questions" in messages[0]["content"]
        assert messages[2]["content"] == "new question"
        assert messages[3]["content"] == "new answer"

    def test_multiple_compact_boundaries(self, tmp_dir):
        """Only the LAST boundary should be used for resume."""
        log = SessionLog("compact2", base_dir=tmp_dir)

        log.append("user", "round 1")
        log.append_compact_boundary("Summary of round 1")

        log.append("user", "round 2")
        log.append_compact_boundary("Summary of rounds 1+2")

        log.append("user", "round 3")

        messages = log.load()

        assert any("Summary of rounds 1+2" in m["content"] for m in messages)
        assert messages[-1]["content"] == "round 3"

    def test_no_boundary_replays_all(self, tmp_dir):
        """Without any boundary, all messages are replayed."""
        log = SessionLog("no-boundary", base_dir=tmp_dir)
        log.append("user", "msg1")
        log.append("assistant", "msg2")
        log.append("user", "msg3")

        messages = log.load()
        assert len(messages) == 3
        assert messages[0]["content"] == "msg1"

    def test_boundary_at_end(self, tmp_dir):
        """Boundary at the very end with no new messages after it."""
        log = SessionLog("boundary-end", base_dir=tmp_dir)
        log.append("user", "question")
        log.append("assistant", "answer")
        log.append_compact_boundary("Conversation about a question")

        messages = log.load()
        assert len(messages) == 2
        assert "[Resumed session" in messages[0]["content"]


class TestJSONLFormat:
    """Verify the file format matches CC conventions."""

    def test_entry_format_matches_cc(self, tmp_dir):
        """Entries should use type=role, have uuid/parent_uuid/session_id/cwd/ts."""
        log = SessionLog("format", base_dir=tmp_dir)
        log.append("user", "hello")
        log.append("assistant", "hi")
        log.append_compact_boundary("summary")

        with open(log.path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # CC format: type is the role directly
        assert entries[0]["type"] == "user"
        assert entries[1]["type"] == "assistant"
        assert entries[2]["type"] == "compact_boundary"

        # All entries have uuid, parent_uuid, timestamp (ISO), session_id, cwd, version, git_branch
        for e in entries:
            assert "uuid" in e
            assert "parent_uuid" in e
            assert "timestamp" in e
            assert isinstance(e["timestamp"], str)  # ISO 8601 string
            assert "T" in e["timestamp"]  # ISO format contains T
            assert "session_id" in e
            assert e["session_id"] == "format"
            assert "cwd" in e
            assert "version" in e
            assert "git_branch" in e

    def test_unicode_content(self, tmp_dir):
        log = SessionLog("unicode", base_dir=tmp_dir)
        log.append("user", "你好世界 🌍")
        log.append("assistant", "こんにちは")

        messages = log.load()
        assert messages[0]["content"] == "你好世界 🌍"
        assert messages[1]["content"] == "こんにちは"

    def test_last_uuid_restored_on_load(self, tmp_dir):
        """After load(), subsequent appends should chain correctly."""
        log1 = SessionLog("chain-restore", base_dir=tmp_dir)
        log1.append("user", "msg1")
        log1.append("assistant", "msg2")

        # New instance (simulates process restart)
        log2 = SessionLog("chain-restore", base_dir=tmp_dir)
        log2.load()  # restores _last_uuid
        log2.append("user", "msg3")

        with open(log2.path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # msg3 should chain to msg2
        assert entries[2]["parent_uuid"] == entries[1]["uuid"]


class TestListSessions:
    """Test session listing for /resume."""

    def test_list_sessions(self, tmp_dir):
        log1 = SessionLog("session-a", base_dir=tmp_dir)
        log1.append("user", "hello a")
        log2 = SessionLog("session-b", base_dir=tmp_dir)
        log2.append("user", "hello b")

        sessions = SessionLog.list_sessions(base_dir=tmp_dir)
        assert len(sessions) == 2
        ids = [s["session_id"] for s in sessions]
        assert "session-a" in ids
        assert "session-b" in ids
        # Each session has required fields
        for s in sessions:
            assert "path" in s
            assert "size_bytes" in s
            assert s["size_bytes"] > 0

    def test_list_sessions_empty(self, tmp_dir):
        sessions = SessionLog.list_sessions(base_dir=tmp_dir)
        assert sessions == []


class TestSessionPreview:
    """session_preview: first user message + user turn count for /resume list."""

    def test_first_user_and_turn_count(self, tmp_dir):
        log = SessionLog("p1", base_dir=tmp_dir)
        log.append("system", "sys-ctx")
        log.append("user", "Build a CLI tool for parsing nginx logs")
        log.append("assistant", "sure")
        log.append("user", "now add tests")
        pv = SessionLog.session_preview(log.path)
        assert pv["first_user"] == "Build a CLI tool for parsing nginx logs"
        assert pv["user_count"] == 2

    def test_empty_session(self, tmp_dir):
        log = SessionLog("p2", base_dir=tmp_dir)
        log.append("system", "sys-ctx")
        pv = SessionLog.session_preview(log.path)
        assert pv["first_user"] == ""
        assert pv["user_count"] == 0

    def test_truncates_long_first_user(self, tmp_dir):
        log = SessionLog("p3", base_dir=tmp_dir)
        long_msg = "x" * 500
        log.append("user", long_msg)
        pv = SessionLog.session_preview(log.path, max_chars=80)
        assert len(pv["first_user"]) == 80
        assert pv["user_count"] == 1

    def test_malformed_lines_skipped(self, tmp_dir):
        log = SessionLog("p4", base_dir=tmp_dir)
        log.append("user", "real first message")
        # Append a couple of garbage lines directly to the file.
        with open(log.path, "a") as f:
            f.write("{not valid json\n\n")
        pv = SessionLog.session_preview(log.path)
        assert pv["first_user"] == "real first message"
        assert pv["user_count"] == 1


class TestToolResultLogging:
    """Test that tool results are properly logged and restored."""

    def test_tool_result_with_metadata(self, tmp_dir):
        log = SessionLog("tool-test", base_dir=tmp_dir)
        log.append("user", "list files")
        log.append("tool", "file1.py\nfile2.py", tool_name="execute", tool_call_id="call-123", is_error=False)
        log.append("assistant", "Found 2 files")

        # Verify JSONL has tool metadata
        with open(log.path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        assert entries[1]["type"] == "tool"
        assert entries[1]["tool_name"] == "execute"
        assert entries[1]["tool_call_id"] == "call-123"
        assert entries[1]["is_error"] is False

        # Verify load restores tool message
        messages = log.load()
        assert len(messages) == 3
        assert messages[1]["role"] == "tool"

    def test_tool_error_metadata_maps_to_replay_field(self, tmp_dir):
        log = SessionLog("tool-error-test", base_dir=tmp_dir)
        log.append("user", "run checks")
        log.append("tool", "command failed", tool_name="execute", is_error=True)

        messages = log.load()

        assert messages[1]["tool_call_error"] is True


class TestSessionIndexDualWrite:
    def test_indexes_regular_messages(self, tmp_dir):
        index = RecordingIndex()
        log = SessionLog("dual-write-msg", base_dir=tmp_dir, search_index=index)

        log.append("user", "hello index")

        assert len(index.calls) == 1
        assert index.calls[0]["session_id"] == "dual-write-msg"
        assert index.calls[0]["role"] == "user"
        assert index.calls[0]["content"] == "hello index"

    def test_indexes_compact_boundary_summary(self, tmp_dir):
        index = RecordingIndex()
        log = SessionLog("dual-write-boundary", base_dir=tmp_dir, search_index=index)

        log.append_compact_boundary("summary for search")

        assert len(index.calls) == 1
        assert index.calls[0]["role"] == "compact_boundary"
        assert index.calls[0]["content"] == "summary for search"

    def test_marks_index_unhealthy_after_failure(self, tmp_dir):
        class FailingIndex:
            def index_message(self, *args, **kwargs):
                raise RuntimeError("fts unavailable")

        log = SessionLog("dual-write-fail", base_dir=tmp_dir, search_index=FailingIndex())

        log.append("user", "hello")

        assert log.search_index_healthy is False


class TestResumeAt:
    """Test query-granularity resume (CC's --resume-session-at)."""

    def test_resume_at_truncates(self, tmp_dir):
        """load(resume_at=uuid) should truncate at that message."""
        log = SessionLog("resume-at", base_dir=tmp_dir)
        u1 = log.append("user", "question 1")
        log.append("assistant", "answer 1")
        u2 = log.append("user", "question 2")
        log.append("assistant", "answer 2")
        log.append("user", "question 3")
        log.append("assistant", "answer 3")

        # Resume at the second user message (discard everything after)
        messages = log.load(resume_at=u2)
        assert len(messages) == 3  # q1 + a1 + q2
        assert messages[0]["content"] == "question 1"
        assert messages[2]["content"] == "question 2"
        assert log._last_uuid == u2

    def test_resume_at_with_boundary(self, tmp_dir):
        """resume_at should respect compact boundaries."""
        log = SessionLog("resume-at-boundary", base_dir=tmp_dir)
        log.append("user", "old")
        log.append("assistant", "old response")
        log.append_compact_boundary("Summary of old conversation")
        u1 = log.append("user", "new q1")
        log.append("assistant", "new a1")
        u2 = log.append("user", "new q2")
        log.append("assistant", "new a2")

        messages = log.load(resume_at=u2)
        assert any("[Resumed session" in m["content"] for m in messages)
        assert messages[-1]["content"] == "new q2"
        assert not any("old" == m["content"] for m in messages)

    def test_resume_at_nonexistent_uuid(self, tmp_dir):
        """resume_at with unknown uuid should return all messages."""
        log = SessionLog("resume-at-bad", base_dir=tmp_dir)
        log.append("user", "q1")
        log.append("assistant", "a1")

        messages = log.load(resume_at="nonexistent-uuid")
        assert len(messages) == 2


class TestFork:
    """Test session forking (CC's --fork-session)."""

    def test_fork_creates_new_session(self, tmp_dir):
        log = SessionLog("original", base_dir=tmp_dir)
        log.append("user", "msg1")
        log.append("assistant", "msg2")
        log.append("user", "msg3")

        forked = log.fork("forked-session")
        assert forked.session_id == "forked-session"
        assert forked.path.exists()

        with open(forked.path, "r") as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 3
        assert all(e["session_id"] == "forked-session" for e in entries)

    def test_fork_at_uuid(self, tmp_dir):
        log = SessionLog("orig2", base_dir=tmp_dir)
        u1 = log.append("user", "msg1")
        log.append("assistant", "msg2")
        log.append("user", "msg3")

        forked = log.fork("forked-at", at_uuid=u1)
        with open(forked.path, "r") as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 1
        assert entries[0]["content"] == "msg1"

    def test_fork_preserves_original(self, tmp_dir):
        log = SessionLog("orig3", base_dir=tmp_dir)
        log.append("user", "msg1")
        log.append("assistant", "msg2")

        log.fork("forked3")
        assert log.entry_count() == 2


class TestListUserMessages:
    """Test user message listing for query-granularity resume picker."""

    def test_list_user_messages(self, tmp_dir):
        log = SessionLog("user-msgs", base_dir=tmp_dir)
        log.append("user", "question 1")
        log.append("assistant", "answer 1")
        log.append("user", "question 2")
        log.append("assistant", "answer 2")
        log.append("user", "question 3")

        msgs = log.list_user_messages()
        assert len(msgs) == 3
        assert "question 3" in msgs[0]["content"]
        assert "question 1" in msgs[2]["content"]
        for m in msgs:
            assert "uuid" in m
            assert "timestamp" in m

    def test_list_user_messages_limit(self, tmp_dir):
        log = SessionLog("limit-msgs", base_dir=tmp_dir)
        for i in range(10):
            log.append("user", f"q{i}")
            log.append("assistant", f"a{i}")

        msgs = log.list_user_messages(limit=3)
        assert len(msgs) == 3


class TestSidecarSessionName:
    """``set_name`` / ``get_name`` / ``clear_name`` form the public sidecar
    API used by ``/rename``. The on-disk layout
    (``<session_id>.meta.json``) is an implementation detail — these tests
    deliberately go through the public methods only so downstream
    callers (CLI, future TUI) get a single stable surface.
    """

    def test_get_name_returns_none_when_no_sidecar(self, tmp_path):
        log = SessionLog("s-no-meta", base_dir=str(tmp_path))
        assert log.get_name() is None

    def test_set_then_get_round_trips_and_strips(self, tmp_path):
        log = SessionLog("s-1", base_dir=str(tmp_path))
        log.set_name("  My research project  ")
        # set_name must strip leading/trailing whitespace so the name
        # displayed in /status matches what the user "really" wrote.
        assert log.get_name() == "My research project"

    def test_set_name_overwrites(self, tmp_path):
        log = SessionLog("s-2", base_dir=str(tmp_path))
        log.set_name("v1")
        log.set_name("v2")
        assert log.get_name() == "v2"

    def test_set_name_rejects_empty(self, tmp_path):
        log = SessionLog("s-3", base_dir=str(tmp_path))
        with pytest.raises(ValueError):
            log.set_name("")
        with pytest.raises(ValueError):
            log.set_name("   ")

    def test_clear_name_removes_sidecar(self, tmp_path):
        log = SessionLog("s-4", base_dir=str(tmp_path))
        log.set_name("temp")
        assert log.clear_name() is True
        assert log.get_name() is None
        # Idempotent: clearing again returns False but does NOT raise.
        assert log.clear_name() is False

    def test_get_name_silently_ignores_corrupt_sidecar(self, tmp_path):
        """A garbled meta file must NOT crash /resume rendering. The
        contract: bad sidecar == no name set."""
        log = SessionLog("s-5", base_dir=str(tmp_path))
        log.meta_path.write_text("{not json", encoding="utf-8")
        assert log.get_name() is None

    def test_list_sessions_includes_name_field(self, tmp_path):
        log = SessionLog("s-listed", base_dir=str(tmp_path))
        log.append("user", "hi")  # need a jsonl entry for list_sessions
        log.set_name("Headline")

        sessions = SessionLog.list_sessions(base_dir=str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s-listed"
        assert sessions[0]["name"] == "Headline"

    def test_list_sessions_name_is_none_when_no_sidecar(self, tmp_path):
        log = SessionLog("s-unnamed", base_dir=str(tmp_path))
        log.append("user", "hi")
        sessions = SessionLog.list_sessions(base_dir=str(tmp_path))
        assert sessions[0]["name"] is None

    def test_profile_round_trips_through_sidecar_and_listing(self, tmp_path):
        log = SessionLog("s-profile", base_dir=str(tmp_path))
        log.append("user", "hi")
        log.set_profile("venus-ds-v4-flash", "project")

        assert log.get_profile_name() == "venus-ds-v4-flash"
        assert log.get_profile_source() == "project"
        sessions = SessionLog.list_sessions(base_dir=str(tmp_path))
        assert sessions[0]["profile_name"] == "venus-ds-v4-flash"
        assert sessions[0]["profile_source"] == "project"

    def test_fork_inherits_profile_metadata(self, tmp_path):
        log = SessionLog("s-source", base_dir=str(tmp_path))
        log.append("user", "hi")
        log.set_profile("claude", "session")

        forked = log.fork("s-fork")

        assert forked.get_profile_name() == "claude"
        assert forked.get_profile_source() == "session"

    def test_rename_session_classmethod(self, tmp_path):
        """``rename_session`` lets non-CLI callers update a name by id
        without instantiating ``SessionLog`` themselves."""
        SessionLog.rename_session("s-by-cls", "via-classmethod", base_dir=str(tmp_path))
        assert SessionLog("s-by-cls", base_dir=str(tmp_path)).get_name() == "via-classmethod"


class TestProjectionLineage:
    """Compact boundary lineage: stale summaries must not survive a
    model/branch switch (Reasonix promptCacheKey port). The boundary IS the
    projection; everything before it on disk is the canonical transcript and
    must replay verbatim when lineage disagrees."""

    SUMMARY = "SUMMARY-MARKER: earlier turns condensed"

    def _write_session(self, base_dir, model="model-a", with_lineage=True):
        log = SessionLog("s-lin", base_dir=base_dir)
        log.append("user", "CANONICAL-Q1")
        log.append("assistant", "CANONICAL-A1")
        if with_lineage:
            log.append_compact_boundary(
                self.SUMMARY, model=model, covered_prefix_hash="hash-abc"
            )
        else:
            log.append_compact_boundary(self.SUMMARY)
            # strip the lineage fields to simulate a pre-feature boundary
            lines = log.path.read_text(encoding="utf-8").splitlines()
            out = []
            for line in lines:
                entry = json.loads(line)
                if entry.get("type") == "compact_boundary":
                    entry.pop("lineage_key", None)
                    entry.pop("model", None)
                    entry.pop("covered_prefix_hash", None)
                out.append(json.dumps(entry, ensure_ascii=False))
            log.path.write_text("\n".join(out) + "\n", encoding="utf-8")
        log.append("user", "TAIL-Q2")
        log.append("assistant", "TAIL-A2")
        return log

    def _contents(self, messages):
        return "\n".join(str(m.get("content", "")) for m in messages)

    def _reload(self, base_dir, model=None):
        log = SessionLog("s-lin", base_dir=base_dir)
        return log.load(model=model)

    def test_boundary_records_lineage_hash_and_model(self, tmp_path):
        log = self._write_session(str(tmp_path))
        boundary = None
        for line in log.path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("type") == "compact_boundary":
                boundary = entry
        assert boundary is not None
        assert boundary["model"] == "model-a"
        assert boundary["covered_prefix_hash"] == "hash-abc"
        key = boundary["lineage_key"]
        assert key.count("|") == 3  # session|cwd|branch|model
        assert key.endswith("|model-a")

    def test_lineage_key_excludes_volatile_parts(self, tmp_path):
        log = SessionLog("s-lin", base_dir=str(tmp_path))
        assert log.lineage_key("m") == log.lineage_key("m")
        assert log.lineage_key("m") != log.lineage_key("other-model")

    def test_load_pre_boundary_returns_canonical_span(self, tmp_path):
        log = self._write_session(str(tmp_path))
        pre = log.load_pre_boundary()
        texts = [e.get("content") for e in pre]
        assert "CANONICAL-Q1" in texts
        assert "CANONICAL-A1" in texts
        assert "TAIL-Q2" not in texts
        assert all(e.get("type") != "compact_boundary" for e in pre)

    def test_same_lineage_resume_keeps_summary(self, tmp_path):
        base = str(tmp_path)
        self._write_session(base)
        messages = self._reload(base, model="model-a")
        joined = self._contents(messages)
        assert "Resumed session" in joined
        assert self.SUMMARY in joined
        assert "TAIL-Q2" in joined
        assert "CANONICAL-Q1" not in joined  # canonical stays off the wire

    def test_model_switch_resume_falls_back_to_canonical(self, tmp_path):
        base = str(tmp_path)
        self._write_session(base)
        messages = self._reload(base, model="model-b")
        joined = self._contents(messages)
        assert "Resumed session" not in joined, "stale summary survived a model switch"
        assert self.SUMMARY not in joined
        assert "CANONICAL-Q1" in joined
        assert "CANONICAL-A1" in joined
        assert "TAIL-Q2" in joined

    def test_resume_without_model_keeps_legacy_behavior(self, tmp_path):
        base = str(tmp_path)
        self._write_session(base)
        messages = self._reload(base, model=None)
        assert "Resumed session" in self._contents(messages)

    def test_legacy_boundary_without_lineage_still_trusted(self, tmp_path):
        base = str(tmp_path)
        self._write_session(base, with_lineage=False)
        messages = self._reload(base, model="model-b")
        assert "Resumed session" in self._contents(messages)


class TestDeriveMessages:
    """``derive_messages`` — the log projection used to assert the trajectory."""

    def test_derive_matches_load_without_since_uuid(self, tmp_dir):
        log = SessionLog("derive-all", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a")
        assert log.derive_messages() == log.load()

    def test_since_uuid_slices_the_last_turn_only(self, tmp_dir):
        log = SessionLog("derive-tail", base_dir=tmp_dir)
        log.append("user", "turn-1 q")
        turn1_end = log.append("assistant", "turn-1 a")
        log.append("user", "turn-2 q")
        log.append("assistant", "", tool_calls=[{"id": "T", "type": "function",
                                                 "function": {"name": "grep", "arguments": "{}"}}])
        log.append("tool", "hit", tool_name="grep", tool_call_id="T")
        log.append("assistant", "turn-2 a")

        tail = log.derive_messages(since_uuid=turn1_end)
        assert [m["role"] for m in tail] == ["user", "assistant", "tool", "assistant"]
        assert tail[0]["content"] == "turn-2 q"

    def test_derive_does_not_disturb_the_append_chain(self, tmp_dir):
        log = SessionLog("derive-chain", base_dir=tmp_dir)
        log.append("user", "q")
        last = log.append("assistant", "a")
        log.derive_messages()
        assert log._last_uuid == last
        log.append("user", "q2")
        entries = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
        assert entries[-1]["parent_uuid"] == last

    def test_unknown_since_uuid_returns_whole_projection(self, tmp_dir):
        log = SessionLog("derive-unknown", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a")
        assert log.derive_messages(since_uuid="no-such-uuid") == log.load()

    def test_since_uuid_before_boundary_excludes_summary_pair(self, tmp_dir):
        log = SessionLog("derive-boundary", base_dir=tmp_dir)
        log.append("user", "old q")
        old = log.append("assistant", "old a")
        log.append_compact_boundary("SUMMARY")
        log.append("user", "new q")
        log.append("assistant", "new a")

        tail = log.derive_messages(since_uuid=old)
        assert [m["content"] for m in tail] == ["new q", "new a"]


class TestTrajectoryEquivalence:
    """The invariant that turns the resume-400 bug class into a test failure.

    ``persist.py`` rebuilds the log at the end of a turn from
    ``run_response.messages``. Nothing used to check that the rebuild projects
    back to what was really sent, so a reordering regression only surfaced as a
    provider 400 on a later ``/resume``.
    """

    @staticmethod
    def _live_multi_round_turn():
        """The live trajectory of a two-round tool turn (read_file then grep)."""
        return [
            Message(role="system", content="you are a helpful agent"),
            Message(role="user", content="read the file then grep it"),
            Message(
                role="assistant",
                tool_calls=[{"id": "A", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}],
            ),
            Message(role="tool", tool_call_id="A", content="file content", tool_name="read_file"),
            Message(
                role="assistant",
                tool_calls=[{"id": "B", "type": "function",
                             "function": {"name": "grep", "arguments": "{}"}}],
            ),
            Message(role="tool", tool_call_id="B", content="matches", tool_name="grep"),
            Message(role="assistant", content="done"),
        ]

    def test_persisted_turn_is_equivalent_to_what_was_sent(self, tmp_dir):
        """Regression: the real persist path must round-trip to the live turn."""
        log = SessionLog("traj-ok", base_dir=tmp_dir)
        live = self._live_multi_round_turn()
        turn_start = log.append("user", "read the file then grep it")
        agent = _FakeAgent(
            log,
            messages=live,
            tools=[
                {"tool_call_id": "A", "tool_name": "read_file", "content": "file content", "replay": True},
                {"tool_call_id": "B", "tool_name": "grep", "content": "matches", "replay": True},
            ],
        )
        Runner._persist_assistant_tool_calls(agent)
        log.append("assistant", "done", model="m1")

        derived = log.derive_messages(since_uuid=turn_start)
        assert [m["role"] for m in derived] == ["assistant", "tool", "assistant", "tool", "assistant"]
        # The interleaving the provider requires, end to end.
        full = log.derive_messages()
        assert [m["role"] for m in full] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant",
        ]
        assert assert_trajectory_equivalent(full, live) is None

    def test_grouped_assistants_then_tools_is_detected(self, tmp_dir):
        """Negative test — the core deliverable.

        Hand-build the log shape the previous implementation produced (all
        assistant tool-call rounds first, then all tool results). It is a valid
        JSONL log and loads fine, but replaying it 400s on OpenAI-compatible
        providers. The invariant must catch it.
        """
        log = SessionLog("traj-bad", base_dir=tmp_dir)
        log.append("user", "read the file then grep it")
        log.append("assistant", "", tool_calls=[{"id": "A", "type": "function",
                                                 "function": {"name": "read_file", "arguments": "{}"}}])
        log.append("assistant", "", tool_calls=[{"id": "B", "type": "function",
                                                 "function": {"name": "grep", "arguments": "{}"}}])
        log.append("tool", "file content", tool_name="read_file", tool_call_id="A")
        log.append("tool", "matches", tool_name="grep", tool_call_id="B")
        log.append("assistant", "done", model="m1")

        derived = log.derive_messages()
        divergence = assert_trajectory_equivalent(derived, self._live_multi_round_turn())
        assert divergence is not None, "mis-ordered log passed the equivalence check"
        assert "not replayable" in divergence
        assert "'A'" in divergence  # names the offending tool_call_id

    def test_orphaned_tool_result_is_detected(self, tmp_dir):
        """A tool result with no requesting assistant is the other 400 shape."""
        log = SessionLog("traj-orphan", base_dir=tmp_dir)
        log.append("user", "read the file")
        log.append("tool", "file content", tool_name="read_file", tool_call_id="A")
        log.append("assistant", "done")

        divergence = assert_trajectory_equivalent(log.derive_messages(), [])
        assert divergence is not None
        assert "not preceded by an assistant carrying tool_calls" in divergence

    def test_missing_tool_round_in_log_is_detected(self, tmp_dir):
        """A dropped tool round is a divergence even though the log stays valid."""
        log = SessionLog("traj-missing", base_dir=tmp_dir)
        log.append("user", "read the file then grep it")
        log.append("assistant", "", tool_calls=[{"id": "A", "type": "function",
                                                 "function": {"name": "read_file", "arguments": "{}"}}])
        log.append("tool", "file content", tool_name="read_file", tool_call_id="A")
        log.append("assistant", "done", model="m1")

        divergence = assert_trajectory_equivalent(
            log.derive_messages(), self._live_multi_round_turn()
        )
        assert divergence is not None
        assert "diverges at index" in divergence

    def test_content_rewrites_are_not_divergences(self, tmp_dir):
        """Compaction/markers rewrite content legally; structure is what counts."""
        log = SessionLog("traj-content", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "answer\n\n[User interrupted the response]")
        live = [
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
            Message(role="assistant", content="answer"),
        ]
        assert assert_trajectory_equivalent(log.derive_messages(), live) is None

    def test_extra_live_user_messages_and_empty_assistant_are_tolerated(self, tmp_dir):
        """The log writes one user entry per turn; the live list may hold more."""
        log = SessionLog("traj-normalize", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a")
        live = [
            Message(role="user", content="q"),
            Message(role="user", content="[injected reminder]"),
            Message(role="assistant", content=""),
            Message(role="assistant", content="a"),
        ]
        assert assert_trajectory_equivalent(log.derive_messages(), live) is None

    def test_tool_call_id_reordering_is_detected(self, tmp_dir):
        """Same roles, wrong pairing: tool B answered under assistant A's round."""
        log = SessionLog("traj-swap", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "", tool_calls=[{"id": "A", "type": "function",
                                                 "function": {"name": "read_file", "arguments": "{}"}}])
        log.append("tool", "matches", tool_name="grep", tool_call_id="B")
        log.append("assistant", "done")

        divergence = assert_trajectory_equivalent(log.derive_messages(), [])
        assert divergence is not None
        assert "does not answer the preceding assistant's tool_calls" in divergence


def _tool_round(call_id, tool_name):
    """One finished tool round as the model layer leaves it in the message list."""
    return [
        Message(
            role="assistant",
            tool_calls=[{"id": call_id, "type": "function",
                         "function": {"name": tool_name, "arguments": "{}"}}],
        ),
        Message(role="tool", tool_call_id=call_id, content=f"{tool_name} output", tool_name=tool_name),
    ]


class TestInTurnPersistence:
    """P2: rounds are written as they finish, end-of-turn write is a backfill.

    Before this, the whole turn was written at the end of the turn, so a
    SIGKILL / OOM kill / power loss lost every round of a long agentic turn.
    """

    QUESTION = "read the file then grep it"

    def _agent(self, log):
        return _FakeAgent(log, messages=[], tools=[])

    def _user_messages(self):
        return [Message(role="user", content=self.QUESTION)]

    def test_flush_then_end_of_turn_write_produces_no_duplicates(self, tmp_dir):
        log = SessionLog("inturn-idempotent", base_dir=tmp_dir)
        log.begin_turn()
        agent = self._agent(log)
        round_a = _tool_round("A", "read_file")
        round_b = _tool_round("B", "grep")

        # Round 1 finishes -> flushed mid-turn (question goes first).
        Runner._flush_turn_tool_rounds(agent, self.QUESTION, self._user_messages(), list(round_a))
        # Round 2 finishes -> flushed; round 1 must not be written twice.
        Runner._flush_turn_tool_rounds(
            agent, self.QUESTION, self._user_messages(), round_a + round_b
        )

        # End of turn: the runner rebuilds the whole turn and backfills.
        final = Message(role="assistant", content="done")
        live = self._user_messages() + round_a + round_b + [final]
        agent.run_response.messages = live
        agent.run_response.tools = [
            {"tool_call_id": "A", "tool_name": "read_file", "content": "read_file output", "replay": True},
            {"tool_call_id": "B", "tool_name": "grep", "content": "grep output", "replay": True},
        ]
        Runner._persist_turn_user_message(agent, self.QUESTION, self._user_messages())
        Runner._persist_assistant_tool_calls(agent)
        log.append("assistant", "done", model="m1")

        derived = log.derive_messages()
        assert [m["role"] for m in derived] == [
            "user", "assistant", "tool", "assistant", "tool", "assistant",
        ]
        assert [m.get("tool_call_id") for m in derived if m["role"] == "tool"] == ["A", "B"]
        assert assert_trajectory_equivalent(derived, live) is None

    def test_hard_kill_keeps_the_finished_rounds(self, tmp_dir):
        """Simulated SIGKILL: no end-of-turn write ever happens."""
        log = SessionLog("inturn-crash", base_dir=tmp_dir)
        log.begin_turn()
        agent = self._agent(log)
        round_a = _tool_round("A", "read_file")
        Runner._flush_turn_tool_rounds(agent, self.QUESTION, self._user_messages(), list(round_a))
        # ... process dies here: no _persist_assistant_tool_calls, no assistant text.

        derived = log.derive_messages()
        assert [m["role"] for m in derived] == ["user", "assistant", "tool"]
        assert derived[0]["content"] == self.QUESTION
        assert derived[2]["content"] == "read_file output"
        # The surviving projection is one a provider accepts: every tool result
        # answers the assistant that requested it.
        assert assert_trajectory_equivalent(derived, derived) is None

    def test_resume_seals_the_turn_the_kill_left_open(self, tmp_dir):
        log = SessionLog("inturn-seal", base_dir=tmp_dir)
        log.begin_turn()
        agent = self._agent(log)
        Runner._flush_turn_tool_rounds(
            agent, self.QUESTION, self._user_messages(), _tool_round("A", "read_file")
        )

        sealed = SessionLog("inturn-seal", base_dir=tmp_dir)
        assert sealed.seal_incomplete_turn() is not None
        derived = sealed.derive_messages()
        assert [m["role"] for m in derived] == ["user", "assistant", "tool", "assistant"]
        assert "[Session ended before the assistant replied]" in derived[-1]["content"]
        # Idempotent: a sealed log needs no second seal.
        assert sealed.seal_incomplete_turn() is None
        # The uuid chain stays intact across the sealing append.
        entries = [json.loads(line) for line in sealed.path.read_text(encoding="utf-8").splitlines()]
        assert entries[-1]["parent_uuid"] == entries[-2]["uuid"]

    def test_seal_leaves_a_complete_log_alone(self, tmp_dir):
        log = SessionLog("inturn-complete", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a")
        before = log.path.read_text(encoding="utf-8")
        assert log.seal_incomplete_turn() is None
        assert log.path.read_text(encoding="utf-8") == before

    def test_seal_closes_a_dangling_user_question(self, tmp_dir):
        """Killed before the first round finished — the question still survives."""
        log = SessionLog("inturn-dangling", base_dir=tmp_dir)
        log.append("user", "q")
        assert log.seal_incomplete_turn() is not None
        assert [m["role"] for m in log.derive_messages()] == ["user", "assistant"]

    def test_unanswered_tool_call_is_never_flushed(self, tmp_dir):
        """An assistant(tool_calls) with no result on disk is the 400 shape."""
        log = SessionLog("inturn-unanswered", base_dir=tmp_dir)
        log.begin_turn()
        agent = self._agent(log)
        pending = [
            Message(
                role="assistant",
                tool_calls=[{"id": "A", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}],
            )
        ]
        Runner._flush_turn_tool_rounds(agent, self.QUESTION, self._user_messages(), pending)
        assert log.entry_count() == 0

    def test_begin_turn_resets_the_per_turn_bookkeeping(self, tmp_dir):
        log = SessionLog("inturn-two-turns", base_dir=tmp_dir)
        agent = self._agent(log)
        log.begin_turn()
        Runner._flush_turn_tool_rounds(
            agent, "turn one", [Message(role="user", content="turn one")], _tool_round("A", "read_file")
        )
        log.append("assistant", "done one")
        log.begin_turn()
        Runner._flush_turn_tool_rounds(
            agent, "turn two", [Message(role="user", content="turn two")], _tool_round("B", "grep")
        )
        log.append("assistant", "done two")

        derived = log.derive_messages()
        assert [m["role"] for m in derived] == [
            "user", "assistant", "tool", "assistant",
            "user", "assistant", "tool", "assistant",
        ]
        assert derived[4]["content"] == "turn two"
        assert assert_trajectory_equivalent(derived, derived) is None

    def test_turn_start_uuid_scopes_the_derived_tail(self, tmp_dir):
        log = SessionLog("inturn-tail", base_dir=tmp_dir)
        log.append("user", "old q")
        log.append("assistant", "old a")
        log.begin_turn()
        agent = self._agent(log)
        Runner._flush_turn_tool_rounds(
            agent, self.QUESTION, self._user_messages(), _tool_round("A", "read_file")
        )
        log.append("assistant", "done")

        tail = log.derive_messages(since_uuid=log._turn_start_uuid)
        assert [m["role"] for m in tail] == ["user", "assistant", "tool", "assistant"]
        assert tail[0]["content"] == self.QUESTION


class TestTrajectoryStats:
    """P3: read the metrics the writer already records — no estimates."""

    def _write_session(self, tmp_dir):
        log = SessionLog("stats", base_dir=tmp_dir)
        log.append("user", "q1")
        log.append(
            "assistant", "",
            tool_calls=[{"id": "A", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}],
            metrics={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        )
        log.append("tool", "content", tool_name="read_file", tool_call_id="A", is_error=False)
        log.append(
            "assistant", "",
            tool_calls=[{"id": "B", "type": "function",
                         "function": {"name": "grep", "arguments": "{}"}}],
            metrics={"input_tokens": 200, "output_tokens": 20, "total_tokens": 220,
                     "prompt_tokens_details": {"cache_read_tokens": 128}},
        )
        log.append("tool", "boom", tool_name="grep", tool_call_id="B", is_error=True)
        log.append("assistant", "done", metrics={"input_tokens": 300, "output_tokens": 30,
                                                 "total_tokens": 330})
        return log

    def test_counts_and_rates(self, tmp_dir):
        stats = self._write_session(tmp_dir).trajectory_stats()
        assert stats["entries"] == 6
        assert stats["turns"] == 1
        assert stats["assistant_messages"] == 3
        assert stats["tool_call_rounds"] == 2
        assert stats["tool_calls"] == 2
        assert stats["tool_errors"] == 1
        assert stats["tool_error_rate"] == 0.5
        assert stats["tools_by_name"] == {"grep": 1, "read_file": 1}

    def test_token_and_cache_sums(self, tmp_dir):
        stats = self._write_session(tmp_dir).trajectory_stats()
        assert stats["input_tokens"] == 600
        assert stats["output_tokens"] == 60
        assert stats["total_tokens"] == 660
        assert stats["cached_tokens"] == 64
        assert stats["cache_read_tokens"] == 128
        assert stats["reasoning_tokens"] == 4
        # This transcript records no cache writes, so the metric is a truthful
        # zero rather than an estimate derived from the hit counters.
        assert stats["cache_write_tokens"] == 0

    def test_cache_write_read_from_the_anthropic_spelling(self, tmp_dir):
        log = SessionLog("stats-write", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a", metrics={
            "input_tokens": 10,
            "prompt_tokens_details": {"cache_creation_tokens": 512},
        })
        assert log.trajectory_stats()["cache_write_tokens"] == 512

    def test_cache_write_read_from_the_inclusive_spelling(self, tmp_dir):
        log = SessionLog("stats-write2", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a", metrics={
            "input_tokens": 10,
            "prompt_tokens_details": {"cache_write_tokens": 256},
        })
        assert log.trajectory_stats()["cache_write_tokens"] == 256

    def test_cache_write_aliases_are_not_double_counted(self, tmp_dir):
        """The two spellings name ONE quantity; summing both would over-bill."""
        log = SessionLog("stats-write3", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a", metrics={
            "prompt_tokens_details": {
                "cache_creation_tokens": 300,
                "cache_write_tokens": 300,
            },
        })
        assert log.trajectory_stats()["cache_write_tokens"] == 300

    def test_cache_write_is_never_folded_into_the_hit_counters(self, tmp_dir):
        """A cache WRITE is billed above an uncached input token, a HIT below it.

        Counting a write as a hit would report cache spend as cache savings, so
        a write-only entry must leave both hit counters at zero.
        """
        log = SessionLog("stats-write4", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a", metrics={
            "prompt_tokens_details": {"cache_creation_tokens": 1024},
        })
        stats = log.trajectory_stats()
        assert stats["cache_write_tokens"] == 1024
        assert stats["cached_tokens"] == 0
        assert stats["cache_read_tokens"] == 0

    def test_compactions_and_audit_entries(self, tmp_dir):
        log = self._write_session(tmp_dir)
        log.append_compact_boundary("summary", model="m1")
        log.append("tool_audit", "audited", tool_name="execute", tool_call_id="C",
                   is_error=True, replay=False)
        stats = log.trajectory_stats()
        assert stats["compactions"] == 1
        assert stats["tool_audit_entries"] == 1
        # tool_audit results are not replayable but they DID run: they count
        # towards the error rate, not towards replayable tool calls.
        assert stats["tool_calls"] == 2
        assert stats["tool_errors"] == 2
        assert stats["tool_error_rate"] == round(2 / 3, 4)

    def test_empty_log_reports_zeros_not_guesses(self, tmp_dir):
        stats = SessionLog("stats-empty", base_dir=tmp_dir).trajectory_stats()
        assert stats["entries"] == 0
        assert stats["tool_error_rate"] == 0.0
        assert stats["input_tokens"] == 0
        assert stats["tools_by_name"] == {}

    def test_metricless_log_reports_zero_tokens(self, tmp_dir):
        """A log written without metrics must not fabricate token counts."""
        log = SessionLog("stats-nometrics", base_dir=tmp_dir)
        log.append("user", "q")
        log.append("assistant", "a")
        stats = log.trajectory_stats()
        assert stats["turns"] == 1
        assert (stats["input_tokens"], stats["output_tokens"], stats["total_tokens"]) == (0, 0, 0)
        assert stats["cached_tokens"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
