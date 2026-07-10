# -*- coding: utf-8 -*-
"""Tests for SessionLog — CC-style append-only JSONL with UUID chain and compact boundaries."""

import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agentica.memory.session_log import SessionLog
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
    API used by ``/session rename``. The on-disk layout
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

    def test_rename_session_classmethod(self, tmp_path):
        """``rename_session`` lets callers update a name without
        instantiating ``SessionLog`` themselves — used by the CLI when the
        user types ``/session rename <id-prefix> <name>``."""
        SessionLog.rename_session("s-by-cls", "via-classmethod", base_dir=str(tmp_path))
        assert SessionLog("s-by-cls", base_dir=str(tmp_path)).get_name() == "via-classmethod"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
