#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Agentica ACP (Agent Client Protocol) implementation.

Guards against: LSP Content-Length framing returning to the stdio transport,
log lines corrupting the JSON-RPC stream, non-standard notification names or
initialize capabilities, and session/load losing a session when the client
starts a fresh agent process for the next turn.

Run with: python -m pytest tests/acp/test_acp.py -v
"""

import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.acp.types import (
    ACPRequest,
    ACPResponse,
    ACPTool,
    ACPToolCall,
    ACPToolResult,
    ACPErrorCode,
    ACPMethod,
    ACP_PROTOCOL_VERSION,
)
from agentica.acp.protocol import ACPProtocolHandler
from agentica.acp.handlers import ACPHandlers, prompt_text
from agentica.acp.session import SessionManager, ACPSession, SessionStatus


class TestACPTypes:
    """Test ACP data types"""

    def test_request_creation(self):
        request = ACPRequest(id=1, method="initialize", params={"protocolVersion": 1})
        assert request.method == "initialize"
        assert request.id == 1

    def test_request_serialization(self):
        request = ACPRequest(id=1, method="initialize", params={"version": "1.0"})
        req_dict = request.to_dict()
        assert req_dict["jsonrpc"] == "2.0"
        assert req_dict["id"] == 1
        assert req_dict["method"] == "initialize"

    def test_request_deserialization(self):
        data = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        req = ACPRequest.from_dict(data)
        assert req.method == "tools/list"
        assert req.id == 1

    def test_response_success(self):
        response = ACPResponse.create_success(id=1, result={"status": "ok"})
        assert response.result == {"status": "ok"}
        assert response.error is None

    def test_response_error(self):
        response = ACPResponse.create_error(
            id=1, code=ACPErrorCode.METHOD_NOT_FOUND, message="Method not found"
        )
        assert response.error is not None
        assert response.error["code"] == -32601
        assert response.result is None

    def test_tool_definition(self):
        tool = ACPTool(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"file_path": {"type": "string"}}},
        )
        tool_dict = tool.to_dict()
        assert tool_dict["name"] == "read_file"
        assert tool_dict["description"] == "Read a file"

    def test_tool_call(self):
        call = ACPToolCall.from_dict({"name": "read_file", "arguments": {"file_path": "tmp.txt"}})
        assert call.name == "read_file"

    def test_tool_result(self):
        result = ACPToolResult(content="test content", isError=False)
        result_dict = result.to_dict()
        assert result_dict["content"] == "test content"
        assert result_dict["isError"] is False

    def test_session_update_is_the_standard_notification(self):
        assert ACPMethod.SESSION_UPDATE.value == "session/update"
        assert ACP_PROTOCOL_VERSION == 1


class TestACPSession:
    """Test ACP Session management"""

    def test_session_creation(self):
        session = ACPSession(mode="default")
        assert session.mode == "default"
        assert session.status == SessionStatus.CREATED
        assert len(session.messages) == 0

    def test_session_add_message(self):
        session = ACPSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_session_status_update(self):
        session = ACPSession()
        session.update_status(SessionStatus.RUNNING)
        assert session.status == SessionStatus.RUNNING

    def test_session_to_dict(self):
        session = ACPSession(mode="chat")
        session.add_message("user", "test")
        data = session.to_dict()
        assert data["id"] == session.id
        assert data["mode"] == "chat"
        assert data["message_count"] == 1

    def test_storage_roundtrip_keeps_history(self):
        session = ACPSession(cwd="/tmp/work")
        session.add_message("user", "first")
        session.add_message("assistant", "second")
        restored = ACPSession.from_storage_dict(session.to_storage_dict())
        assert restored.id == session.id
        assert restored.cwd == "/tmp/work"
        assert [(m.role, m.content) for m in restored.messages] == [
            ("user", "first"),
            ("assistant", "second"),
        ]


class TestSessionManager:
    """Test SessionManager"""

    def test_create_session(self):
        manager = SessionManager()
        session = manager.create_session(mode="default")
        assert session.id is not None
        assert session.mode == "default"
        assert session.id in manager._sessions

    def test_get_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert manager.get_session(session.id) == session
        assert manager.get_session("nonexistent") is None

    def test_list_sessions(self):
        manager = SessionManager()
        manager.create_session(mode="default")
        manager.create_session(mode="chat")
        assert len(manager.list_sessions()) == 2

    def test_delete_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert manager.delete_session(session.id) is True
        assert manager.get_session(session.id) is None
        assert manager.delete_session("nonexistent") is False

    def test_cancel_session(self):
        manager = SessionManager()
        session = manager.create_session()
        session.update_status(SessionStatus.RUNNING)
        assert manager.cancel_session(session.id) is True
        assert session.status == SessionStatus.CANCELLED

    def test_get_stats(self):
        manager = SessionManager()
        session = manager.create_session()
        session.update_status(SessionStatus.RUNNING)
        stats = manager.get_stats()
        assert stats["total"] == 1
        assert stats["running"] == 1

    def test_restore_session_from_another_process(self, tmp_path):
        first = SessionManager(storage_dir=str(tmp_path))
        session = first.create_session(cwd="/tmp/project")
        session.add_message("user", "remember me")
        first.save_session(session)

        second = SessionManager(storage_dir=str(tmp_path))
        restored = second.restore_session(session.id, cwd="/tmp/other")
        assert restored is not None
        assert restored.messages[0].content == "remember me"
        assert restored.cwd == "/tmp/other"

    def test_restore_unknown_session_returns_none(self, tmp_path):
        manager = SessionManager(storage_dir=str(tmp_path))
        assert manager.restore_session("never-created") is None

    def test_delete_removes_stored_metadata(self, tmp_path):
        manager = SessionManager(storage_dir=str(tmp_path))
        session = manager.create_session()
        assert (tmp_path / f"{session.id}.json").exists()
        manager.delete_session(session.id)
        assert not (tmp_path / f"{session.id}.json").exists()


class TestACPProtocol:
    """stdio transport: exactly one JSON-RPC frame per line on stdout"""

    def test_send_success_writes_one_json_line(self):
        out = io.StringIO()
        protocol = ACPProtocolHandler(in_stream=io.StringIO(), out_stream=out)
        protocol.send_success(1, {"message": "test"})
        lines = out.getvalue().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {
            "jsonrpc": "2.0", "id": 1, "result": {"message": "test"}
        }

    def test_send_error_writes_one_json_line(self):
        out = io.StringIO()
        protocol = ACPProtocolHandler(in_stream=io.StringIO(), out_stream=out)
        protocol.send_error(2, ACPErrorCode.INVALID_PARAMS, "Invalid params")
        frame = json.loads(out.getvalue().strip())
        assert frame["error"]["code"] == -32602

    def test_notification_has_method_params_and_no_id(self):
        out = io.StringIO()
        protocol = ACPProtocolHandler(in_stream=io.StringIO(), out_stream=out)
        protocol.send_notification("session/update", {"sessionId": "s1"})
        frame = json.loads(out.getvalue().strip())
        assert frame == {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s1"},
        }

    def test_read_message_parses_one_line(self):
        frame = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": 1},
        }) + "\n"
        protocol = ACPProtocolHandler(in_stream=io.StringIO(frame))
        request = protocol.read_message()
        assert request.method == "initialize"
        assert request.params["protocolVersion"] == 1
        assert protocol.read_message() is None

    def test_read_message_skips_blank_and_invalid_lines(self):
        frame = '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
        protocol = ACPProtocolHandler(in_stream=io.StringIO("\nnot json\n" + frame))
        assert protocol.read_message().method == "ping"

    def test_no_content_length_framing(self):
        out = io.StringIO()
        protocol = ACPProtocolHandler(in_stream=io.StringIO(), out_stream=out)
        protocol.send_success(1, {})
        assert "Content-Length" not in out.getvalue()


class TestACPHandlers:
    """Test ACP method handlers"""

    @pytest.fixture
    def handlers(self):
        return ACPHandlers(protocol=Mock())

    def test_handle_initialize(self, handlers):
        result = handlers.handle_initialize({
            "protocolVersion": 1,
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        })
        assert result["protocolVersion"] == 1
        assert result["agentCapabilities"]["loadSession"] is True
        assert result["agentInfo"]["name"] == "agentica"
        assert result["authMethods"] == []
        assert handlers._initialized is True

    def test_initialize_is_version_1_even_for_older_clients(self, handlers):
        result = handlers.handle_initialize({"protocolVersion": "2024-11-05"})
        assert result["protocolVersion"] == 1

    def test_handle_tools_list(self, handlers):
        handlers._initialized = True
        result = handlers.handle_tools_list({})
        assert "tools" in result
        assert "apply_patch" in {tool["name"] for tool in result["tools"]}
        tool = result["tools"][0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool

    def test_handle_session_new(self, handlers):
        handlers._initialized = True
        result = handlers.handle_session_new({"cwd": "/tmp/work"})
        assert "sessionId" in result
        assert set(result) == {"sessionId"}

    def test_handle_session_prompt(self, handlers):
        handlers._initialized = True
        session_id = handlers.handle_session_new({"cwd": "."})["sessionId"]
        handlers._agent = FakeAgent(["你好", "，世界"])

        result = handlers.handle_session_prompt({
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "说点什么"}],
        })
        assert result == {"stopReason": "end_turn"}
        session = handlers._session_manager.get_session(session_id)
        assert [m.role for m in session.messages] == ["user", "assistant"]
        assert session.messages[1].content == "你好，世界"

    def test_handle_session_prompt_rejects_plain_string(self, handlers):
        handlers._initialized = True
        session_id = handlers.handle_session_new({})["sessionId"]
        with pytest.raises(ValueError, match="content blocks"):
            handlers.handle_session_prompt({"sessionId": session_id, "prompt": "plain text"})

    def test_handle_session_list(self, handlers):
        handlers._initialized = True
        handlers.handle_session_new({"mode": "default"})
        handlers.handle_session_new({"mode": "chat"})
        result = handlers.handle_session_list({})
        assert "sessions" in result
        assert "stats" in result
        assert len(result["sessions"]) == 2


class FakeAgent:
    """Minimal streaming agent: yields incremental text chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def run_stream_sync(self, prompt):
        return [SimpleNamespace(content=chunk) for chunk in self._chunks]


class TestACPStdioContract:
    """Frames a third-party client (Zed, agent-autoloop bridge) relies on."""

    def test_streaming_sends_standard_session_updates(self):
        protocol = Mock()
        handlers = ACPHandlers(protocol=protocol)
        handlers.handle_initialize({"protocolVersion": 1})
        session_id = handlers.handle_session_new({})["sessionId"]
        handlers._agent = FakeAgent(["你", "好"])

        handlers.handle_session_prompt({
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "说两个字"}],
        })

        calls = [call.args for call in protocol.send_notification.call_args_list]
        assert len(calls) == 2
        for method, params in calls:
            assert method == "session/update"
            assert params["sessionId"] == session_id
            update = params["update"]
            assert update["sessionUpdate"] == "agent_message_chunk"
            assert update["content"]["type"] == "text"
        assert "".join(params["update"]["content"]["text"] for _, params in calls) == "你好"

    def test_prompt_text_merges_text_blocks(self):
        assert prompt_text([
            {"type": "text", "text": "first"},
            {"type": "image", "data": "ignored"},
            {"type": "text", "text": "second"},
        ]) == "first\nsecond"

    def test_prompt_text_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            prompt_text("plain string")
        with pytest.raises(ValueError):
            prompt_text([{"type": "text", "text": "  "}])

    def test_session_load_survives_a_new_process(self, tmp_path):
        first = ACPHandlers(protocol=Mock(), session_dir=str(tmp_path))
        first.handle_initialize({"protocolVersion": 1})
        session_id = first.handle_session_new({"cwd": "/tmp/project"})["sessionId"]

        second = ACPHandlers(protocol=Mock(), session_dir=str(tmp_path))
        second.handle_initialize({"protocolVersion": 1})
        loaded = second.handle_session_load({"sessionId": session_id, "cwd": "/tmp/project"})
        assert loaded["sessionId"] == session_id
        assert second._session_manager.get_session(session_id).cwd == "/tmp/project"

    def test_session_load_rejects_unknown_id(self, tmp_path):
        handlers = ACPHandlers(protocol=Mock(), session_dir=str(tmp_path))
        handlers.handle_initialize({"protocolVersion": 1})
        with pytest.raises(RuntimeError, match="Session not found"):
            handlers.handle_session_load({"sessionId": "unknown", "cwd": "."})

    def test_prompt_persists_history_for_next_process(self, tmp_path):
        handlers = ACPHandlers(protocol=Mock(), session_dir=str(tmp_path))
        handlers.handle_initialize({"protocolVersion": 1})
        session_id = handlers.handle_session_new({"cwd": "."})["sessionId"]
        handlers._agent = FakeAgent(["done"])
        handlers.handle_session_prompt({
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "go"}],
        })

        later = SessionManager(storage_dir=str(tmp_path))
        restored = later.restore_session(session_id)
        assert [(m.role, m.content) for m in restored.messages] == [
            ("user", "go"),
            ("assistant", "done"),
        ]


class TestACPIntegration:
    """Integration tests for ACP"""

    def test_full_session_flow(self):
        handlers = ACPHandlers(protocol=Mock())
        init_result = handlers.handle_initialize({
            "protocolVersion": 1,
            "clientInfo": {"name": "test", "version": "1.0.0"},
        })
        assert init_result["protocolVersion"] == 1

        session_id = handlers.handle_session_new({"cwd": "."})["sessionId"]
        assert session_id is not None
        assert len(handlers.handle_session_list({})["sessions"]) == 1
        assert handlers.handle_session_load({"sessionId": session_id})["sessionId"] == session_id
        assert handlers.handle_session_delete({"sessionId": session_id})["deleted"] is True

    def test_tool_execution_flow(self):
        handlers = ACPHandlers(protocol=Mock())
        handlers._initialized = True
        assert len(handlers.handle_tools_list({})["tools"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
