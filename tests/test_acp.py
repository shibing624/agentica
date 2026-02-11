#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Agentica ACP (Agent Client Protocol) implementation.

Run with: python -m pytest tests/test_acp.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica.acp.types import (
    ACPRequest,
    ACPResponse,
    ACPTool,
    ACPToolCall,
    ACPToolResult,
    ACPErrorCode,
    ACPMethod,
)
from agentica.acp.protocol import ACPProtocolHandler
from agentica.acp.handlers import ACPHandlers
from agentica.acp.session import SessionManager, ACPSession, SessionStatus


class TestACPTypes:
    """Test ACP data types"""
    
    def test_request_creation(self):
        """Test ACPRequest creation"""
        request = ACPRequest(
            id=1,
            method="initialize",
            params={"protocolVersion": "2024-11-05"}
        )
        assert request.method == "initialize"
        assert request.id == 1
    
    def test_request_serialization(self):
        """Test ACPRequest serialization"""
        request = ACPRequest(
            id=1,
            method="initialize",
            params={"version": "1.0"}
        )
        req_dict = request.to_dict()
        assert req_dict["jsonrpc"] == "2.0"
        assert req_dict["id"] == 1
        assert req_dict["method"] == "initialize"
    
    def test_request_deserialization(self):
        """Test ACPRequest deserialization"""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        req = ACPRequest.from_dict(data)
        assert req.method == "tools/list"
        assert req.id == 1
    
    def test_response_success(self):
        """Test ACPResponse success creation"""
        response = ACPResponse.create_success(
            id=1,
            result={"status": "ok"}
        )
        assert response.result == {"status": "ok"}
        assert response.error is None
    
    def test_response_error(self):
        """Test ACPResponse error creation"""
        response = ACPResponse.create_error(
            id=1,
            code=ACPErrorCode.METHOD_NOT_FOUND,
            message="Method not found"
        )
        assert response.error is not None
        assert response.error["code"] == -32601
        assert response.result is None
    
    def test_tool_definition(self):
        """Test ACPTool creation"""
        tool = ACPTool(
            name="read_file",
            description="Read a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                }
            }
        )
        tool_dict = tool.to_dict()
        assert tool_dict["name"] == "read_file"
        assert tool_dict["description"] == "Read a file"
    
    def test_tool_call(self):
        """Test ACPToolCall"""
        call = ACPToolCall.from_dict({
            "name": "ls",
            "arguments": {"directory": "."}
        })
        assert call.name == "ls"
        assert call.arguments == {"directory": "."}
    
    def test_tool_result(self):
        """Test ACPToolResult"""
        result = ACPToolResult(content="test content", isError=False)
        result_dict = result.to_dict()
        assert result_dict["content"] == "test content"
        assert result_dict["isError"] == False


class TestACPSession:
    """Test ACP Session management"""
    
    def test_session_creation(self):
        """Test session creation"""
        session = ACPSession(mode="default")
        assert session.mode == "default"
        assert session.status == SessionStatus.CREATED
        assert len(session.messages) == 0
    
    def test_session_add_message(self):
        """Test adding messages to session"""
        session = ACPSession()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
    
    def test_session_status_update(self):
        """Test session status update"""
        session = ACPSession()
        session.update_status(SessionStatus.RUNNING)
        assert session.status == SessionStatus.RUNNING
    
    def test_session_to_dict(self):
        """Test session serialization"""
        session = ACPSession(mode="chat")
        session.add_message("user", "test")
        
        data = session.to_dict()
        assert data["id"] == session.id
        assert data["mode"] == "chat"
        assert data["message_count"] == 1


class TestSessionManager:
    """Test SessionManager"""
    
    def test_create_session(self):
        """Test session creation"""
        manager = SessionManager()
        session = manager.create_session(mode="default")
        
        assert session.id is not None
        assert session.mode == "default"
        assert session.id in manager._sessions
    
    def test_get_session(self):
        """Test getting session by ID"""
        manager = SessionManager()
        session = manager.create_session()
        
        retrieved = manager.get_session(session.id)
        assert retrieved == session
        
        not_found = manager.get_session("nonexistent")
        assert not_found is None
    
    def test_list_sessions(self):
        """Test listing sessions"""
        manager = SessionManager()
        session1 = manager.create_session(mode="default")
        session2 = manager.create_session(mode="chat")
        
        sessions = manager.list_sessions()
        assert len(sessions) == 2
    
    def test_delete_session(self):
        """Test session deletion"""
        manager = SessionManager()
        session = manager.create_session()
        
        success = manager.delete_session(session.id)
        assert success is True
        assert manager.get_session(session.id) is None
        
        # Delete non-existent
        success = manager.delete_session("nonexistent")
        assert success is False
    
    def test_cancel_session(self):
        """Test session cancellation"""
        manager = SessionManager()
        session = manager.create_session()
        session.update_status(SessionStatus.RUNNING)
        
        success = manager.cancel_session(session.id)
        assert success is True
        assert session.status == SessionStatus.CANCELLED
    
    def test_get_stats(self):
        """Test session statistics"""
        manager = SessionManager()
        session = manager.create_session()
        session.update_status(SessionStatus.RUNNING)
        
        stats = manager.get_stats()
        assert stats["total"] == 1
        assert stats["running"] == 1


class TestACPHandlers:
    """Test ACP method handlers"""
    
    @pytest.fixture
    def handlers(self):
        """Create handlers instance"""
        mock_protocol = Mock()
        return ACPHandlers(protocol=mock_protocol)
    
    def test_handle_initialize(self, handlers):
        """Test initialize handler"""
        result = handlers.handle_initialize({
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })
        
        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "agentica"
        assert handlers._initialized is True
    
    def test_handle_tools_list(self, handlers):
        """Test tools/list handler"""
        handlers._initialized = True
        result = handlers.handle_tools_list({})
        
        assert "tools" in result
        assert len(result["tools"]) > 0
        
        # Check tool structure
        tool = result["tools"][0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
    
    def test_handle_tools_call_ls(self, handlers):
        """Test tools/call handler with ls tool"""
        handlers._initialized = True
        result = handlers.handle_tools_call({
            "name": "ls",
            "arguments": {"directory": "."}
        })
        
        assert "content" in result
        assert "isError" in result
    
    def test_handle_session_new(self, handlers):
        """Test session/new handler"""
        handlers._initialized = True
        result = handlers.handle_session_new({
            "mode": "chat",
            "context": {"project": "test"}
        })
        
        assert "sessionId" in result
        assert result["status"] == "created"
        assert result["mode"] == "chat"
    
    def test_handle_session_prompt(self, handlers):
        """Test session/prompt handler"""
        handlers._initialized = True
        
        # Create a session first
        session_result = handlers.handle_session_new({})
        session_id = session_result["sessionId"]
        
        # Mock the agent execution
        with patch.object(handlers, '_execute_sync') as mock_exec:
            mock_exec.return_value = {"content": "test response", "stop_reason": "end_turn"}
            
            result = handlers.handle_session_prompt({
                "sessionId": session_id,
                "prompt": "Hello",
                "stream": False
            })
            
            assert result["sessionId"] == session_id
            assert result["status"] == "completed"
    
    def test_handle_session_list(self, handlers):
        """Test session/list handler"""
        handlers._initialized = True
        
        # Create some sessions
        handlers.handle_session_new({"mode": "default"})
        handlers.handle_session_new({"mode": "chat"})
        
        result = handlers.handle_session_list({})
        
        assert "sessions" in result
        assert "stats" in result
        assert len(result["sessions"]) == 2


class TestACPProtocol:
    """Test ACP protocol handler"""
    
    def test_protocol_handler_creation(self):
        """Test protocol handler creation"""
        protocol = ACPProtocolHandler()
        assert protocol is not None
    
    @patch('sys.stdout')
    def test_send_success(self, mock_stdout):
        """Test sending success response"""
        protocol = ACPProtocolHandler()
        protocol.send_success(1, {"message": "test"})
        
        # Verify stdout was written
        assert mock_stdout.write.called
    
    @patch('sys.stdout')
    def test_send_error(self, mock_stdout):
        """Test sending error response"""
        protocol = ACPProtocolHandler()
        protocol.send_error(1, ACPErrorCode.INVALID_PARAMS, "Invalid params")
        
        # Verify stdout was written
        assert mock_stdout.write.called
    
    @patch('sys.stdout')
    def test_send_notification(self, mock_stdout):
        """Test sending notification"""
        protocol = ACPProtocolHandler()
        protocol.send_notification("notifications/progress", {
            "sessionId": "test-123",
            "type": "start"
        })
        
        # Verify stdout was written
        assert mock_stdout.write.called
        
        # Check the written content includes method and params
        written = mock_stdout.write.call_args_list
        assert len(written) > 0


class TestACPIntegration:
    """Integration tests for ACP"""
    
    def test_full_session_flow(self):
        """Test a complete session flow"""
        handlers = ACPHandlers()
        
        # 1. Initialize
        init_result = handlers.handle_initialize({
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test", "version": "1.0.0"}
        })
        assert init_result["protocolVersion"] == "2024-11-05"
        
        # 2. Create session
        session_result = handlers.handle_session_new({"mode": "default"})
        session_id = session_result["sessionId"]
        assert session_id is not None
        
        # 3. List sessions
        list_result = handlers.handle_session_list({})
        assert len(list_result["sessions"]) == 1
        
        # 4. Load session
        load_result = handlers.handle_session_load({"sessionId": session_id})
        assert load_result["sessionId"] == session_id
        
        # 5. Delete session
        delete_result = handlers.handle_session_delete({"sessionId": session_id})
        assert delete_result["deleted"] is True
    
    def test_tool_execution_flow(self):
        """Test tool execution"""
        handlers = ACPHandlers()
        handlers._initialized = True
        
        # List tools
        tools_result = handlers.handle_tools_list({})
        tools = tools_result["tools"]
        assert len(tools) > 0
        
        # Call a tool
        call_result = handlers.handle_tools_call({
            "name": "ls",
            "arguments": {"directory": "."}
        })
        
        assert "content" in call_result
        assert "isError" in call_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
