# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Protocol Types

Data models for Agent Client Protocol (ACP) communication.
Based on JSON-RPC 2.0 specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum


class ACPMethod(str, Enum):
    """ACP Standard Methods"""
    # Lifecycle
    INITIALIZE = "initialize"
    SHUTDOWN = "shutdown"
    EXIT = "exit"
    
    # Tool methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Agent methods (legacy)
    AGENT_EXECUTE = "agent/execute"
    AGENT_CANCEL = "agent/cancel"
    
    # Session management (ACP core)
    SESSION_NEW = "session/new"
    SESSION_PROMPT = "session/prompt"
    SESSION_LOAD = "session/load"
    SESSION_CANCEL = "session/cancel"
    SESSION_DELETE = "session/delete"
    SESSION_LIST = "session/list"
    
    # File system (Client exposes to Agent)
    FS_READ_TEXT_FILE = "fs/read_text_file"
    FS_WRITE_TEXT_FILE = "fs/write_text_file"
    FS_LIST_DIRECTORY = "fs/list_directory"
    FS_SEARCH = "fs/search"
    
    # Terminal
    TERMINAL_CREATE = "terminal/create"
    TERMINAL_OUTPUT = "terminal/output"
    TERMINAL_RELEASE = "terminal/release"
    TERMINAL_EXECUTE = "terminal/execute"
    
    # Notifications (no response needed)
    NOTIFICATION_INITIALIZED = "notifications/initialized"
    NOTIFICATION_PROGRESS = "notifications/progress"
    NOTIFICATION_COMPLETE = "notifications/complete"
    NOTIFICATION_SESSION_UPDATE = "notifications/session/update"


@dataclass
class ACPMessage:
    """Base ACP Message"""
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass 
class ACPRequest(ACPMessage):
    """ACP Request Message"""
    id: Union[str, int] = field(default=0)
    method: str = field(default="")
    params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ACPRequest:
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", 0),
            method=data.get("method", ""),
            params=data.get("params"),
        )


@dataclass
class ACPResponse(ACPMessage):
    """ACP Response Message"""
    id: Union[str, int] = field(default=0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error is not None:
            result["error"] = self.error
        elif self.result is not None:
            result["result"] = self.result
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ACPResponse:
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", 0),
            result=data.get("result"),
            error=data.get("error"),
        )
    
    @classmethod
    def create_success(cls, id: Union[str, int], result: Dict[str, Any]) -> ACPResponse:
        """Create a success response (factory method)"""
        return cls(id=id, result=result)
    
    @classmethod
    def create_error(cls, id: Union[str, int], code: int, message: str, data: Any = None) -> ACPResponse:
        """Create an error response (factory method)"""
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return cls(id=id, error=error)


@dataclass
class ACPTool:
    """ACP Tool Definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class ACPToolCall:
    """ACP Tool Call"""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ACPToolCall:
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


@dataclass
class ACPToolResult:
    """ACP Tool Result"""
    content: str
    isError: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "isError": self.isError,
        }


@dataclass
class ACPInitializeParams:
    """ACP Initialize Parameters"""
    protocolVersion: str = "2024-11-05"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    clientInfo: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ACPInitializeParams:
        return cls(
            protocolVersion=data.get("protocolVersion", "2024-11-05"),
            capabilities=data.get("capabilities", {}),
            clientInfo=data.get("clientInfo", {}),
        )


@dataclass
class ACPInitializeResult:
    """ACP Initialize Result"""
    protocolVersion: str = "2024-11-05"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    serverInfo: Dict[str, str] = field(default_factory=lambda: {
        "name": "agentica",
        "version": "0.1.0",
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocolVersion": self.protocolVersion,
            "capabilities": self.capabilities,
            "serverInfo": self.serverInfo,
        }


@dataclass
class ACPAgentExecuteParams:
    """ACP Agent Execute Parameters"""
    task: str
    context: Optional[Dict[str, Any]] = None
    files: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ACPAgentExecuteParams:
        return cls(
            task=data["task"],
            context=data.get("context"),
            files=data.get("files"),
        )


# JSON-RPC Error Codes
class ACPErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR_START = -32000
    SERVER_ERROR_END = -32099
