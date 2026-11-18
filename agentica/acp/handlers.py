# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Method Handlers

Handles specific ACP methods like initialize, tools/list, tools/call, agent/execute.
"""


import asyncio
import inspect
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable

from agentica.acp.permissions import PermissionDecision, ToolPermissionPolicy
from agentica.acp.types import (
    ACPTool,
    ACPToolCall,
    ACPToolResult,
    ACPInitializeParams,
    ACPInitializeResult,
    ACPAgentExecuteParams,
    ACPMethod,
)
from agentica.acp.session import SessionManager, ACPSession, SessionStatus
from agentica.utils.log import logger
from agentica.version import __version__

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model
    from agentica.acp.protocol import ACPProtocolHandler


def prompt_text(prompt: Any) -> str:
    """Extract text from an ACP prompt content block array."""
    if not isinstance(prompt, list):
        raise ValueError("prompt must be a list of ACP content blocks")
    parts = [
        block["text"]
        for block in prompt
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]
    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("prompt has no text content")
    return text


class ACPHandlers:
    """Handlers for ACP methods"""
    
    def __init__(self, agent: Optional["Agent"] = None, model: Optional["Model"] = None,
                 protocol: Optional[Any] = None,
                 permission_policy: Optional[Any] = None,
                 permission_callback: Optional[Callable] = None,
                 session_dir: Optional[str] = None):
        self._agent = agent
        self._model = model
        self._protocol = protocol
        self._initialized = False
        self._tools: List[ACPTool] = []
        self._session_dir = session_dir
        self._session_manager = SessionManager(storage_dir=session_dir)
        self._agents: Dict[str, "Agent"] = {}
        # IDE tool-permission gate. Default AUTO = allow all (backward compatible).
        # ``permission_callback`` is an optional async/sync callable invoked for
        # ASK decisions: callback(tool_call) -> bool (True = grant). When a tool
        # needs approval but no callback is set, the call is denied (fail-safe).
        self._permission_policy = permission_policy or ToolPermissionPolicy()
        self._permission_callback = permission_callback

        # Lazy-initialized tool instances (avoid repeated instantiation)
        self._file_tool = None
        self._execute_tool_instance = None
        self._web_search_tool = None
        
        # Initialize default tools
        self._setup_tools()
    
    def _get_file_tool(self):
        """Get or create file tool instance (singleton pattern)"""
        if self._file_tool is None:
            from agentica.tools.builtin import BuiltinFileTool
            self._file_tool = BuiltinFileTool()
        return self._file_tool
    
    def _get_execute_tool(self):
        """Get or create execute tool instance"""
        if self._execute_tool_instance is None:
            from agentica.tools.builtin import BuiltinExecuteTool
            self._execute_tool_instance = BuiltinExecuteTool()
        return self._execute_tool_instance
    
    def _get_web_search_tool(self):
        """Get or create web search tool instance"""
        if self._web_search_tool is None:
            from agentica.tools.builtin import BuiltinWebSearchTool
            self._web_search_tool = BuiltinWebSearchTool()
        return self._web_search_tool
    
    def _new_agent(self, session_id: Optional[str] = None):
        """Build an Agent; session_id pins its transcript for cross-process replay."""
        # Imported here: agentica.agent pulls in the full tool stack.
        from agentica.agent import Agent
        from agentica.model.defaults import create_default_model

        transcript_dir = str(Path(self._session_dir) / "transcripts") if self._session_dir else None
        return Agent(
            model=self._model or create_default_model(),
            name="Agentica-ACP",
            description="Agentica ACP Agent",
            session_id=session_id,
            session_base_dir=transcript_dir if session_id else None,
        )

    def _get_or_create_agent(self):
        """Shared Agent for the legacy agent/execute path."""
        if self._agent is None:
            self._agent = self._new_agent()
        return self._agent

    def _agent_for(self, session: ACPSession):
        """One Agent per ACP session so session/load resumes its transcript."""
        if self._agent is not None:
            return self._agent
        if session.id not in self._agents:
            self._agents[session.id] = self._new_agent(session_id=session.id)
        return self._agents[session.id]
    
    def _setup_tools(self) -> None:
        """Setup available tools for ACP"""
        # File tools
        self._tools.extend([
            ACPTool(
                name="read_file",
                description="Read file content from the filesystem",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "offset": {"type": "integer", "description": "Starting line number"},
                        "limit": {"type": "integer", "description": "Maximum lines to read"},
                    },
                    "required": ["file_path"],
                },
            ),
            ACPTool(
                name="write_file",
                description="Write content to a file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["file_path", "content"],
                },
            ),
            ACPTool(
                name="apply_patch",
                description="Apply one context patch across multiple files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string", "description": "Complete patch envelope"},
                    },
                    "required": ["patch"],
                },
            ),
            ACPTool(
                name="glob",
                description="Find files matching a pattern",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern"},
                        "path": {"type": "string", "description": "Base directory"},
                    },
                    "required": ["pattern"],
                },
            ),
            ACPTool(
                name="grep",
                description="Search for text in files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "path": {"type": "string", "description": "Directory to search"},
                    },
                    "required": ["pattern"],
                },
            ),
            ACPTool(
                name="execute",
                description="Execute a shell command",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                    },
                    "required": ["command"],
                },
            ),
            ACPTool(
                name="web_search",
                description="Search the web",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum results"},
                    },
                    "required": ["queries"],
                },
            ),
        ])
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate protocol version 1 and advertise supported capabilities."""
        init_params = ACPInitializeParams.from_dict(params)
        logger.info(f"ACP initialize from client: {init_params.clientInfo}")
        self._initialized = True
        return ACPInitializeResult().to_dict()
    
    def handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request"""
        logger.info("ACP shutdown requested")
        return {}
    
    def handle_exit(self, params: Dict[str, Any]) -> None:
        """Handle exit notification (no response)"""
        logger.info("ACP exit requested")
        self._initialized = False

    def handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Health check for IDEs and monitors."""
        return {
            "status": "ok",
            "initialized": self._initialized,
            "version": __version__,
            "timestamp": time.time(),
        }
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        return {
            "tools": [tool.to_dict() for tool in self._tools],
        }
    
    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        tool_call = ACPToolCall.from_dict(params)
        
        # Execute the tool
        result = await self._execute_tool(tool_call)
        
        return result.to_dict()
    
    async def _check_permission(self, tool_call: ACPToolCall) -> Optional[ACPToolResult]:
        """Gate a tool call through the permission policy.

        Returns an error ACPToolResult if the call is denied, or None if it may
        proceed. ASK decisions consult ``permission_callback``; absent a
        callback, ASK is treated as denied (fail-safe).
        """
        decision = self._permission_policy.decide(tool_call.name, tool_call.arguments)
        if decision == PermissionDecision.ALLOW:
            return None
        if decision == PermissionDecision.DENY:
            return ACPToolResult(
                content=f"Permission denied by policy for tool '{tool_call.name}'.",
                isError=True,
            )

        # ASK: defer to the IDE/user via the callback.
        granted = False
        if self._permission_callback is not None:
            result = self._permission_callback(tool_call)
            granted = await result if inspect.isawaitable(result) else result
        if not granted:
            return ACPToolResult(
                content=f"Permission not granted for tool '{tool_call.name}'.",
                isError=True,
            )
        return None

    async def _execute_tool(self, tool_call: ACPToolCall) -> ACPToolResult:
        """Execute a tool call (async, since all builtin tools are async)"""
        tool_name = tool_call.name
        arguments = tool_call.arguments

        # Permission gate (no-op in default AUTO mode).
        denied = await self._check_permission(tool_call)
        if denied is not None:
            return denied

        try:
            # File operations - use singleton instance
            if tool_name in (
                "read_file", "write_file", "apply_patch", "glob", "grep"
            ):
                file_tool = self._get_file_tool()
                method = getattr(file_tool, tool_name)
                result = await method(**arguments)
                return ACPToolResult(content=result)
            
            # Shell execution
            elif tool_name == "execute":
                execute_tool = self._get_execute_tool()
                result = await execute_tool.execute(**arguments)
                return ACPToolResult(content=result)
            
            # Web search
            elif tool_name == "web_search":
                web_tool = self._get_web_search_tool()
                result = await web_tool.web_search(**arguments)
                return ACPToolResult(content=result)
            
            else:
                return ACPToolResult(
                    content=f"Unknown tool: {tool_name}",
                    isError=True,
                )
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ACPToolResult(
                content=f"Error executing {tool_name}: {str(e)}",
                isError=True,
            )
    
    async def handle_agent_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent/execute request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        exec_params = ACPAgentExecuteParams.from_dict(params)
        
        logger.info(f"Agent execute task: {exec_params.task[:100]}...")
        
        try:
            agent = self._get_or_create_agent()
            response = await agent.run(exec_params.task)
            
            return {
                "content": response.content if response else "No response",
                "status": "completed",
            }
        
        except Exception as e:
            logger.error(f"Error executing agent task: {e}")
            return {
                "content": f"Error: {str(e)}",
                "status": "error",
            }
    
    def handle_agent_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent/cancel request (legacy)"""
        logger.info("Agent cancel requested (legacy)")
        return {"status": "cancelled"}
    
    # ============ Session Management ============
    
    def handle_session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session; the id is also the agentica transcript id."""
        if not self._initialized:
            raise RuntimeError("Server not initialized")

        session = self._session_manager.create_session(
            mode=params.get("mode", "default"),
            initial_context=params.get("context", {}),
            cwd=params.get("cwd", "."),
            session_key=params.get("_meta", {}).get("sessionKey"),
        )
        logger.info(f"Created session {session.id} (cwd: {session.cwd})")
        return {"sessionId": session.id}
    
    def handle_session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run one turn, streaming the answer as session/update notifications."""
        if not self._initialized:
            raise RuntimeError("Server not initialized")

        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")
        text = prompt_text(params.get("prompt"))

        session = self._session_manager.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")

        if session.active_run_id:
            self._session_manager.cancel_session(session_id)
        run_id = str(uuid.uuid4())
        self._session_manager.set_active_run(session_id, run_id, asyncio.Event())
        session.add_message("user", text)
        session.update_status(SessionStatus.RUNNING)
        logger.info(f"Session {session_id} prompt (run_id={run_id[:8]})")

        result = self._execute_with_streaming(session, text)

        session.update_status(SessionStatus.COMPLETED)
        self._session_manager.clear_active_run(session_id)
        self._session_manager.save_session(session)
        return {"stopReason": result.get("stop_reason", "end_turn")}

    def handle_session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a session created by an earlier process."""
        if not self._initialized:
            raise RuntimeError("Server not initialized")

        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")

        session = self._session_manager.get_session(session_id)
        if session is None:
            session = self._session_manager.restore_session(session_id, params.get("cwd", "."))
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return {"sessionId": session.id}
    
    def handle_session_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/cancel request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")
        
        success = self._session_manager.cancel_session(session_id)
        
        return {
            "sessionId": session_id,
            "cancelled": success,
            "status": "cancelled" if success else "not_found",
        }
    
    def handle_session_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/delete request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")
        
        success = self._session_manager.delete_session(session_id)
        
        return {
            "sessionId": session_id,
            "deleted": success,
        }
    
    def handle_session_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/list request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        sessions = self._session_manager.list_sessions()
        stats = self._session_manager.get_stats()
        
        return {
            "sessions": sessions,
            "stats": stats,
        }
    
    def _execute_with_streaming(self, session: ACPSession, prompt: str) -> Dict[str, Any]:
        """Stream the answer as standard ACP session/update notifications."""
        chunks: List[str] = []
        agent = self._agent_for(session)
        for chunk in agent.run_stream_sync(prompt):
            if chunk and chunk.content:
                chunks.append(chunk.content)
                self._send_notification(ACPMethod.SESSION_UPDATE.value, {
                    "sessionId": session.id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": chunk.content},
                    },
                })

        content = "".join(chunks)
        session.add_message("assistant", content)
        return {"content": content, "stop_reason": "end_turn"}

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a one-way notification over the protocol transport."""
        self._protocol.send_notification(method, params)
