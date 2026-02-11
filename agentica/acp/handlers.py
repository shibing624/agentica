# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Method Handlers

Handles specific ACP methods like initialize, tools/list, tools/call, agent/execute.
"""


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Callable

from agentica.acp.types import (
    ACPTool,
    ACPToolCall,
    ACPToolResult,
    ACPInitializeParams,
    ACPInitializeResult,
    ACPAgentExecuteParams,
    ACPErrorCode,
)
from agentica.acp.session import SessionManager, ACPSession, SessionStatus, SessionMessage
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model
    from agentica.acp.protocol import ACPProtocolHandler


class ACPHandlers:
    """Handlers for ACP methods"""
    
    def __init__(self, agent: Optional["Agent"] = None, model: Optional["Model"] = None,
                 protocol: Optional[Any] = None):
        self._agent = agent
        self._model = model
        self._protocol = protocol
        self._initialized = False
        self._tools: List[ACPTool] = []
        self._session_manager = SessionManager()
        
        # Lazy-initialized tool instances (avoid repeated instantiation)
        self._file_tool = None
        self._execute_tool_instance = None
        self._web_search_tool = None
        
        # Initialize default tools
        self._setup_tools()
    
    def _get_file_tool(self):
        """Get or create file tool instance (singleton pattern)"""
        if self._file_tool is None:
            from agentica.deep_tools import BuiltinFileTool
            self._file_tool = BuiltinFileTool()
        return self._file_tool
    
    def _get_execute_tool(self):
        """Get or create execute tool instance"""
        if self._execute_tool_instance is None:
            from agentica.deep_tools import BuiltinExecuteTool
            self._execute_tool_instance = BuiltinExecuteTool()
        return self._execute_tool_instance
    
    def _get_web_search_tool(self):
        """Get or create web search tool instance"""
        if self._web_search_tool is None:
            from agentica.deep_tools import BuiltinWebSearchTool
            self._web_search_tool = BuiltinWebSearchTool()
        return self._web_search_tool
    
    def _get_or_create_agent(self):
        """Get existing agent or create a new one (centralized initialization)"""
        if self._agent is None:
            from agentica.agent import Agent
            from agentica.model.openai.chat import OpenAIChat
            
            model = self._model or OpenAIChat()
            self._agent = Agent(
                model=model,
                name="Agentica-ACP",
                description="Agentica ACP Agent",
            )
        return self._agent
    
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
                name="edit_file",
                description="Edit a file by replacing text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file"},
                        "old_string": {"type": "string", "description": "Text to replace"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            ),
            ACPTool(
                name="ls",
                description="List directory contents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path"},
                    },
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
                        "include": {"type": "string", "description": "File glob filter, e.g. *.py"},
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
        """Handle initialize request"""
        init_params = ACPInitializeParams.from_dict(params)
        
        logger.info(f"ACP initialize from client: {init_params.clientInfo}")
        
        self._initialized = True
        
        result = ACPInitializeResult(
            protocolVersion=init_params.protocolVersion,
            capabilities={
                "tools": {"listChanged": True},
                "agent": {"execute": True, "cancel": True},
            },
        )
        
        return result.to_dict()
    
    def handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request"""
        logger.info("ACP shutdown requested")
        return {}
    
    def handle_exit(self, params: Dict[str, Any]) -> None:
        """Handle exit notification (no response)"""
        logger.info("ACP exit requested")
        self._initialized = False
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        return {
            "tools": [tool.to_dict() for tool in self._tools],
        }
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        tool_call = ACPToolCall.from_dict(params)
        
        # Execute the tool
        result = self._execute_tool(tool_call)
        
        return result.to_dict()
    
    def _execute_tool(self, tool_call: ACPToolCall) -> ACPToolResult:
        """Execute a tool call"""
        tool_name = tool_call.name
        arguments = tool_call.arguments
        
        try:
            # File operations - use singleton instance
            if tool_name in ("read_file", "write_file", "edit_file", "ls", "glob", "grep"):
                file_tool = self._get_file_tool()
                method = getattr(file_tool, tool_name)
                result = method(**arguments)
                return ACPToolResult(content=result)
            
            # Shell execution
            elif tool_name == "execute":
                execute_tool = self._get_execute_tool()
                result = execute_tool.execute(**arguments)
                return ACPToolResult(content=result)
            
            # Web search
            elif tool_name == "web_search":
                web_tool = self._get_web_search_tool()
                result = web_tool.web_search(**arguments)
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
    
    def handle_agent_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent/execute request"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        exec_params = ACPAgentExecuteParams.from_dict(params)
        
        logger.info(f"Agent execute task: {exec_params.task[:100]}...")
        
        try:
            agent = self._get_or_create_agent()
            response = agent.run(exec_params.task)
            
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
        """
        Handle session/new request.
        
        Creates a new ACP session for the IDE.
        Supports _meta for session key mapping (like acp_tech.md spec).
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        mode = params.get("mode", "default")
        context = params.get("context", {})
        cwd = params.get("cwd", ".")
        
        # Parse _meta for session key (ACP spec)
        meta = params.get("_meta", {})
        session_key = meta.get("sessionKey")
        
        session = self._session_manager.create_session(
            mode=mode,
            initial_context=context,
            cwd=cwd,
            session_key=session_key
        )
        
        logger.info(f"Created session {session.id} (mode: {mode}, cwd: {cwd})")
        
        return {
            "sessionId": session.id,
            "status": "created",
            "mode": mode,
        }
    
    def handle_session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session/prompt request.
        
        This is the core method - processes user prompt and returns response.
        Supports streaming via notifications and cancellation via abort_event.
        """
        import uuid as uuid_module
        import asyncio
        
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        session_id = params.get("sessionId")
        prompt = params.get("prompt", "")
        stream = params.get("stream", True)
        
        if not session_id:
            raise ValueError("sessionId is required")
        if not prompt:
            raise ValueError("prompt is required")
        
        session = self._session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session not found: {session_id}")
        
        # Cancel any existing run
        if session.active_run_id:
            self._session_manager.cancel_session(session_id)
        
        # Create new run with idempotency key
        run_id = str(uuid_module.uuid4())
        abort_event = asyncio.Event()
        self._session_manager.set_active_run(session_id, run_id, abort_event)
        
        # Parse _meta for prefix_cwd option
        meta = params.get("_meta", {})
        prefix_cwd = meta.get("prefixCwd", True)
        
        # Optionally prefix working directory
        if prefix_cwd and session.cwd:
            prompt_with_cwd = f"[Working directory: {session.cwd}]\n\n{prompt}"
        else:
            prompt_with_cwd = prompt
        
        # Add user message to history
        session.add_message("user", prompt)
        session.update_status(SessionStatus.RUNNING)
        
        logger.info(f"Session {session_id} prompt (run_id={run_id[:8]}): {prompt[:100]}...")
        
        try:
            # Execute with streaming support
            if stream and self._protocol:
                result = self._execute_with_streaming(session, prompt_with_cwd)
            else:
                result = self._execute_sync(session, prompt_with_cwd)
            
            session.update_status(SessionStatus.COMPLETED)
            self._session_manager.clear_active_run(session_id)
            
            return {
                "sessionId": session_id,
                "stopReason": result.get("stop_reason", "end_turn"),
                "content": result.get("content", ""),
                "status": "completed",
            }
            
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}")
            session.update_status(SessionStatus.ERROR)
            self._session_manager.clear_active_run(session_id)
            raise
    
    def handle_session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session/load request - load a session by ID"""
        if not self._initialized:
            raise RuntimeError("Server not initialized")
        
        session_id = params.get("sessionId")
        if not session_id:
            raise ValueError("sessionId is required")
        
        session = self._session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Session not found: {session_id}")
        
        return {
            "sessionId": session.id,
            "status": session.status.value,
            "mode": session.mode,
            "messageCount": len(session.messages),
        }
    
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
    
    def _execute_sync(self, session: ACPSession, prompt: str) -> Dict[str, Any]:
        """
        Execute prompt synchronously.
        
        Args:
            session: ACP session
            prompt: User prompt
            
        Returns:
            Execution result
        """
        agent = self._get_or_create_agent()
        response = agent.run_sync(prompt)
        content = response.content if response else ""
        
        # Add assistant message to history
        session.add_message("assistant", content)
        
        return {
            "content": content,
            "stop_reason": "end_turn",
        }
    
    def _execute_with_streaming(self, session: ACPSession, prompt: str) -> Dict[str, Any]:
        """
        Execute prompt with streaming output.
        
        Sends progress notifications during execution.
        
        Args:
            session: ACP session
            prompt: User prompt
            
        Returns:
            Execution result
        """
        full_content = []
        
        # Send initial progress
        self._send_notification("notifications/progress", {
            "sessionId": session.id,
            "type": "start",
            "message": "Processing...",
        })
        
        try:
            agent = self._get_or_create_agent()
            response = agent.run_stream_sync(prompt)
            
            for chunk in response:
                if chunk and chunk.content:
                    full_content.append(chunk.content)
                    
                    # Send progress notification
                    self._send_notification("notifications/session/update", {
                        "sessionId": session.id,
                        "type": "content",
                        "content": chunk.content,
                    })
            
            final_content = "".join(full_content)
            
            # Add assistant message to history
            session.add_message("assistant", final_content)
            
            # Send completion notification
            self._send_notification("notifications/complete", {
                "sessionId": session.id,
                "type": "complete",
            })
            
            return {
                "content": final_content,
                "stop_reason": "end_turn",
            }
            
        except Exception as e:
            # Send error notification
            self._send_notification("notifications/complete", {
                "sessionId": session.id,
                "type": "error",
                "error": str(e),
            })
            raise
    
    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a notification to the client.
        
        Notifications don't require a response.
        """
        if self._protocol:
            from agentica.acp.types import ACPRequest
            notification = ACPRequest(
                id=0,  # Notifications use id=0 or null
                method=method,
                params=params,
            )
            # Use the protocol to send notification
            # This requires protocol to have a send_notification method
            if hasattr(self._protocol, 'send_notification'):
                self._protocol.send_notification(method, params)
