# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Server Implementation

Main ACP server that handles JSON-RPC communication with IDE clients.
Based on Agent Client Protocol (ACP) specification.

Usage:
    from agentica.acp import ACPServer
    
    server = ACPServer()
    server.run()
"""


import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agentica.acp.protocol import ACPProtocolHandler
from agentica.acp.handlers import ACPHandlers
from agentica.acp.types import ACPErrorCode, ACPMethod
from agentica.utils.log import logger, route_console_logging_to_stderr

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model


class ACPServer:
    """
    Agent Client Protocol (ACP) Server
    
    Implements the ACP specification for IDE integration.
    Communicates via JSON-RPC over stdio.
    
    Example IDE configuration:
    {
      "agent_servers": {
        "Agentica": {
          "command": "agentica",
          "args": ["acp"],
          "env": {}
        }
      }
    }
    """
    
    def __init__(self, agent: Optional["Agent"] = None, model: Optional["Model"] = None,
                 session_dir: Optional[str] = None):
        """
        Initialize ACP Server.

        Args:
            agent: Optional pre-configured Agent instance
            model: Optional Model to use for the agent
            session_dir: Where session metadata and transcripts live. Defaults
                to ~/.agentica/acp/sessions; override for tests or embedding.
        """
        self._protocol = ACPProtocolHandler()
        sessions = session_dir or os.environ.get("AGENTICA_ACP_SESSION_DIR") or str(
            Path.home() / ".agentica" / "acp" / "sessions"
        )
        self._handlers = ACPHandlers(
            agent=agent, model=model, protocol=self._protocol, session_dir=sessions
        )
        self._running = False
        
        # Method dispatcher
        self._method_handlers = {
            # Lifecycle
            ACPMethod.INITIALIZE: self._handlers.handle_initialize,
            ACPMethod.SHUTDOWN: self._handlers.handle_shutdown,
            ACPMethod.EXIT: self._handlers.handle_exit,
            ACPMethod.PING: self._handlers.handle_ping,
            
            # Tools
            ACPMethod.TOOLS_LIST: self._handlers.handle_tools_list,
            ACPMethod.TOOLS_CALL: self._handlers.handle_tools_call,
            
            # Agent (legacy)
            ACPMethod.AGENT_EXECUTE: self._handlers.handle_agent_execute,
            ACPMethod.AGENT_CANCEL: self._handlers.handle_agent_cancel,
            
            # Session management
            ACPMethod.SESSION_NEW: self._handlers.handle_session_new,
            ACPMethod.SESSION_PROMPT: self._handlers.handle_session_prompt,
            ACPMethod.SESSION_LOAD: self._handlers.handle_session_load,
            ACPMethod.SESSION_CANCEL: self._handlers.handle_session_cancel,
            ACPMethod.SESSION_DELETE: self._handlers.handle_session_delete,
            ACPMethod.SESSION_LIST: self._handlers.handle_session_list,
        }
    
    def run(self) -> None:
        """
        Run the ACP server.

        Blocks on stdin, reading one JSON-RPC frame per line and writing
        frames to stdout until shutdown or EOF.
        """
        self._running = True
        # stdout is the NDJSON transport. Any other writer — agentica logging,
        # agent printers, third-party print() — would corrupt the frames, so
        # route it to stderr. The protocol handler captured real stdout at
        # construction and keeps writing frames there.
        route_console_logging_to_stderr()
        transport_stdout = sys.stdout
        sys.stdout = sys.stderr
        logger.info("Agentica ACP Server started")

        try:
            while self._running:
                request = self._protocol.read_message()
                if request is None:
                    logger.info("EOF reached, shutting down")
                    break
                self._handle_request(request)
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
        finally:
            self._running = False
            sys.stdout = transport_stdout
            logger.info("Agentica ACP Server stopped")
    
    def _handle_request(self, request) -> None:
        """
        Handle an incoming ACP request.
        
        Args:
            request: The ACPRequest to handle
        """
        method = request.method
        params = request.params or {}
        request_id = request.id
        
        logger.debug(f"Handling method: {method}")
        
        # Check if method exists
        handler = self._method_handlers.get(method)
        
        if handler is None:
            # Method not found
            self._protocol.send_error(
                request_id,
                ACPErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {method}",
            )
            return
        
        # Handle notification methods (no response needed)
        if method == ACPMethod.EXIT:
            handler(params)
            self._running = False
            return
        
        # Handle the method
        try:
            if inspect.iscoroutinefunction(handler):
                result = asyncio.get_event_loop().run_until_complete(handler(params))
            else:
                result = handler(params)
            self._protocol.send_success(request_id, result)
        except Exception as e:
            logger.error(f"Error handling method {method}: {e}")
            self._protocol.send_error(
                request_id,
                ACPErrorCode.INTERNAL_ERROR,
                str(e),
            )
    
    def stop(self) -> None:
        """Stop the ACP server"""
        self._running = False


def main():
    """Main entry point for ACP server"""
    # Check if running in ACP mode
    if len(sys.argv) > 1 and sys.argv[1] == "acp":
        server = ACPServer()
        server.run()
    else:
        print("Usage: agentica acp")
        sys.exit(1)


if __name__ == "__main__":
    main()
