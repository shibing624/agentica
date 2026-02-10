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


import sys
from typing import TYPE_CHECKING, Optional

from agentica.acp.protocol import ACPProtocolHandler
from agentica.acp.handlers import ACPHandlers
from agentica.acp.types import ACPErrorCode, ACPMethod
from agentica.utils.log import logger

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
    
    def __init__(self, agent: Optional["Agent"] = None, model: Optional["Model"] = None):
        """
        Initialize ACP Server.
        
        Args:
            agent: Optional pre-configured Agent instance
            model: Optional Model to use for the agent
        """
        self._protocol = ACPProtocolHandler()
        self._handlers = ACPHandlers(agent=agent, model=model, protocol=self._protocol)
        self._running = False
        
        # Method dispatcher
        self._method_handlers = {
            # Lifecycle
            ACPMethod.INITIALIZE: self._handlers.handle_initialize,
            ACPMethod.SHUTDOWN: self._handlers.handle_shutdown,
            ACPMethod.EXIT: self._handlers.handle_exit,
            
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
        
        This method blocks and listens for JSON-RPC messages on stdin,
        processing them until the server is shut down.
        """
        self._running = True
        logger.info("Agentica ACP Server started")
        
        try:
            while self._running:
                # Read message from stdin
                request = self._protocol.read_message()
                
                if request is None:
                    # EOF reached
                    logger.info("EOF reached, shutting down")
                    break
                
                # Handle the request
                self._handle_request(request)
                
        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self._running = False
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
