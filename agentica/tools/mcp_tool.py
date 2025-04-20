# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model Context Protocol (MCP) client implementations supporting both stdio and SSE transports
"""
from typing import Any, Dict, List, Literal, Optional, Union
from os import environ
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import copy
import asyncio
import threading
import concurrent.futures
from agentica.tools.base import Toolkit, Function
from agentica.utils.log import logger
from agentica.mcp.server import MCPServerStdio, MCPServerSse
from agentica.mcp.client import MCPClient
from agentica.tools.base import Function


class McpTool(Toolkit):
    """
    A toolkit for integrating Model Context Protocol (MCP) servers with Agno agents.
    This allows agents to access tools, resources, and prompts exposed by MCP servers.

    Can be used in three ways:
    1. Direct initialization with a ClientSession
    2. As an async context manager with StdioServerParameters or stdio_command for stdio transport
    3. As an async context manager with sse_server_url for SSE transport
    """

    def __init__(
        self,
        stdio_command: Optional[str] = None,
        sse_server_url: Optional[str] = None,
        *,
        env: Optional[dict[str, str]] = None,
        server_params: Optional[StdioServerParameters] = None,
        session: Optional[ClientSession] = None,
        include_tools: Optional[list[str]] = None,
        exclude_tools: Optional[list[str]] = None,
        sse_headers: Optional[Dict[str, str]] = None,
        sse_timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
    ):
        """
        Initialize the MCP toolkit.

        Args:
            stdio_command: The command to run to start the stdio server. Should be used in conjunction with env.
            sse_server_url: The URL of the SSE endpoint for SSE transport.
            env: The environment variables to pass to the server. Used with stdio_command or for additional SSE config.
            server_params: StdioServerParameters for creating a new stdio session
            session: An initialized MCP ClientSession connected to an MCP server
            include_tools: Optional list of tool names to include (if None, includes all)
            exclude_tools: Optional list of tool names to exclude (if None, excludes none)
            sse_headers: Optional headers for the SSE connection
            sse_timeout: HTTP request timeout for SSE (default: 5 seconds)
            sse_read_timeout: SSE connection timeout (default: 300 seconds)
        """
        super().__init__(name="McpTool")

        # Validate that at least one connection method is provided
        if session is None and server_params is None and stdio_command is None and sse_server_url is None and env is None:
            raise ValueError("Either session, server_params, stdio_command, or sse_server_url must be provided")

        self.session: Optional[ClientSession] = session
        self.server_params: Optional[StdioServerParameters] = server_params

        # Merge provided env with system env
        if env is not None:
            env = {
                **environ,
                **env,
            }
        else:
            env = {**environ}

        # Determine the transport type (stdio or SSE)
        self._transport_type = "stdio"  # Default to stdio
        
        # Check for direct SSE URL parameter first
        self._sse_url = sse_server_url
        
        # If not provided directly, check environment
        if not self._sse_url:
            self._sse_url = env.get("MCP_SERVER_URL")
            
        # Configure SSE if URL is available
        if self._sse_url:
            self._transport_type = "sse"
            
            # Headers provided directly take precedence
            if sse_headers is not None:
                self._sse_headers = sse_headers
            else:
                # Try to get headers from environment
                env_headers = env.get("MCP_SERVER_HEADERS", {})
                if isinstance(env_headers, str):
                    import json
                    try:
                        self._sse_headers = json.loads(env_headers)
                    except:
                        self._sse_headers = {}
                else:
                    self._sse_headers = env_headers
            
            # Timeouts provided directly take precedence
            self._sse_timeout = sse_timeout
            self._sse_read_timeout = sse_read_timeout
            
            # If not provided directly, try to get from environment
            if "MCP_SERVER_TIMEOUT" in env:
                try:
                    self._sse_timeout = float(env.get("MCP_SERVER_TIMEOUT", "5"))
                except ValueError:
                    pass
                    
            if "MCP_SERVER_READ_TIMEOUT" in env:
                try:
                    self._sse_read_timeout = float(env.get("MCP_SERVER_READ_TIMEOUT", "300"))
                except ValueError:
                    pass
        
        # Configure stdio if command is provided and we're not using SSE
        elif stdio_command is not None:
            from shlex import split

            parts = split(stdio_command)
            if not parts:
                raise ValueError("Empty command string")
            cmd = parts[0]
            arguments = parts[1:] if len(parts) > 1 else []
            self.server_params = StdioServerParameters(command=cmd, args=arguments, env=env)
            
            # Store the original command for easier reuse and debugging
            self._stdio_command = stdio_command
            self._cmd_parts = parts

        self._transport_context = None
        self._session_context = None
        self._initialized = False

        self.include_tools = include_tools
        self.exclude_tools = exclude_tools or []
        
        # Store the server configuration separately for tool functions to use directly
        self._server_config = {
            "transport_type": self._transport_type,
            "command": self.server_params.command if self.server_params and self._transport_type == "stdio" else None,
            "args": self.server_params.args if self.server_params and self._transport_type == "stdio" else [],
            "env": env,
            "sse_url": self._sse_url if self._transport_type == "sse" else None,
            "sse_headers": self._sse_headers if self._transport_type == "sse" else None,
            "sse_timeout": self._sse_timeout if self._transport_type == "sse" else None,
            "sse_read_timeout": self._sse_read_timeout if self._transport_type == "sse" else None,
        }

    async def __aenter__(self) -> "McpTool":
        """Enter the async context manager."""

        if self.session is not None:
            # Already has a session, just initialize
            if not self._initialized:
                await self.initialize()
            return self

        try:
            # Handle different transport types
            if self._transport_type == "sse":
                # Create an SSE client connection
                if not self._sse_url:
                    raise ValueError("sse_server_url or MCP_SERVER_URL must be provided for SSE transport")
                
                self._transport_context = sse_client(
                    url=self._sse_url,
                    headers=self._sse_headers,
                    timeout=self._sse_timeout,
                    sse_read_timeout=self._sse_read_timeout
                )
            else:
                # Create a stdio client connection
                if self.server_params is None:
                    raise ValueError("server_params or stdio_command must be provided when using stdio transport")
                
                self._transport_context = stdio_client(self.server_params)
            
            # Create session from transport
            read, write = await self._transport_context.__aenter__()  # type: ignore
            self._session_context = ClientSession(read, write)  # type: ignore
            self.session = await self._session_context.__aenter__()  # type: ignore
            
            # Store the parameters for later use in tool functions
            if self._transport_type == "stdio" and not hasattr(self.session, "_server_params"):
                setattr(self.session, "_server_params", self.server_params)
            elif self._transport_type == "sse" and not hasattr(self.session, "_sse_url"):
                setattr(self.session, "_sse_url", self._sse_url)

            # Initialize with the new session
            await self.initialize()
            return self
        except Exception as e:
            logger.error(f"Failed to enter MCP tool context: {e}")
            # Try to clean up any partially initialized contexts
            if self._session_context is not None:
                try:
                    await self._session_context.__aexit__(None, None, None)
                except:
                    pass
            if self._transport_context is not None:
                try:
                    await self._transport_context.__aexit__(None, None, None)
                except:
                    pass
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        if self._session_context is not None:
            await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
            self.session = None
            self._session_context = None

        if self._transport_context is not None:
            await self._transport_context.__aexit__(exc_type, exc_val, exc_tb)
            self._transport_context = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP toolkit by getting available tools from the MCP server"""
        if self._initialized:
            return

        try:
            if self.session is None:
                raise ValueError("Session is not available. Use as context manager or provide a session.")

            # Initialize the session if not already initialized
            try:
                await asyncio.wait_for(self.session.initialize(), timeout=10.0)
            except Exception as e:
                logger.error(f"Error initializing MCP session: {e}")
                raise ValueError(f"Error initializing MCP session: {e}")

            # Get the list of tools from the MCP server
            try:
                available_tools = await asyncio.wait_for(self.session.list_tools(), timeout=10.0)
                logger.debug(f"Available MCP tools: {[t.name for t in available_tools.tools]}")
            except Exception as e:
                logger.error(f"Error listing MCP tools: {e}")
                raise ValueError(f"Error listing MCP tools: {e}")

            # Filter tools based on include/exclude lists
            filtered_tools = []
            for tool in available_tools.tools:
                if tool.name in self.exclude_tools:
                    continue
                if self.include_tools is None or tool.name in self.include_tools:
                    filtered_tools.append(tool)
            
            # Register the tools with the toolkit
            for tool in filtered_tools:
                try:
                    # Create a custom entrypoint that matches the transport type pattern
                    tool_name = tool.name
                    
                    # Define a wrapper function that creates a fresh connection each time
                    def create_tool_function(t_name):
                        """Create a function for the tool that creates a fresh connection each time"""
                        
                        async def _call_mcp_tool_async(**kwargs):
                            """Async implementation that creates a fresh connection"""
                            logger.debug(f"Creating new MCP connection for tool '{t_name}'")
                            
                            # Create appropriate server based on transport type
                            if self._server_config["transport_type"] == "sse":
                                server = MCPServerSse(
                                    name=t_name,
                                    params={
                                        "url": self._server_config["sse_url"],
                                        "headers": self._server_config["sse_headers"],
                                        "timeout": self._server_config["sse_timeout"],
                                        "sse_read_timeout": self._server_config["sse_read_timeout"]
                                    }
                                )
                            else:
                                server = MCPServerStdio(
                                    name=t_name,
                                    params={
                                        "command": self._server_config["command"],
                                        "args": self._server_config["args"],
                                        "env": copy.deepcopy(self._server_config["env"])
                                    }
                                )
                            
                            try:
                                async with MCPClient(server=server) as client:
                                    # Call the tool (with timeout)
                                    result = await asyncio.wait_for(
                                        client.call_tool(t_name, kwargs),
                                        timeout=30.0
                                    )
                                    return client.extract_result_text(result)
                            except Exception as e:
                                logger.error(f"Error calling MCP tool '{t_name}': {e}")
                                return f"Error calling MCP tool '{t_name}': {e}"

                        def tool_function(**kwargs):
                            """Synchronous wrapper that handles async execution safely"""
                            future = concurrent.futures.Future()

                            def run_in_thread():
                                try:
                                    # Create new event loop
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        # Run the async function
                                        result = loop.run_until_complete(_call_mcp_tool_async(**kwargs))
                                        future.set_result(result)
                                    finally:
                                        loop.close()
                                except Exception as e:
                                    future.set_exception(e)

                            # Start thread and wait for result
                            thread = threading.Thread(target=run_in_thread)
                            thread.start()
                            thread.join(timeout=60)

                            if thread.is_alive():
                                return f"Error: Timeout calling MCP tool '{t_name}'"

                            return future.result()

                        tool_function.__name__ = f"mcp_tool_{t_name}"
                        tool_function.__doc__ = f"Call the MCP tool '{t_name}'"

                        return tool_function

                    # Create the function for this tool
                    entrypoint = create_tool_function(tool_name)
                    
                    # Create parameter schema
                    tool_params = {"type": "object", "properties": {}}
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        if isinstance(tool.inputSchema, dict):
                            if "properties" in tool.inputSchema:
                                tool_params = tool.inputSchema
                            else:
                                tool_params = {
                                    "type": "object",
                                    "properties": tool.inputSchema
                                }

                    f = Function(
                        name=tool_name,
                        description=tool.description,
                        parameters=tool_params,
                        entrypoint=entrypoint,
                        sanitize_arguments=False,  # We'll handle this in the entrypoint
                        skip_entrypoint_processing=True  # Skip processing to preserve our custom entrypoint
                    )
                    self.functions[f.name] = f
                    logger.debug(f"Function: {f.name} registered with {self.name}, Function parameters: {tool_params}")
                except Exception as e:
                    logger.error(f"Failed to register tool {tool.name}: {e}")

            logger.debug(f"{self.name} initialized with {len(filtered_tools)} tools using {self._transport_type} transport")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            raise
