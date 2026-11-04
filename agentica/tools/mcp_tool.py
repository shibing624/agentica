# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model Context Protocol (MCP) client implementations supporting both stdio and SSE transports
"""
import asyncio
import concurrent.futures
import copy
import hashlib
import json
import os
import threading
import time
from os import environ
from typing import TYPE_CHECKING, Dict, Optional, Union

# The mcp SDK (and agentica.mcp.client/server, which wrap it) is imported where
# a connection is made, not here: it costs ~0.2 s, and an agent whose MCP tool
# list comes from the cache (prepare_mcp_servers) never connects at build time.
from agentica.mcp.config import MCPConfig
from agentica.tools.base import Function
from agentica.tools.base import Tool
from agentica.tools.origin import ToolOrigin
from agentica.utils.log import logger

if TYPE_CHECKING:
    from mcp import ClientSession, StdioServerParameters


class McpTool(Tool):
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
            command: Optional[str] = None,
            url: Optional[str] = None,
            *,
            env: Optional[dict[str, str]] = None,
            server_params: Optional["StdioServerParameters"] = None,
            session: Optional["ClientSession"] = None,
            include_tools: Optional[list[str]] = None,
            exclude_tools: Optional[list[str]] = None,
            sse_headers: Optional[Dict[str, str]] = None,
            sse_timeout: float = 5.0,
            sse_read_timeout: float = 300.0,
            terminate_on_close: bool = True,
            defer_schema: bool = False,
    ):
        """
        Initialize the MCP toolkit.

        Args:
            command: The command to run to start the stdio server. Should be used in conjunction with env.
            url: The URL of the SSE or StreamableHttp endpoint.
            env: The environment variables to pass to the server. Used with stdio_command or for additional SSE config.
            server_params: StdioServerParameters for creating a new stdio session
            session: An initialized MCP ClientSession connected to an MCP server
            include_tools: Optional list of tool names to include (if None, includes all)
            exclude_tools: Optional list of tool names to exclude (if None, excludes none)
            sse_headers: Optional headers for the SSE or StreamableHttp connection
            sse_timeout: HTTP request timeout for SSE/StreamableHttp (default: 5 seconds)
            sse_read_timeout: SSE/StreamableHttp connection timeout (default: 300 seconds)
            terminate_on_close: Whether to terminate on close (StreamableHttp only, default: True)
            defer_schema: If True, MCP functions are registered as deferred — they
                stay executable in the host registry but do NOT expand into the
                top-level provider tool schema. Pair with UseCapabilityTool so the
                model can still discover/call them via the stable use_capability
                proxy, keeping the prompt cache prefix stable across MCP inventory
                changes. Default False (current behaviour: expand into schema).
        """
        super().__init__(name="McpTool")

        # Validate that at least one connection method is provided
        if session is None and server_params is None and command is None and url is None and env is None:
            raise ValueError("Either session, server_params, stdio_command, or sse_server_url must be provided")

        self.session: Optional["ClientSession"] = session
        self.server_params: Optional["StdioServerParameters"] = server_params
        # stdio command split into (command, args), set when built from a command
        # string; StdioServerParameters is made from it on first connect.
        self._stdio_spec: Optional[tuple[str, list[str], dict]] = None

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

        # Check for direct URL parameter first
        self._url = url

        # If not provided directly, check environment
        if not self._url:
            self._url = env.get("MCP_SERVER_URL")

        # Configure SSE or StreamableHttp if URL is available
        if self._url:
            # Determine transport type based on URL
            # URLs with "/sse" are treated as SSE
            # All other URLs are treated as StreamableHttp
            if "/sse" in self._url:
                self._transport_type = "sse"
            else:
                # Default to streamable-http for all other URLs
                self._transport_type = "streamable-http"

            # Headers provided directly take precedence
            if sse_headers is not None:
                self._headers = sse_headers
            else:
                # Try to get headers from environment
                env_headers = env.get("MCP_SERVER_HEADERS", {})
                if isinstance(env_headers, str):
                    try:
                        self._headers = json.loads(env_headers)
                    except:
                        self._headers = {}
                else:
                    self._headers = env_headers

            # Timeouts provided directly take precedence
            self._timeout = sse_timeout
            self._read_timeout = sse_read_timeout
            self._terminate_on_close = terminate_on_close

            # If not provided directly, try to get from environment
            if "MCP_SERVER_TIMEOUT" in env:
                try:
                    self._timeout = float(env.get("MCP_SERVER_TIMEOUT", "5"))
                except ValueError:
                    self._timeout = 5.0

            if "MCP_SERVER_READ_TIMEOUT" in env:
                try:
                    self._read_timeout = float(env.get("MCP_SERVER_READ_TIMEOUT", "300"))
                except ValueError:
                    self._read_timeout = 300.0

        # Configure stdio if command is provided and we're not using SSE or StreamableHttp
        elif command is not None:
            from shlex import split

            parts = split(command)
            if not parts:
                raise ValueError("Empty command string")
            cmd = parts[0]
            arguments = parts[1:] if len(parts) > 1 else []
            self._stdio_spec = (cmd, arguments, env)

            # Store the original command for easier reuse and debugging
            self._stdio_command = command
            self._cmd_parts = parts

        self._transport_context = None
        self._session_context = None
        self._initialized = False

        self.include_tools = include_tools
        self.exclude_tools = exclude_tools or []
        self.defer_schema = defer_schema
        # Raw tool specs from the last list_tools (or the cache), unfiltered.
        self._tool_specs: Optional[list[dict]] = None
        # Set by from_config: what identifies this server for the tool cache
        # (the config entry, not the merged process env), and how to rebuild it.
        self._cache_identity: Optional[dict] = None
        self._config_server_name: Optional[str] = None
        self._config_path: Optional[str] = None

        # Store the server configuration separately for tool functions to use directly
        self._server_config = {
            "transport_type": self._transport_type,
            "command": self._stdio_command_and_args()[0] if self._transport_type == "stdio" else None,
            "args": self._stdio_command_and_args()[1] if self._transport_type == "stdio" else [],
            "env": env,
            "url": self._url if self._transport_type in ["sse", "streamable-http"] else None,
            "headers": self._headers if self._transport_type in ["sse", "streamable-http"] else None,
            "timeout": self._timeout if self._transport_type in ["sse", "streamable-http"] else None,
            "read_timeout": self._read_timeout if self._transport_type in ["sse", "streamable-http"] else None,
            "terminate_on_close": self._terminate_on_close if self._transport_type == "streamable-http" else None,
        }

    def _stdio_command_and_args(self) -> tuple[Optional[str], list]:
        if self.server_params is not None:
            return self.server_params.command, list(self.server_params.args)
        if self._stdio_spec is not None:
            return self._stdio_spec[0], list(self._stdio_spec[1])
        return None, []

    def _ensure_server_params(self) -> Optional["StdioServerParameters"]:
        """``server_params`` for a stdio connection, built from the command on first use."""
        if self.server_params is None and self._stdio_spec is not None:
            from mcp import StdioServerParameters

            cmd, arguments, env = self._stdio_spec
            self.server_params = StdioServerParameters(command=cmd, args=arguments, env=env)
        return self.server_params

    async def __aenter__(self) -> "McpTool":
        """Enter the async context manager."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from agentica.mcp.server import streamable_http_transport

        if self.session is not None:
            # Already has a session, just initialize
            if not self._initialized:
                await self.initialize()
            return self

        try:
            # Handle different transport types
            if self._transport_type == "sse":
                # Create an SSE client connection
                if not self._url:
                    raise ValueError("url or MCP_SERVER_URL must be provided for SSE transport")

                self._transport_context = sse_client(
                    url=self._url,
                    headers=self._headers,
                    timeout=self._timeout,
                    sse_read_timeout=self._read_timeout
                )

                # For SSE, we get a tuple of (read, write)
                read, write = await self._transport_context.__aenter__()

            elif self._transport_type == "streamable-http":
                # Create a StreamableHttp client connection
                if not self._url:
                    raise ValueError("url or MCP_SERVER_URL must be provided for StreamableHttp transport")

                self._transport_context = streamable_http_transport(
                    {
                        "url": self._url,
                        "headers": self._headers,
                        "timeout": self._timeout,
                        "sse_read_timeout": self._read_timeout,
                        "terminate_on_close": self._terminate_on_close,
                    }
                )

                # Every mcp 2.x transport yields (read, write)
                read, write = await self._transport_context.__aenter__()

            else:
                # Create a stdio client connection
                if self._ensure_server_params() is None:
                    raise ValueError("server_params or stdio_command must be provided when using stdio transport")

                self._transport_context = stdio_client(self.server_params)

                # For stdio, we get a tuple of (read, write)
                read, write = await self._transport_context.__aenter__()

            # Create session from transport
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()

            # Store the parameters for later use in tool functions
            if self._transport_type == "stdio" and not hasattr(self.session, "_server_params"):
                setattr(self.session, "_server_params", self.server_params)
            elif self._transport_type in ["sse", "streamable-http"] and not hasattr(self.session, "_url"):
                setattr(self.session, "_url", self._url)

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
        try:
            if self._session_context is not None:
                await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
                self.session = None
                self._session_context = None

            if self._transport_context is not None:
                try:
                    await self._transport_context.__aexit__(exc_type, exc_val, exc_tb)
                except GeneratorExit:
                    # This is expected for streamable-http transport during cleanup
                    if self._transport_type == "streamable-http":
                        logger.debug("GeneratorExit during streamable-http transport cleanup (expected)")
                    else:
                        logger.warning("GeneratorExit during transport cleanup")
                except Exception as e:
                    logger.error(f"Error closing transport context: {e}")
                self._transport_context = None
        except Exception as e:
            logger.error(f"Error exiting MCP tool context: {e}")
        finally:
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

            specs = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in available_tools.tools
            ]
            self._register_tool_specs(specs)
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            raise

    def _register_tool_specs(self, specs: list[dict]) -> None:
        """Register one Function per MCP tool spec ``{name, description, input_schema}``.

        Specs come from a live ``list_tools`` or from the on-disk tool cache;
        either way the entrypoint opens its own connection per call, so a
        function registered from the cache works without a startup session.
        """
        self._tool_specs = list(specs)
        filtered_tools = [
            spec for spec in specs
            if spec["name"] not in self.exclude_tools
            and (self.include_tools is None or spec["name"] in self.include_tools)
        ]
        # Register the tools with the toolkit
        for tool in filtered_tools:
            try:
                # Create a custom entrypoint that matches the transport type pattern
                tool_name = tool["name"]

                # Define a wrapper function that creates a fresh connection each time
                def create_tool_function(t_name):
                    """Create a function for the tool that creates a fresh connection each time"""

                    async def _call_mcp_tool_async(**kwargs):
                        """Async implementation that creates a fresh connection"""
                        logger.debug(f"Creating new MCP connection for tool '{t_name}'")
                        from agentica.mcp.client import MCPClient
                        from agentica.mcp.server import (
                            MCPServerSse,
                            MCPServerStdio,
                            MCPServerStreamableHttp,
                        )

                        # Create appropriate server based on transport type
                        if self._server_config["transport_type"] == "sse":
                            server = MCPServerSse(
                                name=t_name,
                                params={
                                    "url": self._server_config["url"],
                                    "headers": self._server_config["headers"],
                                    "timeout": self._server_config["timeout"],
                                    "sse_read_timeout": self._server_config["read_timeout"]
                                },
                                client_session_timeout_seconds=self._server_config["timeout"] or 8.0
                            )
                        elif self._server_config["transport_type"] == "streamable-http":
                            server = MCPServerStreamableHttp(
                                name=t_name,
                                params={
                                    "url": self._server_config["url"],
                                    "headers": self._server_config["headers"],
                                    "timeout": self._server_config["timeout"],
                                    "sse_read_timeout": self._server_config["read_timeout"],
                                    "terminate_on_close": self._server_config["terminate_on_close"]
                                },
                                client_session_timeout_seconds=self._server_config["timeout"] or 8.0
                            )
                        else:
                            server = MCPServerStdio(
                                name=t_name,
                                params={
                                    "command": self._server_config["command"],
                                    "args": self._server_config["args"],
                                    "env": copy.deepcopy(self._server_config["env"])
                                },
                                client_session_timeout_seconds=self._server_config["timeout"] or 8.0
                            )

                        async with MCPClient(server=server) as client:
                            # Call the tool (with timeout)
                            read_timeout = self._server_config["read_timeout"] or 300.0
                            result = await asyncio.wait_for(
                                client.call_tool(t_name, kwargs),
                                timeout=read_timeout
                            )
                            # Check if result is a string (error message) or a CallToolResult
                            if isinstance(result, str):
                                return result  # Already an error message string
                            return client.extract_result_text(result)

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
                            raise TimeoutError(f"Timeout calling MCP tool '{t_name}'")

                        return future.result()

                    tool_function.__name__ = f"mcp_tool_{t_name}"
                    tool_function.__doc__ = f"Call the MCP tool '{t_name}'"

                    return tool_function

                # Create the function for this tool
                entrypoint = create_tool_function(tool_name)

                # Create parameter schema
                tool_params = {"type": "object", "properties": {}}
                input_schema = tool.get("input_schema")
                if input_schema:
                    if "properties" in input_schema:
                        tool_params = input_schema
                    else:
                        tool_params = {
                            "type": "object",
                            "properties": input_schema
                        }

                f = Function(
                    name=tool_name,
                    description=tool.get("description"),
                    parameters=tool_params,
                    entrypoint=entrypoint,
                    skip_entrypoint_processing=True,
                    deferred=self.defer_schema,
                    origin=ToolOrigin(
                        type="mcp",
                        provider_name=self.name,
                        source_tool_name=tool_name,
                    ),
                )
                self.functions[f.name] = f
                logger.debug(f"Function: {f.name} registered with {self.name}, Function parameters: {tool_params}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool_name}: {e}")

        logger.debug(
            f"{self.name} initialized with {len(filtered_tools)} tools using {self._transport_type} transport")

    @classmethod
    def from_config(cls, server_names: Optional[Union[str, list[str]]] = None,
                    config_path: Optional[str] = None):
        """Create McpTool instance from configuration file.

        Args:
            server_names: Optional server name(s) to load. If None, loads all servers
            config_path: Optional path to config file

        Returns:
            McpTool instance with configured servers

        Raises:
            ValueError: If specified servers not found in config
        """
        config = MCPConfig(config_path)
        all_servers = config.servers

        if not all_servers:
            raise ValueError(f"No MCP servers found in configuration: `{config.config_path}`")

        # Convert single server name to list
        if isinstance(server_names, str):
            server_names = [server_names]

        # If no servers specified, use all
        if server_names is None:
            server_names = list(all_servers.keys())

        # Validate servers exist
        invalid_servers = set(server_names) - set(all_servers.keys())
        if invalid_servers:
            raise ValueError(f"Servers not found in MCP configuration: {invalid_servers}. "
                             f"Available servers: {list(all_servers.keys())}")

        tools = []
        for name in server_names:
            server = all_servers[name]
            if server.url:
                # Determine transport type based on URL
                # URLs with "/sse" are treated as SSE
                # All other URLs are treated as StreamableHttp
                is_streamable_http = True  # Default to streamable-http
                transport_type = "streamable-http"

                if "/sse" in server.url:
                    is_streamable_http = False
                    transport_type = "sse"

                # Create tool with appropriate configuration
                tool = cls(
                    url=server.url,
                    sse_headers=server.headers,
                    sse_timeout=server.timeout,
                    sse_read_timeout=server.read_timeout,
                    terminate_on_close=True if is_streamable_http else None
                )

                logger.debug(f"Created {transport_type} tool for server '{name}' with URL: {server.url}")
            else:
                # Create stdio tool
                # Resolve relative paths in args to be relative to config file directory
                config_dir = os.path.dirname(os.path.abspath(config.config_path))
                resolved_args = []
                for arg in (server.args or []):
                    # If arg is a relative path that exists relative to config_dir, resolve it
                    if not os.path.isabs(arg):
                        potential_path = os.path.join(config_dir, arg)
                        if os.path.exists(potential_path):
                            resolved_args.append(os.path.abspath(potential_path))
                        else:
                            resolved_args.append(arg)
                    else:
                        resolved_args.append(arg)
                
                cmd = f"{server.command} {' '.join(resolved_args)}".strip()
                tool = cls(
                    command=cmd,
                    env=server.env
                )
                logger.debug(f"Created stdio tool for server '{name}' with command: {cmd}")
            tool._cache_identity = {
                "version": MCP_TOOLS_CACHE_VERSION,
                "name": name,
                "url": server.url,
                "headers": server.headers,
                "command": server.command,
                "args": server.args,
                "env": server.env,
            }
            tool._config_server_name = name
            tool._config_path = config.config_path
            tools.append(tool)

        logger.info(f"Created {len(tools)} MCP tools({server_names}) from configuration: {config.config_path}")

        # Return single tool or composite
        if len(tools) == 1:
            return tools[0]
        return CompositeMultiMcpTool(tools)


class CompositeMultiMcpTool(Tool):
    """Combines multiple McpTool instances into one.

    Uses composition (not inheritance from McpTool) since it manages
    multiple MCP connections rather than a single one.
    """

    def __init__(self, tools: list[McpTool]):
        super().__init__(name="composite_mcp", description="Multi-MCP tool aggregator")
        self.tools = tools
        self.functions = {}
        self._initialized = False

    async def _enter_tool(self, tool):
        """Helper method to enter a single tool context."""
        try:
            entered_tool = await tool.__aenter__()
            await entered_tool.initialize()
            return entered_tool
        except Exception as e:
            logger.error(f"Error entering tool context: {e}")
            # Don't re-raise, just return None so we can continue with other tools
            return None

    async def __aenter__(self):
        """Enter context for all tools sequentially."""
        entered_tools = []
        errors = []

        for tool in self.tools:
            try:
                entered_tool = await self._enter_tool(tool)
                if entered_tool:
                    entered_tools.append(entered_tool)
            except Exception as e:
                errors.append((tool, e))
                logger.error(f"Failed to initialize tool: {e}")

        self.tools = [t for t in entered_tools if t is not None]

        # Merge functions from all successfully initialized tools
        for tool in self.tools:
            self.functions.update(tool.functions)

        self._initialized = True

        if not self.tools:
            if errors:
                error_msgs = "\n".join([f"- {e}" for _, e in errors])
                raise ValueError(f"Failed to initialize any tools. Errors:\n{error_msgs}")
            else:
                raise ValueError("No tools were successfully initialized")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context for all tools sequentially."""
        errors = []
        for tool in reversed(self.tools):  # Exit in reverse order
            try:
                await tool.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                errors.append(e)
                logger.error(f"Error exiting tool context: {e}")

        if errors and not exc_type:  # Only raise if we're not already handling an exception
            raise errors[0]

    async def initialize(self):
        """Initialize is handled during __aenter__."""
        if self._initialized:
            return

    def _adopt(self, tools: list[McpTool]) -> None:
        """Take ``tools`` (already registered, see ``prepare_mcp_servers``) as the members."""
        self.tools = list(tools)
        self.functions = {}
        for tool in self.tools:
            self.functions.update(tool.functions)
        self._initialized = True


# ============== Startup tool discovery (cached) ==============
#
# Discovering an MCP server's tools means starting it (stdio) or reaching it
# (HTTP), then initialize + list_tools: about a second per server, paid by every
# agent build (each CLI start, each gateway session). Calls do not reuse that
# connection, since every call opens its own, so the only thing startup needs is
# the tool list. It is cached per server config: a cache hit registers the
# tools at once, and a background thread re-lists them for the next build.
# Servers without a cache entry are listed in parallel, not one after another.

MCP_TOOLS_CACHE_VERSION = 1
MCP_DISCOVERY_TIMEOUT_S = 30.0
MCP_REFRESH_DELAY_S = 5.0
_refresh_started: set[str] = set()
_refresh_lock = threading.Lock()


def _tools_cache_enabled() -> bool:
    return os.getenv("AGENTICA_MCP_TOOLS_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _tools_cache_path(tool: McpTool) -> Optional[str]:
    if tool._cache_identity is None or not _tools_cache_enabled():
        return None
    blob = json.dumps(tool._cache_identity, sort_keys=True, default=str).encode("utf-8")
    key = hashlib.sha256(blob).hexdigest()[:24]
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    return os.path.join(home, "cache", "mcp_tools", f"{key}.json")


def load_cached_tool_specs(tool: McpTool) -> Optional[list[dict]]:
    """Tool specs cached for this server's config, or None (no entry / unreadable)."""
    path = _tools_cache_path(tool)
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            specs = json.load(f)["tools"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.debug(f"MCP tool cache unreadable ({path}): {e}")
        return None
    if not isinstance(specs, list) or not all(isinstance(t, dict) and t.get("name") for t in specs):
        return None
    return specs


def write_tool_specs_cache(tool: McpTool, specs: list[dict]) -> None:
    """Persist ``specs`` for this server's config (atomic replace). No-op without a config identity."""
    path = _tools_cache_path(tool)
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"server": tool._config_server_name, "written_at": time.time(), "tools": specs},
                      f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except OSError as e:
        logger.debug(f"MCP tool cache not written ({path}): {e}")


async def _discover(tool: McpTool, timeout: float) -> Optional[McpTool]:
    """Connect, list and register tools, disconnect, all in this one task.

    The mcp transports are anyio contexts that must be exited by the task that
    entered them, so each server's whole round trip stays in one task; that is
    what lets servers be discovered concurrently. Returns None on failure.
    """
    label = tool._config_server_name or tool.name
    try:
        await asyncio.wait_for(tool.__aenter__(), timeout=timeout)
    except Exception as e:
        logger.warning(f"MCP server '{label}' unavailable, its tools are not loaded: {e!r}")
        return None
    try:
        specs = list(tool._tool_specs or [])
    finally:
        await tool.__aexit__(None, None, None)
    write_tool_specs_cache(tool, specs)
    return tool


def _run_blocking(coro, timeout: float):
    """Run ``coro`` to completion from sync code, inside or outside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result(timeout=timeout + 5)


def _refresh_in_background(tools: list[McpTool]) -> None:
    """Re-list cached servers off the startup path; the result applies to the next build.

    Works on fresh instances, never on ``tools``: the running agent's tool list
    must not change under it (the tool schema is in the cached prompt prefix).
    At most once per server config per process.
    """
    todo = []
    with _refresh_lock:
        for tool in tools:
            path = _tools_cache_path(tool)
            if path is None or path in _refresh_started or tool._config_server_name is None:
                continue
            _refresh_started.add(path)
            todo.append(tool)
    if not todo:
        return

    def _work():
        # Start after the build has finished: importing the mcp SDK and
        # spawning stdio servers competes with startup for the GIL and CPU.
        time.sleep(MCP_REFRESH_DELAY_S)
        pairs = []
        for tool in todo:
            try:
                pairs.append((tool, McpTool.from_config(server_names=tool._config_server_name,
                                                        config_path=tool._config_path)))
            except Exception as e:
                logger.debug(f"MCP refresh skipped for '{tool._config_server_name}': {e}")

        async def _all():
            return await asyncio.gather(*(_discover(fresh, MCP_DISCOVERY_TIMEOUT_S) for _, fresh in pairs))

        try:
            results = asyncio.run(_all())
        except Exception as e:
            logger.debug(f"MCP background refresh failed: {e!r}")
            return
        for (old, _), new in zip(pairs, results):
            if new is not None and new._tool_specs != old._tool_specs:
                logger.info(f"MCP server '{old._config_server_name}' tool list changed; "
                            "the new list applies from the next session")

    threading.Thread(target=_work, name="agentica-mcp-refresh", daemon=True).start()


def prepare_mcp_servers(servers: list[McpTool], timeout: float = MCP_DISCOVERY_TIMEOUT_S) -> list[McpTool]:
    """Register tools on each server; return the usable ones, in config order.

    Cached servers are registered from the cache (no connection) and refreshed
    in the background. The rest are discovered concurrently, so startup waits
    for the slowest of them, not their sum. A server that cannot be reached is
    left out with a warning.
    """
    cached, missing = [], []
    for tool in servers:
        specs = load_cached_tool_specs(tool)
        if specs is not None:
            tool._register_tool_specs(specs)
            cached.append(tool)
        else:
            missing.append(tool)

    discovered: list[McpTool] = []
    if missing:
        async def _all():
            return await asyncio.gather(*(_discover(t, timeout) for t in missing))

        discovered = [t for t in _run_blocking(_all(), timeout) if t is not None]

    _refresh_in_background(cached)
    ready = {id(t) for t in cached + discovered}
    return [t for t in servers if id(t) in ready]
