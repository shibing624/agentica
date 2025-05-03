# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from typing import Any, Dict, List, Optional
from mcp import Tool as MCPTool
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
from agentica.mcp.server import MCPServer
from agentica.utils.log import logger

__all__ = ["MCPClient"]


class MCPClient:
    """A client for interacting with MCP servers.

    This client can be used as an async context manager to automatically manage connections
    and resource cleanup.

    Example:
    ```python
    # For stdio-based server
    params = MCPServerStdioParams(command="python", args=["server.py"])
    async with MCPClient(server=MCPServerStdio(params)) as client:
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(result)

    # For SSE-based server
    params = MCPServerSseParams(url="http://localhost:8000/sse")
    async with MCPClient(server=MCPServerSse(params)) as client:
        result = await client.call_tool("get_weather", {"city": "Tokyo"})
        print(result)
    ```
    """

    def __init__(
            self,
            server: MCPServer,
            include_tools: Optional[List[str]] = None,
            exclude_tools: Optional[List[str]] = None
    ):
        """Initialize the MCP client.

        Args:
            server: The MCP server to connect to
            include_tools: Optional list of tool names to include (if None, includes all)
            exclude_tools: Optional list of tool names to exclude (if None, excludes none)
        """
        self.server = server
        self.tools_list: List[MCPTool] = []
        self.tools_by_name: Dict[str, MCPTool] = {}
        self.include_tools = include_tools
        self.exclude_tools = exclude_tools or []

    async def __aenter__(self) -> 'MCPClient':
        """Enter the async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the async context manager."""
        await self.cleanup()

    async def connect(self) -> None:
        """Connect to the MCP server and retrieve the list of available tools."""
        logger.debug(f"Connecting to MCP server: {self.server.name}")

        try:
            await self.server.connect()

            # Get the list of tools from the MCP server
            all_tools = await self.server.list_tools()

            # Filter tools based on include/exclude lists
            self.tools_list = []
            for tool in all_tools:
                if tool.name in self.exclude_tools:
                    continue
                if self.include_tools is None or tool.name in self.include_tools:
                    self.tools_list.append(tool)

            # Create a mapping of tool names to tools
            self.tools_by_name = {tool.name: tool for tool in self.tools_list}

            logger.debug(f"Connected to {self.server.name} with {len(self.tools_list)} tools available")
            for tool in self.tools_list:
                logger.debug(f"  - {tool.name}: {tool.description or 'No description'}")

        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            await self.cleanup()
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: The name of the tool to call
            arguments: The arguments to pass to the tool

        Returns:
            The result of the tool call

        Raises:
            ValueError: If the tool is not available
        """
        if tool_name not in self.tools_by_name:
            available_tools = ", ".join(self.tools_by_name.keys())
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {available_tools}")
        logger.debug(f"Calling tool '{tool_name}' with arguments: {arguments}")
        try:
            result = await self.server.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            msg = f"Error calling tool '{tool_name}': {e}"
            logger.error(msg)
            return msg

    def extract_result_text(self, result: CallToolResult) -> str:
        """Extract text content from a tool call result.

        Args:
            result: The result from a tool call

        Returns:
            The extracted text content
        """
        if result.isError:
            return f"Error: {result.content}"

        text_parts = []
        for content_item in result.content:
            if isinstance(content_item, TextContent):
                text_parts.append(content_item.text)
            elif isinstance(content_item, ImageContent):
                text_parts.append(f"[Image content: {content_item.data}]")
            elif isinstance(content_item, EmbeddedResource):
                text_parts.append(f"[Embedded resource: {content_item.resource.model_dump_json()}]")
            else:
                text_parts.append(f"[Unsupported content type: {content_item.type}]")

        return "\n".join(text_parts)

    async def list_tools(self) -> List[MCPTool]:
        """List the available tools.

        Returns:
            List of available tools
        """
        return self.tools_list

    async def cleanup(self) -> None:
        """Clean up resources and close the connection."""
        try:
            await self.server.cleanup()
            logger.debug(f"Disconnected from MCP server: {self.server.name}")
        except Exception as e:
            logger.error(f"Error cleaning up MCP client: {e}")
