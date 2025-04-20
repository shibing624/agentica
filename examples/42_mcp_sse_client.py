#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP SSE client demo

start the server first:
    python examples/42_mcp_sse_server.py
Then run this client:
    python examples/42_mcp_sse_client.py
"""

import asyncio
import sys
sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.mcp.server import MCPServerSse
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool
from agentica import logger, ShellTool


async def run(server):
    """Run demo with the provided server"""
    async with MCPClient(server=server) as client:
        # List available tools
        tools = await client.list_tools()
        logger.debug(f"Available tools: {[tool.name for tool in tools]}")
        print(f"Available tools: {[tool.name for tool in tools]}")

        try:
            # Call the get_current_weather tool
            result = await client.call_tool("get_current_weather", {"city": "安陆市"})
            logger.info(result)
            weather_result = client.extract_result_text(result)
            logger.info(f"安陆市天气 = {weather_result}")
            print(f"安陆市天气 = {weather_result}")
        except Exception as e:
            print(f"Error calling tool: {e}")
            sys.exit(1)


async def sse_server_demo():
    """
    Demonstrates how to use the MCP client with a SSE-based server.
    """
    print("\n=== Testing SSE-based MCP server (direct connection) ===")
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8081/sse",
            },
    ) as server:
        try:
            await run(server)
        except Exception as e:
            logger.error(f"Error in SSE server demo: {e}")
            sys.exit(1)


async def mcp_toolkit_with_agent_demo():
    """
    Demonstrates how to use the MCPToolkit with an agent over SSE.
    """
    print("\n=== Testing MCPToolkit with SSE and agent ===")
    try:
        # Use McpTool with direct sse_server_url parameter instead of environment variables
        # This demonstrates the new SSE support in the McpTool class
        mcp_tool = McpTool(
            sse_server_url="http://localhost:8081/sse",  # Use direct parameter instead of env
            sse_timeout=5.0,  # HTTP request timeout in seconds
            sse_read_timeout=300.0  # SSE connection timeout in seconds
        )

        async with mcp_tool:
            m = Agent(
                model=OpenAIChat(model="gpt-4o-mini"),
                tools=[ShellTool(), mcp_tool],
                show_tool_calls=False,
                add_datetime_to_instructions=True,
                debug_mode=True
            )

            print("Available tools from agent:")
            for i in m.get_tools():
                print(f" - {i.name}: {list(i.functions.keys())}")

            print("\nTesting weather tool with agent:")
            await m.aprint_response("查询北京市的天气")

            print("\nTesting shell tool with agent:")
            await m.aprint_response("调shell查看本地目录的路径")
    except Exception as e:
        logger.error(f"Error in MCPToolkit with agent demo: {e}")
        sys.exit(1)


async def main():
    """Main function that runs all the examples."""
    print("MCP SSE Client Demo")
    print("===================")

    await sse_server_demo()
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())