# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP StreamableHttp client demo - Demonstrates using MCP with HTTP streaming

Usage:
1. Start the server: python 03_http_server.py
2. Run this client: python 03_http_client.py
"""
import sys
import os
import asyncio
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, logger, ShellTool
from agentica.mcp.server import MCPServerStreamableHttp
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool


async def streamable_http_server_demo() -> None:
    """Demonstrates direct connection to StreamableHttp MCP server."""
    print("\n=== Testing StreamableHttp MCP server (direct connection) ===")

    async with MCPServerStreamableHttp(
            name="Streamable HTTP Python Server",
            params={
                "url": "http://localhost:8000/mcp",
                "timeout": timedelta(seconds=5),
                "sse_read_timeout": timedelta(seconds=300),
                "terminate_on_close": True
            }
    ) as server:
        try:
            async with MCPClient(server=server) as client:
                tools = await client.list_tools()
                logger.debug(f"Available tools: {[tool.name for tool in tools]}")

                city = "北京市"
                result = await client.call_tool("get_current_weather", {"city": city})
                weather_result = client.extract_result_text(result)
                logger.info(f"{city} weather = {weather_result}")
        except Exception as e:
            logger.error(f"Error in StreamableHttp server demo: {e}")
            sys.exit(1)


async def mcp_toolkit_with_agent_demo() -> None:
    """Demonstrates using MCPToolkit with an agent via StreamableHttp."""
    print("\n=== Testing MCPToolkit with StreamableHttp and agent ===")
    try:
        mcp_tool = McpTool(
            url="http://localhost:8000/mcp",
            sse_timeout=5.0,
            sse_read_timeout=300.0
        )

        async with mcp_tool:
            agent = Agent(
                model=OpenAIChat(id="gpt-4o"),
                tools=[ShellTool(), mcp_tool],
                show_tool_calls=True,
                add_datetime_to_instructions=True,
                debug_mode=True
            )

            print("Agent available tools:")
            for tool in agent.get_tools():
                print(f" - {tool.name}: {list(tool.functions.keys())}")

            print("\nTesting agent with weather tool:")
            await agent.aprint_response("查询北京市今天的气温，并用温度的值乘以 314159.14=？")
    except Exception as e:
        logger.error(f"Error in MCPToolkit with agent demo: {e}")
        sys.exit(1)


async def main() -> None:
    """Run all examples."""
    print("MCP StreamableHttp Client Demo")
    print("=" * 60)

    await streamable_http_server_demo()
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
