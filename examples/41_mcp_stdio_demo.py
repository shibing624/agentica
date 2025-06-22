# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP stdio client demo
"""

import asyncio
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.mcp.server import MCPServerStdio
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool
from agentica import logger, ShellTool


async def stdio_server_demo():
    """
    Demonstrates how to use the MCP client with a stdio-based server.
    
    This example assumes you have a Python script called "math_tools.py" that
    implements a simple MCP server with mathematical tools.
    """
    logger.debug("\n=== Testing stdio-based MCP server ===")
    server = MCPServerStdio(
        name="GetWeather",
        params={
            "command": "uv",
            "args": ['run', "weather-forecast-server"],
        }
    )
    # Use the client as an async context manager
    async with MCPClient(server=server) as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        try:
            result = await client.call_tool("get_weather", {"city": "保定"})
            print(f"保定天气 = {client.extract_result_text(result)}")
        except Exception as e:
            print(f"Error calling tool: {e}")


async def mcp_toolkit_with_agent_demo():
    """
    Demonstrates how to use the MCPToolkit with an agent.
    
    This is a more advanced example that shows how to integrate MCP tools with an agent.
    The agent part is commented out as it depends on your agent implementation.
    """
    logger.debug("\n=== Testing MCPToolkit with agent ===")
    try:
        async with McpTool("uv run weather-forecast-server") as mcp_tool:
            m = Agent(
                model=OpenAIChat(model="gpt-4o-mini"),
                tools=[ShellTool(), mcp_tool],
                show_tool_calls=False,
                add_datetime_to_instructions=True,
                debug_mode=True
            )
            r = m.get_tools()
            print(r)
            for i in r:
                print(i.name, i.functions)

            await m.aprint_response("调天气工具 get_weather 查询合肥市天气咋样")
            await m.aprint_response("调shell 查看本地目录的路径")
    except Exception as e:
        logger.error(f"Error in MCPToolkit with agent demo: {e}")
        sys.exit(1)


async def main():
    """Main function that runs all the examples."""
    await stdio_server_demo()
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
