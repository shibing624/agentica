# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP stdio client demo - Demonstrates using MCP with stdio-based servers

This example shows how to:

install dependencies:
uv pip install weather-forecast-server mcp-run-python-code

1. Connect to stdio-based MCP servers
2. List and call tools
3. Integrate MCP tools with agents
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.mcp.server import MCPServerStdio
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool
from agentica import logger, ShellTool


async def get_weather_stdio_demo():
    """Demonstrates how to use the MCP client with a stdio-based server."""
    logger.debug("\n=== Testing stdio-based MCP server ===")
    server = MCPServerStdio(
        name="GetWeather",
        client_session_timeout_seconds=90,
        params={
            "command": "uv",
            "args": ["run", "weather-forecast-server"],
        }
    )
    
    async with MCPClient(server=server) as client:
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        try:
            result = await client.call_tool("get_weather", {"city": "北京"})
            print(f"北京天气 = {client.extract_result_text(result)}")
        except Exception as e:
            print(f"Error calling tool: {e}")


async def run_python_code_server_demo():
    """Demonstrates MCP server for Python code execution."""
    logger.debug("\n=== Testing Python code execution MCP server ===")
    server = MCPServerStdio(
        name="code-interpreter",
        client_session_timeout_seconds=30,
        params={
            "command": "uv",
            "args": ["run", "mcp-run-python-code"],
        }
    )
    
    async with MCPClient(server=server) as client:
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        try:
            result = await client.call_tool(
                "run_python_code", 
                {"code": "c=123*456", 'variable_to_return': 'c'}
            )
            print(f"Result: {client.extract_result_text(result)}")
        except Exception as e:
            print(f"Error calling tool: {e}")


async def mcp_toolkit_with_agent_demo():
    """Demonstrates how to integrate MCP tools with an agent."""
    logger.debug("\n=== Testing MCPToolkit with agent ===")
    try:
        async with McpTool("uv run weather-forecast-server") as mcp_tool1, \
                McpTool("uv run mcp-run-python-code") as mcp_tool2:
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[ShellTool(), mcp_tool1, mcp_tool2],
                add_datetime_to_instructions=True,
                debug_mode=True
            )
            
            # List available tools
            tools = agent.get_tools()
            print(f"Agent tools: {[t.name for t in tools]}")

            await agent.print_response("调天气工具 get_weather 查询合肥市天气咋样")
            await agent.print_response("调shell 查看本地目录的路径")
            await agent.print_response("写python代码并执行计算 123*456*32.132的平方值")
    except Exception as e:
        logger.error(f"Error in MCPToolkit with agent demo: {e}")
        sys.exit(1)


async def main():
    """Main function that runs all the examples."""
    print("=" * 60)
    print("MCP Stdio Demo")
    print("=" * 60)
    
    await get_weather_stdio_demo()
    await run_python_code_server_demo()
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
