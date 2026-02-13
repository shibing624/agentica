# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP StreamableHttp client demo - Demonstrates using MCP with HTTP streaming

No external API needed. Uses calc_server.py which provides math/string/datetime tools.

Usage:
1. Start the server: python 03_http_server.py
2. Run this client:  python 03_http_client.py
"""
import sys
import os
import asyncio
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, logger
from agentica.agent.config import PromptConfig
from agentica.mcp.server import MCPServerStreamableHttp
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool


async def low_level_http_demo():
    """Demo 1: Direct MCPClient connection to StreamableHttp server."""
    print("\n=== Demo 1: Low-level MCPClient with StreamableHttp server ===")

    async with MCPServerStreamableHttp(
            name="CalcServer-HTTP",
            params={
                "url": "http://localhost:8000/mcp",
                "timeout": timedelta(seconds=5),
                "sse_read_timeout": timedelta(seconds=300),
                "terminate_on_close": True,
            }
    ) as server:
        async with MCPClient(server=server) as client:
            tools = await client.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")

            result = await client.call_tool("sqrt", {"number": 144})
            print(f"sqrt(144) = {client.extract_result_text(result)}")

            result = await client.call_tool("power", {"base": 2, "exponent": 10})
            print(f"power(2, 10) = {client.extract_result_text(result)}")

            result = await client.call_tool("current_time", {})
            print(f"current_time() = {client.extract_result_text(result)}")


async def agent_with_http_demo():
    """Demo 2: Agent with MCP tools via StreamableHttp transport."""
    print("\n=== Demo 2: Agent with MCP StreamableHttp tools ===")
    mcp_tool = McpTool(
        url="http://localhost:8000/mcp",
        sse_timeout=5.0,
        sse_read_timeout=300.0,
    )

    async with mcp_tool:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[mcp_tool],
            prompt_config=PromptConfig(add_datetime_to_instructions=True),
        )

        print("Agent available tools:")
        for tool in agent.get_tools():
            print(f"  - {tool.name}: {list(tool.functions.keys())}")

        await agent.print_response("计算 2 的 20 次方，再对结果开平方，使用工具分步完成")
        await agent.print_response("获取当前时间，并统计 'Hello World from Agentica MCP' 有多少个单词")


async def main():
    print("=" * 60)
    print("MCP StreamableHttp Transport Demo (no external API needed)")
    print("=" * 60)

    await low_level_http_demo()
    await agent_with_http_demo()


if __name__ == "__main__":
    asyncio.run(main())
