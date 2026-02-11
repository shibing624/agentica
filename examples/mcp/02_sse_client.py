# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP SSE client demo - Demonstrates using MCP with SSE-based servers

No external API needed. Uses calc_server.py which provides math/string/datetime tools.

Usage:
1. Start the server: python 02_sse_server.py
2. Run this client:  python 02_sse_client.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, logger
from agentica.mcp.server import MCPServerSse
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool


async def low_level_sse_demo():
    """Demo 1: Direct MCPClient connection to SSE server."""
    print("\n=== Demo 1: Low-level MCPClient with SSE server ===")

    async with MCPServerSse(
            name="CalcServer-SSE",
            params={"url": "http://localhost:8081/sse"}
    ) as server:
        async with MCPClient(server=server) as client:
            tools = await client.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")

            result = await client.call_tool("add", {"a": 123, "b": 456})
            print(f"add(123, 456) = {client.extract_result_text(result)}")

            result = await client.call_tool("multiply", {"a": 3.14, "b": 100})
            print(f"multiply(3.14, 100) = {client.extract_result_text(result)}")

            result = await client.call_tool("string_reverse", {"text": "Hello MCP!"})
            print(f'string_reverse("Hello MCP!") = {client.extract_result_text(result)}')


async def agent_with_sse_demo():
    """Demo 2: Agent with MCP tools via SSE transport."""
    print("\n=== Demo 2: Agent with MCP SSE tools ===")
    mcp_tool = McpTool(
        url="http://localhost:8081/sse",
        sse_timeout=5.0,
        sse_read_timeout=300.0,
    )

    async with mcp_tool:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[mcp_tool],
            add_datetime_to_instructions=True,
        )

        print("Agent available tools:")
        for tool in agent.get_tools():
            print(f"  - {tool.name}: {list(tool.functions.keys())}")

        await agent.print_response("计算 (123 + 456) * 789 的结果，分步使用工具完成")
        await agent.print_response("把字符串 'Agentica' 反转，并计算反转后的长度")


async def main():
    print("=" * 60)
    print("MCP SSE Transport Demo (no external API needed)")
    print("=" * 60)

    await low_level_sse_demo()
    await agent_with_sse_demo()


if __name__ == "__main__":
    asyncio.run(main())
