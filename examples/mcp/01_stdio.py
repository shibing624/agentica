# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP stdio client demo - Demonstrates using MCP with stdio-based servers

This example shows how to:
1. Connect to a stdio-based MCP server (no external API needed)
2. List and call tools via MCPClient
3. Integrate MCP tools with an Agent

Install dependencies:
    pip install mcp

Usage:
    python 01_stdio.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, logger
from agentica.agent.config import PromptConfig
from agentica.mcp.server import MCPServerStdio
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool

# Path to the local MCP server script (no external API needed)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_SCRIPT = os.path.join(_SCRIPT_DIR, "calc_server.py")


async def low_level_client_demo():
    """Demo 1: Use MCPClient directly to list and call tools."""
    print("\n=== Demo 1: Low-level MCPClient with stdio server ===")
    server = MCPServerStdio(
        name="CalcServer",
        client_session_timeout_seconds=30,
        params={
            "command": sys.executable,
            "args": [_SERVER_SCRIPT],
        }
    )

    async with MCPClient(server=server) as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call tools directly
        result = await client.call_tool("add", {"a": 123, "b": 456})
        print(f"add(123, 456) = {client.extract_result_text(result)}")

        result = await client.call_tool("multiply", {"a": 3.14, "b": 100})
        print(f"multiply(3.14, 100) = {client.extract_result_text(result)}")

        result = await client.call_tool("string_length", {"text": "Hello MCP!"})
        print(f'string_length("Hello MCP!") = {client.extract_result_text(result)}')

        result = await client.call_tool("current_time", {})
        print(f"current_time() = {client.extract_result_text(result)}")


async def agent_with_mcp_demo():
    """Demo 2: Integrate MCP tools with an Agent."""
    print("\n=== Demo 2: Agent with MCP stdio tools ===")
    async with McpTool(f"{sys.executable} {_SERVER_SCRIPT}") as mcp_tool:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[mcp_tool],
            prompt_config=PromptConfig(add_datetime_to_instructions=True),
        )

        # List tools the agent can use
        tools = agent.get_tools()
        print(f"Agent tools: {[t.name for t in tools]}")

        # Let the agent use MCP tools to answer questions
        await agent.print_response("计算 123 * 456 + 789 的结果，使用工具完成")
        await agent.print_response("告诉我当前时间，并计算字符串 'Agentica MCP Demo' 的长度")


async def main():
    print("=" * 60)
    print("MCP Stdio Transport Demo (no external API needed)")
    print("=" * 60)

    await low_level_client_demo()
    await agent_with_mcp_demo()


if __name__ == "__main__":
    asyncio.run(main())
