# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP JSON Config Demo - Load MCP servers from JSON configuration

No external API needed. Uses calc_server.py as the MCP server.

This example shows how to:
1. Load MCP server configurations from a JSON/YAML file using McpTool.from_config
2. Initialize MCP servers from config
3. Use configured servers with agents

Config file format (mcp_config.json):
{
    "mcpServers": {
        "server_name": {
            "command": "command_to_run",
            "args": ["arg1", "arg2"],
            "env": {"KEY": "VALUE"},
            "url": "http://...",
            "headers": {"Authorization": "Bearer ..."},
            "timeout": 5.0,
            "read_timeout": 300.0
        }
    }
}
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool
from agentica.mcp.config import MCPConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")


async def demo_load_config():
    """Demo 1: Load and display MCP configuration using MCPConfig."""
    print("=" * 60)
    print("Demo 1: Load MCP Configuration")
    print("=" * 60)

    config_path = os.path.join(_DATA_DIR, "mcp_config.json")
    print(f"\nLoading config from: {config_path}")

    try:
        config = MCPConfig(config_path)
        servers = config.list_servers()

        print(f"\nFound {len(servers)} server(s):")
        for name, server_config in servers.items():
            print(f"  - {name}:")
            if server_config.command:
                print(f"      command: {server_config.command}")
            if server_config.args:
                print(f"      args: {server_config.args}")
            if server_config.url:
                print(f"      url: {server_config.url}")
    except Exception as e:
        print(f"Error: {e}")


async def demo_from_config():
    """Demo 2: Load MCP server from config and use with Agent."""
    print("\n" + "=" * 60)
    print("Demo 2: Load MCP Server with from_config and Agent")
    print("=" * 60)

    config_path = os.path.join(_DATA_DIR, "mcp_config.json")

    try:
        mcp_tool = McpTool.from_config(server_names="calc", config_path=config_path)
        print(f"\nLoaded MCP tool for 'calc' server")
        print(f"Transport type: {mcp_tool._transport_type}")

        async with mcp_tool:
            print(f"Available functions: {list(mcp_tool.functions.keys())}")

            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[mcp_tool],
            )

            await agent.print_response("用工具计算 sqrt(144) + power(2, 10) 的结果")

    except ValueError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def demo_all_servers():
    """Demo 3: Load all MCP servers from config."""
    print("\n" + "=" * 60)
    print("Demo 3: Load All MCP Servers from Config")
    print("=" * 60)

    config_path = os.path.join(_DATA_DIR, "mcp_config.json")

    try:
        mcp_tool = McpTool.from_config(config_path=config_path)
        print(f"\nLoaded MCP tool type: {type(mcp_tool).__name__}")

        async with mcp_tool:
            print(f"All available functions: {list(mcp_tool.functions.keys())}")

    except ValueError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    print("MCP JSON Config Demo (no external API needed)")
    print("=" * 60)

    await demo_load_config()
    await demo_from_config()
    await demo_all_servers()


if __name__ == "__main__":
    asyncio.run(main())
