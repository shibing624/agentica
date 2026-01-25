# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP JSON Config Demo - Load MCP servers from JSON configuration

This example shows how to:
1. Load MCP server configurations from a JSON/YAML file using McpTool.from_config
2. Initialize multiple MCP servers using CompositeMultiMcpTool
3. Use configured servers with agents

Config file format (mcp_config.json or mcp_config.yaml):
{
    "mcpServers": {
        "server_name": {
            "command": "command_to_run",  // for stdio
            "args": ["arg1", "arg2"],     // optional
            "env": {"KEY": "VALUE"},      // optional
            "url": "http://...",          // for SSE/HTTP transport
            "headers": {"Authorization": "Bearer ..."},  // optional
            "timeout": 5.0,               // optional
            "read_timeout": 300.0         // optional
        }
    }
}
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool, CompositeMultiMcpTool
from agentica.mcp.config import MCPConfig

pwd_path = os.path.dirname(os.path.abspath(__file__))


async def demo_load_config():
    """Demo 1: Load and display MCP configuration using MCPConfig."""
    print("=" * 60)
    print("Demo 1: Load MCP Configuration")
    print("=" * 60)

    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")

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
        print("\nPlease ensure mcp_config.json exists in examples/data/")


async def demo_single_server():
    """Demo 2: Load single MCP server from config using McpTool.from_config."""
    print("\n" + "=" * 60)
    print("Demo 2: Load Single MCP Server with from_config")
    print("=" * 60)

    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")

    try:
        # Load a single server by name
        mcp_tool = McpTool.from_config(server_names="weather", config_path=config_path)

        print(f"\nLoaded MCP tool for 'weather' server")
        print(f"Transport type: {mcp_tool._transport_type}")

        # Use as context manager to initialize and get tools
        async with mcp_tool:
            print(f"Available functions: {list(mcp_tool.functions.keys())}")

            # Create agent with the MCP tool
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[mcp_tool],
                show_tool_calls=True,
            )

            print("\nAsking about weather...")
            await agent.aprint_response("What's the weather in Beijing? 中文回答")

    except ValueError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def demo_multiple_servers():
    """Demo 3: Load multiple MCP servers using CompositeMultiMcpTool."""
    print("\n" + "=" * 60)
    print("Demo 3: Load Multiple MCP Servers with CompositeMultiMcpTool")
    print("=" * 60)

    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")

    try:
        # Load multiple servers - returns CompositeMultiMcpTool when multiple servers
        mcp_tool = McpTool.from_config(
            server_names=["weather", "minimax"],  # Load multiple servers
            config_path=config_path
        )

        print(f"\nLoaded MCP tool type: {type(mcp_tool).__name__}")

        if isinstance(mcp_tool, CompositeMultiMcpTool):
            print(f"Number of sub-tools: {len(mcp_tool.tools)}")

        # Use as context manager to initialize all tools
        async with mcp_tool:
            print(f"Total available functions: {list(mcp_tool.functions.keys())}")

            # Create agent with the composite MCP tool
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[mcp_tool],
                show_tool_calls=True,
            )

            print("\nAsking about weather...")
            await agent.aprint_response("What's the weather in Shanghai? 中文回答")

    except ValueError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def demo_all_servers():
    """Demo 4: Load all MCP servers from config."""
    print("\n" + "=" * 60)
    print("Demo 4: Load All MCP Servers from Config")
    print("=" * 60)

    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")

    try:
        # Load all servers by not specifying server_names
        mcp_tool = McpTool.from_config(config_path=config_path)

        print(f"\nLoaded MCP tool type: {type(mcp_tool).__name__}")

        async with mcp_tool:
            print(f"All available functions: {list(mcp_tool.functions.keys())}")

    except ValueError as e:
        print(f"Config error: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Main function to run demos."""
    print("MCP JSON Config Demo - Using McpTool.from_config")
    print("=" * 60)

    # Demo 1: Just load and display config
    await demo_load_config()

    # Demo 2: Load single server (uncomment to test)
    await demo_single_server()

    # Demo 3: Load multiple servers (uncomment to test)
    await demo_multiple_servers()

    # Demo 4: Load all servers (uncomment to test)
    await demo_all_servers()


if __name__ == "__main__":
    asyncio.run(main())
