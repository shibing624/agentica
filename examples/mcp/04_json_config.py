# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP JSON Config Demo - Load MCP servers from JSON configuration

This example shows how to:
1. Load MCP server configurations from a JSON file
2. Initialize multiple MCP servers from config
3. Use configured servers with agents
"""
import sys
import os
import json
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.mcp.server import MCPServerStdio
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool

pwd_path = os.path.dirname(os.path.abspath(__file__))


def load_mcp_config(config_path: str) -> dict:
    """Load MCP configuration from JSON file.
    
    Args:
        config_path: Path to the JSON config file
        
    Returns:
        Dictionary containing MCP server configurations
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def parse_server_config(name: str, config: dict) -> MCPServerStdio:
    """Parse a server configuration into MCPServerStdio.
    
    Args:
        name: Server name
        config: Server configuration dictionary
        
    Returns:
        MCPServerStdio instance
    """
    command = config.get("command", "")
    args = config.get("args", [])
    env = config.get("env", {})
    
    # If command contains spaces and no args, split it
    if " " in command and not args:
        parts = command.split()
        command = parts[0]
        args = parts[1:]
    
    return MCPServerStdio(
        name=name,
        params={
            "command": command,
            "args": args,
            "env": env,
        }
    )


async def demo_load_config():
    """Demo: Load and display MCP configuration."""
    print("=" * 60)
    print("Demo 1: Load MCP Configuration from JSON")
    print("=" * 60)
    
    # Config file path
    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")
    
    print(f"\nLoading config from: {config_path}")
    
    try:
        config = load_mcp_config(config_path)
        print("\nLoaded configuration:")
        print(json.dumps(config, indent=2))
        
        # Parse servers
        servers = config.get("mcpServers", {})
        print(f"\nFound {len(servers)} server(s):")
        for name, server_config in servers.items():
            print(f"  - {name}: {server_config.get('command', 'N/A')}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure mcp_config.json exists in examples/data/")


async def demo_use_config_with_client():
    """Demo: Use JSON config to initialize MCP clients."""
    print("\n" + "=" * 60)
    print("Demo 2: Initialize MCP Clients from Config")
    print("=" * 60)
    
    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")
    
    try:
        config = load_mcp_config(config_path)
        servers = config.get("mcpServers", {})
        
        for name, server_config in servers.items():
            print(f"\nInitializing server: {name}")
            
            try:
                server = parse_server_config(name, server_config)
                print(f"  Command: {server.params.get('command')}")
                print(f"  Args: {server.params.get('args', [])}")
                
                # Try to connect and list tools
                async with MCPClient(server=server) as client:
                    tools = await client.list_tools()
                    print(f"  Available tools: {[t.name for t in tools]}")
                    
            except Exception as e:
                print(f"  Error connecting: {e}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")


async def demo_agent_with_config():
    """Demo: Create agent with MCP tools from config."""
    print("\n" + "=" * 60)
    print("Demo 3: Create Agent with MCP Tools from Config")
    print("=" * 60)
    
    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")
    
    try:
        config = load_mcp_config(config_path)
        servers = config.get("mcpServers", {})
        
        # Get the weather server config
        if "weather" not in servers:
            print("Weather server not found in config")
            return
            
        weather_config = servers["weather"]
        command = weather_config.get("command", "")
        
        print(f"\nUsing weather server: {command}")
        
        async with McpTool(command) as mcp_tool:
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[mcp_tool],
                show_tool_calls=True,
            )
            
            print("\nAsking about weather...")
            await agent.aprint_response("What's the weather in Beijing? 中文回答")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def create_sample_config():
    """Create a sample MCP config file."""
    print("\n" + "=" * 60)
    print("Creating Sample MCP Config")
    print("=" * 60)
    
    sample_config = {
        "mcpServers": {
            "weather": {
                "command": "uv",
                "args": ["run", "weather-forecast-server"]
            },
            "code-runner": {
                "command": "uv",
                "args": ["run", "mcp-run-python-code"]
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@anthropic/mcp-filesystem", "/tmp"]
            }
        }
    }
    
    config_path = os.path.join(pwd_path, "..", "data/mcp_config.json")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample config created at: {config_path}")
    print("\nConfig contents:")
    print(json.dumps(sample_config, indent=2))


async def main():
    """Main function to run all demos."""
    print("MCP JSON Config Demo")
    print("=" * 60)
    
    await demo_load_config()
    await demo_use_config_with_client()
    await demo_agent_with_config()  # Uncomment to test with actual servers


if __name__ == "__main__":
    asyncio.run(main())
