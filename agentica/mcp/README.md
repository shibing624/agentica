# Model Context Protocol (MCP) Tools

This module provides a client implementation for the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/), allowing you to easily integrate MCP servers with your Agentica agents.

## Features

- Support for stdio, SSE, and StreamableHttp transport modes
- Explicit parameter names for each transport type
- Async context manager-based API for easy resource management
- Tool caching for improved performance
- Automatic tool registration with your agents

## Installation

Make sure you have the `mcp` package installed:

```bash
pip install mcp
```

## Basic Usage

### Using the McpTool Class

The `McpTool` class is the primary way to integrate MCP services with your Agentica agents:

```python
from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool
from agentica import ShellTool

# For SSE transport (direct connection to running server)
mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    sse_timeout=5.0,
    sse_read_timeout=300.0
)

# For StreamableHttp transport
mcp_tool = McpTool(
    url="http://localhost:8000/mcp",
    sse_timeout=5.0,
    sse_read_timeout=300.0
)

# For stdio transport (launches a subprocess)
mcp_tool = McpTool(
    command="python path/to/your/mcp_server.py"
)

# Use with an agent
async with mcp_tool:
    agent = Agent(
        model=OpenAIChat(model="gpt-4o-mini"),
        tools=[ShellTool(), mcp_tool]
    )

    await agent.aprint_response("Use the weather tool to check the forecast for Beijing")
```

### Using Low-Level MCPClient and Server Classes

For more control, you can directly use the `MCPClient` and server classes:

```python
import asyncio
from agentica.mcp.client import MCPClient
from agentica.mcp.server import MCPServerStdio, MCPServerSse, MCPServerStreamableHttp
from datetime import timedelta

async def stdio_example():
    """Example using stdio transport"""
    server = MCPServerStdio(
        name="MathTools",
        params={
            "command": "python",
            "args": ["path/to/your/mcp_server.py"],
            "env": {"VAR": "value"}
        }
    )
    
    async with MCPClient(server=server) as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Call a tool
        result = await client.call_tool("add", {"a": 5, "b": 7})
        text_result = client.extract_result_text(result)
        print(f"5 + 7 = {text_result}")

async def sse_example():
    """Example using SSE transport"""
    server = MCPServerSse(
        name="WeatherService",
        params={
            "url": "http://localhost:8081/sse",
            "headers": {"Authorization": "Bearer your-token"},  # Optional
            "timeout": 5.0,  # HTTP request timeout
            "sse_read_timeout": 300.0  # SSE connection timeout
        }
    )
    
    async with MCPClient(server=server) as client:
        # Use client as in stdio example
        tools = await client.list_tools()
        result = await client.call_tool("get_weather", {"city": "Beijing"})
        print(client.extract_result_text(result))

async def streamable_http_example():
    """Example using StreamableHttp transport"""
    server = MCPServerStreamableHttp(
        name="WeatherService",
        params={
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "Bearer your-token"},  # Optional
            "timeout": timedelta(seconds=5),  # HTTP request timeout
            "sse_read_timeout": timedelta(seconds=300),  # Connection timeout
            "terminate_on_close": True  # Whether to terminate on close
        }
    )
    
    async with MCPClient(server=server) as client:
        # Use client as in other examples
        tools = await client.list_tools()
        result = await client.call_tool("get_weather", {"city": "Beijing"})
        print(client.extract_result_text(result))

if __name__ == "__main__":
    asyncio.run(stdio_example())
    asyncio.run(sse_example())
    asyncio.run(streamable_http_example())
```

## McpTool Configuration Options

The `McpTool` class supports several configuration options:

### Transport Options

You must provide one of these to specify the transport method:

- `stdio_command`: Command string to run the MCP server via stdio transport
- `sse_server_url`: URL of the SSE endpoint for SSE transport
- `streamable_http_url`: URL of the StreamableHttp endpoint
- `server_params`: Directly provide `StdioServerParameters` for stdio transport
- `session`: Directly provide an initialized `ClientSession`

### SSE and StreamableHttp Configuration

For SSE and StreamableHttp transports, you can configure:

- `sse_headers`: HTTP headers for the connection
- `sse_timeout`: HTTP request timeout in seconds (default: 5.0)
- `sse_read_timeout`: Connection timeout in seconds (default: 300.0)
- `terminate_on_close`: Whether to terminate on close (StreamableHttp only, default: True)

### Tool Filtering

You can filter which tools are exposed to your agent:

- `include_tools`: List of tool names to include (if None, includes all)
- `exclude_tools`: List of tool names to exclude

### Other Options

- `env`: Environment variables to pass to the server process (for stdio transport)

## Example

```python
# Create an McpTool with SSE transport
from agentica.tools.mcp_tool import McpTool

mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    sse_headers={"Authorization": "Bearer token123"},
    include_tools=["get_weather", "get_forecast"],
    exclude_tools=["admin_tool"]
)

# Create an McpTool with StreamableHttp transport
mcp_tool = McpTool(
    url="http://localhost:8000/mcp",
    sse_headers={"Authorization": "Bearer token123"},
    include_tools=["get_weather", "get_forecast"],
    exclude_tools=["admin_tool"]
)

# Create an McpTool with stdio transport
mcp_tool = McpTool(
    command="python weather_server.py --port 8081",
    env={"API_KEY": "your_api_key"},
    include_tools=["get_weather"]
)
```

## Backward Compatibility

The `McpTool` class still supports environment variables for backward compatibility:

```python
from agentica.tools.mcp_tool import McpTool
# Using environment variables for SSE configuration
env = {
    "MCP_SERVER_URL": "http://localhost:8081/sse",
    "MCP_SERVER_HEADERS": '{"Authorization": "Bearer token123"}',
    "MCP_SERVER_TIMEOUT": "5",
    "MCP_SERVER_READ_TIMEOUT": "300"
}

mcp_tool = McpTool(env=env)
```

However, using the direct parameters is recommended for clarity and type safety.

## MCP Server Implementation

If you're implementing your own MCP server, you can use the `FastMCP` class from the `mcp` package:

```python
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

# Create server
mcp = FastMCP("My MCP Server", host="0.0.0.0", port=8000)

# Define a tool
@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a city"""
    # Implementation here
    return f"Weather for {city}: Sunny, 25°C"

# Run the server
if __name__ == "__main__":
    # Use 'stdio', 'sse', or 'streamable-http' transport
    mcp.run(transport="streamable-http")
```

## Examples

Check out the examples directory for more complete examples:

- `examples/41_mcp_stdio_demo.py`: Demonstrates how to use MCP with stdio transport
- `examples/42_mcp_sse_server.py`: A simple MCP server with SSE transport
- `examples/42_mcp_sse_client.py`: Demonstrates how to connect to an MCP SSE server
- `examples/44_mcp_streamable_http_server.py`: A simple MCP server with StreamableHttp transport
- `examples/44_mcp_streamable_http_client.py`: Demonstrates how to connect to an MCP StreamableHttp server
 