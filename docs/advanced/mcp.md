# MCP Integration

[Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) 是一个标准化的工具集成协议，让 Agent 能调用外部服务提供的工具。

## Features

- 支持 stdio 和 SSE 两种传输模式
- Async context manager API
- 工具自动发现和注册
- 工具缓存和过滤

## 安装

```bash
uv add "agentica[mcp]"
# 或只装协议库：uv add mcp
```

## 基本用法

### McpTool（推荐）

```python
from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool

# SSE 传输（连接已运行的服务器）
mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    sse_timeout=5.0,
    sse_read_timeout=300.0,
)

# stdio 传输（启动子进程）
mcp_tool = McpTool(
    command="python path/to/mcp_server.py",
)

# 使用
async with mcp_tool:
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[mcp_tool],
    )
    await agent.print_response("查询北京天气")
```

### MCPClient（低级 API）

```python
from agentica.mcp.client import MCPClient
from agentica.mcp.server import MCPServerStdio, MCPServerSse

async def stdio_example():
    server = MCPServerStdio(
        name="MathTools",
        params={
            "command": "python",
            "args": ["mcp_server.py"],
        }
    )
    async with MCPClient(server=server) as client:
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print(client.extract_result_text(result))
```

### JSON 配置

```python
from agentica import MCPConfig

config = MCPConfig(
    servers=[
        {
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@anthropic/mcp-filesystem", "/path/to/dir"],
        }
    ]
)

agent = Agent(mcp_config=config)
```

## 工具过滤

```python
mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    include_tools=["get_weather", "get_forecast"],
    exclude_tools=["admin_tool"],
)
```

## MCP Server 实现

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My MCP Server", host="0.0.0.0", port=8081)

@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Weather for {city}: Sunny, 25C"

if __name__ == "__main__":
    mcp.run(transport="sse")
```

## 示例

- `examples/41_mcp_stdio_demo.py` -- stdio 传输示例
- `examples/42_mcp_sse_server.py` -- SSE 服务器示例
- `examples/42_mcp_sse_client.py` -- SSE 客户端示例

## 下一步

- [ACP 集成](acp.md) -- Agent Client Protocol
- [Tools](../concepts/tools.md) -- 工具系统
- [Agent 概念](../concepts/agent.md) -- Agent 工具集成
