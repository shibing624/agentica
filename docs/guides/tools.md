# 工具系统

工具赋予 Agent 与外部世界交互的能力。Agentica 提供 40+ 内置工具，并支持自定义工具和 MCP 协议集成。

## 创建自定义工具

### 函数工具（推荐）

任何带类型注解和 docstring 的 Python 函数都可以作为工具：

```python
def get_weather(city: str) -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    # 实际实现...
    return f"{city}：晴，25°C"

agent = Agent(tools=[get_weather])
```

**要点：**

- 函数名即工具名
- docstring 第一行是工具描述
- `Args` 部分描述各参数含义
- 参数类型注解用于生成 JSON Schema
- 返回值建议用 `str`（JSON 格式更佳）

### 异步函数工具

I/O 密集型工具建议使用 async：

```python
import aiohttp

async def fetch_url(url: str) -> str:
    """抓取指定 URL 的网页内容

    Args:
        url: 要抓取的网页 URL
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

agent = Agent(tools=[fetch_url])
```

Agentica 会自动检测 sync/async，sync 函数在 async 上下文中通过 `run_in_executor` 执行。

### 类工具

适合封装一组相关功能：

```python
from agentica import Tool

class DatabaseTool(Tool):
    def __init__(self, connection_string: str):
        super().__init__(name="database")
        self.conn_str = connection_string
        self.register(self.query)
        self.register(self.list_tables)

    def query(self, sql: str) -> str:
        """执行 SQL 查询

        Args:
            sql: SQL 查询语句
        """
        # 实际实现...
        return "query results"

    def list_tables(self) -> str:
        """列出所有数据库表"""
        return "users, orders, products"

agent = Agent(tools=[DatabaseTool("sqlite:///app.db")])
```

### 工具返回值建议

```python
import json

# 推荐：结构化 JSON 返回
def search_products(query: str, max_price: float = None) -> str:
    """搜索产品

    Args:
        query: 搜索关键词
        max_price: 最高价格限制
    """
    results = [
        {"name": "iPhone 16", "price": 7999},
        {"name": "Pixel 9", "price": 5499},
    ]
    return json.dumps(results, ensure_ascii=False)
```

## 内置工具

### 搜索类

| 工具 | 说明 | 依赖 |
|------|------|------|
| `BaiduSearchTool` | 百度搜索 | — |
| `DuckDuckGoTool` | DuckDuckGo 搜索 | `duckduckgo-search` |
| `SearchSerperTool` | Serper 搜索 API | `SERPER_API_KEY` |
| `ExaTool` | Exa 语义搜索 | `exa-py` |
| `SearchBochaTool` | 博查搜索 | — |

```python
from agentica import Agent, BaiduSearchTool

agent = Agent(tools=[BaiduSearchTool()])
```

### 代码与Shell

| 工具 | 说明 |
|------|------|
| `ShellTool` | 执行 Shell 命令 |
| `CodeTool` | Python 代码执行 |
| `PatchTool` | 文件补丁操作 |

```python
from agentica import Agent, ShellTool

agent = Agent(tools=[ShellTool()])
```

### 网页与文件

| 工具 | 说明 |
|------|------|
| `UrlCrawlerTool` | 网页内容抓取 |
| `JinaTool` | Jina Reader API |
| `BrowserTool` | 浏览器自动化 |
| `FileTool` | 文件读写操作 |

### 知识与数据

| 工具 | 说明 |
|------|------|
| `ArxivTool` | Arxiv 论文搜索 |
| `WikipediaTool` | Wikipedia 搜索 |
| `YFinanceTool` | 金融数据查询 |
| `WeatherTool` | 天气查询 |
| `SqlTool` | SQL 数据库查询 |
| `HackerNewsTool` | Hacker News |

### 多媒体

| 工具 | 说明 |
|------|------|
| `DalleTool` | DALL-E 图像生成 |
| `CogViewTool` | 智谱 CogView 图像生成 |
| `CogVideoTool` | 智谱 CogVideo 视频生成 |
| `ImageAnalysisTool` | 图像分析 |
| `OcrTool` | 文字识别 (OCR) |

### 特殊工具

| 工具 | 说明 |
|------|------|
| `UserInputTool` | 运行时向用户提问 (Human-in-the-loop) |
| `SkillTool` | Agent Skill 系统 |
| `McpTool` | MCP 协议集成 |

## MCP 集成

[Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) 是一个标准化的工具集成协议。

### stdio 传输

```python
from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool

mcp_tool = McpTool(
    command="python path/to/mcp_server.py"
)

async with mcp_tool:
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[mcp_tool],
    )
    await agent.print_response("使用工具查询天气")
```

### SSE 传输

```python
mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    sse_timeout=5.0,
    sse_read_timeout=300.0,
)

async with mcp_tool:
    agent = Agent(tools=[mcp_tool])
    await agent.print_response("查询北京天气")
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

### 工具过滤

```python
mcp_tool = McpTool(
    url="http://localhost:8081/sse",
    include_tools=["get_weather", "get_forecast"],
    exclude_tools=["admin_tool"],
)
```

## 工具使用最佳实践

### 1. 清晰的工具描述

模型通过工具描述决定何时调用。描述越清晰，调用越准确：

```python
# 好的描述
def search_database(query: str, table: str = "users") -> str:
    """在数据库中搜索记录

    根据关键词在指定表中搜索匹配的记录。
    返回 JSON 格式的结果列表。

    Args:
        query: 搜索关键词
        table: 目标表名，可选值：users, orders, products
    """
```

### 2. 控制工具数量

工具过多会降低模型的选择准确性：

```python
# 推荐：3-7 个相关工具
agent = Agent(tools=[search, crawl, analyze])

# 避免：过多工具
agent = Agent(tools=[...15个工具...])
```

### 3. 错误处理

工具应返回有意义的错误信息，而非抛出异常：

```python
def call_api(endpoint: str) -> str:
    """调用外部 API"""
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "错误：请求超时，请稍后重试"
    except requests.HTTPError as e:
        return f"错误：HTTP {e.response.status_code}"
```

## 下一步

- [Agent 概念](../concepts/agent.md) — Agent 如何使用工具
- [RAG 指南](rag.md) — 知识库检索
- [安全守卫](guardrails.md) — 工具级安全验证
- [API 参考](../api/agent.md) — 完整 API
