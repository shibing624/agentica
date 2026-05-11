# Quickstart

本指南帮助你在 5 分钟内创建并运行第一个 AI 智能体。

!!! tip "前置条件"
    确保已完成 [安装](installation.md) 并配置了 API Key。

## 第一个 Agent

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(model=ZhipuAI())
    result = await agent.run("一句话介绍北京")
    print(result.content)

asyncio.run(main())
```

输出：

```
北京是中国的首都，是一座拥有三千多年历史的文化名城，也是全国的政治、文化和国际交流中心。
```

## 同步调用

如果你不需要 async，可以使用同步适配器：

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("什么是量子计算?")
print(result.content)
```

## 流式输出

### 异步流式（推荐）

```python
import asyncio
from agentica import Agent, OpenAIChat

async def main():
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
    async for chunk in agent.run_stream("写一首关于春天的诗"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())
```

### 同步流式

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
for chunk in agent.run_stream_sync("写一首关于春天的诗"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
```

## 使用工具

给 Agent 配备工具，让它能执行实际操作：

```python
import asyncio
from agentica import Agent, OpenAIChat, BaiduSearchTool

async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[BaiduSearchTool()],
        instructions=["使用搜索工具获取最新信息，用中文回答"],
    )
    result = await agent.run("今天的科技新闻有哪些?")
    print(result.content)

asyncio.run(main())
```

## 结构化输出

使用 Pydantic 模型获取结构化数据：

```python
import asyncio
from pydantic import BaseModel, Field
from agentica import Agent, OpenAIChat

class CityInfo(BaseModel):
    name: str = Field(description="城市名称")
    country: str = Field(description="所属国家")
    population: str = Field(description="人口")
    highlights: list[str] = Field(description="主要特色")

async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        response_model=CityInfo,
    )
    result = await agent.run("介绍北京")
    city: CityInfo = result.content
    print(f"{city.name} ({city.country})")
    print(f"人口: {city.population}")
    for h in city.highlights:
        print(f"  - {h}")

asyncio.run(main())
```

## 选择模型

Agentica 支持 20+ 模型提供商，详见 [模型提供商指南](../guides/models.md)：

```python
from agentica import (
    OpenAIChat,       # GPT-4o, GPT-4o-mini
    DeepSeek,         # deepseek-chat, deepseek-reasoner
    ZhipuAI,          # glm-4.7-flash (免费)
    Claude,           # claude-sonnet-4-20250514
    Qwen,             # qwen-plus, qwen-turbo
    Moonshot,         # moonshot-v1-128k
    Ark,              # 火山引擎，跑 doubao-pro-32k 等
    Ollama,           # 本地模型
)

agent = Agent(model=DeepSeek(id="deepseek-chat"))
agent = Agent(model=ZhipuAI(id="glm-4.7-flash"))
agent = Agent(model=Ollama(id="llama3.1"))
```

## 下一步

- [Agent 核心概念](../concepts/agent.md) -- 深入理解 Agent 的组成
- [Workflow](../multi-agent/workflow.md) -- 多智能体协作
- [工具系统](../concepts/tools.md) -- 内置工具与自定义工具
- [MCP 集成](../advanced/mcp.md) -- Model Context Protocol
- [CLI 终端指南](terminal.md) -- CLI 完整功能
