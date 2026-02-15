# Agent

Agent 是 Agentica 的核心组件——它将模型、工具和记忆连接在一起，能够思考、决策和执行动作。

## 基本结构

```
Agent
├── Model          # 大脑：LLM 提供推理能力
├── Tools          # 双手：与外部世界交互的能力
├── Memory         # 记忆：会话历史与长期记忆
├── Knowledge      # 知识：外部文档检索 (RAG)
└── Instructions   # 指令：定义 Agent 的行为
```

## 创建 Agent

### 最小示例

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
```

### 完整配置

```python
from agentica import Agent, OpenAIChat, BaiduSearchTool

agent = Agent(
    name="Research Assistant",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "你是一个专业的研究助手",
        "搜索并整理最新信息",
        "用中文回答，条理清晰",
    ],
    tools=[BaiduSearchTool()],
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)
```

### 参数分层

Agent 的参数组织为三层：

**第一层：核心定义**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Model` | LLM 模型实例 |
| `name` | `str` | Agent 名称 |
| `instructions` | `str \| List[str] \| Callable` | 行为指令 |
| `tools` | `List[Tool \| Callable]` | 工具列表 |
| `knowledge` | `Knowledge` | 知识库 |
| `team` | `List[Agent]` | 团队成员 |
| `workspace` | `Workspace` | 工作空间 |
| `response_model` | `Type[BaseModel]` | 结构化输出模型 |

**第二层：通用配置**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_history_to_messages` | `bool` | `False` | 将历史消息加入上下文 |
| `num_history_responses` | `int` | `3` | 历史轮数 |
| `search_knowledge` | `bool` | `True` | 允许 Agent 主动搜索知识库 |
| `markdown` | `bool` | `False` | Markdown 格式输出 |
| `structured_outputs` | `bool` | `False` | 严格结构化输出 |
| `debug` | `bool` | `False` | 调试模式 |

**第三层：打包配置**

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt_config` | `PromptConfig` | 提示词配置（expected_output 等） |
| `tool_config` | `ToolConfig` | 工具配置（并发、压缩等） |
| `memory_config` | `MemoryConfig` | 记忆配置（长期记忆开关等） |
| `team_config` | `TeamConfig` | 团队配置（委派策略等） |

## 运行方式

Agentica 采用 **async-first** 架构，所有核心方法原生 async：

```python
import asyncio
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))

# 1. 异步非流式（推荐）
async def example_async():
    result = await agent.run("你好")
    print(result.content)

# 2. 异步流式
async def example_stream():
    async for chunk in agent.run_stream("你好"):
        if chunk.content:
            print(chunk.content, end="")

# 3. 同步非流式
result = agent.run_sync("你好")
print(result.content)

# 4. 同步流式
for chunk in agent.run_stream_sync("你好"):
    if chunk.content:
        print(chunk.content, end="")
```

### 方法对照

| 方法 | 类型 | 用途 |
|------|------|------|
| `run()` | async | 非流式运行 |
| `run_stream()` | async generator | 流式运行 |
| `run_sync()` | sync | 同步适配器 |
| `run_stream_sync()` | sync iterator | 同步流式适配器 |
| `print_response()` | async | 格式化打印响应 |
| `print_response_stream()` | async | 格式化流式打印 |
| `print_response_sync()` | sync | 同步打印 |
| `print_response_stream_sync()` | sync | 同步流式打印 |

## Model（模型）

Model 是 Agent 的"大脑"，提供推理和生成能力。

### 支持的模型

```python
from agentica import (
    # OpenAI 系列
    OpenAIChat,         # gpt-4o, gpt-4o-mini
    AzureOpenAIChat,    # Azure 部署的 OpenAI 模型

    # 国内模型
    ZhipuAI,            # glm-4.7-flash（免费）, glm-4-plus
    DeepSeek,           # deepseek-chat, deepseek-reasoner
    Qwen,               # qwen-plus, qwen-turbo
    Moonshot,           # moonshot-v1-128k
    Doubao,             # doubao-pro-32k
    Yi,                 # yi-large

    # 海外模型
    Claude,             # claude-3.5-sonnet
    Grok,               # grok-beta
    Together,           # 多种开源模型

    # 本地模型
    Ollama,             # llama3, mistral, qwen2 等

    # 通用适配
    OpenAILike,         # 兼容 OpenAI API 的任意模型
)
```

### 通用参数

```python
model = OpenAIChat(
    id="gpt-4o",              # 模型 ID
    api_key="sk-xxx",         # API 密钥（或通过环境变量）
    base_url="https://...",   # API 地址
    temperature=0.7,          # 温度
    max_tokens=4096,          # 最大输出 token
    timeout=60,               # 超时秒数
)
```

## Memory（记忆）

Agentica 提供两层记忆系统：

### 运行时记忆：AgentMemory

管理当前会话的消息历史，支持 token 感知的截断。

```python
from agentica import Agent, AgentMemory

agent = Agent(
    memory=AgentMemory(),
    add_history_to_messages=True,   # 将历史加入上下文
    num_history_responses=5,        # 保留最近 5 轮
)
```

### 持久化记忆：Workspace

基于文件的持久化存储，使用 Markdown 文件管理上下文和记忆：

```
workspace/
├── AGENT.md      # Agent 上下文信息
├── PERSONA.md    # 用户画像
├── TOOLS.md      # 工具使用记录
├── USER.md       # 用户相关信息
├── MEMORY.md     # 长期记忆
└── users/
    └── {user_id}/  # 多用户隔离
```

```python
from agentica import Agent, Workspace

agent = Agent(
    workspace=Workspace(path="./my_workspace", user_id="alice"),
)
```

### 会话持久化

通过数据库保存会话历史：

```python
from agentica import Agent, AgentMemory, SqliteDb

db = SqliteDb(table_name="sessions", db_file="agent.db")
agent = Agent(
    memory=AgentMemory(db=db),
    session_id="user-123-session",
)
```

## Tools（工具）

工具赋予 Agent 与外部世界交互的能力。

### 函数工具（最简方式）

任何带类型注解和 docstring 的 Python 函数都可以作为工具：

```python
def get_weather(city: str) -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    return f"{city}：晴，25°C"

agent = Agent(tools=[get_weather])
```

### 类工具

继承 `Tool` 基类，适合封装一组相关功能：

```python
from agentica import Tool

class MathTool(Tool):
    def __init__(self):
        super().__init__(name="math")
        self.register(self.add)
        self.register(self.multiply)

    def add(self, a: float, b: float) -> float:
        """两数相加"""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """两数相乘"""
        return a * b

agent = Agent(tools=[MathTool()])
```

### 内置工具

Agentica 提供 40+ 开箱即用的工具，详见 [工具系统指南](../guides/tools.md)。

## Knowledge（知识库）

通过 RAG 让 Agent 基于你的文档回答问题：

```python
from agentica import Agent, Knowledge
from agentica.vectordb import LanceDb
from agentica.emb import OpenAIEmb

knowledge = Knowledge(
    data_path="./documents",
    vector_db=LanceDb(table_name="docs", uri="./lancedb"),
)
knowledge.load(recreate=True)

agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    instructions=["基于知识库回答问题，如无相关信息则明确告知"],
)
```

详见 [RAG 指南](../guides/rag.md)。

## Instructions（指令）

Instructions 定义 Agent 的行为和风格。支持静态和动态两种方式。

### 静态指令

```python
agent = Agent(
    instructions=[
        "你是一个 Python 代码审查专家",
        "检查代码的安全性、性能和可读性",
        "发现问题时提供修复建议",
        "使用中文回复",
    ],
)
```

### 动态指令

```python
def get_instructions(agent):
    base = ["你是一个智能助手"]
    if agent.session_state.get("expert_mode"):
        base.append("使用专业术语详细回答")
    else:
        base.append("用简单易懂的语言回答")
    return base

agent = Agent(instructions=get_instructions)
```

### System Prompt

也可以直接设置完整的系统提示词：

```python
agent = Agent(
    system_prompt="你是一个友好的中文助手，回答简洁准确。",
)
```

## RunResponse

Agent 运行后返回 `RunResponse` 对象：

```python
result = await agent.run("你好")

result.content           # 响应内容（str 或 BaseModel）
result.event             # 事件类型 (RunEvent)
result.model             # 使用的模型名称
result.metrics           # 指标（token 用量、耗时等）
result.tool_calls        # 工具调用信息 List[ToolCallInfo]
result.reasoning_content # 推理内容（思考过程）
```

### 流式事件

流式模式下，每个 chunk 携带不同的事件：

```python
from agentica import RunEvent

async for chunk in agent.run_stream("你好"):
    match chunk.event:
        case RunEvent.run_response:
            print(chunk.content, end="")  # 正常输出
        case RunEvent.tool_call_started:
            print(f"调用工具: {chunk.content}")
        case RunEvent.tool_call_completed:
            print(f"工具完成")
        case RunEvent.run_completed:
            print("完成")
```

## DeepAgent

`DeepAgent` 是预配置的 Agent，内置文件操作、命令执行、网页搜索、子任务委派等工具，适合复杂的多步骤任务：

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(
    model=OpenAIChat(id="gpt-4o"),
    name="My Assistant",
)
result = await agent.run("分析当前目录的代码结构")
```

内置工具包括：`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `web_search`, `fetch_url`, `task`（子任务委派）。

## 下一步

- [Team & Workflow](team.md) — 多智能体协作与工作流
- [工具系统](../guides/tools.md) — 深入了解工具
- [API 参考](../api/agent.md) — 完整 API
