# Agentica API 参考文档

> 完整的 API 参考，包含所有公开类和方法

## 目录

- [Agent](#agent)
- [Model](#model)
- [Memory](#memory)
- [Tools](#tools)
- [Knowledge](#knowledge)
- [VectorDB](#vectordb)
- [Database](#database)
- [Workflow](#workflow)
- [RunResponse](#runresponse)

---

## Agent

### `class Agent`

核心 Agent 类，实现 AI 代理的完整功能。

```python
from agentica import Agent
```

#### 构造参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model` | `Model` | `None` | LLM 模型实例 |
| `name` | `str` | `None` | Agent 名称 |
| `agent_id` | `str` | `None` | Agent 唯一标识（自动生成 UUID） |
| `user_id` | `str` | `None` | 用户 ID |
| `session_id` | `str` | `None` | 会话 ID |
| `description` | `str` | `None` | Agent 描述 |
| `instructions` | `str \| List[str] \| Callable` | `None` | 指令/提示词 |
| `system_prompt` | `str \| Callable` | `None` | 系统提示词 |
| `tools` | `List[Tool \| Callable]` | `None` | 工具列表 |
| `knowledge` | `Knowledge` | `None` | 知识库 |
| `memory` | `AgentMemory` | `AgentMemory()` | 内存管理器 |
| `db` | `BaseDb` | `None` | 持久化数据库 |
| `team` | `List[Agent]` | `None` | Agent 团队 |
| `role` | `str` | `None` | 团队中的角色 |
| `enable_multi_round` | `bool` | `False` | 启用多轮对话策略 |
| `max_rounds` | `int` | `100` | 最大对话轮数 |
| `max_tokens` | `int` | `128000` | 最大 token 数 |
| `response_model` | `Type[BaseModel]` | `None` | 结构化输出模型 |
| `structured_outputs` | `bool` | `False` | 使用结构化输出 |
| `add_history_to_messages` | `bool` | `False` | 添加历史到消息 |
| `num_history_responses` | `int` | `3` | 历史响应数量 |
| `add_references` | `bool` | `False` | 添加知识库引用 |
| `markdown` | `bool` | `False` | Markdown 格式输出 |
| `debug_mode` | `bool` | `False` | 调试模式 |

#### 方法

##### `run(message, stream=False, **kwargs) -> RunResponse | Iterator[RunResponse]`

运行 Agent 处理消息。

```python
# 非流式
response = agent.run("你好")
print(response.content)

# 流式
for chunk in agent.run("你好", stream=True):
    print(chunk.content, end="")
```

**参数:**
- `message`: `str | List | Dict | Message` - 输入消息
- `stream`: `bool` - 是否流式输出
- `**kwargs`: 其他参数传递给模型

**返回:** `RunResponse` 或 `Iterator[RunResponse]`

##### `arun(message, stream=False, **kwargs) -> RunResponse | AsyncIterator[RunResponse]`

异步运行 Agent。

```python
response = await agent.arun("你好")
```

##### `print_response(message, stream=True, **kwargs)`

打印格式化的响应。

```python
agent.print_response("解释量子计算", stream=True)
```

##### `cli_app(user="User", stream=True)`

启动命令行交互界面。

```python
agent.cli_app()
```

##### `deep_copy(update=None) -> Agent`

深拷贝 Agent 实例。

```python
new_agent = agent.deep_copy(update={"name": "NewAgent"})
```

##### `get_chat_history() -> List[Message]`

获取聊天历史。

```python
history = agent.get_chat_history()
```

##### `search_knowledge_base(query, num_documents=None) -> str`

搜索知识库（Agentic RAG）。

```python
results = agent.search_knowledge_base("什么是机器学习？")
```

##### `as_tool(tool_name=None, tool_description=None) -> Function`

将 Agent 转换为工具供其他 Agent 使用。

```python
researcher_tool = researcher_agent.as_tool(
    tool_name="research",
    tool_description="进行深度研究"
)
```

---

## Model

### 基类 `Model`

所有模型的基类。

```python
from agentica import Model
```

#### 通用参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `model` | `str` | 模型 ID |
| `api_key` | `str` | API 密钥 |
| `base_url` | `str` | API 基础 URL |
| `temperature` | `float` | 温度参数 |
| `max_tokens` | `int` | 最大输出 token |
| `timeout` | `int` | 请求超时（秒） |

### OpenAI 模型

```python
from agentica import OpenAIChat

model = OpenAIChat(
    model="gpt-4o",
    api_key="sk-xxx",  # 或设置 OPENAI_API_KEY 环境变量
    temperature=0.7,
)
```

### Azure OpenAI

```python
from agentica import AzureOpenAIChat

model = AzureOpenAIChat(
    model="gpt-4o",
    azure_endpoint="https://xxx.openai.azure.com/",
    api_key="xxx",
    api_version="2024-02-15-preview",
)
```

### DeepSeek

```python
from agentica import DeepSeek

model = DeepSeek(model="deepseek-chat")
```

### Claude

```python
from agentica import Claude

model = Claude(model="claude-3-5-sonnet-20241022")
```

### 其他模型

```python
from agentica import (
    Moonshot,      # 月之暗面
    Qwen,          # 通义千问
    ZhipuAI,       # 智谱 AI
    Doubao,        # 豆包
    Yi,            # 零一万物
    Gemini,        # Google Gemini
    Grok,          # xAI Grok
    Ollama,        # 本地模型
    Together,      # Together AI
    Groq,          # Groq
)
```

---

## Memory

### `class AgentMemory`

Agent 内存管理。

```python
from agentica import AgentMemory
```

#### 构造参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `db` | `BaseDb` | `None` | 持久化数据库 |
| `create_user_memories` | `bool` | `False` | 创建用户长期记忆 |
| `create_session_summary` | `bool` | `False` | 创建会话摘要 |

#### 工厂方法

##### `AgentMemory.with_db(db, user_id=None, create_user_memories=False)`

使用数据库创建内存实例。

```python
from agentica import AgentMemory, SqliteDb

memory = AgentMemory.with_db(
    db=SqliteDb(table_name="agent_sessions", db_file="agent.db"),
    user_id="user-1",
    create_user_memories=True,
)
```

### `class Memory`

单条记忆数据模型。

```python
from agentica import Memory

memory = Memory(
    memory="用户喜欢 Python 编程",
    id="mem-1",
)
```

### `class MemoryManager`

记忆管理器，用于 CRUD 操作。

```python
from agentica import MemoryManager

manager = MemoryManager(db=db, user_id="user-1")
manager.add_memory("用户是高级工程师")
manager.delete_memory("mem-1")
```

---

## Tools

### `class Tool`

工具基类。

```python
from agentica import Tool
```

#### 创建自定义工具

**方式 1: 使用函数**

```python
def get_weather(city: str) -> str:
    """获取城市天气
    
    Args:
        city: 城市名称
    """
    return f"{city}天气晴朗，25°C"

agent = Agent(tools=[get_weather])
```

**方式 2: 继承 Tool 类**

```python
from agentica import Tool

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather")
        self.register(self.get_weather)
    
    def get_weather(self, city: str) -> str:
        """获取城市天气"""
        return f"{city}天气晴朗"
```

### `class Function`

函数定义模型。

```python
from agentica import Function

func = Function.from_callable(my_function, strict=True)
```

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 函数名 |
| `description` | `str` | 函数描述 |
| `parameters` | `Dict` | JSON Schema 参数 |
| `entrypoint` | `Callable` | 实际函数 |
| `strict` | `bool` | 严格模式 |
| `show_result` | `bool` | 显示结果给用户 |
| `stop_after_tool_call` | `bool` | 调用后停止 |

### 内置工具

```python
from agentica.tools import (
    # 搜索
    DuckDuckGoTool,
    BaiduSearchTool,
    SerperTool,
    ExaTool,
    
    # 代码执行
    RunPythonCodeTool,
    ShellTool,
    
    # 文件操作
    FileTool,
    EditTool,
    
    # 网页
    UrlCrawlerTool,
    JinaTool,
    BrowserTool,
    
    # 知识
    ArxivTool,
    WikipediaTool,
    
    # 多媒体
    DalleTool,
    CogViewTool,
    CogVideoTool,
    
    # 其他
    CalculatorTool,
    WeatherTool,
    YFinanceTool,
)
```

---

## Knowledge

### `class Knowledge`

知识库管理。

```python
from agentica import Knowledge
```

#### 构造参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `data_path` | `str \| List[str]` | `None` | 数据文件/目录路径 |
| `vector_db` | `VectorDb` | `None` | 向量数据库 |
| `embedder` | `Embedder` | `None` | 嵌入模型 |
| `num_documents` | `int` | `3` | 检索文档数 |
| `chunk_size` | `int` | `2000` | 分块大小 |
| `chunk` | `bool` | `True` | 是否分块 |

#### 方法

##### `load(recreate=False, upsert=False)`

加载知识库。

```python
knowledge = Knowledge(
    data_path="./docs",
    vector_db=LanceDb(table_name="docs", uri="./lancedb"),
)
knowledge.load(recreate=True)
```

##### `search(query, num_documents=None) -> List[Document]`

搜索知识库。

```python
docs = knowledge.search("什么是 RAG？", num_documents=5)
```

---

## VectorDB

### 基类 `VectorDb`

向量数据库抽象基类。

```python
from agentica import VectorDb
```

### LanceDB

```python
from agentica.vectordb import LanceDb

db = LanceDb(
    table_name="documents",
    uri="./lancedb",
    embedder=OpenAIEmbedder(),
)
```

### ChromaDB

```python
from agentica.vectordb import ChromaDb

db = ChromaDb(
    collection="documents",
    path="./chromadb",
)
```

### 其他向量数据库

```python
from agentica.vectordb import (
    InMemoryVectorDb,  # 内存
    PgVectorDb,        # PostgreSQL
    PineconeDb,        # Pinecone
    QdrantDb,          # Qdrant
)
```

#### 通用方法

| 方法 | 描述 |
|------|------|
| `create()` | 创建数据库/表 |
| `insert(documents)` | 插入文档 |
| `upsert(documents)` | 更新或插入文档 |
| `search(query, limit=5)` | 搜索文档 |
| `delete()` | 删除数据库 |
| `exists()` | 检查是否存在 |

---

## Database

### 基类 `BaseDb`

数据库抽象基类，用于会话和记忆持久化。

```python
from agentica import BaseDb
```

### SQLite

```python
from agentica import SqliteDb

db = SqliteDb(
    table_name="agent_sessions",
    db_file="agent.db",
)
```

### PostgreSQL

```python
from agentica import PostgresDb

db = PostgresDb(
    table_name="agent_sessions",
    db_url="postgresql://user:pass@localhost/db",
)
```

### 其他数据库

```python
from agentica import InMemoryDb, JsonDb
```

---

## Workflow

### `class Workflow`

工作流引擎。

```python
from agentica import Workflow
```

#### 创建工作流

```python
from agentica import Workflow, Agent, RunResponse
from pydantic import Field

class MyWorkflow(Workflow):
    researcher: Agent = Field(...)
    writer: Agent = Field(...)
    
    def run(self, topic: str) -> RunResponse:
        # 研究阶段
        research = self.researcher.run(f"研究: {topic}")
        
        # 写作阶段
        article = self.writer.run(f"基于以下研究写文章:\n{research.content}")
        
        return RunResponse(content=article.content)

# 使用
workflow = MyWorkflow(
    researcher=Agent(name="Researcher", ...),
    writer=Agent(name="Writer", ...),
)
result = workflow.run("人工智能的未来")
```

---

## RunResponse

### `class RunResponse`

Agent 运行响应。

```python
from agentica import RunResponse
```

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `content` | `Any` | 响应内容 |
| `content_type` | `str` | 内容类型 |
| `event` | `str` | 事件类型 |
| `messages` | `List[Message]` | 消息列表 |
| `metrics` | `Dict` | 指标数据 |
| `model` | `str` | 模型名称 |
| `run_id` | `str` | 运行 ID |
| `session_id` | `str` | 会话 ID |
| `tools` | `List[Dict]` | 工具调用 |
| `images` | `List[Image]` | 图片 |
| `reasoning_content` | `str` | 推理内容 |

#### 方法

##### `to_json() -> str`

转换为 JSON 字符串。

##### `to_dict() -> Dict`

转换为字典。

### `class RunEvent`

运行事件枚举。

```python
from agentica import RunEvent

RunEvent.run_started
RunEvent.run_response
RunEvent.run_completed
RunEvent.tool_call_started
RunEvent.tool_call_completed
RunEvent.multi_round_turn
RunEvent.multi_round_completed
```

### `pprint_run_response(response, markdown=True)`

格式化打印响应。

```python
from agentica import pprint_run_response

pprint_run_response(response, markdown=True)
```

---

## MCP (Model Context Protocol)

### `class MCPConfig`

MCP 服务器配置。

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

---

## 实用工具

### Token 计数

```python
from agentica import count_tokens, count_text_tokens

# 计算文本 token
tokens = count_text_tokens("Hello, world!", model_id="gpt-4o")

# 计算消息 token
tokens = count_tokens(messages, tools=tools, model_id="gpt-4o")
```

### 压缩管理

```python
from agentica import CompressionManager

manager = CompressionManager(
    compress_tool_results=True,
    compress_token_limit=50000,
)

agent = Agent(
    compression_manager=manager,
    compress_tool_results=True,
)
```

---

*文档最后更新: 2025-12-20*
