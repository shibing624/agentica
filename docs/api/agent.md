# Agent API 参考

> 基于源码的完整 API 文档。后续将通过 docstring 自动生成。

## Agent

```python
from agentica import Agent
```

核心 Agent 类，将模型、工具和记忆连接在一起。使用 `@dataclass(init=False)` + mixin 多继承架构。

### 构造参数

#### 第一层：核心定义

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `Model` | `None` | LLM 模型实例。未指定时使用默认 OpenAI 模型 |
| `name` | `str` | `None` | Agent 名称，用于日志和团队标识 |
| `agent_id` | `str` | 自动 UUID | Agent 唯一标识 |
| `description` | `str` | `None` | Agent 描述，在 Team 模式下帮助协调者理解成员能力 |
| `instructions` | `str \| List[str] \| Callable` | `None` | 行为指令，注入到 system prompt |
| `tools` | `List[Tool \| Callable]` | `None` | 工具列表 |
| `knowledge` | `Knowledge` | `None` | 知识库实例 (RAG) |
| `team` | `List[Agent]` | `None` | 团队成员列表 |
| `workspace` | `Workspace \| str` | `None` | 工作空间（传 str 会自动创建 Workspace） |
| `response_model` | `Type[BaseModel]` | `None` | Pydantic 模型，启用结构化输出 |

#### 第二层：通用配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_history_to_messages` | `bool` | `False` | 将历史消息加入上下文 |
| `history_window` | `int` | `3` | 保留的历史轮数 |
| `structured_outputs` | `bool` | `False` | 使用 OpenAI 严格结构化输出模式 |
| `debug` | `bool` | `False` | 调试模式（详细日志） |
| `tracing` | `bool` | `False` | 启用 Langfuse 追踪 |

#### 第三层：打包配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt_config` | `PromptConfig` | `PromptConfig()` | 提示词构建配置 |
| `tool_config` | `ToolConfig` | `ToolConfig()` | 工具调用配置 |
| `long_term_memory_config` | `WorkspaceMemoryConfig` | `WorkspaceMemoryConfig()` | 工作空间记忆配置 |
| `team_config` | `TeamConfig` | `TeamConfig()` | 团队配置 |

#### 运行时

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `working_memory` | `WorkingMemory` | `WorkingMemory()` | 工作记忆（会话消息历史） |
| `context` | `Dict[str, Any]` | `None` | 运行时上下文，传递给工具和动态指令 |

### 方法

#### `async run(message, **kwargs) -> RunResponse`

异步非流式运行。

```python
result = await agent.run("你好")
print(result.content)
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `message` | `str \| List \| Dict \| Message` | 输入消息 |
| `stream` | `bool` | 是否流式（内部使用） |
| `stream_intermediate_steps` | `bool` | 流式中间步骤 |
| `session_id` | `str` | 会话 ID |
| `user_id` | `str` | 用户 ID |
| `save_response_to_file` | `str` | 保存响应到文件 |
| `**kwargs` | | 传递给模型的额外参数 |

**返回：** `RunResponse`

---

#### `async run_stream(message, **kwargs) -> AsyncIterator[RunResponse]`

异步流式运行。返回异步生成器。

```python
async for chunk in agent.run_stream("你好"):
    if chunk.content:
        print(chunk.content, end="")
```

参数与 `run()` 相同。

---

#### `run_sync(message, **kwargs) -> RunResponse`

同步非流式运行。内部通过 `asyncio` 适配器调用 `run()`。

```python
result = agent.run_sync("你好")
print(result.content)
```

---

#### `run_stream_sync(message, **kwargs) -> Iterator[RunResponse]`

同步流式运行。通过后台线程驱动异步迭代器。

```python
for chunk in agent.run_stream_sync("你好"):
    if chunk.content:
        print(chunk.content, end="")
```

---

#### `async print_response(message, **kwargs)`

异步非流式打印格式化响应。

```python
await agent.print_response("解释量子计算")
```

---

#### `async print_response_stream(message, **kwargs)`

异步流式打印格式化响应。

```python
await agent.print_response_stream("解释量子计算")
```

---

#### `print_response_sync(message, **kwargs)`

同步非流式打印。

```python
agent.print_response_sync("解释量子计算")
```

---

#### `print_response_stream_sync(message, **kwargs)`

同步流式打印。

```python
agent.print_response_stream_sync("解释量子计算")
```

---

#### `as_tool(tool_name=None, tool_description=None) -> Function`

将 Agent 转换为工具，供其他 Agent 使用。

```python
tool = agent.as_tool(
    tool_name="research",
    tool_description="深度研究指定主题",
)
```

---

#### `cancel()`

取消当前运行。通过协作式取消通知 Agent 在下一个检查点停止。

```python
agent.cancel()
```

---

#### `get_chat_history() -> List[Message]`

获取聊天历史。

---

#### `search_knowledge_base(query, num_documents=None) -> str`

搜索知识库。

---

## PromptConfig

```python
from agentica.agent.config import PromptConfig
```

提示词构建的高级配置。大多数场景只需 `Agent.instructions`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `system_prompt` | `str \| Callable` | `None` | 自定义 system prompt（覆盖默认构建） |
| `task` | `str` | `None` | 任务描述 |
| `role` | `str` | `None` | 角色描述 |
| `guidelines` | `List[str]` | `None` | 行为准则 |
| `expected_output` | `str` | `None` | 期望输出格式 |
| `additional_context` | `str` | `None` | 额外上下文 |
| `add_datetime_to_instructions` | `bool` | `True` | 注入当前时间 |
| `prevent_hallucinations` | `bool` | `False` | 添加防幻觉指令 |
| `prevent_prompt_leakage` | `bool` | `False` | 防止 prompt 泄露 |
| `enable_agentic_prompt` | `bool` | `False` | 启用增强 prompt（适合 DeepAgent） |
| `output_language` | `str` | `None` | 指定输出语言 |
| `markdown` | `bool` | `False` | Markdown 格式输出 |

## ToolConfig

```python
from agentica.agent.config import ToolConfig
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `support_tool_calls` | `bool` | `True` | 是否支持工具调用 |
| `tool_call_limit` | `int` | `None` | 工具调用次数限制 |
| `tool_choice` | `str \| Dict` | `None` | 工具选择策略 |
| `search_knowledge` | `bool` | `True` | 允许 Agent 主动搜索知识库 |
| `add_references` | `bool` | `False` | 添加知识库引用 |
| `compress_tool_results` | `bool` | `False` | 压缩工具结果 |
| `compression_manager` | `CompressionManager` | `None` | 压缩管理器实例 |

## WorkspaceMemoryConfig

```python
from agentica.agent.config import WorkspaceMemoryConfig
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_workspace_context` | `bool` | `True` | 加载 workspace 上下文 |
| `load_workspace_memory` | `bool` | `True` | 加载 workspace 记忆 |
| `memory_days` | `int` | `2` | 记忆回溯天数 |

## TeamConfig

```python
from agentica.agent.config import TeamConfig
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `respond_directly` | `bool` | `False` | 团队成员直接回复（不经协调者） |
| `add_transfer_instructions` | `bool` | `True` | 自动添加委派指令 |
| `team_response_separator` | `str` | `"\n"` | 团队响应分隔符 |

---

## DeepAgent

```python
from agentica import DeepAgent
```

预配置的 Agent，继承自 `Agent`，内置文件操作、命令执行、网页搜索等工具。

### 构造参数

继承 Agent 所有参数，额外提供：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_file_tools` | `bool` | `True` | 启用文件操作工具 |
| `enable_execute_tools` | `bool` | `True` | 启用命令执行工具 |
| `enable_web_tools` | `bool` | `True` | 启用网页搜索工具 |
| `enable_todo_tools` | `bool` | `True` | 启用 TODO 管理 |
| `enable_task_tools` | `bool` | `True` | 启用子任务委派 |

### 内置工具

| 工具 | 函数 | 说明 |
|------|------|------|
| 文件 | `ls`, `glob`, `grep`, `read_file`, `write_file`, `edit_file` | 文件系统操作 |
| 执行 | `execute` | 运行 Shell 命令 |
| 搜索 | `web_search`, `fetch_url` | 网页搜索和抓取 |
| 任务 | `task` | 委派子任务给子 Agent |
| TODO | `read_todos`, `write_todos` | 任务列表管理 |

---

## RunResponse

```python
from agentica import RunResponse
```

Agent 运行的返回值。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `content` | `Any` | 响应内容（str 或 BaseModel） |
| `event` | `RunEvent` | 事件类型 |
| `model` | `str` | 模型名称 |
| `run_id` | `str` | 运行 ID |
| `session_id` | `str` | 会话 ID |
| `messages` | `List[Message]` | 消息列表 |
| `metrics` | `Dict` | 指标（token 用量、耗时） |
| `tool_calls` | `List[ToolCallInfo]` | 工具调用详情 |
| `tool_call_times` | `Dict[str, float]` | 各工具耗时 |
| `reasoning_content` | `str` | 推理内容（思考过程） |
| `images` | `List[Image]` | 生成的图片 |

### 方法

| 方法 | 返回 | 说明 |
|------|------|------|
| `to_json()` | `str` | 转为 JSON 字符串 |
| `to_dict()` | `Dict` | 转为字典 |

---

## RunEvent

```python
from agentica import RunEvent
```

运行事件枚举：

| 事件 | 说明 |
|------|------|
| `run_started` | 运行开始 |
| `run_response` | 正常响应内容 |
| `run_completed` | 运行完成 |
| `tool_call_started` | 工具调用开始 |
| `tool_call_completed` | 工具调用完成 |
| `multi_round_turn` | 多轮对话轮次 |
| `multi_round_completed` | 多轮对话完成 |

---

## ToolCallInfo

```python
from agentica import ToolCallInfo
```

工具调用信息的扁平化访问：

| 属性 | 类型 | 说明 |
|------|------|------|
| `tool_name` | `str` | 工具名称 |
| `tool_args` | `Dict` | 调用参数 |
| `tool_result` | `str` | 返回结果 |
| `call_id` | `str` | 调用 ID |

---

## Workflow

```python
from agentica import Workflow
```

确定性工作流引擎。子类实现 `run()` 方法编排步骤。

### 方法

| 方法 | 类型 | 说明 |
|------|------|------|
| `run()` | async | 子类实现的运行逻辑 |
| `run_sync()` | sync | 同步适配器 |

---

## Model

```python
from agentica import Model
```

所有模型的抽象基类，定义四个 async 抽象方法：

| 方法 | 说明 |
|------|------|
| `invoke()` | 单次调用（不处理工具） |
| `invoke_stream()` | 流式调用 |
| `response()` | 完整响应（含工具循环） |
| `response_stream()` | 流式响应（含工具循环） |

### 通用参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 模型 ID |
| `api_key` | `str` | API 密钥 |
| `base_url` | `str` | API 地址 |
| `temperature` | `float` | 温度 |
| `max_tokens` | `int` | 最大输出 token |
| `timeout` | `int` | 超时秒数 |

### 内置模型

| 类 | 提供商 | 默认模型 |
|----|--------|---------|
| `OpenAIChat` | OpenAI | gpt-4o-mini |
| `AzureOpenAIChat` | Azure OpenAI | — |
| `DeepSeek` | DeepSeek | deepseek-chat |
| `Claude` | Anthropic | claude-3.5-sonnet |
| `ZhipuAI` | 智谱 AI | glm-4.7-flash |
| `Qwen` | 通义千问 | qwen-plus |
| `Moonshot` | 月之暗面 | moonshot-v1-128k |
| `Doubao` | 豆包 | doubao-pro-32k |
| `Yi` | 零一万物 | yi-large |
| `Grok` | xAI | grok-beta |
| `Ollama` | 本地 | llama3 |
| `Together` | Together AI | — |
| `OpenAILike` | 通用适配 | — |

---

## WorkingMemory

```python
from agentica.memory import WorkingMemory
```

运行时工作记忆，维护会话消息历史。

### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `db` | `BaseDb` | `None` | 持久化数据库 |

---

## Tool / Function / FunctionCall

```python
from agentica import Tool, Function, FunctionCall
```

三层工具架构：

| 类 | 说明 |
|----|------|
| `Tool` | 工具容器，注册多个 Function |
| `Function` | 函数定义（schema + 入口点） |
| `FunctionCall` | 单次函数调用实例 |

### Function.from_callable

```python
func = Function.from_callable(my_function, strict=True)
```

从 Python 函数自动生成工具定义。

---

## 数据库

```python
from agentica import SqliteDb, PostgresDb, InMemoryDb, JsonDb
```

| 类 | 说明 |
|----|------|
| `SqliteDb` | SQLite 持久化 |
| `PostgresDb` | PostgreSQL |
| `InMemoryDb` | 内存（测试用） |
| `JsonDb` | JSON 文件 |

```python
db = SqliteDb(table_name="sessions", db_file="agent.db")
```

---

## 实用工具

### Token 计数

```python
from agentica import count_tokens, count_text_tokens

tokens = count_text_tokens("Hello, world!", model_id="gpt-4o")
tokens = count_tokens(messages, tools=tools, model_id="gpt-4o")
```

### 日志

```python
from agentica import set_log_level_to_debug, set_log_level_to_info, logger
```

### 格式化打印

```python
from agentica import pprint_run_response

pprint_run_response(response, markdown=True)
```
