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
| `name` | `str` | `None` | Agent 名称，用于日志和多智能体协作标识 |
| `agent_id` | `str` | 自动 UUID | Agent 唯一标识 |
| `description` | `str` | `None` | Agent 描述，在多智能体协作（Swarm/`as_tool()`）中帮助协调者理解成员能力 |
| `instructions` | `str \| List[str] \| Callable` | `None` | 行为指令，注入到 system prompt |
| `tools` | `List[Tool \| Callable]` | `None` | 工具列表 |
| `knowledge` | `Knowledge` | `None` | 知识库实例 (RAG) |
| `workspace` | `Workspace \| str` | `None` | 工作空间（传 str 会自动创建 Workspace） |
| `response_model` | `Type[BaseModel]` | `None` | Pydantic 模型，启用结构化输出 |

#### 第二层：通用配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_long_term_memory` | `bool` | `False` | 启用长期记忆工具、检索和相关 hooks |
| `enable_experience_capture` | `bool` | `False` | 启用 experience 捕获与自进化 hooks |
| `add_history_to_context` | `bool` | `False` | 将历史消息加入上下文 |
| `num_history_turns` | `int` | `3` | 保留的历史轮数 |
| `use_structured_outputs` | `bool` | `False` | 使用 OpenAI 严格结构化输出模式 |
| `debug` | `bool` | `False` | 调试模式（详细日志） |
| `enable_tracing` | `bool` | `False` | 启用 Langfuse 追踪 |

#### 第三层：打包配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt_config` | `PromptConfig` | `PromptConfig()` | 提示词构建配置 |
| `tool_config` | `ToolConfig` | `ToolConfig()` | 工具调用配置 |
| `long_term_memory_config` | `WorkspaceMemoryConfig` | `WorkspaceMemoryConfig()` | 工作空间记忆配置 |

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

#### `agent.working_memory.messages -> List[Message]`

获取当前会话的消息历史（替代已移除的 `get_chat_history()`）：

```python
# 获取全部消息
messages = agent.working_memory.messages

# 获取最近 N 轮的消息（dict 格式）
recent = agent.working_memory.get_messages_from_last_n_runs(n=5)

# 获取所有消息（dict 格式，含 role/content）
all_msgs = agent.working_memory.get_messages()
```

!!! note "`get_chat_history()` 已移除"
    原 `get_chat_history()` 方法已不存在。直接访问 `agent.working_memory.messages`（返回 `List[Message]`）或调用 `agent.working_memory.get_messages()`（返回 `List[Dict]`）。

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
| `compress_tool_results` | `bool` | `False` | 压缩工具结果（节省 token） |
| `compression_manager` | `CompressionManager` | `None` | 自定义压缩管理器实例 |
| `context_overflow_threshold` | `float` | `0.0` | context_window 使用率触发截断的阈值（0-1，0=禁用，推荐 0.8） |

## WorkspaceMemoryConfig

```python
from agentica.agent.config import WorkspaceMemoryConfig
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_workspace_context` | `bool` | `True` | 加载 workspace 上下文（AGENTS.md 等文件） |
| `load_workspace_memory` | `bool` | `True` | 加载 workspace 相关记忆到 System Prompt |
| `max_memory_entries` | `int` | `5` | 每次注入的最大记忆条数（按相关性排序） |
| `auto_archive` | `bool` | `False` | 每次 run() 后自动归档对话（零 LLM 成本） |
| `auto_extract_memory` | `bool` | `False` | 每次 run() 后自动提取记忆（有 LLM 成本，仅在 LLM 未主动调用 save_memory 时触发） |

---

## DeepAgent

```python
from agentica import DeepAgent
```

`DeepAgent` 是预配置的 product preset，继承自 `Agent`，内置文件操作、命令执行、网页搜索、Workspace memory、压缩和经验捕获等产品默认能力。它适合 CLI、Gateway 和无人值守任务；嵌入式 SDK 集成默认仍应使用 `Agent`。

### 构造参数

继承 Agent 所有参数，额外提供：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_file_tools` | `bool` | `True` | 启用文件操作工具 |
| `include_execute` | `bool` | `True` | 启用命令执行工具 |
| `include_web_search` | `bool` | `True` | 启用网页搜索工具 |
| `include_fetch_url` | `bool` | `True` | 启用网页抓取工具 |
| `include_todos` | `bool` | `True` | 启用 TODO 管理 |
| `include_task` | `bool` | `True` | 启用子任务委派 |
| `include_skills` | `bool` | `True` | 启用 Skill 查询工具 |
| `include_ask_user_question` | `bool` | `False` | 启用 Human-in-the-loop 工具 |

### 内置工具

| 工具 | 函数 | 说明 |
|------|------|------|
| 文件 | `ls`, `glob`, `grep`, `read_file`, `write_file`, `edit_file`, `multi_edit_file` | 文件系统操作 |
| 执行 | `execute` | 运行 Shell 命令 |
| 搜索 | `web_search`, `fetch_url` | 网页搜索和抓取 |
| 任务 | `task` | 委派子任务给子 Agent |
| TODO | `write_todos` | 任务列表管理（写入即读取，单工具设计） |
| 记忆 | `save_memory`, `search_memory` | 持久化记忆读写（需配合 Workspace） |
| 用户输入 | `ask_user_question` | Human-in-the-loop 请求用户确认或输入 |

!!! note "`read_todos` 已移除"
    历史版本有 `read_todos` 工具，当前版本只有 `write_todos`。`write_todos` 采用"写即读"设计：
    每次调用返回完整的当前 todo 列表状态，LLM 通过返回值获取最新状态，无需单独的读工具。

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

运行事件枚举（`str` + `Enum`，事件值为驼峰字符串）：

| 事件 | 值 | 说明 |
|------|-----|------|
| `run_started` | `"RunStarted"` | 运行开始 |
| `run_response` | `"RunResponse"` | 正常响应内容 token |
| `run_completed` | `"RunCompleted"` | 运行完成（最后一个 chunk） |
| `tool_call_started` | `"ToolCallStarted"` | 工具调用开始 |
| `tool_call_completed` | `"ToolCallCompleted"` | 工具调用完成 |
| `reasoning_started` | `"ReasoningStarted"` | 推理链开始（DeepSeek-R1 等） |
| `reasoning_step` | `"ReasoningStep"` | 推理链步骤 token |
| `reasoning_completed` | `"ReasoningCompleted"` | 推理链完成 |
| `updating_memory` | `"UpdatingMemory"` | 正在更新记忆 |
| `workflow_started` | `"WorkflowStarted"` | Workflow 开始 |
| `workflow_completed` | `"WorkflowCompleted"` | Workflow 完成 |

```python
from agentica import Agent, ZhipuAI, RunEvent

agent = Agent(model=ZhipuAI())
async for chunk in agent.run_stream("帮我写一个排序"):
    match chunk.event:
        case RunEvent.run_response:
            print(chunk.content, end="", flush=True)
        case RunEvent.tool_call_started:
            print(f"\n🔧 {chunk.content}")
        case RunEvent.reasoning_step:
            print(f"[think] {chunk.content}", end="")  # 推理模型思考过程
        case RunEvent.run_completed:
            print(f"\n✓ tokens: {chunk.metrics}")
```

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
| `OpenAIChat` + `base_url` | 通用适配 | — |

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
