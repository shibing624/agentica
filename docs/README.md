# OpenAI Agents Python Demo 技术文档

本文档整理了 `openai-agents-python` 项目中 `examples/` 目录的设计逻辑、核心 Demo 清单及其技术原理，供复现参考。

---

## 一、Examples 设计逻辑

`examples/` 目录按**功能模块**组织，从简单到复杂，覆盖 Agent SDK 的核心能力：

```
examples/
├── basic/              # 基础用法：Agent 创建、工具、流式输出
├── agent_patterns/     # 常见设计模式：路由、并行、Guardrails、LLM-as-Judge
├── handoffs/           # Agent 间交接（Handoff）
├── memory/             # 会话记忆（Session）
├── tools/              # 内置工具：Web Search、File Search、Code Interpreter
├── mcp/                # MCP 协议集成
├── model_providers/    # 自定义模型提供者（兼容第三方 LLM）
├── reasoning_content/  # 推理内容展示
├── customer_service/   # 完整示例：客服系统
├── research_bot/       # 完整示例：研究机器人
├── financial_research_agent/  # 完整示例：金融研究
├── realtime/           # 实时语音 --- 不要
└── voice/              # 语音交互 --- 不要
```

---

## 二、核心 Demo 清单与技术原理

### 2.1 基础模块 (`basic/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **Hello World** | `hello_world.py` | 最简 Agent 运行 | `Agent` + `Runner.run()` 异步执行 |
| **工具调用** | `tools.py` | Agent 调用自定义函数 | `@function_tool` 装饰器将 Python 函数转为 LLM 可调用工具 |
| **流式输出** | `stream_text.py` | 逐字流式返回 | `Runner.run_streamed()` + `stream_events()` 异步迭代器 |
| **动态指令** | `dynamic_system_prompt.py` | 运行时动态生成 system prompt | `instructions` 参数支持 `Callable`，接收 `RunContextWrapper` |
| **结构化输出** | `non_strict_output_type.py` | Agent 输出为 Pydantic 模型 | `output_type=BaseModel` 强制 LLM 输出 JSON Schema |
| **生命周期钩子** | `lifecycle_example.py` | 监控 Agent/Tool/LLM 各阶段 | `RunHooks` / `AgentHooks` 回调类 |
| **用量追踪** | `usage_tracking.py` | 统计 token 消耗 | `context.usage` 累计 `input_tokens` / `output_tokens` |

**关键代码示例 - Hello World:**
```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You only respond in haikus.")
result = await Runner.run(agent, "Tell me about recursion.")
print(result.final_output)
```

**关键代码示例 - 工具调用:**
```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"{city}: Sunny, 20°C"

agent = Agent(name="Weather Bot", tools=[get_weather])
result = await Runner.run(agent, "What's the weather in Tokyo?")
```

---

### 2.2 Agent 设计模式 (`agent_patterns/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **确定性流程** | `deterministic.py` | 多 Agent 串行流水线 | 手动编排 `Runner.run()` 调用顺序，前一个输出作为后一个输入 |
| **路由/Handoff** | `routing.py` | 根据条件交接给不同 Agent | `Agent.handoffs=[agent1, agent2]`，LLM 自动选择 |
| **Agents as Tools** | `agents_as_tools.py` | 将 Agent 封装为工具 | `agent.as_tool()` 将子 Agent 包装成 `@function_tool` |
| **Agents as Tools (流式)** | `agents_as_tools_streaming.py` | 子 Agent 流式回调 | `as_tool(on_stream=callback)` 监听嵌套 Agent 事件 |
| **并行执行** | `parallelization.py` | 多 Agent 并发运行 | `asyncio.gather()` 并行调用 `Runner.run()` |
| **LLM-as-Judge** | `llm_as_a_judge.py` | 评估-反馈循环 | 生成 Agent + 评估 Agent，循环直到评估通过 |
| **输入 Guardrail** | `input_guardrails.py` | 输入安全检查 | `@input_guardrail` 装饰器，`tripwire_triggered=True` 时抛异常 |
| **输出 Guardrail** | `output_guardrails.py` | 输出安全检查 | `@output_guardrail` 装饰器，检查 `final_output` |

**关键代码示例 - Agents as Tools:**
```python
spanish_agent = Agent(name="spanish_agent", instructions="Translate to Spanish")

orchestrator = Agent(
    name="orchestrator",
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate text to Spanish",
        )
    ],
)
```

**关键代码示例 - 输入 Guardrail:**
```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def math_guardrail(context, agent, input) -> GuardrailFunctionOutput:
    # 用另一个 Agent 检查是否是数学作业
    result = await Runner.run(guardrail_agent, input)
    return GuardrailFunctionOutput(
        tripwire_triggered=result.final_output.is_math_homework
    )

agent = Agent(name="Support", input_guardrails=[math_guardrail])
```

---

### 2.3 Handoff 交接 (`handoffs/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **消息过滤** | `message_filter.py` | 交接时过滤/修改历史消息 | `handoff(agent, input_filter=filter_func)` |
| **流式交接** | `message_filter_streaming.py` | 流式模式下的交接 | 同上，配合 `run_streamed()` |

**Handoff vs Agents as Tools 区别:**
- **Handoff**: 子 Agent 接管对话，接收完整历史
- **Agents as Tools**: 子 Agent 作为工具被调用，只接收生成的输入，主 Agent 保持控制

---

### 2.4 会话记忆 (`memory/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **SQLite Session** | `sqlite_session_example.py` | 本地持久化会话 | `SQLiteSession(session_id)` 自动保存/加载历史 |
| **Redis Session** | `redis_session_example.py` | Redis 分布式会话 | `RedisSession` 适合多实例部署 |
| **OpenAI Session** | `openai_session_example.py` | OpenAI 托管会话 | 使用 OpenAI 服务端存储 |
| **加密 Session** | `encrypted_session_example.py` | 加密存储 | 敏感数据加密 |

**关键代码示例:**
```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("conversation_123")

# 第一轮
result = await Runner.run(agent, "My name is Alice", session=session)

# 第二轮 - 自动记住上下文
result = await Runner.run(agent, "What's my name?", session=session)
# 输出: Your name is Alice
```

---

### 2.5 内置工具 (`tools/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **Web Search** | `web_search.py` | 网络搜索 | `WebSearchTool` 调用 OpenAI 托管搜索 |
| **File Search** | `file_search.py` | 向量检索 | `FileSearchTool` + Vector Store |
| **Code Interpreter** | `code_interpreter.py` | 执行 Python 代码 | `CodeInterpreterTool` 沙箱执行 |
| **Shell** | `shell.py` / `local_shell.py` | 执行 Shell 命令 | 本地命令执行（需谨慎） |
| **Image Generator** | `image_generator.py` | 生成图片 | DALL-E 集成 |

**关键代码示例 - Web Search:**
```python
from agents import Agent, Runner, WebSearchTool

agent = Agent(
    name="Searcher",
    tools=[WebSearchTool(user_location={"type": "approximate", "city": "Beijing"})]
)
result = await Runner.run(agent, "Search for latest AI news")
```

---

### 2.6 MCP 协议 (`mcp/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **Filesystem** | `filesystem_example/` | 文件系统访问 | `MCPServerStdio` 启动 MCP 服务 |
| **Git** | `git_example/` | Git 操作 | MCP Git Server |
| **SSE** | `sse_example/` | Server-Sent Events | MCP SSE 传输 |
| **Streamable HTTP** | `streamablehttp_example/` | HTTP 流式 | MCP HTTP 传输 |

**关键代码示例:**
```python
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="Filesystem Server",
    params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]},
) as server:
    agent = Agent(name="Assistant", mcp_servers=[server])
    result = await Runner.run(agent, "List files")
```

---

### 2.7 自定义模型 (`model_providers/`)

| Demo | 文件 | 功能 | 技术原理 |
|------|------|------|----------|
| **Agent 级别** | `custom_example_agent.py` | 单个 Agent 使用自定义模型 | `Agent(model=custom_model)` |
| **全局级别** | `custom_example_global.py` | 全局默认模型 | `set_default_openai_client()` |
| **Provider 级别** | `custom_example_provider.py` | 自定义 Provider | 实现 `ModelProvider` 接口 |
| **LiteLLM** | `litellm_provider.py` | 多模型统一接口 | LiteLLM 适配各种 LLM |

TODO: LiteLLM 需要我们支持。
---

### 2.8 完整应用示例

#### 2.8.1 客服系统 (`customer_service/`)

**架构:**
```
用户 → Triage Agent → FAQ Agent (FAQ 查询)
                   → Seat Booking Agent (座位预订)
```

**技术要点:**
- 多 Agent Handoff 协作
- `@function_tool` 模拟业务逻辑
- `RunContextWrapper` 共享上下文状态
- `on_handoff` 钩子在交接时执行逻辑

#### 2.8.2 研究机器人 (`research_bot/`)

**架构:**
```
用户输入 → Planner Agent (规划搜索) 
        → Search Agent × N (并行搜索)
        → Writer Agent (生成报告)
```

**技术要点:**
- `asyncio.gather()` 并行搜索
- `WebSearchTool` 网络检索
- 结构化输出 (`WebSearchPlan`, `ReportData`)
- `trace()` 追踪整个工作流 --- 我用langfuse实现的

---

## 三、复现 Demo 清单（按优先级）

### 必须实现（核心能力）

| 优先级 | Demo | 验证能力 |
|--------|------|----------|
| P0 | `basic/hello_world.py` | Agent 基础运行 |
| P0 | `basic/tools.py` | 工具调用 |
| P0 | `basic/stream_text.py` | 流式输出 |
| P0 | `agent_patterns/routing.py` | Handoff 路由 |
| P0 | `agent_patterns/agents_as_tools.py` | Agent 组合 |

### 建议实现（进阶能力）

| 优先级 | Demo | 验证能力 |
|--------|------|----------|
| P1 | `agent_patterns/parallelization.py` | 并行执行 |
| P1 | `agent_patterns/input_guardrails.py` | 安全检查 |
| P1 | `agent_patterns/llm_as_a_judge.py` | 评估循环 |
| P1 | `memory/sqlite_session_example.py` | 会话持久化 |
| P1 | `basic/lifecycle_example.py` | 生命周期钩子 |

### 可选实现（完整示例）

| 优先级 | Demo | 验证能力 |
|--------|------|----------|
| P2 | `customer_service/main.py` | 多 Agent 协作 |
| P2 | `research_bot/` | 端到端工作流 |
| P2 | `tools/web_search.py` | 内置工具 |
| P2 | `mcp/filesystem_example/` | MCP 协议 |

---

### 4.3 工具定义方式

```python
# 方式 1: 装饰器
@function_tool
def my_tool(param: str) -> str:
    """Tool description"""
    return result

# 方式 2: Agent 转工具
sub_agent.as_tool(tool_name="name", tool_description="desc")

# 方式 3: 内置工具
WebSearchTool(), FileSearchTool(), CodeInterpreterTool()
```

---
为了让agentica的examples更可读、方便使用。参考examples/README.md 文件，和
参考上面的openai-agents的demo清单，我需要你帮我梳理下agentica的demo需要新增删除修改的demo。