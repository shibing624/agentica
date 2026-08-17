# Architecture Overview

Agentica 采用分层架构，将 Agent 的**身份定义**与**执行引擎**解耦，同时内置了上下文压缩、会话持久化和并发安全工具执行等生产级能力。

## 五层架构

```
+------------------------------------------------------------------+
|                        Application Layer                          |
|  CLI (agentica)  /  FastAPI  /  ACP Server  /  Web UI            |
+------------------------------------------------------------------+
|                        Orchestration Layer                         |
|  as_tool (composition) | Workflow (pipeline) | Swarm (mesh)       |
|  BuiltinTaskTool / SubagentRegistry (subagent runtime)            |
+------------------------------------------------------------------+
|                          Agent Layer                               |
|  Agent  <--->  Runner (execution engine)                          |
|  PromptsMixin | ToolsMixin | AsToolMixin | PrinterMixin           |
+------------------------------------------------------------------+
|                          Model Layer                               |
|  OpenAI | Anthropic | ZhipuAI | DeepSeek | Ollama | LiteLLM    |
|  Tool loop | Compression | Death spiral | Cost budget            |
+------------------------------------------------------------------+
|                        Infrastructure Layer                        |
|  Tools | Knowledge(RAG) | Memory | VectorDB | MCP | Hooks        |
+------------------------------------------------------------------+
```

## Agent 与 Runner 的分离

**Agent** 定义"我是谁、我能做什么"（静态配置）；  
**Runner** 负责"怎么执行"（动态执行引擎）。

两者解耦的好处：Runner 可独立测试，Agent 配置可以序列化，不同的 Runner 策略可以插拔。

```python
# agentica/agent/base.py（简化）
@dataclass(init=False)
class Agent(PromptsMixin, AsToolMixin, ToolsMixin, PrinterMixin, GoalMixin):
    def __init__(self, ...):
        self._runner = Runner(self)   # Runner 持有 Agent 弱引用

    async def run(self, message, **kw) -> RunResponse:
        return await self._runner.run(message, **kw)
```

`GoalMixin`（`agent/goal_mixin.py`）持有 standing-goal 闭环（`get_goal_manager` /
`enable_goal_tool` / `run_goal` / `run_goal_step`），使 `base.py` 只留公开 run API
面与薄委托，不再内嵌 goal 闭环。

### 模块布局（诚实拆包）

大文件拆成包后，**不保留“旧路径假装一切都还在 `__init__`”的兼容层**。SDK 对外入口仍是 `agentica` / `Agent` / `Runner`；CLI 内部与测试应直接引用真实子模块。

| 包 | 包根导出 | 主要子模块 |
|----|----------|------------|
| `agentica/runner/` | `Runner`, `LoopBreak`, `ModelCallResult`, `ToolHandlingResult` | `core.py`, `loop.py`（`_run_impl`）, `compress.py`, `retry_fallback.py` |
| `agentica/cli/commands/` | `CommandContext`, `PendingQueue`, `COMMAND_REGISTRY` / `COMMAND_HANDLERS` | `session.py`, `model_config.py`, `runtime.py`, `goal.py`, `registry.py` |
| `agentica/cli/display/` | TUI 公共渲染 API | `stream.py`, `console.py`, `messages.py`, `status_bar.py` |
| `agentica/cli/interactive/` | `run_interactive` | `app.py`, `console_io.py`, `stream_loop.py`, `attachments.py` |
| `agentica/tools/builtin/` | `get_builtin_tools` + 内置工具类 | `file_tool.py`, `execute_tool.py`, `delegate_tool.py`, `task_state_tools.py`, `web_tools.py` |

示例：

```python
from agentica.runner import Runner
from agentica.cli.interactive import run_interactive
from agentica.cli.commands.runtime import _cmd_steer   # 私有命令实现走子模块
from agentica.compression import evict_context          # 压缩逻辑在 compression 包
```

Agent 通过 Mixin 组合获得各类能力，每个 Mixin 只是方法容器，**状态全部存在 Agent 的 dataclass fields 上**：

| Mixin | 职责 |
|-------|------|
| `PromptsMixin` | System Prompt 三区组装、PromptBuilder 集成 |
| `ToolsMixin` | 工具注册、tool system prompt 合并、builtin tool 管理 |
| `AsToolMixin` | `Agent.as_tool()` — 把当前 Agent 包装成可被其它 Agent 调用的 Function |
| `PrinterMixin` | 流式事件打印、格式化输出 |

## Runner 执行流程

```
User Message
    │
    ▼
[InputGuardrail]          ← 拦截不合规输入
    │
    ▼
Runner._run_impl()
    │
    ├─► 构建 System Prompt  (PromptsMixin.get_system_message)
    │       ├── Static Zone:    description + instructions
    │       ├── Semi-static:    workspace context + git status
    │       └── Dynamic Zone:   workspace memory + datetime
    │
    ├─► 上下文压缩检查       (Runner._maybe_compress_messages)
    │       ├── Layer 1: evict_context() — 淘汰旧结果 + 收缩参数（免费；`enable_evict`）
    │       └── Layer 2: CompressionManager.auto_compact() — LLM 摘要（`enable_auto_compact`）
    │
    ├─► LLM API 调用         (Model.response / response_stream)
    │
    ▼
Model Response
    │
    ├─[有 tool_calls]──────────────────────────────────────┐
    │                                                       │
    │       ┌── concurrency_safe=True ──────┐              │
    │       │  asyncio.gather() 并发执行    │              │
    │       └──────────────────────────────┘              │
    │       [ToolInputGuardrail] → 执行工具 → [ToolOutputGuardrail]
    │                                   │                  │
    │                      工具结果加入消息历史              │
    │                                   │                  │
    │                      Session Log 追加                │
    │                                   │                  │
    │                      继续循环 ────┘                  │
    │                                                       │
    └─[无 tool_calls]──────────────────────────────────────┘
                │
                ▼
    [OutputGuardrail]
                │
                ▼
           RunResponse
```

## 上下文压缩机制

让一个超窗口的请求装下只有两种办法，按代价从低到高尝试：

```
Layer 1 淘汰（免费，无 LLM）:
    触发：token_count >= context_window × 0.7
    按最旧优先把 tool result 内容换成写明调用的占位符
    同时收缩过大的 tool_call 参数字符串（JSON 仍然合法）
    降回 context_window × 0.5 就停；模型还没看过的当前批次永不淘汰

Layer 2 摘要（一次 LLM 调用，不可逆）:
    淘汰兜不住时才走这层
    整个对话历史 → LLM 摘要 → [Context compressed]\n{summary}
    保留 system prompt 和从最后一条 user 消息起的整个尾部
    CompactBoundary 写入 Session Log，恢复时从此处开始
    provider 支持时优先用服务端原生 compact 做同一件事
```

淘汰没有「保留最近 N 条」这类计数参数：任何固定条数都会输给 N+1 大小的并行批次，最近的结果靠「够到之前就停」自然幸存。

两层都以「一条工具结果」而非「一条消息」为单位判断，因为 provider 的打包形态不同：OpenAI 系是一条结果一条 `role="tool"` 消息，Anthropic 则把一整轮塞进单条 `role="user"` 消息的 `tool_result` block 列表。细节见 [Compression](../advanced/compression.md)。

**Layer 0 工具输出预算**（`tool_result_storage`）：在结果产生的那一刻限住单条结果和本轮批次
（批次上限为 `0.25 × context_window`）。收缩形态取决于这个 session 有没有 `read_file` / `execute`
能把副本读回来——能，就落盘并给出路径；不能，就直接如实截断，不留一个没人能打开的路径：

```
~/.agentica/projects/<user>/<cwd>/<session_id>/tool-results/<tool_use_id>.txt
```

## Session Log（JSONL 会话日志）

基于追加写 JSONL 的持久化会话日志，每次 `run()` 自动追加：

```jsonl
{"type":"user","uuid":"a1b2...","parent_uuid":null,"session_id":"sess-001","timestamp":"2026-04-05T10:00:00.000Z","content":"分析这段代码"}
{"type":"assistant","uuid":"c3d4...","parent_uuid":"a1b2...","timestamp":"...","content":"好的...","model":"gpt-4o","usage":{"input_tokens":1024,"output_tokens":256}}
{"type":"tool","uuid":"e5f6...","parent_uuid":"c3d4...","tool_name":"read_file","content":"..."}
{"type":"compact_boundary","uuid":"g7h8...","parent_uuid":null,"summary":"...会话摘要..."}
```

**恢复机制**：重新创建相同 `session_id` 的 Agent 时，从最后一个 `compact_boundary` 开始加载消息，跳过历史数据（大文件优化）。

**存储路径**：`~/.agentica/projects/<sanitized-cwd>/<session-id>.jsonl`

## 工具并发执行

LLM 一次性返回多个工具调用时，Agentica 自动并发执行安全的只读工具：

```
LLM 返回: [read_file("a.py"), grep("TODO"), read_file("b.py")]

                    asyncio.gather()
                    ┌─────────────────────┐
  ──► read_file("a.py")  (concurrency_safe=True)
  ──► grep("TODO")       (concurrency_safe=True)
  ──► read_file("b.py")  (concurrency_safe=True)
                    └─────────────────────┘
                    等待全部完成，一起写入消息历史

  ──► execute("git commit")  (concurrency_safe=False) → 串行执行
```

**安全标注**：每个工具函数通过 `is_read_only`、`is_destructive`、`concurrency_safe` 元数据声明其行为语义，Guardrail 和权限系统据此决策。

## 安全机制

### 四层 Guardrails

```
输入 → [InputGuardrail] → Agent → [OutputGuardrail] → 输出

工具调用: Agent → [ToolInputGuardrail] → 工具执行 → [ToolOutputGuardrail] → Agent
```

```python
from agentica.guardrails.agent import InputGuardrail, OutputGuardrail

def check_language(message: str) -> bool:
    # 拦截非中文输入
    return True  # False = 拦截

agent = Agent(
    model=ZhipuAI(),
    input_guardrails=[InputGuardrail(check_language)],
)
```

### Death Spiral 检测

当连续 N 轮（默认 5）所有工具调用都失败时，自动停止，防止无限 error-retry 循环。

### Cost Budget

通过 `RunConfig` 设置单次运行的成本上限：

```python
from agentica.run_config import RunConfig

result = await agent.run(
    "复杂的分析任务",
    config=RunConfig(
        max_cost_usd=0.5,      # 最多花 0.5 美元
        max_tokens=10000,      # 最多输出 10000 tokens
        timeout=120,           # 最长运行 120 秒
    ),
)
```

## 多 Agent 编排

### as_tool（编排器模式）

```
Orchestrator Agent
    ├── tools=[search_agent.as_tool(), analyst_agent.as_tool()]
    └── 由 LLM 决定调用哪个 worker、传什么参数
```

任意 Agent 通过 `agent.as_tool(tool_name=..., tool_description=...)` 转换为可被
其他 Agent 调用的 `Function`。每次调用内部会 `clone()` 一个新 worker 运行任务，
保持调用方与被调用方的隔离。

```python
from agentica import Agent, ZhipuAI

search_agent = Agent(name="Searcher", tools=[DuckDuckGoTool()])
analyst_agent = Agent(name="Analyst", instructions=["分析数据，输出洞察"])

leader = Agent(
    model=ZhipuAI(),
    tools=[
        search_agent.as_tool(tool_name="search"),
        analyst_agent.as_tool(tool_name="analyze"),
    ],
)
```

### Subagent（受治理的 spawn）

通过 `BuiltinTaskTool` / `SubagentRegistry.spawn()` 启动结构化子 Agent，附带：
工具白/黑名单、`MAX_DEPTH=2` 嵌套限制、超时控制、并发上限、注册表追踪。
适合那些需要"沙箱化执行 + 复用父 Agent 模型/工具/工作区"的子任务。

### Workflow（管道模式）

```
Input
  → Agent A (数据收集)
  → Agent B (数据分析)
  → Agent C (报告生成)
  → Output
```

确定性管道，每步输出作为下步输入，适合结构化多步骤任务。

### Swarm（网状模式）

多 Agent 自主协作，动态决定任务分配，适合复杂、不可预测的任务场景。

## 钩子系统（Hooks）

生命周期钩子允许在 Agent 运行的各个阶段插入自定义逻辑：

```python
from agentica.hooks import AgentHooks, RunHooks
from agentica.run_response import RunResponse

class MyHooks(RunHooks):
    async def on_agent_start(self, agent, message):
        print(f"开始处理: {message[:50]}")

    async def on_agent_end(self, agent, response: RunResponse):
        print(f"完成 | tokens: {response.metrics.get('total_tokens')}")

    async def on_tool_call_completed(self, agent, tool_name, result):
        print(f"工具 {tool_name} 完成")

agent = Agent(model=ZhipuAI())
result = await agent.run("你好", hooks=MyHooks())
```

内置 Hooks 实现：

| Hook 类 | 功能 |
|---------|------|
| `MemoryExtractHooks` | 对话结束后自动提取记忆 |
| `ConversationArchiveHooks` | 对话结束后自动归档 |

## 下一步

- [Agent 核心概念](../concepts/agent.md) -- Agent API 详解
- [Tools](../concepts/tools.md) -- 工具系统深度解析
- [Memory & Workspace](../concepts/memory.md) -- 记忆和持久化
- [Hooks](../advanced/hooks.md) -- 生命周期钩子完整 API
- [Compression](../advanced/compression.md) -- 上下文压缩配置
