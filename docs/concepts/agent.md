# Agent

Agent 是 Agentica 的核心组件——它将模型、工具和记忆连接在一起，能够思考、决策和执行动作。

## 基本结构

```
Agent
├── Model          # 大脑：LLM 提供推理能力
├── Tools          # 双手：与外部世界交互的能力
├── Memory         # 运行时记忆：当前会话消息历史
├── Workspace      # 持久化记忆：跨会话长期记忆
├── Knowledge      # 知识库：外部文档检索 (RAG)
├── Instructions   # 指令：定义 Agent 的行为和风格
└── Runner         # 执行引擎（自动创建，无需手动管理）
```

## 创建 Agent

### 最小示例

```python
from agentica import Agent, ZhipuAI

agent = Agent(model=ZhipuAI())
result = agent.run_sync("一句话介绍北京")
print(result.content)
```

### 完整配置示例

```python
import asyncio
from agentica import Agent, OpenAIChat
from agentica.tools.duckduckgo_tool import DuckDuckGoTool
from agentica.tools.url_crawler_tool import UrlCrawlerTool
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig
from agentica import Workspace

async def main():
    agent = Agent(
        # 核心定义
        name="Research Assistant",
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "你是一个专业的研究助手，擅长收集和整理最新信息",
            "搜索时使用多个关键词进行验证，确保信息准确",
            "用中文回答，条理清晰，附上信息来源",
        ],
        tools=[DuckDuckGoTool(), UrlCrawlerTool()],

        # Prompt 配置
        prompt_config=PromptConfig(
            markdown=True,               # 输出 Markdown 格式
            add_datetime_to_instructions=True,  # 注入当前日期
        ),

        # 工具配置
        tool_config=ToolConfig(
            tool_call_limit=20,          # 单次最多调用 20 次工具
            compress_tool_results=True,  # 自动压缩大工具结果
        ),

        # 长期记忆
        workspace=Workspace("./workspace"),
        long_term_memory_config=WorkspaceMemoryConfig(
            load_workspace_memory=True,
            max_memory_entries=5,
        ),

        # 会话历史
        add_history_to_context=True,
        num_history_turns=5,
    )

    result = await agent.run("2025 年 AI 领域最重要的进展有哪些？")
    print(result.content)

asyncio.run(main())
```

## 参数分层

### 第一层：核心定义

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `Model` | LLM 模型实例（必填） |
| `name` | `str` | Agent 名称，注入到 System Prompt |
| `description` | `str` | Agent 描述，作为 System Prompt 开头 |
| `instructions` | `str \| List[str] \| Callable` | 行为指令，支持动态计算 |
| `tools` | `List[Tool \| Callable]` | 工具列表 |
| `knowledge` | `Knowledge` | RAG 知识库 |
| `workspace` | `Workspace` | 持久化工作空间 |
| `session_id` | `str` | 会话 ID（用于会话持久化和恢复） |
| `response_model` | `Type[BaseModel]` | 结构化输出（Pydantic 模型） |

### 第二层：会话历史

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_long_term_memory` | `bool` | `False` | 启用长期记忆工具、检索和相关 hooks |
| `enable_experience_capture` | `bool` | `False` | 启用 experience 捕获与自进化 hooks |
| `add_history_to_context` | `bool` | `False` | 将历史消息加入上下文 |
| `num_history_turns` | `int` | `3` | 保留最近 N 轮历史 |
| `use_structured_outputs` | `bool` | `False` | 严格 JSON 结构化输出（部分 API 支持） |
| `debug` | `bool` | `False` | 启用调试日志 |
| `enable_tracing` | `bool` | `False` | 启用 Langfuse tracing |

### 第三层：打包配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt_config` | `PromptConfig` | 提示词工程配置 |
| `tool_config` | `ToolConfig` | 工具调用行为配置 |
| `long_term_memory_config` | `WorkspaceMemoryConfig` | 工作空间记忆配置 |

## 运行方式

Agentica 采用 **async-first** 架构：

```python
import asyncio
from agentica import Agent, ZhipuAI

agent = Agent(model=ZhipuAI())

# 1. 异步非流式（推荐，生产环境使用）
async def example_async():
    result = await agent.run("你好")
    print(result.content)

# 2. 异步流式（实时显示，长响应适用）
async def example_stream():
    async for chunk in agent.run_stream("写一首诗"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

# 3. 同步非流式（脚本/Jupyter 适用）
result = agent.run_sync("你好")
print(result.content)

# 4. 同步流式（Jupyter / 非 async 环境）
for chunk in agent.run_stream_sync("写一首诗"):
    if chunk.content:
        print(chunk.content, end="", flush=True)

asyncio.run(example_async())
```

| 方法 | 类型 | 适用场景 |
|------|------|----------|
| `run(msg)` | `async` | FastAPI、异步服务、生产环境 |
| `run_stream(msg)` | `async generator` | 实时流式响应 |
| `run_sync(msg)` | `sync` | 脚本、Jupyter、CLI |
| `run_stream_sync(msg)` | `sync iterator` | 同步代码中的流式输出 |
| `print_response(msg)` | `async` | 快速测试（自带格式化打印） |

## Instructions（指令）

指令定义 Agent 的行为、风格和约束。

### 静态指令

```python
agent = Agent(
    instructions=[
        "你是一个 Python 代码审查专家",
        "重点检查：安全漏洞、性能瓶颈、代码可读性",
        "发现问题时，提供具体的修复建议和代码示例",
        "使用中文回复，格式清晰",
    ],
)
```

### 动态指令（基于上下文计算）

```python
from datetime import date

def get_instructions(agent) -> list:
    """运行时动态计算指令"""
    instructions = ["你是一个智能助手"]
    today = date.today()
    if today.weekday() >= 5:  # 周末
        instructions.append("今天是周末，优先考虑娱乐和生活类问题")
    if agent.session_state.get("expert_mode"):
        instructions.append("使用专业术语，提供深度分析")
    return instructions

agent = Agent(instructions=get_instructions)
# 每次 run() 前，Callable 会被调用并传入 agent 实例
```

### 完整 System Prompt

如果需要完全控制 System Prompt（绕过 Agentica 的 prompt 组装逻辑）：

```python
from agentica.agent.config import PromptConfig

agent = Agent(
    prompt_config=PromptConfig(
        system_prompt="你是一个友好的中文助手。回答要简洁准确，控制在 200 字以内。",
    ),
)

# 也可以是 Callable，运行时动态计算：
agent = Agent(
    prompt_config=PromptConfig(
        system_prompt=lambda agent: f"你是 {agent.name}，今天是{date.today()}。",
    ),
)
```

## PromptConfig（进阶）

`PromptConfig` 控制 System Prompt 的构建细节：

```python
from agentica.agent.config import PromptConfig

agent = Agent(
    model=ZhipuAI(),
    prompt_config=PromptConfig(
        # 内容补充（追加到 instructions 后）
        task="分析用户提供的数据并给出洞察",
        role="数据分析师",
        guidelines=["使用统计方法验证结论", "图表描述要详细"],
        expected_output="分析报告：问题→数据→结论→建议",

        # 格式控制
        markdown=True,                # 要求 LLM 输出 Markdown
        output_language="中文",       # 强制中文输出
        add_datetime_to_instructions=True,  # 注入当前日期

        # 安全开关
        prevent_hallucinations=True,  # 注入"不确定时说不知道"指令
        prevent_prompt_leakage=True,  # 注入"不泄露 system prompt"指令
        limit_tool_access=True,       # 注入"只使用已提供的工具"指令

        # Agentic Prompt 模式（见下文）
        enable_agentic_prompt=False,  # 默认关闭

        # 极简模式（最低 token 消耗）
        minimal=False,
    ),
)
```

### `enable_agentic_prompt`：自主 Agent 模式

开启后，System Prompt 注入四个行为模块，让 Agent 具备自主完成复杂任务的能力：

```python
from agentica.agent.config import PromptConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    prompt_config=PromptConfig(enable_agentic_prompt=True),
)
```

注入的四个模块：

| 模块 | 文件 | 效果 |
|------|------|------|
| **Soul** | `soul.md` | 核心行为准则：技术准确性优先、有主见、避免过度工程 |
| **Tools Guide** | `tools.md` | 工具使用策略：禁用 cat/grep shell 等效命令，强制使用内置工具 |
| **Heartbeat** | `heartbeat.md` | 迭代控制：必须持续直到问题彻底解决，不半途而废 |
| **Self Verification** | `self_verification.md` | 代码变更后必须运行 lint/test 验证 |

**适用场景**：`DeepAgent`（CLI 模式）默认开启；普通 `Agent` 默认关闭。开启后 Agent 更擅长多步骤代码任务，但 prompt token 会增加约 800 tokens。

### `minimal` 模式

最低 token 消耗，跳过所有 prompt 组装，只输出一行系统提示：

```python
agent = Agent(
    model=ZhipuAI(),
    prompt_config=PromptConfig(minimal=True),
)
# System Prompt: "You are Agent-Name, an AI assistant.\nCWD: ...\nDate: ..."
# 约 20 tokens，适合大批量、成本敏感的场景
```

也可通过环境变量启用：`AGENTICA_SIMPLE=1 python script.py`

## ToolConfig（工具配置）

```python
from agentica.agent.config import ToolConfig

agent = Agent(
    model=ZhipuAI(),
    tools=[...],
    tool_config=ToolConfig(
        tool_call_limit=30,            # 单次 run 最多调用工具 30 次（防止无限循环）
        compress_tool_results=True,    # 大工具结果自动压缩（节省 token）
        context_overflow_threshold=0.8, # 上下文达 80% 时触发压缩
        search_knowledge=True,         # 允许 Agent 主动搜索知识库
    ),
)
```

## WorkspaceMemoryConfig（长期记忆配置）

```python
from agentica.agent.config import WorkspaceMemoryConfig
from agentica import Workspace

agent = Agent(
    workspace=Workspace("./workspace"),
    long_term_memory_config=WorkspaceMemoryConfig(
        load_workspace_context=True,   # 加载 AGENTS.md/PERSONA.md 等上下文文件
        load_workspace_memory=True,    # 加载相关记忆到 System Prompt
        max_memory_entries=5,          # 最多注入 5 条最相关的记忆
        auto_archive=True,             # 每次 run 后自动归档对话（无 LLM 成本）
        auto_extract_memory=False,     # 自动提取关键信息存记忆（有 LLM 成本）
    ),
)
```

## RunResponse

`run()` 返回 `RunResponse` 对象，包含完整的执行信息：

```python
from agentica import Agent, ZhipuAI, RunEvent

agent = Agent(model=ZhipuAI())
result = await agent.run("分析这段代码的时间复杂度")

# 基础内容
result.content           # str 或 BaseModel（结构化输出时）
result.model             # 使用的模型 ID
result.run_id            # 本次运行的唯一 ID
result.agent_id          # Agent 的唯一 ID

# 性能指标
result.metrics           # dict，包含 token 用量、延迟等
result.metrics["time"]   # 总耗时（秒）
result.metrics["prompt_tokens"]     # 输入 token 数
result.metrics["completion_tokens"] # 输出 token 数

# 工具调用信息
result.tool_calls        # List[ToolCallInfo]
for tc in result.tool_calls:
    print(f"{tc['tool_name']}: {tc['tool_call_error']}")

# 推理过程（DeepSeek-R1 等推理模型）
result.reasoning_content # str，LLM 的思考链
```

### 流式事件

```python
from agentica import RunEvent

async for chunk in agent.run_stream("帮我写一个排序算法"):
    match chunk.event:
        case RunEvent.run_response:
            print(chunk.content, end="", flush=True)  # 正常输出 token
        case RunEvent.tool_call_started:
            print(f"\n🔧 调用工具: {chunk.content}")
        case RunEvent.tool_call_completed:
            print(f" ✓")
        case RunEvent.run_completed:
            print(f"\n完成 | tokens: {chunk.metrics}")
```

## DeepAgent

`DeepAgent` 是 batteries-included product preset，专为 CLI、Gateway、日常 dogfood、复杂代码和研究任务设计。它继承自 `Agent`，但默认开启更多产品能力；库集成和最小 SDK 用法仍建议从普通 `Agent` 开始，再显式打开需要的能力。

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(
    model=OpenAIChat(id="gpt-4o"),
    name="My AI Assistant",
    work_dir="./my-project",        # 文件操作基准目录
    workspace="./workspace",        # 持久化记忆目录
)

result = await agent.run("""
分析当前项目的代码结构，找出潜在的性能瓶颈，
并为最重要的问题提供优化建议和修改后的代码
""")
```

**DeepAgent 内置工具**（`ls`, `read_file`, `write_file`, `edit_file`, `multi_edit_file`, `glob`, `grep`, `execute`, `web_search`, `fetch_url`, `write_todos`, `task`, `ask_user_question`, `save_memory`, `search_memory`）

**DeepAgent 默认开启**：
- `enable_agentic_prompt=True`（Soul + Tools Guide + Heartbeat + Self Verification）
- `compress_tool_results=True`（大工具结果自动压缩）
- `context_overflow_threshold=0.8`（上下文 80% 触发压缩）
- Workspace 长期记忆
- Session Log 会话持久化

!!! tip "SDK Core vs Product Preset"
    如果你在应用里嵌入 Agentica，默认选 `Agent`。如果你在做 CLI/Gateway/无人值守任务，想要一键获得工具、记忆、压缩和经验捕获，再选 `DeepAgent`。

## 结构化输出

通过 Pydantic 模型获取严格结构化的 JSON 数据：

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List
from agentica import Agent, ZhipuAI

class CodeReview(BaseModel):
    summary: str = Field(description="代码总体评价")
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议")
    score: int = Field(description="代码质量评分 1-10", ge=1, le=10)

async def main():
    agent = Agent(
        model=ZhipuAI(),
        response_model=CodeReview,
    )
    result = await agent.run("def add(a, b): return a+b")
    review: CodeReview = result.content  # 自动解析为 CodeReview 对象
    print(f"评分: {review.score}/10")
    for issue in review.issues:
        print(f"  - {issue}")

asyncio.run(main())
```

## 多轮对话

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(
        model=ZhipuAI(),
        add_history_to_context=True,  # 开启会话历史
        num_history_turns=10,          # 保留最近 10 轮
    )

    # 第一轮
    r1 = await agent.run("我叫小明，是一名 Python 开发者")
    print(r1.content)

    # 第二轮（Agent 记得上一轮的内容）
    r2 = await agent.run("你还记得我的名字吗？")
    print(r2.content)  # "你叫小明，是一名 Python 开发者"

asyncio.run(main())
```

## 会话持久化（Session Log）

通过 `session_id` 参数启用 JSONL 会话日志，支持跨进程恢复：

```python
import asyncio
from agentica import Agent, ZhipuAI

SESSION_ID = "my-research-session-001"

async def first_run():
    agent = Agent(model=ZhipuAI(), session_id=SESSION_ID)
    # 消息自动写入 ~/.agentica/projects/<cwd>/<session_id>.jsonl
    result = await agent.run("我在研究机器学习优化算法")
    print(result.content)

async def second_run():
    # 用相同 session_id 创建，自动恢复上次会话
    agent = Agent(model=ZhipuAI(), session_id=SESSION_ID)
    result = await agent.run("继续上次的话题")
    print(result.content)  # Agent 记得上次的内容

asyncio.run(first_run())
asyncio.run(second_run())
```

## clone()：并行 Agent

`clone()` 创建当前 Agent 的副本，重置运行时状态（`run_id`、消息历史、`_running` 标记），同时给每个 clone 独立的 Model 实例，适合并发场景：

```python
import asyncio
from agentica import Agent, OpenAIChat

async def main():
    base = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=["你是一个代码审查专家"],
    )

    # 并发处理多个文件
    files = ["auth.py", "database.py", "api.py"]
    tasks = [
        base.clone().run(f"审查 {f}")
        for f in files
    ]
    results = await asyncio.gather(*tasks)
    for f, r in zip(files, results):
        print(f"=== {f} ===\n{r.content}")

asyncio.run(main())
```

## System Prompt 三区结构

Agentica 将 System Prompt 分为三个区域，最大化 LLM prefix-cache 命中率，节省 API 费用：

```
┌─ STATIC ZONE（不变，始终可被 cache）─────────────────────┐
│  description, instructions, guidelines, expected_output  │
│  → 固定内容，跨 run 完全相同                              │
├─ SEMI-STATIC ZONE（少变）───────────────────────────────┤
│  workspace context (AGENTS.md 等), git status             │
│  → 文件内容变化时才更新                                   │
├─ DYNAMIC ZONE（每轮可变，必须在最后）────────────────────┤
│  workspace memory（相关记忆），session summary            │
│  datetime（日期，精确到天）                              │
│  → 每次 run 可能不同                                      │
└─────────────────────────────────────────────────────────┘
```

日期精确到**天**（`YYYY-MM-DD`），同一天的请求 datetime 字段完全相同，前缀 cache 可复用。

## SandboxConfig（沙箱安全）

对不受信任的代码执行场景，启用沙箱限制：

```python
from agentica.agent.config import SandboxConfig
from agentica import DeepAgent, ZhipuAI

agent = DeepAgent(
    model=ZhipuAI(),
    sandbox_config=SandboxConfig(
        enabled=True,
        writable_dirs=["./output"],        # 只允许写这个目录
        allowed_commands=["python", "pip"], # 只允许运行 Python 和 pip
        max_execution_time=60,             # 命令最长执行 60 秒
    ),
)
```

!!! warning "沙箱说明"
    `SandboxConfig` 是尽力而为的安全屏障，**不是**真正的系统级沙箱。对不受信任的代码，应使用 Docker/seccomp 等 OS 级隔离。

## 下一步

- [Workflow](../multi-agent/workflow.md) -- 确定性工作流编排
- [Subagent](../multi-agent/subagent.md) -- 受治理的子 Agent（spawn / task）
- [工具系统](tools.md) -- 工具开发和配置
- [Memory & Workspace](memory.md) -- 长期记忆机制
- [Hooks](../advanced/hooks.md) -- 生命周期钩子
- [Guardrails](../advanced/guardrails.md) -- 输入/输出安全守卫
- [RunConfig](../advanced/run-config.md) -- 运行时配置（Cost Budget、超时）
- [API 参考](../api/agent.md) -- 完整 API 文档
