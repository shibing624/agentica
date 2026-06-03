[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: Build AI Agents
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.12%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**Agentica** 不是套一层 LLM API 的聊天壳，而是一个 Async-First 的 agent harness。
它让 Agent 能真正跑起来: 调工具、跑长任务、做多智能体协作、跨会话保留记忆，并通过 Skill system 接入可演进的 self-learn 工作流。

| 能力 | 说明 |
|------|------|
| **Long-running Agent Loop** | `Runner` 驱动的 LLM ↔ Tool 循环，内置压缩、重试、成本预算、死循环防护 |
| **Works Beyond Chat** | 文件、执行、搜索、浏览器、MCP、多智能体、Workflow，不依附单一 IDE 场景 |
| **Memory That Survives Sessions** | Workspace 记忆按条目存储、相关性召回，并可把确认过的偏好同步到 `~/.agentica/AGENTS.md` |
| **Skill-Based Self-Learn** | SkillTool 可加载外部技能；内置 Agent 持续学习策略 |
| **Self-Evolution（自进化）** | 工具失败 / 用户纠正 / 成功序列 → 经验卡片 → 自动生成 SKILL.md，跨会话复用 |
| **Open, Composable Harness** | 模型、工具、记忆、Skill、Guardrails、MCP 都是可替换部件，而不是封闭 SaaS 黑盒 |

## 🔥 News

- [2026/06/03] **v1.4.6**：支持fallback模型可配置，支持多个fallback模型；支持 LSP， CLI 开启 LSP 开关（`--enable-diagnostics`/`--diagnostics-server`）；支持 `agentica doctor`；支持 `/goal` 长程任务。详见 [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**：MemoryExtractHooks 优化，新增 `auto_extract_memory_background` 后台抽取（不再阻塞 `on_agent_end`），memory 抽取优先走更快更便宜的 `auxiliary_model`。详见 [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**：Skill 生命周期重构 + VaG 解耦，新增 `SkillLifecycleHooks` 统一扩展点。详见 [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

## 架构

Agentica 提供了从底层模型路由到顶层多智能体协作的完整抽象：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### 核心执行引擎 (Agentic Loop)

Agentica 的单体 Agent 运行在一个纯粹的基于控制流的 `while(true)` 引擎中，严格依据工具调用来驱动，并内置了防死循环、成本追踪、上下文微压缩（Compaction）和四层安全护栏：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## 安装

```bash
pip install -U agentica
```

## 快速开始

不用学 asyncio 也能跑。`run_sync` 在内部跑完整 agentic loop（工具并发、流式、压缩、重试都在），从外面看就是一个普通同步函数：

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("一句话介绍北京")
print(result.content)
```

```
北京是中国的首都，是一座拥有三千多年历史的文化名城，也是全国的政治、文化和国际交流中心。
```

需要先设置 LLM 提供商的 API Key：

```bash
export OPENAI_API_KEY="sk-xxx"                      # OpenAI API Key
export OPENAI_BASE_URL="https://api.openai.com/v1"  # OpenAI API Base URL
```

可选设置 DeepSeek API Key、Zhipu API Key、Claude API Key 等：

```bash
export DEEPSEEK_API_KEY="your-api-key"              # DeepSeek API Key
export ZHIPUAI_API_KEY="your-api-key"               # Zhipu API Key
export ANTHROPIC_API_KEY="your-api-key"             # Claude API Key
```

### 同步 vs 异步

| 场景 | 推荐 API |
|---|---|
| 普通脚本 / Jupyter / FastAPI 路由（默认） | `agent.run_sync(...)`、`agent.print_response_sync(...)`、`for chunk in agent.run_stream_sync(...)` |
| 已经在 asyncio 事件循环里 / 想 `gather` N 个 agent 并发 | `await agent.run(...)`、`async for chunk in agent.run_stream(...)` |

`run_sync` 内部就是 `asyncio.run(self.run(...))`，工具调用层用 `asyncio.gather` 并发——**同步 API 不会牺牲性能**，只是把事件循环藏起来了。

```python
import asyncio
from agentica import Agent, OpenAIChat

async def main():
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
    result = await agent.run("一句话介绍上海")
    print(result.content)

asyncio.run(main())
```

### 推荐导入方式

核心 SDK + 内置工具都已经顶层导出，不用记长路径：

```python
from agentica import (
    Agent, DeepAgent, Workspace, tool,
    OpenAIChat,                                       # openai 是核心依赖
    BuiltinFileTool, BuiltinExecuteTool,              # 文件 / 执行
    BuiltinFetchUrlTool, BuiltinWebSearchTool,        # 网页
    BuiltinTodoTool, BuiltinTaskTool,                 # 任务清单 / 子 Agent
    HistoryConfig, WorkspaceMemoryConfig, RunConfig,  # 配置
)

# 其他模型 / 重型工具走子模块（避免启动时拉重依赖）
from agentica.model.anthropic.claude import Claude   # pip install anthropic
from agentica.model.ollama.chat import Ollama
from agentica.tools.shell_tool import ShellTool
```

## 功能特性

- **Async-First** — 原生 async API，`asyncio.gather()` 并行工具执行，同步适配器兼容
- **Runner Agentic Loop** — LLM ↔ 工具调用自动循环，多轮链式推理、死循环检测、成本预算、压缩 pipeline、API 重试
- **20+ 模型** — OpenAI / DeepSeek / Claude / 智谱 / Qwen / Moonshot / Ollama / LiteLLM 等
- **40+ 内置工具** — 搜索、代码执行、文件操作、浏览器、OCR、图像生成
- **RAG** — 知识库管理、混合检索、Rerank，集成 LangChain / LlamaIndex
- **多智能体** — `Agent.as_tool()`（轻量组合）、Swarm（并行/自治）和 Workflow（确定性编排）
- **Actor-Critic 精炼** — `refine()` + 多 Critic 并行评审，`SchemaCritic` 程序级零成本验证 / `AgentCritic` 异构强模型把关，循环检测自动早停
- **`/goal` 长任务循环** — `await agent.run_goal("xxx")` 持续推进，自动判断完成、续跑、暂停；支持 token / wall-clock / turn 三种 hard cap；CLI `/goal /subgoal` 即开即用，详见 [文档](https://shibing624.github.io/agentica/advanced/goals)
- **安全守卫** — 输入/输出/工具级 Guardrails，流式实时检测
- **MCP / ACP** — Model Context Protocol 和 Agent Communication Protocol 支持
- **Skill 系统** — 基于 Markdown 的技能注入，支持项目级、用户级和外部托管 skill 目录
- **多模态** — 文本、图像、音频、视频理解
- **持久化记忆** — 索引/内容分离、相关性召回、四类型分类、drift 防御，并可同步长期偏好到全局 `AGENTS.md`

## Workspace 记忆

Workspace 提供跨会话的持久化记忆，采用索引/召回设计；需要时还可以把确认过的用户/反馈记忆编译进全局 `~/.agentica/AGENTS.md`，让新 session 自动继承：

```python
from agentica import Workspace

workspace = Workspace("./workspace")
workspace.initialize()

# 写入带类型的记忆条目（每条独立文件，自动更新索引）
await workspace.write_memory_entry(
    title="Python Style",
    content="User prefers concise, typed Python.",
    memory_type="feedback",              # user|feedback|project|reference
    description="python coding style",   # 相关性匹配关键词
    sync_to_global_agent_md=True,        # 同步到 ~/.agentica/AGENTS.md 的 Learned Preferences 区块
)

# 相关性召回（根据当前 query 返回最相关的 ≤5 条）
memory = await workspace.get_relevant_memories(query="how to write python")
```

Agent 自动根据当前 query 召回最相关记忆，而非全量注入：

```python
from agentica import DeepAgent, Workspace
from agentica.agent.config import WorkspaceMemoryConfig

agent = DeepAgent(
    workspace=Workspace("./workspace"),
    long_term_memory_config=WorkspaceMemoryConfig(
        max_memory_entries=5,  # 最多注入 5 条相关记忆
        sync_memories_to_global_agent_md=True,
    ),
)
```

`DeepAgent` 默认启用 `SkillTool(auto_load=True)`，会自动发现 `~/.agentica/skills/` 和 `.agentica/skills/` 目录下的 skill；同时默认开启 `tool_config.auto_load_mcp=True`，启动时会自动读取工作目录里的 `mcp_config.json/yaml/yml`。这样 DeepAgent 开箱就是带 skills + MCP + memory 的一键完全体。

## 自进化（Self-Evolution）

Agentica 不止"记住事实"，还能**记住做事方式**。Agent 在跑工具的过程中产生的所有信号——工具失败、用户纠正、成功序列——都会被采集成 **Experience 事件**，按规则编译成 **经验卡片（cards）**，等同一个规则被反复确认 N 次后，**自动生成一个 `SKILL.md`** 并落到 workspace 的 `generated_skills/` 目录。下一次启动新 Agent 时，`SkillTool` 自动发现并注入这个技能，模型就能在新会话里直接复用之前学到的做事方式。

整条链路本地、可审计、零外部依赖，确定性采集（tool error / success）零 LLM 成本，只有"用户纠正分类"和"是否生成新技能"这两步用 `auxiliary_model` 做 LLM 判定。

### 流程图

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

事件采集（黄色）→ 经验编译与生命周期（蓝色存储 + 灰色虚线框）→ Skill 生成闸门 + LLM 判定（粉色）→ 下一会话自动复用，全链路可审计、零外部依赖。

### 启用方式

最小改动：把 `ExperienceCaptureHooks` 挂到 Agent 上，配置 `ExperienceConfig.skill_upgrade=SkillUpgradeConfig(mode="shadow")` 即开启完整自进化闭环。

```python
from agentica import Agent, Workspace, OpenAIChat
from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig
from agentica.hooks import (
    ConversationArchiveHooks,
    ExperienceCaptureHooks,
    MemoryExtractHooks,
    _CompositeRunHooks,
)

workspace = Workspace("./workspace", user_id="alice")
workspace.initialize()

model = OpenAIChat(id="gpt-4o-mini")

hooks = _CompositeRunHooks([
    ConversationArchiveHooks(),                # 自动归档对话
    MemoryExtractHooks(),                      # LLM 抽取长期记忆
    ExperienceCaptureHooks(
        ExperienceConfig(
            capture_tool_errors=True,          # 确定性，零 LLM 成本
            capture_success_patterns=True,     # 确定性，零 LLM 成本
            capture_user_corrections=True,     # 用 auxiliary_model 分类
            feedback_confidence_threshold=0.6,
            promotion_count=3,                 # 同一规则被确认 3 次 → tier=hot
            skill_upgrade=SkillUpgradeConfig(
                mode="shadow",                 # off | draft | shadow
                min_repeat_count=3,            # 至少重复 3 次才考虑生成 skill
                min_tier="warm",
                min_success_applications=1,    # 冷启动 demo 可设 0
            ),
        )
    ),
])

agent = Agent(
    model=model,
    auxiliary_model=model,                     # 给纠正分类 / skill 判定用
    workspace=workspace,
)
agent._default_run_hooks = hooks

# 跑你正常的业务，所有失败 / 纠正 / 成功都会沉淀到 workspace
agent.run_sync("帮我读一下 ./docs/agent.md")
```

跑完 N 个 session 后，workspace 长这样：

```
workspace/users/alice/
├── experiences/
│   ├── events.jsonl                        # 所有原始事件（append-only）
│   ├── EXPERIENCE.md                       # 卡片索引
│   └── <hash>__list_dir_before_read.md     # 编译后的经验卡片
├── generated_skills/
│   ├── INDEX.md                            # L1 关键词路由
│   └── list-dir-before-read/
│       ├── SKILL.md                        # 自动生成的可复用技能
│       └── meta.json                       # status: shadow / draft / promoted
└── reports/learning/                       # 每次 run 的学习报告
```

完整 e2e demo（Session 1 自进化生成技能 → Session 2 全新 Agent 跨会话复用）：[`examples/self_evolution/01_self_evolution_e2e.py`](examples/self_evolution/01_self_evolution_e2e.py)。

> **配置取舍**：`mode="shadow"` 自动安装到 workspace 本地不影响其他用户；`mode="draft"` 只生成草稿不安装，适合人审；`mode="off"` 等价于不开启 skill 自动生成（但仍采集经验卡片）。`min_success_applications` 是"必须先有 ≥N 次 tool_recovery 才允许生成 skill"的安全闸——避免给 Agent 永远做不对的事情生成技能；冷启动 demo 把它设成 `0`。

## Actor-Critic 精炼（refine）

Agentica 提供 **协议级 Actor-Critic 范式**：Actor 出草稿，多个 Critic 并行评审，不通过则反馈让 Actor 修订，直到所有 Critic 同意或触发早停。这是 [CarePilot 论文（arXiv:2603.24157）](https://arxiv.org/abs/2603.24157) 验证过的"小模型 + 好框架 > 大模型零样本"路径——在医疗 GUI 任务上 7B 微调模型 48.9% 任务完成率，比 GPT-5 高近 13 个百分点；其消融数据显示去掉 Critic 闭环 → 任务准确率从 48.9% 暴跌到 12.5%。

agentica 的设计原则是 **"SDK 提供协议而非能力"**：基模型的自我反思能力交给 LLM 本身，SDK 负责无 LLM 能替代的三件事——

- **业务约束注入**：`SchemaCritic` 用 Pydantic 做程序级零成本验证（任何 LLM 都打不过的硬约束）
- **审计反思 trail**：`RefineResult.history` 记录每轮 draft + verdicts，全流程可观测、可重放
- **异构组合**：便宜 Actor + 强 Critic、多 Critic 并行评审、确定性 + LLM 验证混搭，仅 SDK 层能表达

```python
import asyncio
from typing import Literal
from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat
from agentica.critic import SchemaCritic, AgentCritic, CritiqueStyle, refine


class IntentReply(BaseModel):
    intent: Literal["question", "command", "complaint", "smalltalk"]
    confidence: float = Field(ge=0.0, le=1.0)


async def main():
    actor = Agent(
        name="classifier",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=(
            "把用户消息分类为 question/command/complaint/smalltalk 之一。"
            '只输出 JSON: {"intent": "<四类之一>", "confidence": <0~1 浮点数>}。'
            "不要解释，不要 markdown 代码块。"
        ),
    )
    reviewer = Agent(
        name="reviewer",
        model=OpenAIChat(id="gpt-4o"),
        instructions="检查分类是否合理，合理回复 APPROVED，否则列出问题。",
    )

    result = await refine(
        actor,
        task='分类下列消息:\n"我的手机一直死机，请帮我退款!"',
        critics=[
            SchemaCritic(IntentReply),                              # 程序级 schema 验证（零 LLM 成本）
            AgentCritic(reviewer, style=CritiqueStyle.STRICT),      # LLM 级语义审核（可调风格）
        ],
        max_iter=3,
    )

    print(result.final_draft)        # 例: {"intent": "complaint", "confidence": 0.95}
    print(result.approved, result.stopped_reason, result.iterations)
    for r in result.history:         # 每轮 draft + 每个 critic 的 verdict（可审计）
        print(r.draft, [(v.critic_name, v.approved) for v in r.verdicts])


asyncio.run(main())
```

**关键特性**：

- `Critic` Protocol — duck-typed 接口，自定义 critic 不到 20 行（regex / API call / 任何业务规则）
- 多 Critic 并行（`asyncio.gather`），强模型 critic 与零成本程序 critic 自由混搭
- `CritiqueStyle.STRICT/NEUTRAL/LENIENT` — 控制 LLM critic 的评审温度（论文建议默认 NEUTRAL）


完整示例：

- [`examples/agent_patterns/13_actor_critic_refine.py`](examples/agent_patterns/13_actor_critic_refine.py) — `refine()` 标准用法，`SchemaCritic` + `AgentCritic` 并行混搭，可审计 history trail
- [`examples/agent_patterns/04_debate.py`](examples/agent_patterns/04_debate.py) — 多 agent 辩论场景，用 `AgentCritic` 把对方包装成结构化反驳者

## Agent 配方（Recipes）

`Agent` 参数多，但常用组合就那几个。直接 copy-paste：

### 一次性脚本（最简）

```python
agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
print(agent.run_sync("一句话介绍北京").content)
```

### 多轮对话

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=5,
)
agent.run_sync("我叫 Alice，是 ML 工程师。")
agent.run_sync("我叫什么？")  # 模型记得
```

### 工具型 Agent（自定义工具组合）

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("帮我搜 Python 3.13 新特性，写到 features.md")
```

### 多用户 + 长期记忆 + 会话归档

每个用户一个 Agent 实例，`session_id` 一般直接复用 `user_id`：

```python
from agentica import Agent, OpenAIChat, Workspace, WorkspaceMemoryConfig

def create_agent(user_id: str) -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=Workspace("~/.agentica/workspace", user_id=user_id),
        session_id=user_id,                      # 会话日志按 user 切到 ~/.agentica/projects/.../{user_id}.jsonl
        enable_long_term_memory=True,            # ← 关键：必须显式开启
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,                   # 每次 run 后归档对话
            auto_extract_memory=True,            # LLM 自动抽取记忆
        ),
        add_history_to_context=True,
        num_history_turns=5,
    )
```

> **常见踩坑**：只配 `long_term_memory_config` 但忘了 `enable_long_term_memory=True`，所有记忆/归档都会被静默忽略。从 v1.3.7 起 `Agent.__init__` 会在这种情况下打 warning，但请直接按上面写就不会踩。

### 长会话省 token：定制历史消息

搜索类工具结果通常巨大且后续轮次用不上，可以在历史里删掉；AI 回复可以截断：

```python
from agentica import Agent, OpenAIChat, HistoryConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=10,
    history_config=HistoryConfig(
        excluded_tools=["search_*", "web_search"],   # 整条 tool 结果丢掉，对应的 tool_calls 自动同步剥离
        assistant_max_chars=200,                      # AI 回复截断到 200 字
    ),
)
```

更复杂的过滤（剥用户 prompt 前缀、按 metadata 删消息等）用 `history_filter` 回调，看 `examples/memory/03_history_filter.py`。

### 完全体（CLI / Gateway / 长任务）

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ 内置工具 + 压缩 + 长期记忆 + skills + MCP，开箱即用
```

## CLI

```bash
agentica --model_provider zhipuai --model_name glm-4.7-flash
```

安装外部 skill 集合时，**推荐使用新的 `skills` 命令**：

```bash
agentica skills install https://github.com/obra/superpowers
agentica skills list
agentica skills remove learn-from-experience
agentica skills reload
```

如果你已经进入交互式 CLI，也可以直接在会话里安装、查看和卸载 skills：

```text
> /skills install https://github.com/obra/superpowers
> /skills list
> /skills inspect learn-from-experience
> /skills remove learn-from-experience
> /skills reload
```

也支持安装本地目录或指定目标目录：

```bash
agentica skills install /path/to/skill-repo --target-dir ~/.agentica/skills
```

如果你安装到自定义目录而不是标准搜索路径，记得把这个目录加入 `AGENTICA_EXTRA_SKILL_PATH`，这样 `DeepAgent` 和 CLI 才会自动发现它。

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

### 长任务：`/goal`

让 Agent 持续向一个目标推进，每轮结束自动判断是否完成，没完成就续跑——直到 judge 判 done、预算耗尽、或用户主动停下。

CLI：

```text
/goal 实现 xxx 功能并跑通 pytest    # 设置目标 + 自动开跑
/goal status                       # 显示状态、预算、subgoals
/goal pause | resume | clear
/subgoal 必须补单测                  # 给目标加验收条件
```

SDK 一行起飞（推荐用 pro + flash 双模型省钱）：

```python
from agentica import Agent, DeepSeekChat

agent = Agent(
    session_id="my-task",
    model=DeepSeekChat(id="deepseek-v4-pro"),       # 主模型干活
    auxiliary_model=DeepSeekChat(
        id="deepseek-v4-flash",
        max_completion_tokens=4096,   # judge JSON 输出预算；reasoning judge 必须显式给够
    ),
)

# 不传任何 budget = 只有默认 100 turns 安全网，token / wall-clock 不设限
result = await agent.run_goal("compute 17+9+16 and state the answer")

# 推荐：长编码任务显式设 token / wall-clock 上限
result = await agent.run_goal(
    "实现 xxx 功能并跑通 pytest",
    token_budget=100000,        # ~$0.1 (取决于模型)
    wall_clock_budget_sec=1800,  # 30 分钟
)

print(result.status, result.reason)   # complete / paused / budget_limited
print(result.response_content)        # == result.run_response.content or ""
```

三个预算之间是 **"任一触发即停"**（取最严的）：

| 参数 | 默认 | 不传时的行为 | 推荐取值 |
|---|---|---|---|
| `turn_budget` | `100` | 用默认 100（安全网，防 runaway） | 简单任务 5–20，复杂任务用默认 |
| `token_budget` | `None` | **不限**（仅 turn_budget 兜底） | 编码任务 `50_000`–`200_000` |
| `wall_clock_budget_sec` | `None` | **不限** | 长任务 `1800`–`3600` 秒 |

优先级 `budget > tool > judge`。模型也能通过受限工具 `update_goal(status="complete"|"paused")` 自己结束循环。完整说明：[Standing Goal Loop 文档](https://shibing624.github.io/agentica/advanced/goals)，运行示例 `examples/cli/03_goal_loop_demo.py`。

## Web UI / Gateway

**Gateway 现在已经集成到 `agentica` 主库中**。

安装 Gateway 运行时：

```bash
pip install -U "agentica[gateway]"
```

设置一个最小可运行配置后直接启动：

```bash
export AGENTICA_MODEL_PROVIDER=zhipuai
export AGENTICA_MODEL_NAME=glm-4.7-flash
export GATEWAY_TOKEN=change-me
agentica-gateway
```

默认会启动在 `http://127.0.0.1:8789/chat`。如需改监听地址，可设置 `HOST` 和 `PORT`；如需接入 Telegram / Discord / Slack，可分别安装 `agentica[telegram]`、`agentica[discord]`、`agentica[slack]`。

Gateway 内置了 Web UI、API、WebSocket、cron scheduler，以及飞书 / Telegram / Discord 等 channel 接入能力，适合把 `agentica` 从本地 CLI 扩展到常驻服务。

## 示例

查看 [examples/](https://github.com/shibing624/agentica/tree/main/examples) 获取完整示例，涵盖：

| 类别 | 内容 |
|------|------|
| **基础用法** | Hello World、流式输出、结构化输出、多轮对话、多模态、**Agentic Loop 对比** |
| **工具** | 自定义工具、Async 工具、搜索、代码执行、并行工具、并发安全、成本追踪、沙箱隔离、压缩 |
| **Agent 模式** | Agent 作为工具、并行执行、团队协作、辩论、路由分发、Swarm、子 Agent、模型层钩子、会话恢复 |
| **安全护栏** | 输入/输出/工具级 Guardrails、流式护栏 |
| **记忆** | 会话历史、WorkingMemory、上下文压缩、Workspace 记忆、LLM 自动记忆 |
| **RAG** | PDF 问答、高级 RAG、LangChain / LlamaIndex 集成 |
| **工作流** | 数据管道、投资研究、新闻报道、代码审查 |
| **MCP** | Stdio / SSE / HTTP 传输、JSON 配置 |
| **可观测性** | Langfuse、Token 追踪、Usage 聚合 |
| **应用** | LLM OS、深度研究、客服系统、**金融研究（6-Agent 流水线）** |

[→ 查看完整示例目录](https://github.com/shibing624/agentica/blob/main/examples/README.md)

## 文档

完整使用文档：**https://shibing624.github.io/agentica**

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/agentica/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 Agentica，请引用：

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 致谢

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
