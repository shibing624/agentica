# Agentica: 基于结构化并发的异步优先智能体框架——面向自主研究与软件工程

> **目标会议:** AAAI 2027
>
> **状态:** 草稿 — 实验结果部分以 `[TODO]` 标记待填充

---

## 摘要

大语言模型（LLM）智能体在自主研究和软件工程任务中展现出巨大潜力。然而，现有智能体框架存在根本性的架构局限：同步执行瓶颈导致独立工具调用被串行化、缺乏结构化并发保证、会话管理碎片化（执行状态与智能体本身分离）、以及长程任务中的上下文管理不足。本文提出 **Agentica**，一个基于 Python 3.12 的异步优先智能体框架，引入以下关键创新：(1) 基于 `asyncio.TaskGroup` 的**结构化并发工具执行**，提供逐任务异常隔离的形式化安全保证；(2) **三层工具抽象**（Tool → Function → FunctionCall），采用内存安全的弱引用和透明的同步/异步边界桥接；(3) **Agent-as-Session 统一状态模型**，智能体自身作为会话载体，通过 token 感知的运行历史管理消除 Runner/Session 分离带来的架构碎片化；(4) **DeepAgent 深度研究扩展**，具备双阈值滞回上下文管理（附形式化收敛性质）和 HEARTBEAT 式迭代控制；(5) **双层级护栏系统**，同时支持智能体级和工具级的输入/输出验证。我们将智能体执行模型形式化为状态转换系统，并证明了异常隔离、终止保证和上下文边界强制等关键安全性质。在三个基准测试——GAIA、SWE-bench Verified 和 Terminal-Bench 上的评估表明，智能体框架的架构设计显著影响任务完成率、效率和可靠性。实验结果显示，结构化并发在多工具任务上实现了 `[TODO: X%]` 的加速，双阈值上下文管理将长程任务完成率提升了 `[TODO: X%]`，整体性能达到或超过当前最优智能体系统。

---

## 1. 引言

基于 LLM 的自主智能体的出现，推动了复杂任务处理方式的范式转变——从软件工程（Jimenez et al., 2024）到深度研究（Xu & Peng, 2025）再到通用助手（Mialon et al., 2023）。不同于产生单轮回复的简单聊天机器人，智能体需要进行多步推理、调用外部工具、协调子智能体，并在长时间交互中维护持久状态。

尽管进展迅速，当前智能体框架面临若干架构层面的挑战，限制了其在困难长程任务上的表现：

**挑战 1：顺序工具执行。** 当 LLM 在单次回复中返回多个工具调用（如同时进行网络搜索和文件读取），大多数框架将其串行执行。MetaGPT（Hong et al., 2023）以流水线方式处理动作；CrewAI 使用顺序任务执行；LangGraph 的 `ToolNode` 逐一调度工具。OpenAI Agents SDK 未公开其内部工具并发模型。仅 AutoGen v0.4 采用异步优先设计，但基于 Actor 消息传递模型而非结构化并发。这种串行化在可安全并行的 I/O 密集型操作上浪费了大量时间。

**挑战 2：缺乏结构化并发保证。** 即使支持异步执行的框架（如通过 `asyncio.gather`）也缺乏结构化并发语义。当一个工具调用失败时，`asyncio.gather` 在 `return_exceptions=False` 模式下会取消所有兄弟任务，导致部分结果丢失；在 `return_exceptions=True` 模式下，错误传播成为开发者的负担。两种方式对生产级智能体系统均不理想。我们将此形式化为*部分故障隔离*性质（定义 1），并证明带逐任务异常捕获的 `asyncio.TaskGroup` 满足该性质。

**挑战 3：会话管理碎片化。** 主流设计模式将执行（Runner）与状态（Session）分离，以 OpenAI Agents SDK 为代表——Agent 是声明式配置，Runner 驱动执行，Session 存储对话历史。这种三方分离引入架构摩擦：开发者必须显式协调 runner-session 生命周期，状态不由推理实体内在拥有，多轮上下文需手动通过外部 session 对象传递。

**挑战 4：长程任务的上下文溢出。** 深度研究任务通常需要数十次工具调用，累积的上下文超出模型有效窗口。硬截断丢失关键信息；朴素摘要可能遗漏重要细节。深度研究综述（Xu & Peng, 2025）将上下文管理列为三大核心工程挑战之一。单阈值机制在上下文在阈值附近波动时会出现压缩振荡。

**挑战 5：工具级操作的安全机制不足。** 虽然 OpenAI Agents SDK 等框架提供了智能体级护栏（验证整体输入/输出），但与文件系统、数据库和外部 API 直接交互的单个工具调用缺乏细粒度验证。一个格式错误的工具调用即可破坏状态或泄露敏感数据。

为应对上述挑战，我们提出 **Agentica**，一个异步优先智能体框架，主要贡献如下：

1. **具有形式化安全保证的结构化并发工具执行**，使用 Python 3.12 的 `asyncio.TaskGroup`。我们将智能体执行循环形式化为状态转换系统（第 3.2 节），并证明三阶段执行协议满足*部分故障隔离*性质——单个工具失败不会取消兄弟执行，所有结果（成功或失败）以原始顺序保留。据我们所知，Agentica 是首个将结构化并发应用于工具执行的智能体框架。

2. **三层工具抽象**（Tool → Function → FunctionCall），分离注册、模式定义和调用。Function 通过弱引用（`weakref`）持有对父 Agent 的内存安全引用，打破 `Agent → Model → functions → Function._agent → Agent` 循环引用链。`FunctionCall.execute()` 自动检测同步/异步入口并通过 `run_in_executor()` 路由同步函数，使异步边界对工具作者透明。

3. **Agent-as-Session 统一状态模型**，将执行、推理和会话状态合并到 Agent 自身。不同于 OpenAI Agents SDK 的 Runner/Session 分离，Agentica 的 `AgentMemory` 直接在智能体内部存储结构化运行历史（`runs: List[AgentRun]`），通过 token 感知的历史检索（`get_messages_from_last_n_runs()`）提供自动预算受限的上下文组装，消除会话管理样板代码（第 3.5 节）。

4. **DeepAgent 双阈值滞回上下文管理**。引入带形式化收敛分析的双阈值机制（第 3.6 节）：软阈值（$\theta_s$）触发压缩，硬阈值（$\theta_h$）强制生成回答，滞回间隙（$\theta_h - \theta_s$）提供可证明的振荡避免。结合 HEARTBEAT 式迭代检查点，实现可靠的长程任务完成。

5. **双层级护栏系统**，同时提供智能体级（`InputGuardrail`、`OutputGuardrail`）和工具级（`ToolInputGuardrail`、`ToolOutputGuardrail`）验证——首个提供细粒度逐工具调用安全验证的框架。

6. **基于文件的工作空间记忆**，使用 Markdown 文件（`AGENT.md`、`MEMORY.md`、`USER.md` 等），具备人类可读、可审计、可 Git 版本控制的特性，支持 `users/{user_id}/` 目录下的多用户隔离。

我们在三个互补的基准测试上进行评估：
- **GAIA**（Mialon et al., 2023）：165 个通用 AI 助手任务，测试多步推理和工具使用
- **SWE-bench Verified**（Jimenez et al., 2024; OpenAI, 2024）：500 个来自 GitHub 的真实软件工程任务
- **Terminal-Bench**（Merrill et al., 2026）：89 个基于 Docker 终端环境的高难度任务，涵盖软件工程、ML/AI、网络安全和系统管理

---

## 2. 相关工作

### 2.1 智能体框架

**MetaGPT**（Hong et al., 2023）引入了 SOP-as-Prompt 范式，将人类标准操作流程编码为结构化提示序列用于多智能体协作。其发布-订阅通信机制和可执行反馈循环是重要进展。然而，MetaGPT 的编排层执行模型本质上是顺序的——智能体按流水线顺序执行。虽然各角色使用 `async def run()`，但仅用于非阻塞 LLM I/O；智能体间协调仍为同步。发表为 ICLR 2024 Oral。

**AutoGen**（Wu et al., 2023）提出了基于可对话智能体的多智能体对话框架。AutoGen v0.4（2025 年 1 月）引入了事件驱动的 Actor 模型运行时。虽然通过 Actor 消息总线实现异步优先，但其并发通过消息传递而非结构化并发原语实现，单轮内的工具执行仍为顺序。

**OpenAI Agents SDK**（OpenAI, 2025）继承实验性 Swarm 框架，提供轻量级智能体抽象——Agent 为声明式配置对象，无状态 Runner 类驱动执行。支持基于 handoff 的多智能体协调和 tripwire 模式护栏。其 Python 3.9 最低要求排除了 `asyncio.TaskGroup`（Python 3.11 引入）的使用，内部工具执行并发模型未公开。托管工具（WebSearch、FileSearch、CodeInterpreter）实际上造成对 OpenAI Responses API 的厂商锁定。

**LangGraph**（LangChain, 2023）将智能体工作流建模为有状态有向图。虽然对复杂工作流编排功能强大，但其 `ToolNode` 组件顺序执行工具，图抽象对简单智能体模式引入额外开销。

**CrewAI**（Moura, 2024）使用基于角色的智能体设计。同步优先，无原生异步架构或并行工具执行。

表 1 总结了架构对比。

**表 1：智能体框架架构对比**

| 维度 | Agentica | MetaGPT | AutoGen v0.4 | OpenAI Agents SDK | LangGraph | CrewAI |
|------|----------|---------|-------------|-------------------|-----------|--------|
| 执行模型 | 异步优先（原生） | 异步 I/O，同步编排 | 异步（Actor 模型） | 异步为主 | 双模式 | 同步优先 |
| 工具并发 | `TaskGroup`（结构化） | 顺序 | 顺序（每轮） | 未公开 | 顺序（`ToolNode`） | 顺序 |
| 工具抽象 | 三层层次结构 | Action 级 | 函数注册 | `@function_tool` 扁平 | `@tool` 装饰器 | `BaseTool` 类 |
| 会话模型 | Agent-as-Session（统一） | 环境共享状态 | ConversableAgent 状态 | Runner + Session（分离） | Checkpointer（外部） | Memory（外部） |
| 多智能体 | `as_tool()` 委托 | 发布-订阅流水线 | GroupChat + Actor | Handoffs + agents-as-tools | 图子节点 | 顺序/层次 |
| 护栏 | 智能体 + 工具级 | 无 | 无 | 仅智能体级 | 无 | 无 |
| 记忆 | AgentMemory + Workspace | 三层（工作/共享/长期） | 消息历史 | Sessions（SQLite/Redis） | Checkpoints + Store | SQLite + Entity |
| Python 版本 | >= 3.12 | >= 3.9 | >= 3.10 | >= 3.9 | >= 3.9 | >= 3.10 |
| 学术论文 | 本文 | ICLR 2024 Oral | arXiv 2023 | 无 | 无 | 无 |

### 2.2 深度研究系统

深度研究（Deep Research）代表了 AI 智能体自动化端到端研究工作流的新范式（Xu & Peng, 2025; Wang et al., 2025）。Xu & Peng（2025）的综合综述分析了 80+ 种实现，提出了四维分类体系：基础模型、工具利用、任务规划和知识综合。Wang et al.（2025）定义了三个演进阶段：智能搜索、集成研究和全栈 AI 科学家。

商业系统包括 OpenAI Deep Research（基于 o3 与强化学习）、Google Gemini Deep Research（百万 token 上下文）、Perplexity Deep Research 和 Anthropic Claude Research。开源替代方案包括 OpenDeepResearcher、Auto-Deep-Research 和 OpenManus。

Agentica 的 DeepAgent 模块以其双阈值上下文管理机制和 HEARTBEAT 式迭代控制为该领域做出贡献——这些工程创新解决了被识别为核心挑战的上下文溢出和过早终止问题。

### 2.3 智能体评估基准

**GAIA**（Mialon et al., 2023）在 466 个真实世界任务（165 个公开验证集，301 个私有测试集）上评估通用 AI 助手，需要多步推理、多模态处理和工具使用。人类基线约 92%；顶级 AI 系统平均不到 30%，凸显了巨大的能力差距。

**SWE-bench**（Jimenez et al., 2024）在来自 12 个 Python 仓库的真实 GitHub issue 上评估软件工程智能体。SWE-bench Verified（OpenAI, 2024）是 500 个人工验证子集。JoyCode Agent 当前以 74.6% 领先，表明智能体脚手架优于裸模型能力（Claude 3.7 Sonnet 无框架时为 70.3%）。

**Terminal-Bench**（Merrill et al., 2026）是 89 个在 Docker 终端环境中执行的高难度任务基准。任务涵盖软件工程（32.6%）、计算（21.3%）、ML/AI（11.2%）、网络安全（9.0%）、DevOps（4.5%）等领域。任务难度从数小时到数天的专家人力。当前最佳系统（GPT-5.3-Codex + Simple Codex agent）达到 75.1%。评估使用基于结果的容器状态检查，每配置至少 5 次运行（共 32,155 次），报告 95% 置信区间。

---

## 3. 架构

### 3.1 概述

Agentica 建立在三个设计原则之上：

1. **异步优先**：所有核心方法（`run()`、`response()`、`execute()`、`invoke()`）均为原生 `async`。同步适配器（`run_sync()`、`run_stream_sync()`）通过后台线程 + 事件循环模式包装异步实现。

2. **通过 mixin 分离关注点**：`Agent` 类使用 `@dataclass(init=False)` 与纯方法容器 mixin 的多重继承：

```python
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, SessionMixin,
            TeamMixin, ToolsMixin, PrinterMixin, MediaMixin):
```

每个 mixin 提供特定能力（执行、提示词、会话管理、多智能体委托、工具管理、输出格式化、媒体处理），不携带状态或 `__init__`。

3. **Pydantic 管数据，dataclass 管行为**：数据结构（`Model`、`Tool`、`Function`、`RunResponse`）使用 Pydantic `BaseModel`。`Agent` 类刻意使用 `@dataclass`——它有可变字段（Callable、lists）、复杂初始化和别名处理，不需要 Pydantic 的验证开销。

图 1 展示了整体架构。

```
┌─────────────────────────────────────────────────────────┐
│                      Agent 层                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │RunnerMix │ │PromptMix │ │SessionMix│ │ TeamMixin │  │
│  │ run()    │ │ sys_msg  │ │ load()   │ │ as_tool() │  │
│  │ stream() │ │ usr_msg  │ │ save()   │ │ transfer()│  │
│  └────┬─────┘ └──────────┘ └──────────┘ └───────────┘  │
│       │                                                 │
│  ┌────▼─────────────────────────────────────────────┐   │
│  │           _run_impl() [异步生成器]                │   │
│  │        统一执行引擎 + Langfuse 链路追踪            │   │
│  └────┬─────────────────────────────────────────────┘   │
├───────┼─────────────────────────────────────────────────┤
│       │            Model 层                              │
│  ┌────▼─────────────────────────────────────────────┐   │
│  │  Model（异步抽象：invoke, response,               │   │
│  │         invoke_stream, response_stream）           │   │
│  │                                                   │   │
│  │  ┌─────────────────────────────────────────────┐  │   │
│  │  │ run_function_calls() [asyncio.TaskGroup]    │  │   │
│  │  │  阶段 1：发送 started 事件（顺序）           │  │   │
│  │  │  阶段 2：并行执行（TaskGroup）               │  │   │
│  │  │  阶段 3：顺序处理结果                        │  │   │
│  │  └─────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Tool 层                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │   Tool   │───>│ Function │───>│   FunctionCall   │   │
│  │ (容器)   │    │(模式 +   │    │    (调用)        │   │
│  │          │    │ weakref) │    │   .execute()     │   │
│  └──────────┘    └──────────┘    │ 自动同步/异步    │   │
│                                  └──────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                  Memory 层                               │
│  ┌─────────────┐         ┌───────────────────────────┐  │
│  │ AgentMemory │         │      Workspace            │  │
│  │ (运行时,    │         │  AGENT.md | PERSONA.md    │  │
│  │  token      │         │  users/{id}/MEMORY.md     │  │
│  │  感知)      │         │  users/{id}/USER.md       │  │
│  └─────────────┘         └───────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                 Guardrail 层                             │
│  ┌──────────────────┐  ┌────────────────────────────┐   │
│  │  智能体级          │  │     工具级                  │   │
│  │  InputGuardrail   │  │  ToolInputGuardrail        │   │
│  │  OutputGuardrail  │  │  ToolOutputGuardrail       │   │
│  └──────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 形式化执行模型

我们将智能体执行循环形式化为状态转换系统，以支持对安全性质的严格推理。

**定义 1（智能体状态）。** 智能体状态是一个四元组 $S = (M, H, C, \sigma)$，其中：
- $M = [m_1, ..., m_k]$ 为消息历史
- $H \subseteq \mathcal{R}$ 为存储在 `AgentMemory.runs` 中的已完成运行集合
- $C \in \mathbb{N}$ 为估计的上下文 token 数
- $\sigma \in \{idle, running, terminated\}$ 为执行状态

**定义 2（执行步骤）。** 单个执行步骤将状态 $S_i$ 转换为 $S_{i+1}$：

$$S_i \xrightarrow{\text{LLM}} (r, T) \xrightarrow{\text{Execute}} S_{i+1}$$

其中 $r$ 为模型文本回复，$T = \{t_1, ..., t_n\}$ 为工具调用集合。执行分三个阶段：

- **阶段 1**：对每个 $t_j \in T$，发送 `started` 事件（顺序，保序）
- **阶段 2**：并发执行所有 $t_j$：$\text{results} = \text{TaskGroup}(\{t_1, ..., t_n\})$
- **阶段 3**：按原始顺序处理结果（顺序）

**定理 1（部分故障隔离）。** 在带逐任务异常捕获的 `asyncio.TaskGroup` 执行模型下，若工具调用 $t_j$ 抛出异常 $e_j$，则对所有 $t_k$（$k \neq j$）：(a) $t_k$ 不被取消，(b) $t_k$ 的结果（成功或其自身异常）被保留，(c) 结果以原始顺序 $[r_1, ..., r_n]$ 返回。

*证明概要。* 每个任务 $t_j$ 被包装在 `async def _execute_one(j, fc)` 中，该函数在本地捕获 `ToolCallException` 和通用 `Exception`，存储在 `results[j]` 中。`TaskGroup` 仅在*未捕获*异常逃逸任务时传播 `BaseExceptionGroup`。由于所有异常均在 `_execute_one` 内被捕获，TaskGroup 正常退出，所有任务完成。阶段 3 按索引顺序 $0, 1, ..., n-1$ 遍历 `results`，保持原始序列。$\square$

这严格强于 `asyncio.gather(return_exceptions=True)`——后者提供 (b) 和 (c)，但不提供所有任务在明确界定范围内完成的结构化生命周期保证。

**定义 3（终止条件）。** 智能体在以下任一条件满足时终止：(a) 模型不产生工具调用（自然完成），(b) 上下文 $C > \theta_h$（硬阈值），(c) 步数超过 `max_rounds`，(d) 抛出 `StopAgentRun` 流控异常。

### 3.3 结构化并发工具执行

当 LLM 在单次回复中返回 $N$ 个工具调用时，Agentica 分三阶段执行：

**阶段 1（顺序）：发送事件。** 对每个工具调用 $t_i$，按顺序发送 `tool_call_started` 事件，确保流式消费者获得确定性事件顺序。

**阶段 2（并行）：执行。** 所有 $N$ 个工具调用在 `asyncio.TaskGroup` 中并发执行：

```python
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(_execute_one(i, fc))
             for i, fc in enumerate(function_calls)]
```

每个任务在 try/except 中捕获异常，防止单个失败取消兄弟任务。

对于同步工具入口，`FunctionCall._call_func()` 通过 `inspect.iscoroutinefunction()` 自动检测并通过 `run_in_executor()` 路由：

```python
async def _call_func(self, func, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, **kwargs))
```

**阶段 3（顺序）：处理结果。** 按原始顺序处理结果，构建工具响应消息，处理流控异常（`StopAgentRun`、`RetryAgentRun`），发送 `tool_call_completed` 事件。

三阶段设计在阶段 2 实现最大并行性，同时在阶段 1 和 3 维持确定性顺序。

### 3.4 三层工具抽象

工具系统使用分离关注点的三层层次结构：

**Tool** 是将相关函数分组的容器（如 `BuiltinFileTool` 包含 `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`）。

**Function** 持有模式（名称、描述、JSON Schema 参数）和入口可调用对象。`Function.from_callable()` 工厂方法使用 `inspect.signature()` 和 `get_type_hints()` 从 Python 函数自动生成工具定义。关键设计：Function 通过**弱引用**（`weakref.ref`）持有对父 Agent 的引用，打破 `Agent → Model → functions → Function._agent → Agent` 循环引用链，避免内存泄漏。

`_safe_validate_call()` 包装 Pydantic 的 `validate_call`，剥离不可解析的前向引用注解（如 mixin 方法上的 `self: "Agent"`），使 Pydantic 验证能用于 PEP 563 延迟注解的方法。

**FunctionCall** 表示单次调用，包含参数、结果、错误和调用 ID。其 `execute()` 方法是唯一执行入口——仅异步，支持前/后钩子和第 3.3 节描述的同步/异步自动检测。

### 3.5 Agent-as-Session：统一状态模型

Agentica 与其他框架在会话状态管理上存在根本的设计分歧。我们识别出现有框架中的三种模式：

**模式 A：无状态 Runner + 外部 Session**（OpenAI Agents SDK）。Agent 为声明式配置对象；无状态 `Runner` 类驱动执行；独立的 `Session` 对象（SQLite 或 Redis 后端）存储对话历史。开发者须显式传递 session 引用：

```python
# OpenAI Agents SDK 模式
agent = Agent(name="assistant", model="gpt-4o", tools=[...])
result = await Runner.run(agent, "query", session=session)
# 没有 session，Agent 无法记住之前的交互
```

**模式 B：环境中介状态**（MetaGPT、AutoGen）。智能体通过共享环境或消息总线通信，状态分散在环境、各智能体记忆和通信通道中。

**模式 C：Agent-as-Session**（Agentica）。Agent 自身即为会话载体。`AgentMemory` 直接嵌入智能体内部，维护结构化运行历史：

```python
class AgentMemory(BaseModel):
    runs: List[AgentRun] = []      # 结构化运行历史
    messages: List[Message] = []    # 当前上下文窗口
    summary: Optional[SessionSummary] = None  # 压缩历史
```

关键机制是 `get_messages_from_last_n_runs()`，提供**token 预算感知的历史组装**：

1. 从最近到最早遍历运行记录
2. 对每次运行，提取对话消息（过滤工具调用）
3. 应用渐进式截断：越早的运行被更积极地截断
4. 当 token 预算耗尽时停止

该设计相对模式 A 的优势：

- **单一事实来源**：智能体拥有自己的状态，无需协调 runner、session 和 agent 对象
- **token 感知上下文组装**：历史消息自动受预算约束，防止多轮交互中的上下文溢出
- **渐进式信息衰减**：较早的运行被更积极地截断（工具结果摘要至 `tool_result_max_chars`），模拟人类记忆中近期事件回忆更详细的特性
- **零配置多轮对话**：多轮对话默认可用，无需显式会话管理代码

会话持久化由 `SessionMixin` 正交处理，通过 `write_to_storage()` 将智能体状态（含 `AgentMemory`）序列化到存储后端。这分离了状态所有权（Agent）和状态持久化（SessionMixin + 存储后端）的关注点。

**表 2：会话管理模式对比**

| 属性 | 无状态 Runner（OpenAI） | Agent-as-Session（本文） |
|------|------------------------|------------------------|
| 状态所有权 | 外部 Session 对象 | Agent 自身 |
| 多轮设置 | 显式传递 session | 自动（内置） |
| Token 预算 | 手动管理 | 自动（`max_tokens` 参数） |
| 历史截断 | 应用层 | 框架层（渐进式） |
| 持久化 | 必须有 Session 后端 | 可选（SessionMixin） |
| 序列化 | Session.to_dict() | AgentMemory.to_dict() → SessionRow |
