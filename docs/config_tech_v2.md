# Agentica Agent V2 架构升级方案（最终版）

> 融合 config_tech.md（渐进迁移）和原 config_tech_v2.md（断裂重构）两个方案的优势。
> 核心决策：**保持 dataclass 基类 + V2 参数分层 + V1 的 RunConfig + Session 剥离 + 一次性 breaking change**。

---

## 一、现状诊断

### 1.1 核心数据

| 指标 | 值 |
|------|---|
| Agent `__init__` 参数总数 | **57**（含 7 个兼容别名） |
| 去掉别名后实际参数 | **50** |
| 示例中使用过的参数 | **~25** |
| 90% 的示例只用到的参数 | `model` + `instructions` + `tools` |

### 1.2 使用频率 Top 10

| 参数 | 出现次数 | 说明 |
|------|---------|------|
| `model` | 25+ | 几乎所有示例 |
| `instructions` | 12 | 第二高频 |
| `name` | 9 | 多 Agent / Workflow |
| `response_model` | 7 | 结构化输出 |
| `tools` | 6 | 函数调用 |
| `description` | 5 | Agent 人格 |
| `add_history_to_messages` | 5 | 多轮对话 |
| `knowledge` | 5 | RAG |
| `search_knowledge` | 5 | RAG 搜索 |
| `debug` | 5 | 调试 |

### 1.3 核心问题

**不是"参数多"，而是混合了不同关注层次的概念到同一个构造函数。**

当前 Agent 同时承担了：
- Agent 定义（我是谁、我能做什么）
- Prompt 工程（怎么构建 system message）
- Session 管理（会话创建/恢复/持久化）
- Memory 管理（历史消息、用户记忆）
- 运行时状态（run_id, run_response）

---

## 二、设计决策与取舍

### 2.1 基类选择：保持 `@dataclass(init=False)`

**不迁移到 Pydantic BaseModel。** 原因：

1. 当前 Agent 使用 7 个 mixin 的直接多继承，Pydantic 的 `ModelMetaclass` 对 MRO 和 `__init__` 的接管与 mixin 体系有兼容性风险
2. Agent 持有 `Callable`、可变列表、`AgentMemory` 等复杂类型，dataclass 的宽松类型约束更适合
3. 手写 `__init__` 虽然长，但从 300 行精简到 ~80 行后完全可控
4. 数据模型（RunResponse, Tool, Function）已经用 BaseModel，Agent 作为"行为容器"用 dataclass 更合适

序列化需求通过 `to_dict()` / `from_dict()` 方法手动实现，不依赖 BaseModel。

### 2.2 兼容性策略：一次性 breaking change

项目处于 async-first 重构期，已经做了破坏性变更（去掉 `arun`/`aresponse`）。在此窗口期：
- 直接删除 7 个兼容别名，不做 deprecation 周期
- 直接删除废弃参数，不加 warning
- 所有 examples 和 tests 同步更新

### 2.3 参考框架

| 维度 | OpenAI SDK | MetaGPT | Agentica (当前) | Agentica V2 |
|------|-----------|---------|----------------|-------------|
| 构造参数 | ~15 | ~15 | ~57 | **~22** |
| Session 管理 | 外部 | Environment | Agent 内部 | **外部 SessionManager** |
| 运行时配置 | Runner 参数 | RoleContext | Agent 字段 | **RunConfig** |
| Prompt 构建 | instructions | goal+constraints | ~16 细粒度参数 | **PromptConfig 打包** |

---

## 三、V2 架构设计

### 3.1 设计原则

1. **Agent 是定义，不是容器** — 只描述"我是谁、我能做什么"
2. **分层配置** — 核心参数平铺，高级配置打包成 Config 对象
3. **运行时状态不暴露** — run_id, run_response 等内部管理
4. **RunConfig 管运行时** — stream、timeout 等每次 run 可能不同的参数独立出来
5. **Session 外置** — 由独立 SessionManager 管理，Agent 不持有 session/db 字段

### 3.2 参数分层架构

```
Agent V2 (~22 参数)
│
├── 第一层：核心定义（开发者必须知道的）
│   ├── model              # 模型
│   ├── name               # 名字
│   ├── agent_id           # 唯一标识
│   ├── instructions       # 指令 (str | List[str] | Callable)
│   ├── description        # 描述（团队中用于交接说明）
│   ├── tools              # 工具列表
│   ├── knowledge          # 知识库
│   ├── team               # 团队成员
│   ├── workspace          # 工作空间
│   └── response_model     # 结构化输出类型
│
├── 第二层：常用配置（按需设置，有合理默认值）
│   ├── add_history_to_messages  # 多轮对话
│   ├── num_history_responses    # 历史轮数
│   ├── search_knowledge         # Agentic RAG
│   ├── output_language          # 输出语言
│   ├── markdown                 # Markdown 格式
│   ├── structured_outputs       # 结构化输出模式
│   ├── debug               # 调试模式
│   └── monitoring               # 监控（Langfuse）
│
├── 第三层：打包配置（高级用户）
│   ├── prompt_config: PromptConfig      # Prompt 工程细节
│   ├── tool_config: ToolConfig          # 工具调用细节
│   ├── memory_config: MemoryConfig      # 记忆配置
│   └── team_config: TeamConfig          # 团队协作细节
│
├── 运行时配置（不在 Agent 构造函数，在 run() 调用处）
│   └── RunConfig  # stream, timeout, response_model 覆盖, tool_choice 覆盖
│
└── 外置模块（不在 Agent 构造函数）
    ├── SessionManager     # 会话持久化 CRUD
    └── RunResponse        # 运行结果
```

### 3.3 核心类定义

```python
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, TeamMixin, ToolsMixin, PrinterMixin):
    """AI Agent — 定义智能体的身份和能力。

    Agent 只关心"我是谁、我能做什么"，不管会话持久化。
    Session 管理由外部 SessionManager 处理。

    Example - 最简用法:
        >>> agent = Agent(instructions="You are a helpful assistant.")
        >>> response = await agent.run("Hello!")

    Example - 完整配置:
        >>> agent = Agent(
        ...     name="analyst",
        ...     model=OpenAIChat(id="gpt-4o"),
        ...     instructions="You are a data analyst.",
        ...     tools=[web_search, calculator],
        ...     knowledge=my_knowledge,
        ...     response_model=AnalysisReport,
        ... )
    """

    # ============================
    # 第一层：核心定义（~10 参数）
    # ============================
    model: Optional[Model] = None
    name: Optional[str] = None
    agent_id: str = ""                     # default_factory in __init__
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None
    knowledge: Optional[Knowledge] = None
    team: Optional[List["Agent"]] = None
    workspace: Optional[Workspace] = None
    response_model: Optional[Type[Any]] = None

    # ============================
    # 第二层：常用配置（~8 参数）
    # ============================
    add_history_to_messages: bool = False
    num_history_responses: int = 3
    search_knowledge: bool = True
    output_language: Optional[str] = None
    markdown: bool = False
    structured_outputs: bool = False
    debug: bool = False
    monitoring: bool = False

    # ============================
    # 第三层：打包配置
    # ============================
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    team_config: TeamConfig = field(default_factory=TeamConfig)

    # ============================
    # 运行时（内部管理，不暴露给构造函数）
    # ============================
    memory: AgentMemory = field(default_factory=AgentMemory)
    run_id: Optional[str] = field(default=None, init=False, repr=False)
    run_input: Optional[Any] = field(default=None, init=False, repr=False)
    run_response: RunResponse = field(default_factory=RunResponse, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)

    def __init__(self, *, model=None, name=None, agent_id=None, ...):
        # ~80 行，只做字段赋值 + workspace_path 快捷方式
        self.agent_id = agent_id or str(uuid4())
        self.model = model
        self.name = name
        # ... 其余字段直接赋值，无别名映射 ...
```

### 3.4 RunConfig（从 config_tech.md 方案吸收）

```python
@dataclass
class RunConfig:
    """Per-run configuration. Overrides Agent defaults when provided.

    把"每次 run 可能不同"的参数从 Agent 构造函数移到 run() 调用处。
    Agent 上的同名字段（如 response_model）仍作为默认值，RunConfig 覆盖。

    Example:
        >>> response = await agent.run("分析数据", config=RunConfig(
        ...     stream=True,
        ...     run_timeout=30,
        ...     response_model=AnalysisReport,
        ... ))
    """
    stream: bool = False
    stream_intermediate_steps: bool = False
    response_model: Optional[Type[Any]] = None
    structured_outputs: Optional[bool] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_timeout: Optional[float] = None
    first_token_timeout: Optional[float] = None
    save_response_to_file: Optional[str] = None
```

**为什么需要 RunConfig：**
- `stream`/`timeout` 是"这一次运行"的配置，不是 Agent 的固有属性
- 避免 `run()` 签名无限膨胀——新增运行时参数只需加到 RunConfig
- Agent 上的 `response_model`、`structured_outputs` 是默认值，RunConfig 可以覆盖

### 3.5 Config 对象定义

```python
@dataclass
class PromptConfig:
    """Prompt 构建相关的高级配置。

    大部分用户只需 Agent.instructions，以下参数供高级定制。
    """
    # 自定义 system prompt（覆盖默认构建逻辑）
    system_prompt: Optional[Union[str, Callable]] = None
    system_prompt_template: Optional[PromptTemplate] = None
    use_default_system_message: bool = True
    system_message_role: str = "system"
    user_message_role: str = "user"
    user_prompt_template: Optional[PromptTemplate] = None
    use_default_user_message: bool = True

    # System message 构建细节
    task: Optional[str] = None
    role: Optional[str] = None
    guidelines: Optional[List[str]] = None
    expected_output: Optional[str] = None
    additional_context: Optional[str] = None
    introduction: Optional[str] = None
    references_format: Literal["json", "yaml"] = "json"

    # Prompt 行为开关
    add_name_to_instructions: bool = False
    add_datetime_to_instructions: bool = True
    prevent_hallucinations: bool = False
    prevent_prompt_leakage: bool = False
    limit_tool_access: bool = False
    enable_agentic_prompt: bool = False


@dataclass
class ToolConfig:
    """工具调用相关的高级配置。"""
    support_tool_calls: bool = True
    tool_call_limit: Optional[int] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    auto_load_mcp: bool = False
    read_chat_history: bool = False
    read_tool_call_history: bool = False
    update_knowledge: bool = False
    add_references: bool = False
    compress_tool_results: bool = False
    compression_manager: Optional[Any] = None


@dataclass
class MemoryConfig:
    """记忆与 Workspace 相关的高级配置。"""
    load_workspace_context: bool = True
    load_workspace_memory: bool = True
    memory_days: int = 2


@dataclass
class TeamConfig:
    """团队协作相关的高级配置。"""
    role: Optional[str] = None
    respond_directly: bool = False
    add_transfer_instructions: bool = True
    team_response_separator: str = "\n"
```

### 3.6 SessionManager（从 Agent 剥离）

**设计选择：** SessionManager 只管数据加载/保存，`agent.run()` 仍然是执行主体。
不引入 `sm.run()` 方法，避免"两种 run 方式"的认知负担。

```python
class SessionManager:
    """独立的会话管理器。

    Agent 是无状态定义，SessionManager 管理会话的生命周期。
    SessionManager 只负责 load/save，不参与执行。

    Example:
        >>> agent = Agent(instructions="You are helpful.")
        >>> sm = SessionManager(db=SqliteDb("agent.db"))
        >>>
        >>> # 创建新会话
        >>> session = await sm.create_session(agent, user_id="alice")
        >>>
        >>> # 加载会话 → 运行 → 保存
        >>> await sm.load(agent, session)
        >>> response = await agent.run("Hello!")
        >>> await sm.save(agent, session)
        >>>
        >>> # 恢复已有会话
        >>> session = await sm.get_session(session_id="xxx")
        >>> await sm.load(agent, session)
        >>> response = await agent.run("Continue our conversation")
        >>> await sm.save(agent, session)
    """

    def __init__(self, db: BaseDb, *, user_id: Optional[str] = None):
        self.db = db
        self.user_id = user_id

    async def create_session(
        self, agent: Agent, user_id: Optional[str] = None
    ) -> Session:
        """创建新会话。"""
        ...

    async def get_session(self, session_id: str) -> Session:
        """从 DB 获取会话元数据。"""
        ...

    async def load(self, agent: Agent, session: Session) -> None:
        """从 DB 加载会话，恢复 agent.memory 状态。"""
        ...

    async def save(self, agent: Agent, session: Session) -> None:
        """把 agent.memory 持久化到 DB。"""
        ...

    async def list_sessions(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Session]:
        ...

    async def delete_session(self, session_id: str) -> None:
        ...

    async def rename_session(self, session_id: str, name: str) -> None:
        ...


@dataclass
class Session:
    """会话数据对象。"""
    session_id: str
    agent_id: str
    user_id: Optional[str] = None
    session_name: Optional[str] = None
    memory: AgentMemory = field(default_factory=AgentMemory)
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

### 3.7 Agent.run() 接口

```python
class Agent:
    async def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        audio: Optional[Any] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
    ) -> RunResponse:
        """执行一轮对话。

        如果需要多轮对话且不需要持久化，Agent 内部 memory 自动维护上下文。
        如果需要持久化，使用 SessionManager.load() / save() 在 run 前后管理。
        """
        ...

    async def run_stream(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        audio: Optional[Any] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
    ) -> AsyncIterator[RunResponse]:
        ...

    def run_sync(self, message=None, **kwargs) -> RunResponse:
        """同步适配器，内部调用 run()。"""
        ...

    def run_stream_sync(self, message=None, **kwargs) -> Iterator[RunResponse]:
        """同步流式适配器。"""
        ...
```

---

## 四、删除清单

### 4.1 直接删除的参数

| 参数 | 理由 | 去向 |
|------|------|------|
| `llm` | 兼容别名 | 用 `model` |
| `knowledge_base` | 兼容别名 | 用 `knowledge` |
| `add_chat_history_to_messages` | 兼容别名 | 用 `add_history_to_messages` |
| `add_knowledge_references_to_prompt` | 兼容别名 | 用 `ToolConfig.add_references` |
| `output_model` | 兼容别名 | 用 `response_model` |
| `output_file` | 兼容别名 | 用 `RunConfig.save_response_to_file` |
| `debug` | 兼容别名 | 用 `debug` |
| `enable_user_memories` | 已废弃 | Workspace 替代 |
| `images` | 构造时不需要 | `run(images=...)` |
| `videos` | 构造时不需要 | `run(videos=...)` |
| `agent_data` | 无使用场景 | 删除 |
| `user_data` | 属于会话 | `Session.state` |
| `session_id` | 属于会话 | `SessionManager` |
| `session_name` | 属于会话 | `SessionManager` |
| `session_state` | 属于会话 | `Session.state` |
| `session_data` | 属于会话 | `Session` |
| `db` | 属于会话 | `SessionManager` |
| `user_id` | 属于会话 | `SessionManager` / `Session` |
| `stream` | 运行时配置 | `run()` vs `run_stream()` 区分 |
| `stream_intermediate_steps` | 运行时配置 | `RunConfig` |
| `run_timeout` | 运行时配置 | `RunConfig` |
| `first_token_timeout` | 运行时配置 | `RunConfig` |
| `save_response_to_file` | 运行时配置 | `RunConfig` |
| `add_messages` | 运行时输入 | `run(messages=...)` |
| `context` | 运行时输入 | `run()` 内部处理 |
| `add_context` | 内部逻辑 | 有 context 就用 |
| `resolve_context` | 内部逻辑 | 删除 |
| `parse_response` | 内部逻辑 | 有 response_model 自动 parse |
| `workspace_path` | 快捷方式 | `workspace=Workspace(path)` |
| `retriever` | Knowledge 内部 | `Knowledge` 自行管理 |

### 4.2 移入 Config 对象的参数

| 从 Agent 顶层 | 移入 |
|--------------|------|
| `system_prompt`, `system_prompt_template`, `use_default_system_message`, `system_message_role`, `user_message_role`, `user_prompt_template`, `use_default_user_message` | `PromptConfig` |
| `task`, `role`, `guidelines`, `expected_output`, `additional_context`, `introduction` | `PromptConfig` |
| `add_name_to_instructions`, `add_datetime_to_instructions`, `prevent_hallucinations`, `prevent_prompt_leakage`, `limit_tool_access`, `enable_agentic_prompt` | `PromptConfig` |
| `references_format` | `PromptConfig` |
| `support_tool_calls`, `tool_call_limit`, `tool_choice`, `auto_load_mcp` | `ToolConfig` |
| `read_chat_history`, `read_tool_call_history`, `update_knowledge`, `add_references` | `ToolConfig` |
| `compress_tool_results`, `compression_manager` | `ToolConfig` |
| `load_workspace_context`, `load_workspace_memory`, `memory_days` | `MemoryConfig` |
| `respond_directly`, `add_transfer_instructions`, `team_response_separator` | `TeamConfig` |

---

## 五、Mixin 重构

### 5.1 继承链变化

```python
# 当前 (V1)
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, SessionMixin, TeamMixin, ToolsMixin, PrinterMixin, MediaMixin)

# V2
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, TeamMixin, ToolsMixin, PrinterMixin)
```

删除：
- **SessionMixin** — 全部迁移到独立 `SessionManager`
- **MediaMixin** — `images`/`videos` 移到 `run()` 参数，`add_image()`/`add_video()` 方法不再需要

### 5.2 RunnerMixin 瘦身

`_run_impl` 中移除 session IO 步骤：

```
# 当前 _run_impl 流程
1. self.update_model()
2. self._resolve_context()
3. await self.read_from_storage()      ← 删除
4. self.add_introduction()
5. await self.get_messages_for_run()
6. await self.model.response()
7. self.memory.add_messages()
8. await self.write_to_storage()       ← 删除
9. self.save_run_response_to_file()

# V2 _run_impl 流程（纯执行）
1. update_model()
2. get_messages_for_run()
3. model.response() / model.response_stream()
4. memory.add_messages()
5. return RunResponse
```

Session 加载/保存由调用方通过 SessionManager 控制。

### 5.3 PromptsMixin 适配

Mixin 内部从 `self.prevent_hallucinations` 改为读取 `self.prompt_config.prevent_hallucinations`。
同理其他被移入 Config 的字段。这是机械性替换，不涉及逻辑变化。

### 5.4 Mixin 状态总览

| Mixin | 状态 | 说明 |
|-------|------|------|
| PromptsMixin | **适配** | 从 `self.xxx` 改为 `self.prompt_config.xxx` |
| RunnerMixin | **瘦身** | 移除 session IO，支持 RunConfig 参数 |
| TeamMixin | **适配** | 从 `self.xxx` 改为 `self.team_config.xxx` |
| ToolsMixin | **适配** | 从 `self.xxx` 改为 `self.tool_config.xxx` |
| PrinterMixin | 保持不变 | |
| SessionMixin | **删除** | → `SessionManager` |
| MediaMixin | **删除** | → `run()` 参数 |

---

## 六、用户 API 对比

### 6.1 最简用法（无变化）

```python
agent = Agent(model=OpenAIChat(id="gpt-4o"))
response = agent.run_sync("Hello!")
```

### 6.2 带工具和指令（无变化）

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="analyst",
    instructions="You are a data analyst.",
    tools=[web_search, calculator],
    description="Data analysis expert",
)
```

### 6.3 多轮对话 + 持久化

```python
# V1 — Session 混在 Agent 里
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are helpful.",
    db=SqliteDb("agent.db"),
    session_id="session-123",
    user_id="alice",
    add_history_to_messages=True,
)
response = agent.run_sync("Hello!")
response = agent.run_sync("Continue...")

# V2 — Session 独立管理
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are helpful.",
    add_history_to_messages=True,
)
sm = SessionManager(db=SqliteDb("agent.db"))
session = await sm.create_session(agent, user_id="alice")
await sm.load(agent, session)
response = await agent.run("Hello!")
response = await agent.run("Continue...")
await sm.save(agent, session)

# V2 快捷方式 — 不需要持久化时
agent = Agent(instructions="You are helpful.", add_history_to_messages=True)
response = agent.run_sync("Hello!")       # 内部 memory 自动维护上下文
response = agent.run_sync("Continue...")  # 自动保持多轮
```

### 6.4 RAG（无变化）

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=my_knowledge,
    search_knowledge=True,
    add_history_to_messages=True,
    markdown=True,
)
```

### 6.5 高级 Prompt 定制

```python
# V1 — 所有参数平铺
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="...",
    prevent_hallucinations=True,
    prevent_prompt_leakage=True,
    add_datetime_to_instructions=True,
    output_language="zh",
    markdown=True,
    system_message_role="system",
)

# V2 — 常用的保留顶层，其余打包
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="...",
    output_language="zh",
    markdown=True,
    prompt_config=PromptConfig(
        prevent_hallucinations=True,
        prevent_prompt_leakage=True,
        add_datetime_to_instructions=True,
        system_message_role="system",
    ),
)
```

### 6.6 团队

```python
# V1
leader = Agent(
    name="leader",
    team=[analyst, writer],
    add_transfer_instructions=True,
    respond_directly=False,
)

# V2
leader = Agent(
    name="leader",
    team=[analyst, writer],
    team_config=TeamConfig(
        add_transfer_instructions=True,
        respond_directly=False,
    ),
)
```

### 6.7 RunConfig 运行时覆盖（新能力）

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="You are a data analyst.",
    response_model=BriefReport,        # Agent 默认输出类型
)

# 这次要流式 + 不同输出类型 + 超时控制
response = await agent.run(
    "分析这段代码",
    config=RunConfig(
        stream=True,
        response_model=DetailedReport,  # 覆盖默认
        run_timeout=30,
    ),
)

# 不传 config 时使用 Agent 上的默认值
response = await agent.run("快速分析")  # 用 BriefReport
```

---

## 七、迁移计划

### Phase 1：参数分层 + Config 对象 + RunConfig

1. 创建 `PromptConfig`, `ToolConfig`, `MemoryConfig`, `TeamConfig` 四个 dataclass → `agentica/agent/config.py`
2. 创建 `RunConfig` dataclass → `agentica/run_config.py`
3. 重写 Agent `__init__`：删除别名、删除废弃参数、将高级参数收入 Config 对象
4. 适配所有 Mixin：`self.xxx` → `self.prompt_config.xxx` / `self.tool_config.xxx` 等
5. `images`/`videos`/`add_messages` 从构造参数移到 `run()` 参数
6. `run()` / `run_stream()` 签名增加 `config: Optional[RunConfig] = None`
7. 运行时字段（`run_id`, `run_input`, `run_response`）保留为 `field(init=False)`
8. 删除 `MediaMixin`，其逻辑内联到 RunnerMixin
9. 更新所有 tests 和 examples

### Phase 2：Session 剥离

1. 创建独立 `SessionManager` 类 → `agentica/session.py`
2. 创建 `Session` 数据对象
3. 将 `SessionMixin` 的方法迁移到 `SessionManager`
4. 从 Agent 删除 session 相关字段：`session_id`, `session_name`, `session_state`, `session_data`, `db`, `user_id`, `user_data`, `_agent_session`
5. 从继承链中移除 `SessionMixin`
6. RunnerMixin 中移除 `read_from_storage()` / `write_to_storage()` 调用
7. Agent 保留内部 `memory: AgentMemory` 用于无持久化的临时多轮对话
8. 更新所有 examples

### Phase 3：清理 + 测试

1. 删除无用参数：`agent_data`, `resolve_context`, `add_context`, `parse_response`
2. 删除 `enable_user_memories`
3. `workspace_path` 快捷方式内联到 `__init__`（`if workspace_path: self.workspace = Workspace(path=workspace_path)`）
4. 全量跑 `pytest tests/ -v`
5. 更新所有 examples 和 README

---

## 八、最终 Agent 签名一览

Phase 3 完成后：

```python
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, TeamMixin, ToolsMixin, PrinterMixin):
    def __init__(
        self,
        *,
        # ---- 核心定义 ----
        model: Optional[Model] = None,
        name: Optional[str] = None,
        agent_id: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[Union[str, List[str], Callable]] = None,
        tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
        knowledge: Optional[Knowledge] = None,
        team: Optional[List["Agent"]] = None,
        workspace: Optional[Union[Workspace, str]] = None,  # str → Workspace(path=str)
        response_model: Optional[Type[Any]] = None,
        # ---- 常用配置 ----
        add_history_to_messages: bool = False,
        num_history_responses: int = 3,
        search_knowledge: bool = True,
        output_language: Optional[str] = None,
        markdown: bool = False,
        structured_outputs: bool = False,
        debug: bool = False,
        monitoring: bool = False,
        # ---- 打包配置 ----
        prompt_config: Optional[PromptConfig] = None,
        tool_config: Optional[ToolConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        team_config: Optional[TeamConfig] = None,
        # ---- 运行时内部 ----
        memory: Optional[AgentMemory] = None,
    ):
        ...
```

**参数计数：10（核心）+ 8（常用）+ 4（Config）+ 1（memory）= 23 个。**

---

## 九、风险控制

| 风险 | 概率 | 缓解方案 |
|------|------|---------|
| 外部用户在用别名参数 | 低（早期项目） | 直接删除，CHANGELOG 说明 |
| Session 剥离导致多轮对话 API 变化 | 高 | Agent 保留内部 memory，无 SessionManager 时自动临时多轮 |
| Mixin 内部 `self.xxx` 大量改为 `self.prompt_config.xxx` | 中 | 机械性替换，每个 Phase 跑完整 tests |
| RunConfig 与 Agent 默认值的合并逻辑 | 低 | `_run_impl` 开头做 `config = config or RunConfig()`，字段为 None 时 fallback 到 Agent 默认 |
| dataclass 不支持 `model_dump_json()` 序列化 | 低 | 手动实现 `to_dict()` / `from_dict()`，满足配置导出需求 |

---

## 十、预期收益

| 指标 | V1（当前） | V2 |
|------|-----------|-----|
| 构造参数数 | 57 | **23**（核心可见） |
| `__init__` 代码行数 | 300+ | **~80** |
| 新手理解成本 | 读半天文档 | 看前 10 个参数即可上手 |
| IDE 提示 | 刷 3 屏 | 一屏搞定 |
| Session 管理 | 与 Agent 耦合 | 独立模块，可独立测试 |
| 运行时配置 | 构造时固定 | RunConfig 按需覆盖 |
| 并发安全 | Agent 持有运行时状态，不可并发 | 运行时状态在栈帧内，可并发（Phase 3 后） |
| 多轮对话 | 必须在构造时配置 db/session_id | 运行时通过 SessionManager 管理 |
| Mixin 继承链 | 7 个 | **5 个** |
