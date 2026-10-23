# 架构与内部实现（开发者选读）

> **阅读提示**：本文档面向想要了解 Agentica 内部执行机制、Prompt Cache 优化原理、上下文压缩底层不变量以及参与本仓库开发的工程师。**日常使用 CLI、Web 或 Python SDK 无需阅读本页**，请直接看 [快速上手](../index.md)。

---

## 五层架构总览

Agentica 采用自底向上的模块化分层设计：

```text
+-------------------------------------------------------------------+
|                        编排层 (Orchestration)                      |
|   Agent.as_tool()  ·  Subagent (task)  ·  Workflow  ·  Swarm      |
+-------------------------------------------------------------------+
|                           代理层 (Agent)                           |
|   Agent / DeepAgent  ·  Runner (Agentic Loop)  ·  PromptsMixin    |
+-------------------------------------------------------------------+
|                工具层 (Tools)       |        记忆层 (Memory)        |
|   Function / Tool / @tool          |   WorkingMemory / SessionLog |
|   Builtin Tools / MCP              |   Workspace (MEMORY/AGENTS)  |
+-------------------------------------------------------------------+
|                     安全与可观测层 (Safety & Trace)                 |
|   Guardrails  ·  OS Sandbox  ·  ApprovalRegistry  ·  Langfuse v5  |
+-------------------------------------------------------------------+
|                           模型层 (Model)                           |
|   OpenAIChat / OpenAIResponses / Claude / DeepSeek / Ollama / ... |
+-------------------------------------------------------------------+
```

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Agent Loop 架构图" />
</div>

---

## Runner 执行循环 (`agentica/runner/`)

每次 `Agent.run()` / `Agent.run_stream()` 都由 `Runner` 驱动：

```text
User Message
    │
    ▼
[InputGuardrail]
    │
    ▼
Runner._run_impl()
    ├─► 构建 System Prompt (PromptsMixin.get_system_message)
    │       ├── Static Zone:      description / instructions / guidelines
    │       ├── Semi-static Zone: workspace context (AGENTS.md) + skills 目录（首轮冻结）
    │       └── Volatile Zone:    冻结的 memory/experiences + YYYY-MM-DD 日期
    │
    ├─► 上下文压缩检查 (CompressMixin)
    │       ├── Layer 1: evict_context() — 80% 水位起免费淘汰旧工具结果与超长参数
    │       └── Layer 2: CompressionManager.auto_compact() — 95% 水位换空窗 + 重写 notes
    │
    ├─► 模型调用 (RetryMixin._call_with_retry)
    │       └── 自动处理流式首包重试、跨 provider fallback、thinking 签名修复、reactive compact
    │
    ▼
Model Response
    ├─[有 tool_calls]─► 拆分为并发组 (concurrency_safe=True → asyncio.gather)
    │                   与串行组 (按顺序执行，支持 sibling-abort)
    │                   ├─► ToolInputGuardrail & Approval 审批检查
    │                   ├─► 执行工具 → Layer 0 输出预算限流 (tool_result_storage)
    │                   ├─► ToolOutputGuardrail
    │                   └─► 追加结果并继续下一轮循环（中途可注入 steer / peer 消息）
    │
    └─[无 tool_calls]─► OutputGuardrail ─► 写入 WorkingMemory & SessionLog ─► 返回 RunResponse
```

---

## System Prompt 前缀缓存与会话快照 (`agentica/agent/prompts.py`)

Prompt Cache（无论是 Anthropic 的显式 `cache_control` 断点，还是 OpenAI 的隐式前缀匹配 + `prompt_cache_key`）都要求**连续请求的前缀逐字节一致**。因为 System Message 位于后续每一个消息缓存断点的最前面，一旦 System Message 里有任何一行每轮变动，失效的不只是系统提示词本身，而是**整段历史对话的缓存全部重算**。

因此，Agentica 的设计原则是**让整个 System Message 在一次会话期间保持静止**：

1. **会话首轮一次性冻结快照**：
   - `Workspace.freeze_snapshots()` 在第一轮运行时冻结 `AGENTS.md`、记忆索引（`MEMORY.md`）和经验卡片（`EXPERIENCE.md`）。
   - `Agent.freeze_session_guidance()` 冻结技能目录（避免后台技能升级钩子中途刷新系统提示词导致缓存失效）。
   - 快照中会附带磁盘文件的真实路径说明，告知模型若需最新状态可自行调用 `read_file`。
2. **不往 System Prompt 注入每轮变化的 Git 状态**：
   - 分支和脏文件列表每经过一次编辑就会改变。Agentica 不在系统提示词里塞 `git status`，模型需要时自己调工具查看。
3. **日期精度到天（`YYYY-MM-DD`）**，放在 `VOLATILE_SYSTEM_MARKER` 之后的末尾。
4. **连接池每轮关闭，亲和性靠应用层路由**：
   - CLI 每轮运行在独立的事件循环中，`Runner` 在每轮结束调用 `Model.close_client()`。因此缓存亲和性不依赖底层的 HTTP 连接复用，而是依赖 OpenAI 的 `prompt_cache_key`（默认填入 `session_id`）或代理网关的粘性路由头（`default_headers` / `cache_control_session_header`）。

---

## 上下文压缩的底层机制与不变量 (`agentica/compression/`)

### 1. 为什么淘汰和打包以「单条工具结果」为单位？

OpenAI 与 Anthropic 对工具结果的协议形状完全不同：
- **OpenAI 系**：一条工具结果是一条独立的 `role="tool"` 消息。
- **Anthropic (`Claude`)**：同一轮的所有工具结果被打包进**一条** `role="user"` 消息的 `content` 数组（每个元素是 `{"type": "tool_result", "tool_use_id": ...}`）。

如果只扫描 `role == "tool"`，Layer 1 在 Claude 上就永远不会生效。因此 `evict.py` 使用 `carries_tool_results()` / `tool_result_blocks()` 按单条结果遍历，并通过发起调用的 assistant `tool_calls` 反查 `tool_use_id` 对应的工具名。同理，Layer 2 在保留「最后一条 user 消息起的尾部」时，必须跳过承载 `tool_result` 的 user 消息，否则会留下失去 `tool_use` 配对的孤儿结果块，导致 API 报 400。

### 2. 保护当前工具轮次 (`live_tool_round_start`)

Layer 1 绝不淘汰模型还没看过的尾部工具结果，也绝不收缩正在执行的调用参数。`live_tool_round_start` 在定位尾部边界时会先跳过带 `_injected=True` 标记的中途插话消息（`_inject_steering` / `_inject_peer_messages`），防止插话把边界抬高而误伤刚返回的工具结果。

此外，被收缩的超长工具参数叶节点会被替换为 JSON 对象 `{"$evicted": N}`，而不是字符串标记（防止模型把字符串占位符当成正文内容抄进下一次 `write_file` 或 `send_message`）。

### 3. 水位校准 (`ContextCalibration`)

本地 `count_tokens`（基于 tiktoken）看不见服务端额外计入的开销（如 OpenAI Responses 回放的加密 reasoning 项）。各模型适配器将服务端返回的完整输入 token 数写入 `Message.metrics["context_tokens"]`，`ContextCalibration` 计算 `max(1.0, 实测 / 本地估算)` 比例，后续水位判断全部乘以该比例（只向上放大，不向下缩小）。

### 4. 三处消息存储的一致性

换窗（Layer 2）发生时，以下三处存储必须同步更新：
1. **`working_memory.runs[].response.messages`**：下一轮请求的 prompt 来源，必须调用 `WorkingMemory.collapse_runs()` 折叠旧轮次。
2. **`working_memory.messages`**：供 `/history`、`/export` 读取。
3. **`<session>.jsonl`**：必须写入空的 `compact_boundary` 并追加保留的尾部消息，否则 `/resume` 时会从头回放压缩前的全部历史。同时 `LoopState.context_collapsed` 置为 `True`，回合结束时直接持久化存活列表，不再按压缩前的旧下标切片。

---

## 配置解析与 CLI 装配内部链路 (`agentica/config/` & `agentica/cli/`)

- **核心读写函数（`agentica/config/profiles.py`）**：`global_config_path`、`load_global_config`、`save_global_config`、`get_profile`、`get_profiles`、`get_active_profile_name`、`set_active_profile`、`upsert_profile`、`delete_profile`、`resolve_active_profile_name`、`apply_global_config`。写操作使用 `ruamel.yaml` round-trip 保留注释。
- **Profile 解析顺序**：`resolve_active_profile_name(work_dir)` 按 `项目级 override (project.json)` > `config.yaml active_profile` > `default` 解析。
- **CLI 会话级状态隔离**：`/reasoning`、`/statusbar`、`/debug`、`/peername`、`/permissions`、`/tools` 存储在 `~/.agentica/projects/<user>/<slug>/project.json` 与 `<session_id>.meta.json` 中，不污染全局 `config.yaml`。

---

## Gateway 审批与鉴权内部设计 (`agentica/gateway/`)

1. **机器令牌与浏览器会话分离**：
   - 每次启动生成的随机 `token` 写入 `0600` 的 `runtime.json`，仅供本机进程（Desktop 壳、CLI、本机脚本）通过 `Authorization: Bearer` 调用。
   - 浏览器登录后下发 `agentica_session` HttpOnly Cookie，服务端 `auth.json` 只存其 SHA-256 摘要，密码使用 `hashlib.scrypt` 存储。这样重启 Gateway 不会让浏览器掉线，修改密码也能立即吊销其他浏览器的会话。
2. **工具审批绑定在 `LiveTurn` 上 (`gateway/services/live_turn.py`)**：
   - `ApprovalRegistry` 挂在当前正在运行的 `LiveTurn` 上，而不挂在 LRU 缓存的 `Agent` 实例或当前 SSE 连接数上。
   - 浏览器刷新瞬间 SSE 订阅数为 0，但 `LiveTurn` 仍在运行，审批卡不会被误拒；重连时通过 `GET /api/chat/runs/{run_id}/events?after=` 按序号回放事件并调用 `republish_pending_approvals()` 重新推送未决审批卡。
   - 没有交互前端的通道（IM、定时任务、非流式 `/api/chat`）没有 `LiveTurn`，需要审批的调用会立即返回 `deny`，不会挂死后台协程。

---

## 延伸阅读

- [内置工具设计判据](../concepts/builtin-tool-bar.md) — 什么能力应该放进内置工具、CLI 与 SDK 的物理边界
- [Shell Hooks 协议](../getting-started/shell-hooks.md) / [Notify Sink 协议](../getting-started/notify-sink.md) / [Attach Socket 协议](../getting-started/attach.md)
