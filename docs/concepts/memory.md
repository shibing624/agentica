# Memory & Workspace

Agentica 提供两层记忆系统：**运行时记忆（WorkingMemory）** 和 **持久化记忆（Workspace）**。

## 运行时记忆：WorkingMemory

管理当前会话的消息历史，支持 token 感知的截断。

```python
from agentica import Agent

agent = Agent(
    add_history_to_context=True,   # 将历史加入上下文
    num_history_turns=5,            # 保留最近 5 轮
)
```

### 会话摘要

WorkingMemory 支持自动生成会话摘要，在多轮对话后保留关键信息：

```python
from agentica.memory import WorkingMemory

agent = Agent(
    working_memory=WorkingMemory(
        create_session_summary=True,            # 每轮结束后生成摘要
        update_session_summary_after_run=True,  # 自动更新
        max_messages=200,                       # 消息软上限（FIFO 淘汰）
    ),
)
```

会话摘要会注入到 System Prompt 末尾，同时被 `CompressionManager.auto_compact()` 直接复用——压缩时无需额外 LLM 调用。

---

## 持久化记忆：Workspace

基于文件的持久化记忆，存储跨会话的用户偏好、项目上下文和反馈记录：

```
workspace/
+-- users/
    +-- {user_id}/   # 多用户隔离（CLI 的 user_id 是 default）
        +-- AGENTS.md        # 这个 user 的常驻规则（人可改，agent 也可用 apply_patch 改）
        +-- MEMORY.md        # 记忆索引（仅存条目链接，≤200行/25KB）
        +-- memory/          # 记忆内容文件（每条独立 .md）
            +-- feedback_python_style.md
            +-- project_deadline.md
            +-- user_background.md
        +-- conversations/   # 对话归档
            +-- 2026-04-01.md
```

没有 workspace 根目录的 `AGENTS.md`。常驻规则的真实文件在 `users/{user_id}/AGENTS.md`；默认 CLI 用户额外提供 `~/.agentica/AGENTS.md` symlink，指向 `users/default/AGENTS.md`，用于兼容主流 coding agent。项目规则在 repo 根的 `AGENTS.md` 链。

### 基本用法

```python
from agentica import Agent, Workspace

agent = Agent(
    workspace=Workspace(path="./my_workspace", user_id="alice"),
)
```

---

## 记忆写入：write_memory_entry()

推荐使用 `write_memory_entry()` 写入带类型的记忆条目。每条记忆写入独立文件，并自动更新 `MEMORY.md` 索引。

```python
workspace = Workspace("./workspace")
workspace.initialize()

# 写入用户偏好
await workspace.write_memory_entry(
    title="Python Style",
    content="User prefers concise, typed Python. Avoid unnecessary comments.",
    memory_type="feedback",           # user | feedback | project | reference
    description="python coding style typed concise",  # 相关性匹配关键词
)

# 写入项目上下文
await workspace.write_memory_entry(
    title="Release Deadline",
    content="v2.0 release is due end of April 2026.",
    memory_type="project",
    description="v2 release deadline april 2026",
)
```

每条记忆文件带 YAML frontmatter：

```markdown
---
name: Python Style
description: python coding style typed concise
type: feedback
---

User prefers concise, typed Python. Avoid unnecessary comments.
```

### 四类型分类法

| 类型 | 存储内容 | 典型触发 |
|------|---------|---------|
| `user` | 用户角色、偏好、技术背景 | "我是数据科学家"、"我用 Python 10 年了" |
| `feedback` | 对 AI 行为的纠正和确认 | "别 mock 数据库"、"这个方案很好" |
| `project` | 非代码可推导的项目上下文 | "合并冻结从周四开始"、"这是合规要求" |
| `reference` | 外部系统指针 | "pipeline bugs 在 Linear INGEST 项目" |

> `feedback` 类型同时记录失败（"不要这样做"）和成功（"对，就这样"）——只记录纠错会导致 AI 行为随时间漂移。

---

## 常驻规则：直接改 AGENTS.md，没有专门工具

「记住：以后都要 X」和「记住这件事 X」不是一回事，落盘位置不同：

| | 常驻规则 | 一个事实 |
|---|---|---|
| 落盘位置 | 用户级：`~/.agentica/workspace/users/{user_id}/AGENTS.md`（CLI 的 user_id 是 `default`，也可通过 symlink `~/.agentica/AGENTS.md` 改同一份文件）；项目级：`<repo 根>/AGENTS.md` | `memory/*.md` + `MEMORY.md` 索引 |
| 何时进 system prompt | **下个会话起**全量注入（见下）；本会话靠对话历史 | **下个会话起**注入 MEMORY.md 索引（标题 + hook + 相对路径，最多 60 行 / 4KB）；正文 `search_memory` / `read_file` |
| 谁来写 | 人手写，或 agent 用 `apply_patch` / `write_file` | `save_memory` |
| 适合 | "always ..." / "never ..." / "从现在开始 ..." | 用户是谁、某个决定为什么这么定、环境怎么搭的 |

规则这一侧**没有专门的工具**，也不需要固定格式：`AGENTS.md` 就是一个 markdown 文件，人和 agent 用同一种方式改它。Web 设置 › 记忆的上半块编辑的就是这份用户级文件；下半块列出 `MEMORY.md` 抽出来的条目。写法说明在 bundled `agentica` skill 里（`self_manage` 只管 `config.yaml` / `.env` / 升级，不管常驻规则）。

外部 workflow（如 `learn-from-experience`）要把确认偏好写成常驻规则时，直接往 `~/.agentica/workspace/users/{user_id}/AGENTS.md` 追加普通行即可；默认 CLI 用户也可以写 `~/.agentica/AGENTS.md`，它是指向同一份文件的 symlink。经验卡片若进 prompt，是直接从 EXPERIENCE 相关性召回注入（`## Learned Experiences`），**只保留用户纠正类**（`correction`）；`tool_error` / `success_pattern` 不进 system prompt。事实记忆走 MEMORY.md 索引注入 + `search_memory`，也不写进 AGENTS.md。

---

## 记忆注入：MEMORY.md 索引

CLI / SDK / Gateway 同一条路径：`freeze_snapshots()` 把 **MEMORY.md 索引**（不是 topic 正文）冻进 system prompt。每条是 `- [Title](memory/x.md) — hook`；正文按需 `read_file` 或 `search_memory`。注入副本有更紧的预算（60 行 / 4KB），超出截断并点名完整索引的绝对路径。磁盘上的索引仍是 200 行 / 25KB。

`get_relevant_memories()` 仍可按 query 拉 topic 正文，给脚本和调试用，**不再**进生产 system prompt。

---

## 记忆漂移防御

`get_relevant_memories()` / 读到的 topic 正文会追加一条提示，防止过时引用造成幻觉：

```
Note: memories reflect the state at write time. If a memory references a specific
file path, function, or flag, verify it still exists before recommending it.
```

---

## 会话快照与 prompt cache

System Prompt 里所有从「实时状态」读出来的部分，都在会话第一轮一次性冻结：

| 内容 | 冻结入口 |
|------|----------|
| 工作区上下文（AGENTS.md 等） | `Workspace.freeze_snapshots()` |
| 工作区记忆 | `Workspace.freeze_snapshots()` |
| 经验（experiences） | `Workspace.freeze_snapshots()` |
| skills 目录（session guidance） | `Agent.freeze_session_guidance()` |

原因是 prompt cache 按**字节精确的前缀**匹配，而 system message 位于后续每一个缓存断点的前缀里：中途改一行，失效的不只是 system 那个断点，而是**连同整段对话历史一起**重新计价。

经验和 skills 尤其要冻，因为它们是**这个 agent 自己在会话中途写的**：捕获钩子会在工具出错、用户纠正、批量 judge 时写入新的经验卡片，skill upgrade 钩子会在后台调 `refresh_tool_system_prompts()` 重排 skills 目录。也就是说，没有任何人提出要求，一次后台写入就让整段对话重新计价。

代价用一行指针补回来：注入的经验块会写明这是会话开始时选中的，并给出 `EXPERIENCE.md` 索引路径（`Workspace.experience_index_path`），需要最新的经验时 agent 自己 `read_file` 一次即可。skills 则不需要指针——新装的 skill 会立刻生效，因为 `/skills` 会重建 agent，而新 agent 会重新冻结。

---

## Git 上下文（为什么不注入）

Workspace **不会**把 Git 状态注入 System Prompt。分支、未提交变更、最近 commit 由 agent 需要时自己跑一次 git 获取（`execute`）。

原因是 prompt cache 按**字节精确的前缀**匹配，而 system message 位于后续每一个缓存断点的前缀里：`git status --short` 每轮变一行，失效的不只是 tools+system 那个断点，而是**连同整段对话历史一起**按 1.25x 重写。改成会话开始时冻结一次也不划算——省下了钱，换来的是一份会越来越旧的文件列表，而模型本来就从自己的工具结果里看见了每一次编辑。

---

## 对话归档

使用 `ConversationArchiveHooks` 自动将对话归档到每日日志文件：

```python
from agentica import Agent, Workspace
from agentica.agent.config import WorkspaceMemoryConfig

agent = Agent(
    workspace=Workspace("./workspace"),
    long_term_memory_config=WorkspaceMemoryConfig(auto_archive=True),
)
```

归档写入 `users/{user_id}/conversations/YYYY-MM-DD.md`，使用 per-file asyncio.Lock 防止并发写冲突。

---

## Session Log（JSONL）

CLI、Web 轨迹、SDK **写的是同一份** append-only JSONL：

`~/.agentica/projects/<user>/<project-slug>/<session_id>.jsonl`

对话行（`user` / `assistant` / `tool`）给 `/resume` 回放；`type=event` 行只给观测，不进模型上下文。Web `/traces`、CLI `/trace`、SDK `agent.session_log.analyze()` 都是对这份文件的**读时分析**。

```python
from agentica import Agent

agent = Agent(session_id="my-session-001")  # 不传 session_id 则不落盘
log = agent.session_log
print(log.path)
print(log.format_trace())          # 与 CLI /trace 同一份文本
log.export("my-session-001.jsonl") # 完整轨迹
log.export("trace.json", kind="analysis")  # 与 Web /traces 同一份 JSON
```

支持 `compact_boundary`（压缩边界）：恢复时从最后一个边界之后开始加载，跳过历史数据。

---

## WorkspaceConfig

可自定义文件布局：

```python
from agentica.workspace import Workspace, WorkspaceConfig

config = WorkspaceConfig(
    agent_md="AGENTS.md",        # workspace 级与 user 级共用同一个文件名
    memory_md="MEMORY.md",      # 记忆索引文件
    memory_dir="memory",         # 记忆内容文件目录
    users_dir="users",
    conversations_dir="conversations",
)

workspace = Workspace(path="./workspace", config=config)
```

---

## 下一步

- [Agent 核心概念](agent.md) — Agent 如何使用记忆
- [Hooks](../advanced/hooks.md) — ConversationArchiveHooks 详解
- [Context Compression](../advanced/compression.md) — 上下文压缩与会话摘要复用
