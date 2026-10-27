# 记忆与常驻规则 (`Memory & Workspace`)

在使用 Agentica 时，「记住刚才聊了什么」「记住项目的某个背景事实」和「记住我以后的代码风格要求」分别存放在三个不同的地方。分清这三者，就能让 Agent 既听话又不乱记东西。

---

## 一表分清三种记忆

| 你想让 Agent 记住什么 | 存在哪里 | 怎么写入 | 怎么查回 |
|---|---|---|---|
| **1. 当前这次对话的上下文**<br>（刚才问了什么、刚读过哪个文件、刚跑了什么命令） | **会话历史 (`WorkingMemory` / `SessionLog`)**<br>`~/.agentica/projects/<user>/<项目>/<session_id>.jsonl` | 对话时自动记录 | 当前窗口内自动带入；换窗后用 **`search_session`** 检索，或用 `/resume` 恢复整个会话；之前会话的提问和未完成工作项在新会话中通过 `<session_recall>` 自动带入，也可用 `search_session(scope="project")` 检索 |
| **2. 跨会话的事实与经验**<br>（比如「我是数据科学家」「发布截止日是月底」「上次修这个报错的方案」） | **Workspace 长期记忆**<br>`~/.agentica/workspace/users/<user_id>/MEMORY.md` 及 `memory/*.md` | 对话中让 Agent 记下（调用 `save_memory`），或开启自动抽取 | 索引自动带入新会话；每轮由辅助模型挑出相关条目，正文附在你的消息后（`<memory_recall>`）；也可用 **`search_memory`** 检索 |
| **3. 每次都要遵守的常驻规则**<br>（比如「永远用简体中文回答」「不要用 mock 数据库」「提交前先跑测试」） | **`AGENTS.md` 规则文件**<br>项目级：`<仓库根目录>/AGENTS.md`<br>用户级：`~/.agentica/AGENTS.md` | 你直接用编辑器改，或在对话里说「把这条写进 AGENTS.md」 | 每个新会话启动时**全文自动加载**进系统提示词（修改后从**下一个新会话**起生效） |

---

## 在 CLI 和 Web 里怎么用

### 1. 写常驻规则（`AGENTS.md`）

只要是「以后每次都要遵守」的指令，直接写进 Markdown 文件即可，不需要任何特殊格式：

- **只对当前代码仓库生效的规则**：写在仓库根目录的 `AGENTS.md`（或 `CLAUDE.md`）。
- **对你本机所有项目都生效的个人习惯**：写在 `~/.agentica/AGENTS.md`（在 Web 端打开「设置 › 记忆」上半部分编辑的也是同一个文件）。

```markdown
# 项目规则
- 回答和解释一律使用简体中文。
- 修改代码后先运行 `pytest tests/unit` 验证通过。
- 不要主动执行 `git push`。
```

> **提示**：为了保证长对话期间的缓存命中率（不因为中途改文件导致整段历史重新计费），`AGENTS.md` 和记忆索引在**会话启动时读取一次**。如果你中途修改了 `AGENTS.md`，敲 `/new` 开启新会话即可生效。

### 2. 存跨会话的长期记忆（`save_memory` / `search_memory`）

在 CLI 或 Web 里，你可以直接对 Agent 说：

```text
> 记住：我们的测试环境数据库端口是 15432， staging 分支每周四下午封版。
```

Agent 会自动调用 `save_memory` 把它分类存进 `~/.agentica/workspace/users/default/memory/`，并更新 `MEMORY.md` 索引。之后你新开任何会话问起「测试库端口是多少」，它都能找出来。

**每轮自动召回**：CLI、Web 和 `DeepAgent` 默认开启。每次你发消息前，辅助模型（`auxiliary_model`，未配置时用主模型）会读一次**最新的** `MEMORY.md` 索引，挑出这条消息真正需要的条目（最多 5 条），把正文放进 `<memory_recall>` 块，拼在你这条消息的**前面**，你的原话留在消息最底部（长上下文模型对中间部分注意力最弱，最新的问题放在最后读）。用 SDK 时，不论传 `message="..."` 还是 `messages=[历史..., 本轮 user]`（IM 适配器每轮自己重建历史的写法）都会触发召回，后者只改写列表最后一条 user 消息的副本，不动调用方传进来的对象。要点：
- 系统提示词里仍然只有会话开始时冻结的索引，召回块追加在本轮用户消息上，不改动已发送的前缀，不影响缓存命中。
- 读的是实时索引，所以**另一个会话刚存下的记忆，这个会话下一轮就能召回**。
- 当前窗口里已经召回过的条目不会重复召回；换窗后旧召回块被移出，相关时会再次召回。
- 大多数消息不会召回任何条目；选择调用失败、超时（默认 5 秒）或返回无法解析的内容时，本轮照常进行，只是没有召回。同一会话内连续失败 3 次后，本会话不再发起召回（新会话自动恢复），避免辅助模型挂掉时每轮都白等。
- 召回块不写入会话日志，`search_session` 搜不到它。
- 召回按用户进行，不按项目过滤：在 A 项目存下的偏好，在 B 项目里同样会被召回。每条记忆会记下写入时所在的目录（frontmatter `origin_work_dir`），召回块里显示这个目录（如 `(written while working in ~/Documents/Codes/agentica)`，家目录缩写成 `~`），`search_memory` 结果里也有 `origin_work_dir`。模型可以据此判断这条记忆是否适用于当前项目，需要时直接打开那个目录核对。之前写入的旧记忆没有这一行，照常召回。

**密钥过滤**：写入记忆前（以及读取旧记忆时）会屏蔽密钥形态的内容：厂商前缀 key、JWT、私钥块、`XXX_API_KEY=...` 这类凭据赋值（数字取值如 `max_tokens=50` 保留），以及无前缀的高熵 token。端口、路径、commit hash 不受影响。

长期记忆分为四类：
- `user`：你的角色、技术栈、个人偏好
- `feedback`：你对 Agent 做法的纠正（「别这样做」）或肯定（「对，就这样处理」）
- `project`：从代码里直接看不出来的项目背景（排期、业务术语、外部对接人）
- `reference`：外部系统线索（如某个看板或文档库的位置）

**相互矛盾的纠正**：高置信度的纠正会直接写成经验卡片（`EXPERIENCE.md`）。中文、英文、中英混合的纠正都会生成卡片。新卡片写入后，最多 5 张旧纠正卡片会交给辅助模型判断是否矛盾：先取有共同关键词的，剩下的名额给最新的，跨语言的矛盾（「回复一律用中文」和「Always answer in English」）也能判出来。矛盾的两张卡片都会保留，并互相标注 `conflict_with`。下一个会话的提示词里，这两条规则后面会附一句：两条规则冲突，按当前对话判断，并向你确认用哪条。

查看当前工作区与记忆状态：
- CLI：敲 `/workspace` 查看路径与记忆条数，敲 `/memory` 查看当前会话消息。
- Web / Desktop：点左侧或设置里的「记忆」面板查看和编辑。

### 3. 接上之前的会话（`<session_recall>` 与工作项）

长期记忆只存事实，"做到一半的事"不进去。可新开一个会话时，最需要的恰恰是"上次停在哪"：「继续」「推一下」这种话本身没内容，指的就是上次没收尾的那件事。CLI、Web 和 `DeepAgent` 默认开启这两项：

- **工作项（`track_intents`）**：每 3 轮由辅助模型（`intent_track` 任务）在后台把这几轮整理成工作项，包括你要做什么、做到哪（commit hash、文件、数字）、状态（`open` / `done` / `dropped`）和下一步。同一件事会更新已有条目，不会重复新建；你说"已经推了"，对应条目就改成 `done`。待整理的轮次先落盘，会话来不及整理就退出的，下一个会话在同一项目里的第一轮补上。
- **会话召回（`recall_sessions`）**：每次发消息前，在你的消息前面拼一个 `<session_recall>` 块。这一步不调用模型，只查本地 SQLite 索引（`~/.agentica/projects/<user>/session_index.db`），真实数据上耗时约 60 ms。
  - 新会话的第一轮：本项目最近 14 天里 `open` 的工作项（最多 3 条），以及最近两个会话最后问的几句话。
  - 每一轮：和这条消息有共同关键词的旧工作项和旧提问。其他项目的条目只在消息里提到那个项目目录名时才出现，在 A 仓库说"push"不会带出 B 仓库没推的改动。
  - 每行都带标记（`[intent:12]`、`[q:996ba8f8]`）和会话 id。已经出现在上下文里的行不会再出现；用 `search_session(session_id=..., uuid=...)` 能取回原话，摘要不确定时回原始记录核对。超过两周没动的 `open` 条目标为 `stale`，不再列为待办。

`search_session` 新增 `scope="project" | "all"` 参数，用来搜之前会话的提问和工作项；`query` 留空时列出本项目最近几个会话最后问的话。新增的 `session_id` 参数用来搜索或读取另一个会话的完整记录。

索引平时随会话日志同步写入。第一次用时，后台会把已有的会话记录补进索引，之后只读新增部分（实测约 6000 条提问，首次 3.6 秒，之后约 0.05 秒）。

---

## 在 Python SDK 里怎么用

### 1. 开启会话内多轮历史（`add_history_to_context`）

默认情况下，普通 `Agent()` 每次 `run()` 是独立的单轮调用。做多轮对话时，打开 `add_history_to_context=True`：

```python
from agentica import Agent, OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    # num_history_turns=10,  # 可选：只保留最近 10 轮；默认 None 重放当前窗口全部轮次
)

agent.run_sync("我叫小明，在做一个 FastAPI 项目")
res = agent.run_sync("你还记得我在做什么项目吗？")
print(res.content)
```

如果希望把会话落盘到 `.jsonl` 以便下次进程重启后接着聊，传入 `session_id` 即可：

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    session_id="user-1001-chat",
    add_history_to_context=True,
)
```

### 2. 开启跨会话长期记忆（`Workspace`）

给 `Agent` 绑定一个 `Workspace` 并开启 `enable_long_term_memory=True`，Agent 就会自动获得 `save_memory` 和 `search_memory` 工具，并在每次会话启动时加载该用户的 `AGENTS.md` 与 `MEMORY.md` 索引：

```python
from agentica import Agent, OpenAIChat, Workspace
from agentica.agent.config import WorkspaceMemoryConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    workspace=Workspace(path="./my_workspace", user_id="alice"),
    enable_long_term_memory=True,
    long_term_memory_config=WorkspaceMemoryConfig(
        load_workspace_context=True,   # 加载 AGENTS.md 常驻规则
        load_workspace_memory=True,    # 加载 MEMORY.md 记忆索引
        recall_memories=True,          # 每轮由辅助模型挑选相关记忆，正文拼在用户消息前（普通 Agent 默认关闭）
        recall_sessions=True,          # 每轮拼 <session_recall>：未完成工作项 + 之前会话的相关提问（不调用模型）
        track_intents=True,            # 后台整理跨会话工作项（open/done/dropped），每 intent_every_n_turns 轮一次
        max_memory_entries=5,          # 单轮最多召回条数
        auto_archive=True,             # 每轮结束后将原始对话归档到 conversations/YYYY-MM-DD.md（不调 LLM）
        auto_extract_memory=False,     # 是否在后台自动用副模型抽取记忆条目（会消耗少量 token）
    ),
)
```

> **多租户隔离**：在服务端为不同用户创建 Agent 时，传入各自的 `Workspace(path="./workspace", user_id=user_id)`，每个用户的 `AGENTS.md`、`MEMORY.md` 和对话归档都会隔离在 `users/<user_id>/` 目录下。

### 3. 在代码里直接读写记忆

你也可以不经过 LLM，直接用 Python 操作 `Workspace`：

```python
from agentica import Workspace

ws = Workspace("./my_workspace", user_id="alice")
ws.initialize()

# 写入一条记忆
await ws.write_memory_entry(
    title="Python 风格偏好",
    content="优先使用类型注解和简洁的函数拆分，避免多余的 try/except。",
    memory_type="feedback",
    description="python coding style type hints",
)

# 搜索记忆
hits = ws.search_memory("python 风格", limit=5)
```

### 4. 导出与分析会话日志（`session_log`）

当设置了 `session_id` 时，可以在代码中导出完整 JSONL 或查看统计分析（与 CLI `/trace`、Web `/traces` 同源）：

```python
agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), session_id="my-session-001")
await agent.run("帮我检查 README")

log = agent.session_log
print(log.path)                            # JSONL 文件路径
print(log.format_trace())                  # 打印与 CLI /trace 相同的轮次耗时与 token 统计
log.export("my-session-001.jsonl")         # 导出原始 JSONL
log.export("trace.json", kind="analysis")  # 导出结构化分析 JSON
```

---

## 下一步

- [Agent 与 DeepAgent](agent.md) — Agent 的完整配置方式
- [上下文压缩 (/compact)](../advanced/compression.md) — 长会话换窗与 `search_session` 用法
- [技能系统 (Skills)](../advanced/skills.md) — 用 `SKILL.md` 沉淀可复用的领域工作流
