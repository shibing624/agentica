# 上下文压缩与长会话管理 (`/compact`)

在长对话、连续读大文件或跑复杂重构时，你**不需要手动清理上下文**。Agentica 默认开启全自动上下文管理：尽量保留近期对话和关键细节，同时把完整原始记录留在本地会话日志里，需要时随时查回。

---

## 对用户来说，它是怎么工作的？

Agentica 按从轻到重的顺序自动处理上下文增长：

| 阶段 | 触发时机 | 做了什么 | 你怎么拿回细节 |
|---|---|---|---|
| **1. 超大输出自动落盘** | 单条工具输出过大（如 `execute` 输出超过 20,000 字符），或单轮并行工具总输出超过窗口的 25% | 完整输出存进本地 `tool-results/<id>.txt` 文件，上下文里只留 2,000 字首尾预览和文件路径 | Agent 看到路径后，直接用 `read_file` 或 `grep` 按需读取对应片段 |
| **2. 淘汰旧工具输出（Layer 1）** | 当前窗口占用达到 **80%** | 按从旧到新的顺序，把早期轮次的工具结果和超长调用参数替换为简短占位符，直到窗口降回 **50%**。**当前正在执行的这一轮工具调用绝不触碰** | 占位符写明了原本调用的工具和参数；有会话日志且带 `search_session` 时，还会带上 tool-call id，Agent 用 `search_session(uuid="<id>")` 取回原始输出（重调不等价：非幂等命令结果会变，参数已被清掉的调用无法重调）；否则可原样重调 |
| **3. 开启新窗口并交接笔记（Layer 2）** | 窗口占用接近 **95%**（或你手动敲 `/compact`） | 先由主模型根据当前窗口更新交接笔记 `<session>.notes.md`（保留目标、约束、关键 ID、已完成与待办事项），然后开启一个新的活动窗口，把笔记和被折叠轮次的简表带入新窗口 | 完整历史对话与工具结果都留在会话 JSONL 日志里，Agent 用 `search_session` 按关键词或 `[uuid]` 查回全文 |

> **为什么不会因为模型估算偏差而漏触发？**
> 不同厂商对长上下文、推理块或图片的 token 计数可能高于本地估算。Agentica 会根据模型接口实际返回的 prompt token 数自动校准水位比例，防止本地估算偏低导致一直不触发压缩。

---

## 在 CLI 和 Web 里怎么用

### 1. 日常使用（全自动）

默认什么都不用配：
- 状态栏会实时显示当前上下文用量（如 `120.4K/512K 24%`）。
- 占用升高时自动执行淘汰与换窗；发生换窗时终端会提示新窗口编号及交接笔记状态。

### 2. 手动换一个干净窗口：`/compact`

当你完成了一个阶段的任务、想腾出上下文开始下一阶段，但又不想丢失之前的项目背景时，在 CLI 或 Web 输入框敲：

```text
> /compact
  New context window 1: 40 messages -> 6. Prior turns stay in the session log.
```

`/compact` 会先让主模型把当前进展整理进 `<session>.notes.md`，然后开启新窗口。新窗口里仍然保留你的系统指令和这份交接笔记。

### 3. 限制单次窗口的工作上限：`compact_token_limit`

有些模型支持 1M 甚至更长的上下文窗口，但长上下文每轮请求更贵、更慢。如果你希望在达到某个 token 数（例如 30 万）时就提前淘汰和换窗，可以设置 `compact_token_limit`：

**在 CLI 里设置：**

```text
> /config set compact_token_limit 300000
```

或启动时传入：

```bash
agentica --compact-token-limit 300000
```

**在 `~/.agentica/config.yaml` 里设置：**

可以写在某个模型的 `profile` 下，也可以写在全局 `settings` 下：

```yaml
profiles:
  default:
    model_provider: deepseek
    model_name: deepseek-v4-flash
    context_window: 1000000
    compact_token_limit: 300000   # 在 300k token 处按比例触发淘汰与换窗

settings:
  enable_evict: true              # 是否开启 80% 淘汰旧工具结果（默认 true）
  enable_auto_compact: true       # 是否开启 95% 自动换窗（默认 true）
  # compact_token_limit: 300000   # 全局默认工作上限（可选）
```

设置了 `compact_token_limit: 300000` 后：
- Layer 1 淘汰在 `300,000 × 80% = 240,000` token 处触发；
- Layer 2 换窗在 `min(300,000, 窗口 × 95%)` 处触发。

### 4. 关闭自动压缩（可选）

如果你在跑基准评测，或者希望超窗时直接报错而不是自动压缩：

```bash
agentica --no-evict --no-auto-compact
```

（即使关掉 `--no-auto-compact`，你依然可以手动敲 `/compact`。）

---

## 换窗后怎么查回以前的内容？

换窗不会删除本地日志。每个带会话记录的会话在同一目录下都有两份文件：

```text
~/.agentica/projects/<user>/<项目目录>/<session-id>.jsonl
~/.agentica/projects/<user>/<项目目录>/<session-id>.notes.md
```

### `search_session` 与 `search_memory` 的区别

Agent 内置了两个检索工具，分工明确：

| 工具 | 查什么 | 什么时候用 |
|---|---|---|
| **`search_session`** | **当前这一次会话**的完整 JSONL 日志（含所有换窗之前的提问、回答、工具调用参数与工具输出结果），以及当前最新的 `<session>.notes.md` | 换窗后找回本会话早期的某个报错、读过的文件片段、执行过的命令输出，或者查看本会话之前问过什么问题 |
| **`search_memory`** | **跨会话**的长期记忆（`MEMORY.md` 下的记忆条目与近期对话归档） | 查找以前其他会话里记下的用户偏好、项目规范或历史结论 |

**实用技巧：**
- 想看本会话早期的某次工具输出：Agent 会调用 `search_session(query="关键词", role="tool")`，拿到带 8 位 `[uuid]` 的命中行后，再调 `search_session(uuid="...")` 取出那条记录的完整原文。`uuid` 也接受被淘汰工具结果占位符里的 tool-call id（精确匹配，取最新一条）。
- **多次换窗时，旧的 `notes.md` 会丢吗？** 每次换窗重写 `notes.md` 时，都会把上一份 `notes.md` 一并交给模型修订——仍然成立的目标、约束和关键 ID 会保留下来，已经做完或作废的条目会被清理，避免笔记无限膨胀。如果某个早期细节没写进新笔记，原始对话和工具输出仍然完整保存在 `.jsonl` 里，随时可以用 `search_session` 搜出来。

---

## 在 Python SDK 中配置

### 1. 默认配置与工作上限

`Agent` 和 `DeepAgent` 默认都开启了 Layer 1 淘汰和 Layer 2 自动换窗。如果希望换窗后能用 `search_session` 查回历史并自动维护 `<session>.notes.md`，只需传入 `session_id`：

```python
from agentica import Agent, OpenAIChat, CompressionManager
from agentica.agent.config import ToolConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    session_id="my-long-task-001",        # 开启本地会话日志与 notes.md
    add_history_to_context=True,          # 默认重放当前窗口内的全部轮次
    tool_config=ToolConfig(
        enable_evict=True,                # 默认 True
        enable_auto_compact=True,         # 默认 True
        compact_token_limit=300000,       # 可选：限制单窗工作 token 上限
    ),
)
```

> **提示**：如果需要自定义淘汰触发比例（默认 `0.8`），可在环境变量或 `.env` 中设置 `AGENTICA_EVICT_THRESHOLD_RATIO=0.75`。

### 2. 检查某轮是否发生了换窗

在 SDK 中，可以通过 `RunResponse.context_compactions` 查看本轮运行期间换了几次活动窗口：

```python
response = await agent.run("继续分析剩余模块")
if response.context_compactions:
    print(f"本轮自动换了 {response.context_compactions} 次上下文窗口")
```

### 3. 监听压缩生命周期（Hooks）

如果需要在换窗前后执行自定义逻辑（例如打日志或统计），可以使用 `RunHooks`：

```python
from agentica.agent.hooks import RunHooks

class CompactionLogger(RunHooks):
    async def on_pre_compact(self, agent, messages, **kwargs):
        print(f"即将换窗，当前消息数: {len(messages)}")

    async def on_post_compact(self, agent, messages, **kwargs):
        print(f"换窗完成，保留消息数: {len(messages)}")
```

---

## 下一步

- [记忆与常驻规则](../concepts/memory.md) — 跨会话长期记忆与 `AGENTS.md`
- [统一配置文件 (config.yaml)](../guides/config.md) — 在配置文件中设置 `compact_token_limit`
- [架构与内部实现（开发者选读）](../introduction/architecture.md) — 了解 Layer 0/1/2 的底层实现、OpenAI 与 Anthropic 消息结构差异及缓存校准机制
