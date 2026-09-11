# Context Compression

Agentica 提供两层上下文压缩策略，防止长对话或大量工具输出导致 token 超限。Layer 2 对齐 Codex TokenBudget compact：满窗时换一个空的活动窗，**不再**调用 LLM 或 `/responses/compact` 做摘要。旧对话留在 session JSONL，用 `search_session` 查。

## 两层设计

压缩本质上只有两种操作，按代价从低到高尝试：

| 层 | 做什么 | 代价 | 可逆性 |
|----|--------|------|--------|
| Layer 1 淘汰 | 把更早回合的工具结果换成占位符；超长 tool_call 参数换成 `<evicted-tool-arg>`。**正在跑的那一轮（未返回的调用 + 末批 result）不动**，含 SDK / CLI / Web 自定义工具 | 免费，无 LLM | 模型可重发那次调用 |
| Layer 2 换窗 | 丢掉活动窗里的旧轮次，装上 `<context_window>` + session notes 摘录 | 免费，无 LLM | 原文在 JSONL，不在 prompt 里 |

在这两层之前还有一个 **Layer 0**，它不是压缩而是工具输出策略：单条结果超过阈值时在产生的那一刻就落盘，从不以全量进入上下文。

```
Tool 输出 (可能很大)
    |
    v
[Layer 0 Tool Result Storage] -- 超大输出产生时即落盘
    |
    v
Context Messages
    |
    v
[Layer 1 淘汰] -- 窗口紧时免费收缩 tool result / 超长参数
    |
    v
[Layer 2 换窗] -- 空摘要 compact_boundary；同一 session_id
    |
    +--> auto / /compact：保留 system + 正在问的尾巴
    +--> prompt-too-long: 强制换窗后再重试
```

### Layer 1：淘汰（`agentica.compression.evict`）

由 `ToolConfig.enable_evict` 控制（默认开）。关掉后这一层完全不跑，窗口会更快涨到 Layer 2。

只有两个参数，没有「保留最近 N 条」这类计数：

- **`EVICT_THRESHOLD_RATIO = 0.7`** — 占用低于窗口 70% 时一条都不动。清掉一条窗口本来放得下的结果是净亏：省下的上下文没人要，模型却要重跑工具才能拿回来。
- **`EVICT_TARGET_RATIO = 0.5`** — 超过阈值后按最旧优先淘汰，降回 50% 就停。目标低于阈值是为了迟滞，否则每轮刚跌破阈值又超，变成持续抖动。

最近的结果之所以幸存，是因为淘汰在够到它们之前就停了。**消息尾部那一段连续的工具结果（模型还没看过的当前批次）整体排除在外**：任何固定条数都会输给 count+1 大小的并行批次，这正是「读了又读」死循环的成因。CLI `--tools`、SDK `tools=`、Web extra、MCP 与内置工具走同一条边界，不按名字开白名单。

占位符写明是哪个调用（`read_file(file_path=..., offset=...)`），模型据此可以原样重发。它**不**先把内容复制到磁盘——取回同样是一次工具调用，而对文件读取来说原路径上的内容比快照更新鲜。

#### 淘汰的单位是「一条结果」，不是「一条消息」

两种 provider 对结果的打包方式不一样，这是这一层唯一容易出错的地方：

| 形态 | 结构 |
|------|------|
| OpenAI 系 | 一条结果 = 一条 `role="tool"` 消息 |
| Anthropic | 一整轮结果打包进**一条** `role="user"` 消息的 content 列表，每条是 `{"type": "tool_result", "tool_use_id": ...}` block |

只扫 `role="tool"` 意味着 Anthropic 路径上这一层从来没生效过——不报错，只是静默失效。所以遍历以「结果」为单位展开，两种形态都覆盖。`tool_result` block 本身不带工具名，占位符通过发起调用的那条 assistant 消息的 `tool_calls` 反查 `tool_use_id` 得到。

同一个形态差异还影响另一处，已按同样口径处理：

- Layer 2 保留「最后一条 user 消息之后的整段尾巴」。Anthropic 的工具轮本身就是 user 消息，从那里切会留下一批 `tool_result`，而它们对应的 `tool_use` 在刚被换窗丢掉的 assistant 消息里——这种孤儿 block 会被 API 直接拒绝。所以判断尾巴时跳过承载工具结果的 user 消息。

### Layer 2：换窗（`CompressionManager`）

淘汰兜不住时才走这层。不再调用摘要模型，也不再走 provider-native `/responses/compact`。`ToolConfig.compression_manager` 留空时自动创建（给 `/compact` 和跨 provider fallback 用）。自动触发由 `ToolConfig.enable_auto_compact` 控制（默认开）；关掉后 runner / `prompt_too_long` 后的 reactive 都不跑，超窗就把 provider 错误抛出。`/compact` 不受此开关影响。

每个 Agent 会挂上 `BuiltinContextTool`。换了几个窗也只扫**同一份** JSONL，不按窗建第二套档案。

```
~/.agentica/projects/<user>/<sanitized-cwd>/<session-id>.jsonl
~/.agentica/projects/<user>/<sanitized-cwd>/<session-id>.notes.md
```

- `search_session` — 查整份 JSONL，**包括**每一条 `compact_boundary` 之前。关键词路径对齐 Codex `history.search_contents`：`query` 先当字面子串，中文问法再叠字（`工单号` 能命中 `工单 ZX-41827`），命中按相关度排序。每次结果都附带最近用户问题（倒序最多 20 条、截断；跳过 `<context_window>` preamble）。空 query 只返回这份索引。不要扫 JSONL。不提供 `read_session_item`（Codex `read_item` 是服务端大条目分页，本地一行通常短于 snippet）
- `<session-id>.notes.md` 是 **standing state**（goals / constraints / IDs / decisions），模型用已有文件工具写，不是第二份 transcript。第一次触及 Layer 2 阈值时若文件仍空，先注入 fallback 催写并推迟约 4% 窗口。真正切窗时：文件已有内容则注入 `<session_notes>`；仍空则把丢掉的那一段按时间交织成 skim（user/assistant，其次 tool args/result，带时间戳）注入 `<dropped_span>`，**不写进 notes.md**。写进去会让 notes 变成 JSONL 缩写，第二次换窗还冻住第一窗的 skim。`search_session` 搜 JSONL，并搜模型手写的 notes（措辞可能和原文不同）

油表是 `<context_window>` user 片段：新窗写满窗身份；剩余 token 降到工作窗口的 25% 时每窗提醒一次。不写进冻结的 system 前缀。

## CompressionManager 配置

```python
from agentica import Agent, OpenAIChat, CompressionManager
from agentica.agent.config import ToolConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tool_config=ToolConfig(
        compression_manager=CompressionManager(
            compact_token_limit=300000,  # 可选工作阈值；不配 = 窗口×0.95 才换窗
        ),
    ),
)
```

`compact_token_limit` 可省略：不配则约 95% 窗口才换窗。换窗会保留 **system prompt**（否则本轮剩下的调用没有任何指令）和 **从最后一条 user 消息开始的整个尾部**（否则对话以 assistant 结尾，provider 会直接拒绝）。

## 开关（默认都开）

两层都可以关。比例旋钮仍然只有 `AGENTICA_EVICT_THRESHOLD_RATIO`；这两个布尔管的是「要不要自动做」，不是「做多狠」。

| 开关 | 默认 | 关掉之后 |
|------|------|----------|
| `ToolConfig.enable_evict` | `True` | Layer 1 不淘汰。窗口更容易涨到 Layer 2 |
| `ToolConfig.enable_auto_compact` | `True` | runner 自动换窗、`prompt_too_long` 后的 reactive 都不跑。`/compact` 仍可用 |

```python
from agentica import Agent, OpenAIChat
from agentica.agent.config import ToolConfig

# SDK：评测 / 成本敏感服务可以关自动换窗
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tool_config=ToolConfig(enable_evict=False, enable_auto_compact=False),
)
```

CLI：`--no-evict` / `--no-auto-compact`（也认 `--evict` / `--auto-compact` 强制打开）。未传 flag 时读 `~/.agentica/config.yaml`：

```yaml
settings:
  enable_evict: true
  enable_auto_compact: true
  # compact_token_limit: 300000   # optional working cap; see below
```

Gateway 读同一对 settings。SDK 的 `Agent()` **不**读 config.yaml——只认构造时传入的 `ToolConfig`。

### 工作阈值 `compact_token_limit`

`model.context_window` 是服务商硬上限，不要把它填小来“早点压缩”。另设绝对 token 帽：

```
Layer 2 触发 = min(compact_token_limit 或 ∞, int(window × 0.95))
Layer 1 的 0.8 / 0.5 相对 min(compact_token_limit 或 ∞, window)
```

不配则和现在完全一样（约 95% 窗口才换窗）。1M 窗口配 `300000` 就在 30 万处换窗；32k 窗口配 `128000` 仍被窗口挡住。写在 profile 上（每个模型可以不同），或 `settings.compact_token_limit` 做全局默认。CLI：`/config set compact_token_limit 300000`、`--compact-token-limit`。SDK：`ToolConfig(compact_token_limit=300000)`。

## Layer 0：工具输出预算

不是压缩，是输出策略——在**结果产生的那一刻**（`Model.run_function_calls`）就把它限住，
所以超大输出一次都不会完整进入上下文。两条规则：

| 规则 | 阈值 | 说明 |
|------|------|------|
| 单条结果 | `Function.max_result_size_chars`（`execute` 为 `max_output_length`，默认 20,000 字符） | 单个 tool result 超过此值就收缩。`read_file` 为 `None`（不收缩），否则它会去读自己的落盘文件，形成循环。`execute` 在读管道时就会封顶（硬顶 64MiB 后杀进程），所以 `cat` 一个 60 万行的文件不会整份进入 live round |
| 单轮批次 | `0.25 × model.context_window` | 本轮全部新结果加起来超过窗口的这个份额时，从最大的开始收缩。Layer 1 从不动尾部批次（模型还没看过），所以一轮并行 6 个大调用只有这里能兜 |

批次预算按**窗口比例**而不是固定字符数：固定 200K 字符在 512K token 的窗口上会误伤，
在 8K token 的窗口上又完全不触发。

### 收缩成什么形态，取决于这个 session 能不能取回

`can_recover_spill(model.functions)` 检查是否注册了 `read_file` 或 `execute`：

- **能取回**（CLI、带文件工具的 agent）：写入磁盘，上下文里换成预览 + 路径（`<persisted-output>`），
  模型一次 `read_file` 就能拿回全量。`execute` 撞上 64MiB 硬顶被杀掉时，落盘文件只有前 64MiB，
  头文案写 INCOMPLETE，不要把它当全文读。

  ```
  ~/.agentica/projects/<user>/<project-hash>/<session-id>/tool-results/<tool_use_id>.txt
  ```

- **不能取回**（只挂业务工具的服务型 agent）：**不写盘**，直接截断成 `<truncated-output>`，
  并说明"本 session 没有能读取副本的工具"。给一个没人能打开的路径既丢了数据，
  又会诱导模型去调一个它根本没有的工具。

目录按 `session_id` 分（缺省 `"default"`），按 `workspace.user_id` 隔离租户——
**不按 `run_id`**：run_id 每轮一个新 uuid，会把同一次会话打散成几十个目录。

预览统一 2,000 字符（40% 头 + 60% 尾），写盘和预览都先过一遍敏感信息脱敏。

## Hooks 集成

压缩前后可以通过 Hooks 插入自定义逻辑：

```python
from agentica.hooks import RunHooks

class CompactionTracker(RunHooks):
    async def on_pre_compact(self, agent, messages, **kwargs):
        print(f"Before: {len(messages)} messages")

    async def on_post_compact(self, agent, messages, **kwargs):
        print(f"After: {len(messages)} messages")
```

## 自动压缩触发

`CompressionManager.auto_compact` 在以下条件触发：

1. 当前 token 数达到 `min(compact_token_limit or ∞, int(window × 0.95))`（notes 仍空时先 fallback 催写，推迟到约 99%）
2. 用户执行 `/compact`（始终强制换窗；多余参数不再当摘要指令）
3. provider 返回 `prompt_too_long` 之后的 reactive 换窗

本地换窗几乎不会失败；失败则对话不变。

## 观测压缩是否发生

换窗会从 prompt 里拿掉早期轮次（JSONL 仍在）。SDK 调用方没有 CLI 的事件回调，所以次数直接挂在响应上：

```python
response = await agent.run("...")
if response.context_compactions:
    logger.info(f"本轮换了 {response.context_compactions} 次活动窗")
```

Layer 2 与 `prompt_too_long` 之后的 reactive 都会计数；Layer 1 淘汰是免费且可通过重跑工具恢复的，不计数。

## 下一步

- [RunConfig](run-config.md) -- 超时和成本控制
- [Hooks](hooks.md) -- on_pre_compact / on_post_compact
- [Agent 概念](../concepts/agent.md) -- Agent 上下文管理
