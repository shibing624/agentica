# Context Compression

Agentica 提供两层上下文压缩策略，防止长对话或大量工具输出导致 token 超限。使用 `OpenAIResponses` 且服务端支持 `/responses/compact` 时，优先保留模型原生状态；其他模型继续使用可跨 provider 的本地压缩。

## 两层设计

压缩本质上只有两种操作，按代价从低到高尝试：

| 层 | 做什么 | 代价 | 可逆性 |
|----|--------|------|--------|
| Layer 1 淘汰 | 把最旧的工具结果换成写明调用的占位符；收缩过大的 tool_call 参数 | 免费，无 LLM | 模型可重发那次调用 |
| Layer 2 摘要 | 把整段历史换成一份 LLM 摘要 | 一次 LLM 调用 | 不可逆 |

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
[Layer 2 Native Compact] -- Responses 原生 checkpoint（服务端做同一件事，优先）
    |
    +--> 成功：保留 portable transcript，后续原样回放 checkpoint
    +--> 失败/不支持：[Layer 1 淘汰] -> [Layer 2 LLM 摘要]
                                              |
                                              +--> prompt-too-long: 强制 Layer 2
```

### Layer 1：淘汰（`agentica.compression.evict`）

由 `ToolConfig.enable_evict` 控制（默认开）。关掉后这一层完全不跑，窗口会更快涨到 Layer 2。

只有两个参数，没有「保留最近 N 条」这类计数：

- **`EVICT_THRESHOLD_RATIO = 0.7`** — 占用低于窗口 70% 时一条都不动。清掉一条窗口本来放得下的结果是净亏：省下的上下文没人要，模型却要重跑工具才能拿回来。
- **`EVICT_TARGET_RATIO = 0.5`** — 超过阈值后按最旧优先淘汰，降回 50% 就停。目标低于阈值是为了迟滞，否则每轮刚跌破阈值又超，变成持续抖动。

最近的结果之所以幸存，是因为淘汰在够到它们之前就停了。**消息尾部那一段连续的工具结果（模型还没看过的当前批次）整体排除在外**：任何固定条数都会输给 count+1 大小的并行批次，这正是「读了又读」死循环的成因。

占位符写明是哪个调用（`read_file(file_path=..., offset=...)`），模型据此可以原样重发。它**不**先把内容复制到磁盘——取回同样是一次工具调用，而对文件读取来说原路径上的内容比快照更新鲜。

#### 淘汰的单位是「一条结果」，不是「一条消息」

两种 provider 对结果的打包方式不一样，这是这一层唯一容易出错的地方：

| 形态 | 结构 |
|------|------|
| OpenAI 系 | 一条结果 = 一条 `role="tool"` 消息 |
| Anthropic | 一整轮结果打包进**一条** `role="user"` 消息的 content 列表，每条是 `{"type": "tool_result", "tool_use_id": ...}` block |

只扫 `role="tool"` 意味着 Anthropic 路径上这一层从来没生效过——不报错，只是静默失效。所以遍历以「结果」为单位展开，两种形态都覆盖。`tool_result` block 本身不带工具名，占位符通过发起调用的那条 assistant 消息的 `tool_calls` 反查 `tool_use_id` 得到。

同一个形态差异还影响另一处，已按同样口径处理：

- Layer 2 保留「最后一条 user 消息之后的整段尾巴」。Anthropic 的工具轮本身就是 user 消息，从那里切会留下一批 `tool_result`，而它们对应的 `tool_use` block 在刚被摘要替换掉的 assistant 消息里——这种孤儿 block 会被 API 直接拒绝。所以判断尾巴时跳过承载工具结果的 user 消息。

### Layer 2：摘要（`CompressionManager`）

淘汰兜不住时才走这层。`ToolConfig.compression_manager` 留空时自动创建（给 `/compact` 和跨 provider fallback 用）。自动触发由 `ToolConfig.enable_auto_compact` 控制（默认开）；关掉后 runner / 原生 compact / `prompt_too_long` 后的 reactive 都不跑，超窗就把 provider 错误抛出。`/compact` 不受此开关影响。

## Responses 原生 Compact

`OpenAIResponses` 通过 OpenAI SDK 的 `client.responses.compact()` 调用 `<base_url>/responses/compact`。例如：

```python
from agentica import Agent, OpenAIResponses

agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.6-sol",
        base_url="https://api.example.com/v1",
        reasoning="high",
        max_output_tokens=8192,
    ),
)
```

自动触发阈值同时考虑 `context_window`、`max_output_tokens`、安全 buffer 和 `CompressionManager.compress_token_limit`，保证 compact 请求仍能被服务端接受。

返回的 `response.compaction.output` 是下一轮的 canonical input，不是人类可读摘要。Agentica 会：

- 原样保存并回放完整 output，不裁剪 opaque `compaction` item。
- 每轮重新附加当前 system instructions。
- 在 SessionLog 中持久化 checkpoint，`resume` 后继续使用。
- 同时保留普通 role/content transcript，供 CLI 展示和跨 provider fallback。
- 仅在 provider、model 和 `base_url` 全部一致时使用 checkpoint；其他 provider 不会收到 opaque item。

跨 provider fallback 前，Agentica 会用现有本地 summary 压缩 portable transcript。原生 endpoint 不支持、请求失败或 checkpoint 身份不匹配时，也不会把 opaque 数据伪装成普通摘要。

## CompressionManager 配置

```python
from agentica import Agent, OpenAIChat, CompressionManager
from agentica.agent.config import ToolConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tool_config=ToolConfig(
        compression_manager=CompressionManager(
            model=OpenAIChat(id="gpt-4o-mini"),  # 用便宜模型做摘要
            compress_token_limit=100000,         # 触发摘要的 token 阈值
            compress_target_token_limit=60000,   # 摘要后的目标 token 数
        ),
    ),
)
```

两个阈值都可以省略：留空时运行时按 `model.context_window` 推导（80% 触发 / 50% 目标）。`model` 省略时使用调用方传入的活跃模型。

摘要会保留两样东西：**system prompt**（否则本轮剩下的调用没有任何指令）和**从最后一条 user 消息开始的整个尾部**（否则对话以 assistant 结尾，provider 会直接拒绝）。

## 开关（默认都开）

两层都可以关。比例旋钮仍然只有 `AGENTICA_EVICT_THRESHOLD_RATIO`；这两个布尔管的是「要不要自动做」，不是「做多狠」。

| 开关 | 默认 | 关掉之后 |
|------|------|----------|
| `ToolConfig.enable_evict` | `True` | Layer 1 不淘汰。窗口更容易涨到 Layer 2 |
| `ToolConfig.enable_auto_compact` | `True` | runner 自动摘要、原生 compact、`prompt_too_long` 后的 reactive 都不跑。`/compact` 仍可用 |

```python
from agentica import Agent, OpenAIChat
from agentica.agent.config import ToolConfig

# SDK：评测 / 成本敏感服务可以关自动摘要
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

不配则和现在完全一样（约 95% 窗口才摘要）。1M 窗口配 `300000` 就在 30 万处摘要；32k 窗口配 `128000` 仍被窗口挡住。写在 profile 上（每个模型可以不同），或 `settings.compact_token_limit` 做全局默认。CLI：`/config set compact_token_limit 300000`、`--compact-token-limit`。SDK：`ToolConfig(compact_token_limit=300000)`。

## Layer 0：工具输出预算

不是压缩，是输出策略——在**结果产生的那一刻**（`Model.run_function_calls`）就把它限住，
所以超大输出一次都不会完整进入上下文。两条规则：

| 规则 | 阈值 | 说明 |
|------|------|------|
| 单条结果 | `Function.max_result_size_chars`（`execute` 为 `max_output_length`，默认 20,000 字符） | 单个 tool result 超过此值就收缩。`read_file` 为 `None`（不收缩），否则它会去读自己的落盘文件，形成循环 |
| 单轮批次 | `0.25 × model.context_window` | 本轮全部新结果加起来超过窗口的这个份额时，从最大的开始收缩。Layer 1 从不动尾部批次（模型还没看过），所以一轮并行 6 个大调用只有这里能兜 |

批次预算按**窗口比例**而不是固定字符数：固定 200K 字符在 512K token 的窗口上会误伤，
在 8K token 的窗口上又完全不触发。

### 收缩成什么形态，取决于这个 session 能不能取回

`can_recover_spill(model.functions)` 检查是否注册了 `read_file` 或 `execute`：

- **能取回**（CLI、带文件工具的 agent）：写入磁盘，上下文里换成预览 + 路径（`<persisted-output>`），
  模型一次 `read_file` 就能拿回全量。

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

原生 compact 或 `CompressionManager.auto_compact` 在以下条件触发：

1. 当前 token 数超过安全阈值或 `compress_token_limit`
2. 原生 compact 不可用或失败时，进入本地压缩
3. 本地 auto-compact 带有 circuit-breaker，防止连续失败重复调用

手动 `/compact [instructions]` 使用相同优先级：Responses 原生 endpoint 优先，失败后回退到本地 LLM summary，再失败则如实报告失败、不改动历史。

## 观测压缩是否发生

摘要不可逆，而且要花一次 LLM 调用。SDK 调用方没有 CLI 的事件回调，所以次数直接挂在响应上：

```python
response = await agent.run("...")
if response.context_compactions:
    logger.info(f"本轮压缩了 {response.context_compactions} 次历史")
```

原生 compact、本地 Layer 2、以及 `prompt_too_long` 之后的 reactive 压缩都会计数；
Layer 1 淘汰是免费且可通过重跑工具恢复的，不计数。

## 下一步

- [RunConfig](run-config.md) -- 超时和成本控制
- [Hooks](hooks.md) -- on_pre_compact / on_post_compact
- [Agent 概念](../concepts/agent.md) -- Agent 上下文管理
