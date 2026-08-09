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

同一个形态差异还影响另外两处，都已按同样口径处理：

- `sanitize_tool_pairs` 只认 `role="tool"` 形态。Anthropic transcript 在它眼里像是每个调用都没有回复，重建会给每个调用插一条占位 `role="tool"` 消息、把本来没坏的 transcript 弄坏，因此这类 transcript 原样返回。
- Layer 2 保留「最后一条 user 消息之后的整段尾巴」。Anthropic 的工具轮本身就是 user 消息，从那里切会留下一批 `tool_result`，而它们对应的 `tool_use` block 在刚被摘要替换掉的 assistant 消息里——这种孤儿 block 会被 API 直接拒绝。所以判断尾巴时跳过承载工具结果的 user 消息。

### Layer 2：摘要（`CompressionManager`）

淘汰兜不住时才走这层。它总是可用的（`ToolConfig.compression_manager` 留空时自动创建），否则长会话除了被 provider 拒绝没有别的出路。

## Responses 原生 Compact

`OpenAIResponses` 通过 OpenAI SDK 的 `client.responses.compact()` 调用 `<base_url>/responses/compact`。例如：

```python
from agentica import Agent, OpenAIResponses

agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.6-sol",
        base_url="https://v2.open.venus.woa.com/llmproxy/v1",
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

## Tool Result Storage

当单个工具输出超过阈值时，自动持久化到磁盘：

```
~/.agentica/projects/<project-hash>/<session-id>/tool-results/
+-- {tool_use_id}.txt    # 完整输出
```

Context 中只保留前 2000 字符的预览 + 文件路径。

### 两层预算

| 层级 | 阈值 | 说明 |
|------|------|------|
| 单工具限制 | 50,000 字符 | 单个 tool result 超过此值 -> 持久化 |
| 消息预算 | 200,000 字符 | 单条消息中所有 tool_result 总和超过此值 -> 持久化最大的几个 |

### 配置

通过 `Function.max_result_size_chars` 控制单工具阈值：

- 默认阈值：50,000 字符
- 设为 `None` 禁用持久化
- 预览长度：2,000 字符

### 工作流程

1. 工具执行完成，返回输出字符串
2. Layer 1: `maybe_persist_result()` 检查单工具大小，超限 -> 写入磁盘 -> 返回预览
3. Layer 2: `enforce_tool_result_budget()` 检查本轮所有 tool_result 总大小，超限 -> 持久化最大的结果

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

手动 `/compact [instructions]` 使用相同优先级：Responses 原生 endpoint 优先，失败后回退到本地 LLM summary，再失败则使用 CLI 内置的拼接式摘要。

## 下一步

- [RunConfig](run-config.md) -- 超时和成本控制
- [Hooks](hooks.md) -- on_pre_compact / on_post_compact
- [Agent 概念](../concepts/agent.md) -- Agent 上下文管理
