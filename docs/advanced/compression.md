# Context Compression

Agentica 提供多层上下文压缩策略，防止长对话或大量工具输出导致 token 超限。使用 `OpenAIResponses` 且服务端支持 `/responses/compact` 时，优先保留模型原生状态；其他模型继续使用可跨 provider 的本地压缩。

## 压缩架构

```
Tool 输出 (可能很大)
    |
    v
[Tool Result Storage] -- 超大输出持久化到磁盘
    |
    v
Context Messages
    |
    v
[Native Compact] -- Responses 原生 checkpoint（优先）
    |
    +--> 成功：保留 portable transcript，后续原样回放 checkpoint
    +--> 失败：[Micro] -> [Rule-based] -> [LLM Summary]
                                      |
                                      +--> prompt-too-long: Reactive Compact
```

实际执行顺序如下：

1. 超大工具结果持久化到磁盘。
2. `OpenAIResponses` 尝试 provider-native compact。
3. Micro-compact 清理旧工具结果。
4. Rule-based 截断旧结果并丢弃旧轮次。
5. 本地 LLM summary 压缩为可移植文本。
6. API 返回 `prompt_too_long` 时强制执行本地 reactive compact 后重试。

原生 compact 成功后会直接跳过第 3～5 步，避免先破坏历史再调用官方端点。原生调用失败会输出明确 warning 并继续本地流水线；第 6 步始终使用本地压缩，因为 `/responses/compact` 的输入本身也必须仍在模型 context window 内。

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

## CompressionManager

### 基本配置

```python
from agentica import Agent, OpenAIChat, CompressionManager

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    compression_manager=CompressionManager(
        compress_tool_results=True,
        compress_token_limit=100000,       # 触发压缩的 token 阈值
        compress_target_token_limit=60000, # 压缩后的目标 token 数
    ),
)
```

### 本地两阶段压缩策略

**Stage 1 -- Rule-based（免费，始终先执行）**：

- 截断最旧的未压缩工具结果到 `truncate_head_chars` 字符
- 如果仍超限，丢弃最旧的消息轮次，只保留最近 `keep_recent_rounds` 轮

**Stage 2 -- LLM-based（可选，消耗 token）**：

- 使用轻量级 LLM 智能摘要工具结果
- 保留关键信息：数字、日期、实体、标识符
- 删除冗余内容：过渡语、元评论、格式

```python
manager = CompressionManager(
    model=OpenAIChat(id="gpt-4o-mini"),  # 用便宜模型做摘要
    compress_tool_results=True,
    use_llm_compression=True,
)
```

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

手动 `/compact [instructions]` 使用相同优先级：Responses 原生 endpoint 优先，失败后回退到本地 LLM summary，再失败则使用 rule-based 压缩。

## 下一步

- [RunConfig](run-config.md) -- 超时和成本控制
- [Hooks](hooks.md) -- on_pre_compact / on_post_compact
- [Agent 概念](../concepts/agent.md) -- Agent 上下文管理
