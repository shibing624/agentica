# OpenAI Responses API

Agentica 的 `OpenAIResponses` 适配 OpenAI Responses API，同时保留与其他模型一致的 `Agent` 运行方式。它支持普通和流式文本、reasoning summary、图片输入、函数工具调用、结构化输出、多轮工具状态回放以及原生 `/responses/compact`。

如果现有 OpenAI-compatible 服务只实现 Chat Completions，请继续使用 `OpenAIChat`。Responses API 不是简单的 URL 别名，两者的请求结构、推理参数和工具结果格式不同。

## SDK 快速开始

```python
from agentica import Agent, OpenAIResponses

agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.6-sol",
        reasoning="high",
        max_output_tokens=4096,
    )
)

result = agent.run_sync("解释这个仓库的主要模块")
print(result.content)
```

API Key 默认读取 `OPENAI_API_KEY`，也可以显式传入：

```python
model = OpenAIResponses(
    id="gpt-5.6-sol",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```

自定义 `base_url` 适用于真正兼容 Responses API 的代理或私有端点；只兼容 `/chat/completions` 的服务仍应使用 `OpenAIChat`。

`OpenAIResponses` 依赖 `openai>=2.51.0`。Agentica 的安装依赖已包含该下限。

## 主要参数

| 参数 | 说明 |
|------|------|
| `reasoning` | 推理强度：`none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` |
| `max_output_tokens` | Responses API 的最大输出 token 数 |
| `parallel_tool_calls` | 是否允许模型请求并行函数调用 |
| `truncation` | Responses API 的截断策略 |
| `base_url` | OpenAI 或兼容端点的 API 根地址 |

`OpenAIResponses` 继承 `OpenAIChat` 的通用模型配置，包括 API Key、工具、流式输出和结构化输出等能力。但推理参数必须使用 `reasoning`，不能传 Chat Completions 的 `reasoning_effort`，同时传入会直接报错。

在多轮工具调用中，Agentica 会把工具结果映射为 `function_call_output`，并回放响应中的函数调用和 reasoning 状态。启用 reasoning 且不使用服务端存储时，会请求 `reasoning.encrypted_content`，让后续轮次能够继续推理而不暴露内部推理文本。

## 原生上下文压缩

当上下文接近安全阈值时，`OpenAIResponses` 优先调用：

```text
POST <base_url>/responses/compact
```

例如配置 `base_url: https://api.example.com/v1` 时，请求地址自动为 `https://api.example.com/v1/responses/compact`，不需要单独配置 compact URL。

服务端返回的完整 `response.compaction.output` 会作为下一轮 canonical input 原样回放。Agentica 不解析或裁剪 opaque `compaction` item，并在 SessionLog 中保存 checkpoint，使 `/resume` 后仍可继续。普通 role/content transcript 会同时保留，CLI 历史和跨 provider fallback 不依赖 opaque 数据。

以下情况自动使用现有本地压缩：

- `/responses/compact` 不可用或请求失败。
- fallback 切换到不同 provider、model 或 `base_url`。
- 已收到 `prompt_too_long`；此时 compact 请求本身也可能超限。

CLI 的 `/compact [instructions]` 同样原生优先。`instructions` 会传给 compact endpoint；若原生调用失败，则作为本地 summary 指令继续使用。详细执行顺序见 [Context Compression](../advanced/compression.md)。

OpenAI 官方参考：[Compaction 指南](https://developers.openai.com/api/docs/guides/compaction) · [`responses.compact` API](https://developers.openai.com/api/reference/resources/responses/methods/compact)

## CLI 和 Gateway 配置

CLI 与 Gateway 共用 `~/.agentica/config.yaml`。通过 `wire_api: responses` 选择 Responses API：

```yaml
active_profile: responses

profiles:
  responses:
    model_provider: openai
    model_name: gpt-5.6-sol
    api_key: sk-...
    base_url: https://api.openai.com/v1
    wire_api: responses
    reasoning: high
    max_tokens: 4096
    context_window: 200000
```

配置文件中的 `max_tokens` 会映射为 Responses API 的 `max_output_tokens`。`wire_api: responses` 只支持 `model_provider: openai`；其他 provider 使用该字段会在配置校验时失败。

省略 `wire_api` 时默认使用 `chat_completions`。同一配置块中不要同时设置 `reasoning` 和 `reasoning_effort`：

```yaml
# Responses API
wire_api: responses
reasoning: high

# Chat Completions
wire_api: chat_completions
reasoning_effort: high
```

## 为 Subagent 配置辅助模型

`model_tier: auxiliary` 的 Subagent 会优先使用 task 专用模型，其次使用 `auxiliary_model`，最后回退到主模型。辅助模型也可以独立选择 Responses API：

```yaml
active_profile: default

profiles:
  default:
    model_provider: openai
    model_name: gpt-5.6-sol
    wire_api: responses
    reasoning: high

    auxiliary_model:
      model_provider: openai
      model_name: gpt-5-mini
      wire_api: responses
      reasoning: medium
      max_tokens: 4096
```

这样主对话使用较高推理强度，资料搜索、上下文压缩和默认 Subagent 等辅助任务使用单独模型。Subagent 的 `model_tier` 配置见 [Subagent 文档](../multi-agent/subagent.md)。

## Chat Completions 还是 Responses

| 场景 | 建议 |
|------|------|
| OpenAI 新模型，并需要原生 reasoning 状态或 Responses 工具协议 | `OpenAIResponses` |
| 现有代码依赖 `/chat/completions` | `OpenAIChat` |
| OpenAI-compatible 代理只声明兼容 Chat Completions | `OpenAIChat` |
| 代理明确实现 `/responses` 和对应事件格式 | `OpenAIResponses(base_url=...)` |

两种类都使用统一的 `Agent.run()` / `run_sync()` 接口，因此切换 wire API 不需要修改 Agent 调用流程。
