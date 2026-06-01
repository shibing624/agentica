# Model

Model 是 Agent 的"大脑"，提供推理和生成能力。Agentica 支持 20+ 模型提供商。

## 支持的模型

```python
from agentica import (
    # OpenAI 系列
    OpenAIChat,         # gpt-4o, gpt-4o-mini
    AzureOpenAIChat,    # Azure 部署的 OpenAI 模型

    # 国内模型
    ZhipuAI,            # glm-4.7-flash（免费）, glm-4-plus
    DeepSeek,           # deepseek-chat, deepseek-reasoner
    Qwen,               # qwen-plus, qwen-turbo
    Moonshot,           # moonshot-v1-128k
    Ark,                # 火山引擎，跑 doubao-pro-32k 等
    Yi,                 # yi-large

    # 海外模型
    Claude,             # claude-3.5-sonnet
    Grok,               # grok-beta
    Together,           # 多种开源模型

    # 本地模型
    Ollama,             # llama3, mistral, qwen2 等

)
```

> 兼容 OpenAI API 的任意自定义端点：直接用 `OpenAIChat(id=..., api_key=..., base_url=...)`。

## 通用参数

所有 Model 子类共享以下参数：

```python
model = OpenAIChat(
    id="gpt-4o",              # 模型 ID
    api_key="sk-xxx",         # API 密钥（或通过环境变量）
    base_url="https://...",   # API 地址
    temperature=0.7,          # 温度
    max_tokens=4096,          # 最大输出 token
    timeout=60,               # 超时秒数
    context_window=128000,    # 上下文窗口大小
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 模型 ID |
| `api_key` | `str` | API 密钥 |
| `base_url` | `str` | API 地址 |
| `temperature` | `float` | 生成温度 (0-2) |
| `max_tokens` | `int` | 最大输出 token 数 |
| `timeout` | `int` | 请求超时秒数 |
| `context_window` | `int` | 上下文窗口大小 |
| `response_format` | `Any` | 响应格式 |

## 模型架构

`Model` 基类使用 `@dataclass` + `ABC`：

```python
@dataclass
class Model(ABC):
    id: str = "not-provided"
    context_window: int = 128000
    tools: Optional[List[ModelTool]] = None
    ...

    @abstractmethod
    async def response(self, messages) -> ModelResponse: ...

    @abstractmethod
    async def response_stream(self, messages) -> AsyncIterator[ModelResponse]: ...
```

### 核心提供商

| 目录 | 提供商 |
|------|--------|
| `model/openai/` | OpenAI |
| `model/anthropic/` | Anthropic (Claude) |
| `model/kimi/` | Moonshot |
| `model/ollama/` | Ollama (本地) |
| `model/litellm/` | LiteLLM (通用适配) |
| `model/azure/` | Azure OpenAI |

### OpenAI 兼容提供商

直接从 `agentica` 顶层导入对应的工厂类，每个工厂内部硬编码了 `base_url` 与默认环境变量。

```python
from agentica import DeepSeekChat, QwenChat, ZhipuAIChat, ArkChat

# api_key 不传则自动读对应 env（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ZAI_API_KEY / ARK_API_KEY）
model = DeepSeekChat(id="deepseek-v4-flash")
model = QwenChat(id="qwen-max")
model = ZhipuAIChat(id="glm-4.7-flash")
model = ArkChat(id="doubao-1.5-pro-32k")                  # 火山引擎

# 私有部署 / 代理：传 base_url 覆盖
model = DeepSeekChat(id="deepseek-v4-pro", base_url="https://my-proxy/api")
```

支持的工厂：`DeepSeekChat`, `QwenChat`, `ZhipuAIChat`, `MoonshotChat`, `ArkChat`, `TogetherChat`, `GrokChat`,
`YiChat`, `NvidiaChat`, `SambanovaChat`, `OpenRouterChat`, `FireworksChat`, `InternLMChat`。

需要按字符串 slug 派发（gateway / multi-tenant 场景）时使用 `agentica.PROVIDER_FACTORIES`：

```python
from agentica import PROVIDER_FACTORIES
model = PROVIDER_FACTORIES["deepseek"](id="deepseek-v4-flash")
```

## Model 内置安全机制

Model 层集成了多项运行时安全机制：

| 机制 | 说明 |
|------|------|
| **Death Spiral 检测** | 连续 5 轮全部工具调用失败时自动停止 |
| **Cost Budget** | 通过 `CostTracker` 追踪成本，超预算自动停止 |
| **Max Tokens Recovery** | 输出被截断时自动重试（最多 3 次） |
| **API 重试** | 可重试错误（429, 503, timeout 等）指数退避重试（最多 3 次） |
| **Context 压缩** | token 超限时自动触发 3 层压缩（micro → auto → reactive） |

当 Death Spiral / Max Turns / Cost Budget 任一安全检查中断循环时，Runner **不会**把英文错误文本拼进回复内容，而是写入 `RunResponse` 的结构化字段，下游据此分支处理，无需正则剥离：

```python
resp = await agent.run("...")
if not resp.is_complete:                 # 被安全检查中断
    if resp.break_reason == RunBreakReason.DEATH_SPIRAL.value:
        ...  # 软降级 / 重试 / 告警；resp.content 保持干净
    # resp.break_message 为人类可读详情（仅用于日志，不要直接发给用户）
```

`break_reason` 取值见 `RunBreakReason`：`death_spiral` / `max_turns` / `cost_budget`。

### 断点自动兜底（fallback_on_break）

若不想让下游自己处理 break，可开启 `fallback_on_break`：循环被安全检查中断时，Runner 会用 `fallback_models` 链做**一次无工具补跑**，回放完整历史（含失败的 tool 调用，让模型看到失败原因），把回复填进 `content`：

```python
agent = Agent(
    model=primary_model,
    fallback_models=[cheap_model],   # 兜底模型（无需绑定工具）
    fallback_on_break=True,          # 仅 break 触发，异常抛出仍交给调用方
)
resp = await agent.run("...")
reply = resp.content                 # 永远可用，下游无需写 fallback 代码
# 观测：resp.break_reason 仍记录中断原因；resp.fallback_used=True；
#       resp.model = 实际应答的兜底模型 id（主模型仍是 agent.model.id）
```

补跑只在 loop-break 时触发（异常抛出不触发，交由调用方 try/except）；不注入任何人设提示，纯靠历史上下文。

**下游契约（重要）**：开启 `fallback_on_break` 后，`is_complete` 仍为 `False`（循环确实被中断过，保留供观测）。**判断"有没有可发的回复"要看 `content` / `fallback_used`，不要再用 `if not is_complete: reply=""` 那套**——否则会把已经兜好的回复丢掉。两种模式二选一：

- 关闭（默认）：`content` 在 break 时为空/残缺，下游自己决定怎么降级（查 `is_complete`）。
- 开启：库已兜底，下游直接读 `content`；`break_reason` / `fallback_used` / `model` 仅用于日志告警。

其它注意点：

- **流式**：兜底回复会作为一个 chunk yield 出来，同时写入最终 `run_response.content`。若主模型在 break 前已 yield 过可见文本，消费者累加的分块会是"旧文本+兜底"，**以最终 `run_response.content` 为准**。
- **成本**：`cost_budget` 触发时，兜底仍会多产生**一次**有界 LLM 调用，会略微超出预算上限（计入 `cost_tracker`）。
- **模型选择**：兜底模型应是**与主模型不同的实例**（最好跨 provider）。若把绑了工具的主模型自己塞进 `fallback_models`，兜底可能只产 tool_calls、`content` 为空（工具不执行），等于兜底失效。
- **结构化输出**：`response_model` 只绑在主模型上，兜底回复是纯文本、不保证能解析成目标模型。

## 下一步

- [Agent 核心概念](agent.md) -- Agent 如何使用 Model
- [模型提供商完整指南](../guides/models.md) -- 每个提供商的详细配置
- [RunConfig](../advanced/run-config.md) -- 运行时配置
