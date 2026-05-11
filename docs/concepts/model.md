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

    # 通用适配
    OpenAILike,         # 兼容 OpenAI API 的任意模型
)
```

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

通过 `model/providers.py` 的注册工厂创建：

```python
from agentica.model.providers import create_provider

# Full param shape: provider slug + id (model name) + api_key + base_url(optional)
model = create_provider(
    "deepseek",
    id="deepseek-v4-flash",
    api_key="sk-xxx",                              # 也可不传，自动读 DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com",           # 可选；私有部署/代理时覆盖
)

# 其他 provider 同样调用方式，slug + id 即可
model = create_provider("qwen",    id="qwen-max",          api_key="sk-xxx")
model = create_provider("zhipuai", id="glm-4.7-flash",     api_key="sk-xxx")
model = create_provider("ark",     id="doubao-1.5-pro-32k", api_key="sk-xxx")  # 火山引擎
```

支持: deepseek, qwen, zhipuai, moonshot, ark, together, xai, yi, nvidia, sambanova, groq, cerebras, mistral

## Model 内置安全机制

Model 层集成了多项运行时安全机制：

| 机制 | 说明 |
|------|------|
| **Death Spiral 检测** | 连续 5 轮全部工具调用失败时自动停止 |
| **Cost Budget** | 通过 `CostTracker` 追踪成本，超预算自动停止 |
| **Max Tokens Recovery** | 输出被截断时自动重试（最多 3 次） |
| **API 重试** | 可重试错误（429, 503, timeout 等）指数退避重试（最多 3 次） |
| **Context 压缩** | token 超限时自动触发 3 层压缩（micro → auto → reactive） |

## 下一步

- [Agent 核心概念](agent.md) -- Agent 如何使用 Model
- [模型提供商完整指南](../guides/models.md) -- 每个提供商的详细配置
- [RunConfig](../advanced/run-config.md) -- 运行时配置
