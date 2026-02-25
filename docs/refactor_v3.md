# Agentica v3 重构方案

> 本文档是 v3 架构重构的完整设计方案，涵盖 Model 层精简、结构化输出、Agent dataclass 化、Tool 注册机制、Runner 拆分、Guardrails 统一抽象、`__init__.py` 拆分等所有改动。

---

## 0. 工程方案：Git Worktree

使用 `git worktree` 而非 `git branch` 进行大规模重构，保留 main 分支可随时对比运行。

```bash
cd /Users/xuming/Documents/Codes/agentica
git branch refactor/v3
git worktree add ../agentica-refactor refactor/v3

# 所有重构在 ../agentica-refactor 目录进行
# main 目录保持不动，随时可跑测试对比
# 完成后合并回 main
```

---

## 1. Model 层精简与 OpenAI-Compatible 注册机制

### 1.1 现状分析

当前 `agentica/model/` 下有 **24 个 provider 子目录**，分为三类：

| 类型 | Provider | 代码量 | 特点 |
|------|---------|--------|------|
| **保留（核心实现）** | `openai/` | ~968行 | OpenAIChat + OpenAILike，完整 async 链路 |
| **保留（独立 SDK）** | `anthropic/` | ~500行 | Claude，独立 Anthropic SDK |
| **保留（Azure 特化）** | `azure/` | ~100行 | AzureOpenAIChat，有自己的 client 初始化 |
| **保留（统一网关）** | `litellm/` | ~400行 | LiteLLM，覆盖 100+ provider |
| **保留（本地模型）** | `ollama/` | ~400行 | Ollama，独立 SDK |
| **删除（OpenAILike 薄壳）** | deepseek, doubao, qwen, xai, yi, nvidia, internlm, moonshot, zhipuai, sambanova, openrouter, fireworks | 各 24-30行 | 仅设置 id/base_url/api_key |
| **删除（有自定义逻辑）** | together | 179行 | monkey_patch 流式解析，LiteLLM 可替代 |
| **删除（独立 SDK 实现）** | google, groq, cohere, mistral, huggingface | 各 390-671行 | LiteLLM 完全覆盖 |
| **删除（云平台 SDK）** | aws, vertexai | 各 480-650行 | LiteLLM 完全覆盖 |
| **删除（继承 Claude）** | kimi | 53行 | 改指 Kimi API 的 Claude 薄壳 |

### 1.2 保留目录（5 个 provider + 公共文件）

```
agentica/model/
├── __init__.py          # 精简后的导出
├── base.py              # Model 基类（改为 @dataclass）
├── base_audio_model.py  # 音频模型基类
├── content.py           # Media/Image/Audio/Video
├── message.py           # Message 定义
├── response.py          # ModelResponse/ModelResponseEvent
├── usage.py             # Usage 跟踪
├── openai/              # OpenAIChat + OpenAILike
│   ├── chat.py
│   └── like.py
├── anthropic/           # Claude
│   └── claude.py
├── azure/               # AzureOpenAIChat
│   └── openai_chat.py
├── litellm/             # LiteLLMChat
│   └── chat.py
├── ollama/              # OllamaChat + Hermes + OllamaTools
│   ├── chat.py
│   ├── hermes.py
│   └── tools.py
└── providers.py         # 【新增】OpenAI-Compatible 预置 provider 注册表
```

### 1.3 删除目录（19 个）

```
deepseek/ doubao/ qwen/ xai/ yi/ nvidia/ internlm/ moonshot/
zhipuai/ sambanova/ openrouter/ fireworks/ together/ kimi/
google/ groq/ cohere/ mistral/ huggingface/ aws/ vertexai/
```

### 1.4 OpenAI-Compatible Provider 注册/工厂机制

**新增 `agentica/model/providers.py`**，用一个注册表替代 12 个目录：

```python
"""OpenAI-Compatible provider 预置配置注册表。

消除了 deepseek/、qwen/ 等 12 个仅配置 base_url/api_key 的薄壳目录，
统一通过 create_provider() 工厂函数或直接使用 OpenAILike 创建。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from os import getenv


@dataclass
class ProviderConfig:
    """OpenAI-Compatible provider 的预置配置。"""
    name: str
    default_model: str
    base_url: str
    api_key_env: str                       # 环境变量名
    api_key_env_fallback: Optional[str] = None  # 备选环境变量名
    provider: Optional[str] = None         # 显示用的 provider 名
    context_window: int = 128000
    max_output_tokens: Optional[int] = None


# ── 全局注册表 ──────────────────────────────────────────────
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}


def register_provider(key: str, config: ProviderConfig) -> None:
    """注册一个 OpenAI-Compatible provider 配置。"""
    PROVIDER_REGISTRY[key] = config


def get_provider_config(key: str) -> ProviderConfig:
    """获取已注册的 provider 配置。"""
    if key not in PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown provider '{key}'. "
            f"Available: {sorted(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[key]


def create_provider(key: str, **overrides):
    """工厂函数：根据注册表创建 OpenAILike 实例。

    Args:
        key: 注册表中的 provider 名称，如 "deepseek", "qwen"
        **overrides: 覆盖默认配置的参数，如 id="deepseek-reasoner"

    Returns:
        OpenAILike 实例

    Example:
        model = create_provider("deepseek")
        model = create_provider("deepseek", id="deepseek-reasoner")
        model = create_provider("qwen", id="qwen-turbo")
    """
    from agentica.model.openai.like import OpenAILike

    config = get_provider_config(key)
    api_key = getenv(config.api_key_env)
    if api_key is None and config.api_key_env_fallback:
        api_key = getenv(config.api_key_env_fallback)

    params = {
        "id": config.default_model,
        "name": config.name,
        "provider": config.provider or config.name,
        "api_key": api_key,
        "base_url": config.base_url,
    }
    if config.max_output_tokens is not None:
        params["max_output_tokens"] = config.max_output_tokens

    params.update(overrides)
    return OpenAILike(**params)


# ── 内置 provider 注册 ──────────────────────────────────────
# 以下注册替代了原来 12 个独立目录中的 24 行薄壳类

register_provider("deepseek", ProviderConfig(
    name="DeepSeek",
    default_model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
))

register_provider("doubao", ProviderConfig(
    name="Doubao",
    default_model=getenv("ARK_MODEL_NAME", "doubao-1.5-pro-32k"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key_env="ARK_API_KEY",
    provider="ByteDance",
))

register_provider("qwen", ProviderConfig(
    name="Qwen",
    default_model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="DASHSCOPE_API_KEY",
    provider="Alibaba",
))

register_provider("xai", ProviderConfig(
    name="Grok",
    default_model="grok-beta",
    base_url="https://api.x.ai/v1",
    api_key_env="XAI_API_KEY",
    provider="xAI",
))

register_provider("yi", ProviderConfig(
    name="Yi",
    default_model="yi-lightning",
    base_url="https://api.lingyiwanwu.com/v1",
    api_key_env="YI_API_KEY",
    provider="01.ai",
))

register_provider("nvidia", ProviderConfig(
    name="Nvidia",
    default_model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="NVIDIA_API_KEY",
))

register_provider("internlm", ProviderConfig(
    name="InternLM",
    default_model="internlm2.5-latest",
    base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions",
    api_key_env="INTERNLM_API_KEY",
))

register_provider("moonshot", ProviderConfig(
    name="MoonShot",
    default_model="kimi-k2.5",
    base_url="https://api.moonshot.cn/v1",
    api_key_env="MOONSHOT_API_KEY",
))

register_provider("zhipuai", ProviderConfig(
    name="ZhipuAI",
    default_model="glm-4.7-flash",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="ZHIPUAI_API_KEY",
    api_key_env_fallback="ZAI_API_KEY",
))

register_provider("sambanova", ProviderConfig(
    name="Sambanova",
    default_model="Meta-Llama-3.1-8B-Instruct",
    base_url="https://api.sambanova.ai/v1",
    api_key_env="SAMBANOVA_API_KEY",
))

register_provider("openrouter", ProviderConfig(
    name="OpenRouter",
    default_model="gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    max_output_tokens=16384,
))

register_provider("fireworks", ProviderConfig(
    name="Fireworks",
    default_model="accounts/fireworks/models/firefunction-v2",
    base_url="https://api.fireworks.ai/inference/v1",
    api_key_env="FIREWORKS_API_KEY",
))

register_provider("kimi", ProviderConfig(
    name="Kimi",
    default_model="kimi-k2.5",
    base_url="https://api.moonshot.cn/v1",
    api_key_env="MOONSHOT_API_KEY",
    provider="MoonShot",
))
```

### 1.5 向后兼容别名

在 `agentica/model/__init__.py` 中提供向后兼容的别名：

```python
# 向后兼容：保留旧类名作为工厂函数别名
def DeepSeekChat(**kwargs):
    """Backward-compatible alias. Use create_provider('deepseek') instead."""
    return create_provider("deepseek", **kwargs)

def QwenChat(**kwargs):
    return create_provider("qwen", **kwargs)

def ZhipuAIChat(**kwargs):
    return create_provider("zhipuai", **kwargs)

# ... 其余类似
```

### 1.6 迁移示例

```python
# ── Before ──
from agentica import DeepSeekChat
model = DeepSeekChat()

# ── After（推荐）──
from agentica.model.providers import create_provider
model = create_provider("deepseek")
model = create_provider("deepseek", id="deepseek-reasoner")

# ── After（直接使用 OpenAILike）──
from agentica import OpenAILike
model = OpenAILike(
    id="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ── After（通过 LiteLLM 覆盖所有已删的独立 SDK provider）──
from agentica import LiteLLMChat
model = LiteLLMChat(id="gemini/gemini-2.0-flash")       # 替代 google/
model = LiteLLMChat(id="groq/llama-3.1-70b-versatile")  # 替代 groq/
model = LiteLLMChat(id="cohere/command-r-plus")          # 替代 cohere/
model = LiteLLMChat(id="mistral/mistral-large-latest")   # 替代 mistral/
model = LiteLLMChat(id="huggingface/meta-llama/...")     # 替代 huggingface/
model = LiteLLMChat(id="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")  # 替代 aws/
model = LiteLLMChat(id="vertex_ai/gemini-2.0-flash")     # 替代 vertexai/

# ── After（向后兼容，不推荐长期使用）──
from agentica import DeepSeekChat  # 实际是工厂函数别名
model = DeepSeekChat()
```

---

## 2. Model Provider 异步接口一致性

### 2.1 现状

| Provider | invoke | invoke_stream | response | response_stream | handle_tool_calls | handle_stream_tool_calls |
|----------|--------|---------------|----------|-----------------|-------------------|--------------------------|
| OpenAIChat | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async |
| Claude | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async |
| LiteLLM | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async (_前缀) | ✅ async (_前缀) |
| Ollama | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async | ✅ async |
| Azure | 继承 OpenAIChat | 继承 | 继承 | 继承 | 继承 | 继承 |

### 2.2 需要统一的接口

实际上五个保留 provider 都已经实现了完整的 async 链路。需要做的是：

1. **接口规范化**：在 `Model` 基类中用 `@abstractmethod` 明确标记必须实现的方法：

```python
from abc import ABC, abstractmethod

@dataclass
class Model(ABC):
    """LLM 模型抽象基类。"""

    @abstractmethod
    async def invoke(self, messages: List[Message]) -> Any:
        """调用 LLM API，返回原始响应。"""
        ...

    @abstractmethod
    async def invoke_stream(self, messages: List[Message]) -> AsyncIterator[Any]:
        """流式调用 LLM API。"""
        ...

    @abstractmethod
    async def response(self, messages: List[Message]) -> ModelResponse:
        """完整响应（含工具调用循环）。"""
        ...

    @abstractmethod
    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """流式响应（含工具调用循环）。"""
        ...
```

2. **方法命名统一**：LiteLLM 的 `_handle_tool_calls` / `_handle_stream_tool_calls` 去掉下划线前缀，与其他 provider 保持一致。

3. **返回值类型统一**：确保所有 provider 的 `response()` 返回 `ModelResponse`，`response_stream()` 返回 `AsyncIterator[ModelResponse]`，`invoke()` 和 `invoke_stream()` 的返回值类型可以各自不同（SDK 原始类型）。

4. **错误处理标准化**：在 `Model` 基类中定义统一的异常捕获和重试策略接口。

---

## 3. 结构化输出统一走 API 原生能力

### 3.1 现状

当前有两条路径：
- `structured_outputs=True` → OpenAI `beta.chat.completions.parse` → `model_response.parsed`
- `structured_outputs=False` → LLM 返回文本 → `parse_structured_output()` 正则/花括号提取 → Pydantic 验证

### 3.2 重构方案

**主路径：各 Model 的 `response()` / `response_stream()` 直接返回 `model_response.parsed`**

| Provider | 原生结构化输出方式 | 实现状态 |
|----------|-------------------|---------|
| OpenAIChat | `beta.chat.completions.parse(response_format=ResponseModel)` | ✅ 已实现 |
| Claude | Anthropic SDK `tool_use` 模式（传 tool schema，强制调用单一 tool） | ❌ 需新增 |
| LiteLLM | `response_format={"type": "json_schema", "json_schema": {...}}` | ❌ 需接入 |
| Ollama | `format=schema_dict`（Ollama 0.5+ 原生支持） | ❌ 需接入 |
| Azure | 继承 OpenAIChat，自动支持 | ✅ 已有 |

**具体改动：**

1. **OpenAIChat**：已经支持 `beta.chat.completions.parse`，保持不变。

2. **Claude**：新增结构化输出支持（通过 tool_use 模式）：
   ```python
   # 在 Claude.response() 中，当 response_model 存在时：
   # 1. 将 Pydantic model 转为 tool schema
   # 2. 设置 tool_choice={"type": "tool", "name": "structured_output"}
   # 3. 从 tool_use content block 中提取并解析 JSON
   # 4. 设置 model_response.parsed = ResponseModel(**parsed_data)
   ```

3. **LiteLLM**：接入原生 `response_format`：
   ```python
   # 在 LiteLLMChat.invoke() 中，当 response_format 存在时：
   # 传递 response_format={"type": "json_schema", "json_schema": schema}
   ```

4. **Ollama**：接入原生 `format` 参数：
   ```python
   # 在 OllamaChat.invoke() 中，当 response_format 存在时：
   # 传递 format=response_model.model_json_schema()
   ```

5. **Model 基类新增**：`get_response_format(response_model)` 方法，各子类覆写。

6. **Runner 侧简化**：
   - `_consume_run()` 中的 `parse_structured_output()` 文本后解析 **降级为 warning 级别的最后安全网**（不删除，保持兜底）
   - 主路径优先使用 `model_response.parsed`
   - `prompts.py` 中构建 JSON field description 的逻辑大幅简化（API 原生能力已约束格式）

7. **`utils/string.py` 中的 `parse_structured_output()`**：标记为 deprecated，保留作为兜底。

---

## 4. Agent 和 Model 的 dataclass 化

### 4.1 现状

- `Agent`：已是 `@dataclass(init=False)`
- `Model`：Pydantic `BaseModel`，使用了 `Field(alias="model")`、`ConfigDict`、`PrivateAttr`
- `Function`/`FunctionCall`/`ModelTool`：Pydantic `BaseModel`（需要序列化能力）
- `ModelResponse`：已是 `@dataclass`

### 4.2 改动范围

| 类 | Before | After | 理由 |
|----|--------|-------|------|
| `Model` | `BaseModel` | `@dataclass` | 与 Agent 一致，去掉 Pydantic 依赖 |
| `OpenAIChat` | `Model(BaseModel)` | `@dataclass` (继承 Model) | 同上 |
| `Claude` | `Model(BaseModel)` | `@dataclass` | 同上 |
| `LiteLLMChat` | `Model(BaseModel)` | `@dataclass` | 同上 |
| `OllamaChat` | `Model(BaseModel)` | `@dataclass` | 同上 |
| `AzureOpenAIChat` | `OpenAIChat` | `@dataclass` | 同上 |
| `Function` | `BaseModel` | **保留 BaseModel** | 需要 JSON Schema 生成和序列化 |
| `FunctionCall` | `BaseModel` | **保留 BaseModel** | 需要序列化 |
| `ModelTool` | `BaseModel` | **保留 BaseModel** | 需要序列化 |
| `Message` | `BaseModel` | **保留 BaseModel** | 需要序列化 |
| `RunResponse` | `BaseModel` | **保留 BaseModel** | 需要序列化 |

### 4.3 Model 基类改造细节

```python
# ── Before ──
class Model(BaseModel):
    id: str = Field(..., alias="model")
    name: Optional[str] = None
    provider: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    _client: Optional[Any] = PrivateAttr(default=None)

# ── After ──
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class Model(ABC):
    id: str = "not-provided"       # 去掉 alias，统一用 id
    name: Optional[str] = None
    provider: Optional[str] = None
    # ... 其他字段

    # 私有属性用 field(init=False, repr=False)
    _client: Optional[Any] = field(init=False, repr=False, default=None)

    def to_dict(self) -> dict:
        """替代 model_dump()。"""
        return {k: v for k, v in dataclasses.asdict(self).items()
                if not k.startswith("_")}
```

### 4.4 破坏性变更

- `Model(model="gpt-4o")` → `Model(id="gpt-4o")`（去掉 `alias="model"`）
- `model.model_dump()` → `model.to_dict()`
- `model.model_copy()` → `copy.copy(model)` 或手写浅拷贝
- `Field(default=...)` → `field(default=...)`
- `PrivateAttr()` → `field(init=False, repr=False)`

---

## 5. Tool 注册机制

### 5.1 保留裸函数自动解析

现有 `Function.from_callable()` 机制完整保留，用户可以继续传裸函数：

```python
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent(tools=[calculate])  # 自动解析 docstring + type hints → Function
```

### 5.2 新增 @tool 装饰器

```python
# agentica/tools/decorators.py（新增文件）
from functools import wraps

def tool(
    name: str = None,
    description: str = None,
    show_result: bool = False,
    sanitize_arguments: bool = True,
):
    """装饰器：为函数附加工具元数据，传给 Agent 时自动识别。

    Example:
        @tool(name="web_search", description="Search the web")
        def search(query: str, max_results: int = 5) -> str:
            ...

        agent = Agent(tools=[search])
    """
    def decorator(func):
        func._tool_metadata = {
            "name": name or func.__name__,
            "description": description or (func.__doc__ or "").strip(),
            "show_result": show_result,
            "sanitize_arguments": sanitize_arguments,
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._tool_metadata = func._tool_metadata
        return wrapper
    return decorator
```

### 5.3 全局 Tool Registry

```python
# agentica/tools/registry.py（新增文件）
from typing import Dict, Union, Callable

_TOOL_REGISTRY: Dict[str, Union[Callable, "Tool"]] = {}


def register_tool(name: str, tool_or_func) -> None:
    """注册一个工具到全局注册表。"""
    _TOOL_REGISTRY[name] = tool_or_func


def get_tool(name: str):
    """按名称获取已注册的工具。"""
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found. Available: {sorted(_TOOL_REGISTRY.keys())}")
    return _TOOL_REGISTRY[name]


def list_tools() -> list:
    """列出所有已注册的工具名称。"""
    return sorted(_TOOL_REGISTRY.keys())
```

### 5.4 Function.from_callable() 增强

在 `Function.from_callable()` 中增加对 `_tool_metadata` 属性的检测：

```python
@classmethod
def from_callable(cls, c: Callable, strict: bool = False) -> "Function":
    # 检测 @tool 装饰器元数据
    metadata = getattr(c, "_tool_metadata", None)
    if metadata:
        return cls(
            name=metadata["name"],
            description=metadata["description"],
            entrypoint=c,
            show_result=metadata.get("show_result", False),
            sanitize_arguments=metadata.get("sanitize_arguments", True),
        )
    # 原有逻辑：从 docstring + type hints 自动解析
    ...
```

---

## 6. Runner 拆分

### 6.1 现状

`RunnerMixin`（724 行）是一个混入类，包含所有执行逻辑。`Agent` 通过多继承获得这些方法。

### 6.2 拆分方案

**新建 `agentica/runner.py`，将 `RunnerMixin` 拆分为独立的 `Runner` 类：**

```python
# agentica/runner.py
class Runner:
    """Agent 的执行引擎。处理单轮/多轮 LLM 调用 + 工具执行。

    Runner 与 Agent 解耦：
    - Agent 负责定义（model, tools, instructions, prompt 构建）
    - Runner 负责执行（LLM 调用、工具调用、流式处理、记忆更新）
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent

    async def _run_impl(
        self, message, *, stream, audio, images, videos, messages, ...
    ) -> AsyncIterator[RunResponse]:
        """核心执行引擎（从 RunnerMixin._run_impl 迁移）。"""
        ...

    async def _consume_run(self, message, ...) -> RunResponse:
        """消费 _run_impl 生成器并返回最终响应。"""
        ...

    async def run(self, message=None, **kwargs) -> RunResponse:
        """主要异步 API（非流式）。"""
        ...

    async def run_stream(self, message=None, **kwargs) -> AsyncIterator[RunResponse]:
        """流式异步 API。"""
        ...

    def run_sync(self, message=None, **kwargs) -> RunResponse:
        """run() 的同步包装器。"""
        ...

    def run_stream_sync(self, message=None, **kwargs) -> Iterator[RunResponse]:
        """run_stream() 的同步包装器。"""
        ...
```

**Agent 侧改造（优雅委托）：**

```python
# agentica/agent/base.py
@dataclass(init=False)
class Agent(PromptsMixin, TeamMixin, ToolsMixin, PrinterMixin):
    # 注意：去掉了 RunnerMixin

    def __init__(self, ...):
        ...
        self._runner = Runner(self)

    # ── 公共 API 不变，用户代码零改动 ──

    async def run(self, message=None, **kwargs) -> RunResponse:
        return await self._runner.run(message, **kwargs)

    async def run_stream(self, message=None, **kwargs) -> AsyncIterator[RunResponse]:
        async for chunk in self._runner.run_stream(message, **kwargs):
            yield chunk

    def run_sync(self, message=None, **kwargs) -> RunResponse:
        return self._runner.run_sync(message, **kwargs)

    def run_stream_sync(self, message=None, **kwargs) -> Iterator[RunResponse]:
        return self._runner.run_stream_sync(message, **kwargs)
```

### 6.3 优势

- **Agent 公共 API 完全不变**：`agent.run()` / `agent.run_stream()` / `agent.run_sync()` 签名保持一致
- **Runner 可独立测试**：mock Agent 即可单独测试执行逻辑
- **可扩展**：`DeepAgent` 可以用不同的 Runner 实现（如 `DeepRunner`），无需修改 Agent 基类
- **关注点分离**：Agent 负责「是什么」（定义），Runner 负责「怎么做」（执行）

---

## 7. Guardrails 统一抽象

### 7.1 现状分析

`guardrails/base.py`（615 行）和 `guardrails/tool.py`（638 行）有大量结构重复：

| 结构 | base.py (Agent 级) | tool.py (Tool 级) |
|------|-------------------|-------------------|
| 异常基类 | `GuardrailTripwireTriggered` | `ToolGuardrailTripwireTriggered` |
| 输入异常 | `InputGuardrailTripwireTriggered` | `ToolInputGuardrailTripwireTriggered` |
| 输出异常 | `OutputGuardrailTripwireTriggered` | `ToolOutputGuardrailTripwireTriggered` |
| 函数输出 | `GuardrailFunctionOutput` | `ToolGuardrailFunctionOutput` |
| 输入结果 | `InputGuardrailResult` | `ToolInputGuardrailResult` |
| 输出结果 | `OutputGuardrailResult` | `ToolOutputGuardrailResult` |
| 输入 Guard | `InputGuardrail` | `ToolInputGuardrail` |
| 输出 Guard | `OutputGuardrail` | `ToolOutputGuardrail` |
| 装饰器 | `@input_guardrail` / `@output_guardrail` | `@tool_input_guardrail` / `@tool_output_guardrail` |
| 执行函数 | `run_input_guardrails()` / `run_output_guardrails()` | `run_tool_input_guardrails()` / `run_tool_output_guardrails()` |

两者核心逻辑几乎相同：**guard → validate → raise/reject 模式**。

### 7.2 重构方案：三层抽象

```
guardrails/
├── __init__.py          # 公共 API 导出
├── core.py              # 【新增】统一抽象层
├── agent.py             # 原 base.py，精简后继承 core
└── tool.py              # 精简后继承 core
```

**核心抽象层 `core.py`：**

```python
"""Guardrail 统一抽象层。

提供基础异常、行为定义、Guard 基类，
agent.py 和 tool.py 通过继承/组合实现具体逻辑。
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, Any, Callable, Union
from enum import Enum


# ── 行为定义（统一 Agent 级和 Tool 级） ──

class GuardrailAction(str, Enum):
    """Guardrail 触发后的行为。"""
    ALLOW = "allow"              # 通过
    BLOCK = "block"              # 阻断（Agent 级：抛异常；Tool 级：抛异常）
    REJECT_CONTENT = "reject"    # 替换内容继续（仅 Tool 级使用）


@dataclass
class GuardrailOutput:
    """Guardrail 函数的统一输出。"""
    action: GuardrailAction
    output_info: Any = None
    reject_message: Optional[str] = None  # 仅 REJECT_CONTENT 时使用

    @classmethod
    def allow(cls, output_info=None):
        return cls(action=GuardrailAction.ALLOW, output_info=output_info)

    @classmethod
    def block(cls, output_info=None):
        return cls(action=GuardrailAction.BLOCK, output_info=output_info)

    @classmethod
    def reject(cls, message: str, output_info=None):
        return cls(action=GuardrailAction.REJECT_CONTENT,
                   reject_message=message, output_info=output_info)


# ── 统一异常体系 ──

class GuardrailTriggered(Exception):
    """Guardrail 触发的基础异常。"""
    def __init__(self, guardrail_name: str, output: GuardrailOutput):
        self.guardrail_name = guardrail_name
        self.output = output
        super().__init__(f"Guardrail '{guardrail_name}' triggered: {output.action.value}")


# ── Guard 基类 ──

TData = TypeVar("TData")

@dataclass
class BaseGuardrail(Generic[TData]):
    """Guard 基类：封装验证函数 + 执行逻辑。"""
    guardrail_function: Callable
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = getattr(self.guardrail_function, "__name__", "unnamed_guardrail")

    async def run(self, data: TData) -> GuardrailOutput:
        """执行 guardrail 函数，支持同步和异步函数。"""
        import asyncio
        result = self.guardrail_function(data)
        if asyncio.iscoroutine(result):
            result = await result

        # 兼容：如果返回 bool，转为 GuardrailOutput
        if isinstance(result, bool):
            return GuardrailOutput.allow() if result else GuardrailOutput.block()
        if isinstance(result, GuardrailOutput):
            return result
        raise TypeError(
            f"Guardrail '{self.name}' must return bool or GuardrailOutput, "
            f"got {type(result)}"
        )


async def run_guardrails(guardrails, data, exception_class=GuardrailTriggered):
    """统一的 guardrail 执行引擎。

    顺序执行所有 guardrails，根据 action 决定行为：
    - ALLOW: 继续
    - BLOCK: 抛出 exception_class
    - REJECT_CONTENT: 返回 (reject_message, guardrail_result)
    """
    results = []
    for guard in guardrails:
        output = await guard.run(data)
        results.append(output)
        if output.action == GuardrailAction.BLOCK:
            raise exception_class(guard.name, output)
        if output.action == GuardrailAction.REJECT_CONTENT:
            return output, results  # 调用方处理替换
    return None, results  # 全部通过
```

**精简后的 `agent.py`（原 base.py）：**

```python
"""Agent 级 Guardrails：验证 Agent 输入/输出。"""

from .core import BaseGuardrail, GuardrailTriggered, GuardrailOutput, run_guardrails


class InputGuardrailTriggered(GuardrailTriggered):
    """输入 guardrail 触发异常。"""
    pass


class OutputGuardrailTriggered(GuardrailTriggered):
    """输出 guardrail 触发异常。"""
    pass


class InputGuardrail(BaseGuardrail):
    """Agent 输入验证器。"""
    run_in_parallel: bool = False


class OutputGuardrail(BaseGuardrail):
    """Agent 输出验证器。"""
    pass


async def run_input_guardrails(guardrails, data):
    return await run_guardrails(guardrails, data, InputGuardrailTriggered)


async def run_output_guardrails(guardrails, data):
    return await run_guardrails(guardrails, data, OutputGuardrailTriggered)


# 装饰器
def input_guardrail(func=None, *, name=None, run_in_parallel=False):
    ...  # 同现有逻辑，但代码量大幅减少

def output_guardrail(func=None, *, name=None):
    ...
```

**精简后的 `tool.py`：**

```python
"""Tool 级 Guardrails：验证工具调用输入/输出。"""

from .core import BaseGuardrail, GuardrailTriggered, GuardrailOutput, run_guardrails


class ToolInputGuardrailTriggered(GuardrailTriggered):
    pass


class ToolOutputGuardrailTriggered(GuardrailTriggered):
    pass


class ToolInputGuardrail(BaseGuardrail):
    pass


class ToolOutputGuardrail(BaseGuardrail):
    pass


async def run_tool_input_guardrails(guardrails, data):
    return await run_guardrails(guardrails, data, ToolInputGuardrailTriggered)


async def run_tool_output_guardrails(guardrails, data):
    return await run_guardrails(guardrails, data, ToolOutputGuardrailTriggered)


# 装饰器
def tool_input_guardrail(func=None, *, name=None):
    ...

def tool_output_guardrail(func=None, *, name=None):
    ...
```

### 7.3 预估效果

| 文件 | Before | After |
|------|--------|-------|
| base.py → agent.py | 615 行 | ~120 行 |
| tool.py | 638 行 | ~120 行 |
| core.py（新增） | - | ~100 行 |
| **总计** | **1253 行** | **~340 行** |

减少约 **73%** 的代码量。

---

## 8. `__init__.py` 简化与按模块拆分导出

### 8.1 现状

`agentica/__init__.py` 有 534 行，混合了：
- 即时导入（Model provider、Agent、核心类）
- 懒加载逻辑（90+ 个名称，threading.Lock 双重检查锁）
- `__all__` 列表
- `TYPE_CHECKING` 块

### 8.2 重构方案：按模块拆分导出

**原则**：根 `__init__.py` 只导出最核心的顶层 API，其余按子模块按需导入。

**用户视角的导入方式变化：**

```python
# ── 顶层 API（从 agentica 直接导入，不变）──
from agentica import Agent, DeepAgent, Workflow, RunConfig
from agentica import OpenAIChat, OpenAILike, LiteLLMChat, AzureOpenAIChat
from agentica import Message, RunResponse, ModelResponse
from agentica import Tool, Function

# ── Model 相关（从子模块导入）──
from agentica.model import Claude, OllamaChat
from agentica.model.providers import create_provider

# ── 工具（从子模块导入）──
from agentica.tools import ShellTool, CodeTool, DalleTool
from agentica.tools import SearchSerperTool, DuckDuckGoTool

# ── 数据库（从子模块导入）──
from agentica.db import SqliteDb, PostgresDb, InMemoryDb

# ── 知识库 & 向量库（从子模块导入）──
from agentica.knowledge import Knowledge, LlamaIndexKnowledge
from agentica.vectordb import InMemoryVectorDb, SearchType

# ── Embedding（从子模块导入）──
from agentica.embedding import OpenAIEmbedding, OllamaEmbedding

# ── Guardrails（从子模块导入）──
from agentica.guardrails import InputGuardrail, OutputGuardrail

# ── MCP/ACP（从子模块导入）──
from agentica.mcp import MCPConfig, McpTool
from agentica.acp import ACPServer, ACPTool
```

**精简后的 `agentica/__init__.py`**：

```python
"""Agentica: Build AI Agents with ease.

Core API:
    from agentica import Agent, OpenAIChat

Sub-module imports:
    from agentica.model import Claude, OllamaChat
    from agentica.model.providers import create_provider
    from agentica.tools import ShellTool, CodeTool
    from agentica.db import SqliteDb
"""

# ── 配置 ──
from agentica.config import (
    AGENTICA_HOME,
    AGENTICA_DOTENV_PATH,
    AGENTICA_LOG_LEVEL,
)

# ── 日志 ──
from agentica.utils.log import logger, set_log_level_to_debug

# ── 核心 Model ──
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.model.content import Media, Image, Audio, Video
from agentica.model.usage import Usage
from agentica.model.openai.chat import OpenAIChat
from agentica.model.openai.like import OpenAILike
from agentica.model.azure.openai_chat import AzureOpenAIChat

# ── 核心 Agent ──
from agentica.agent.base import Agent
from agentica.deep_agent.base import DeepAgent
from agentica.workflow.workflow import Workflow

# ── 运行配置 & 响应 ──
from agentica.run.response import RunResponse, RunResponseExtraData
from agentica.run.config import RunConfig

# ── 工具基类 ──
from agentica.tools.base import Tool, Function, FunctionCall, ModelTool

# ── 记忆 ──
from agentica.memory.base import AgentRun, SessionSummary, WorkingMemory

# ── 数据库基类 ──
from agentica.db.base import BaseDb

# ── 向后兼容别名（provider 工厂函数）──
from agentica.model.providers import create_provider

def DeepSeekChat(**kwargs):
    return create_provider("deepseek", **kwargs)

def QwenChat(**kwargs):
    return create_provider("qwen", **kwargs)

def ZhipuAIChat(**kwargs):
    return create_provider("zhipuai", **kwargs)

def MoonshotChat(**kwargs):
    return create_provider("moonshot", **kwargs)

def DoubaoChat(**kwargs):
    return create_provider("doubao", **kwargs)

def GrokChat(**kwargs):
    return create_provider("xai", **kwargs)

def YiChat(**kwargs):
    return create_provider("yi", **kwargs)

# ── 懒加载（仅保留少量高频但重型的依赖）──
def __getattr__(name):
    _LAZY = {
        "Claude": "agentica.model.anthropic.claude",
        "OllamaChat": "agentica.model.ollama.chat",
        "LiteLLMChat": "agentica.model.litellm.chat",
    }
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'agentica' has no attribute '{name}'")
```

### 8.3 预估效果

| 指标 | Before | After |
|------|--------|-------|
| `__init__.py` 行数 | 534 行 | ~100 行 |
| 即时导入数量 | ~50+ 个 | ~25 个（核心 API） |
| 懒加载名称数 | ~90 个 | ~3 个（其余走子模块导入） |
| 启动 import 耗时 | 较重 | 显著减轻 |

---

## 9. 执行计划与阶段划分

### Phase 1: Model 层精简（优先级最高）

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 1.1 | 创建 `model/providers.py`，注册 13 个 OpenAI-Compatible provider | 新增 1 文件 |
| 1.2 | 删除 19 个 provider 子目录 | 删除 19 目录 ~40 文件 |
| 1.3 | 在 `model/__init__.py` 添加向后兼容别名 | 修改 1 文件 |
| 1.4 | 更新 `agentica/__init__.py`，删除已删 provider 的导入 | 修改 1 文件 |
| 1.5 | 更新所有 examples（DeepSeekChat ×3, ZhipuAIChat ×5） | 修改 8 文件 |

### Phase 2: Model/Agent dataclass 化

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 2.1 | `Model` 基类 `BaseModel` → `@dataclass` + ABC | 修改 `model/base.py` |
| 2.2 | 去掉 `Field(alias="model")`，统一用 `id` | 修改 `model/base.py` |
| 2.3 | OpenAIChat, Claude, LiteLLMChat, OllamaChat, AzureOpenAI 同步改 | 修改 5 文件 |
| 2.4 | `model_dump()` → `to_dict()`，`PrivateAttr` → `field()` | 修改 base + 5 provider |
| 2.5 | Model 基类加 `@abstractmethod` 标记 | 修改 `model/base.py` |

### Phase 3: 异步接口一致性 + 结构化输出

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 3.1 | LiteLLM `_handle_tool_calls` 去掉下划线前缀 | 修改 `litellm/chat.py` |
| 3.2 | Claude 新增结构化输出（tool_use 模式） | 修改 `anthropic/claude.py` |
| 3.3 | LiteLLM 接入 `response_format` 原生 | 修改 `litellm/chat.py` |
| 3.4 | Ollama 接入 `format=schema` 原生 | 修改 `ollama/chat.py` |
| 3.5 | Model 基类新增 `get_response_format()` 接口 | 修改 `model/base.py` |
| 3.6 | Runner 侧：主路径优先用 `parsed`，文本解析降级为兜底 | 修改 `agent/runner.py` |
| 3.7 | `parse_structured_output()` 标记 deprecated | 修改 `utils/string.py` |

### Phase 4: Tool 注册机制

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 4.1 | 新增 `tools/decorators.py`（@tool 装饰器） | 新增 1 文件 |
| 4.2 | 新增 `tools/registry.py`（全局注册表） | 新增 1 文件 |
| 4.3 | `Function.from_callable()` 增加 `_tool_metadata` 检测 | 修改 `tools/base.py` |
| 4.4 | `tools/__init__.py` 导出新增 API | 修改 1 文件 |

### Phase 5: Runner 拆分

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 5.1 | 新建 `agentica/runner.py`，从 `agent/runner.py` 迁移逻辑 | 新增 1 文件 |
| 5.2 | Agent 去掉 RunnerMixin 继承，改为持有 `_runner: Runner` | 修改 `agent/base.py` |
| 5.3 | Agent 的 `run()`/`run_stream()` 等委托给 Runner | 修改 `agent/base.py` |
| 5.4 | 删除 `agent/runner.py`（原 RunnerMixin） | 删除 1 文件 |
| 5.5 | DeepAgent 适配新 Runner | 修改 `deep_agent/base.py` |

### Phase 6: Guardrails 统一抽象

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 6.1 | 新建 `guardrails/core.py`（统一抽象层） | 新增 1 文件 |
| 6.2 | `guardrails/base.py` → `guardrails/agent.py`，继承 core | 重写 1 文件 |
| 6.3 | `guardrails/tool.py` 精简，继承 core | 重写 1 文件 |
| 6.4 | 更新 `guardrails/__init__.py` 导出 | 修改 1 文件 |

### Phase 7: `__init__.py` 简化

| 步骤 | 任务 | 文件变更 |
|------|------|---------|
| 7.1 | 重写 `agentica/__init__.py`（~100 行） | 修改 1 文件 |
| 7.2 | 确保各子模块 `__init__.py` 导出完整 | 修改多个 `__init__.py` |
| 7.3 | 验证所有 `from agentica import X` 仍能正常工作 | 测试 |

### Phase 8: 测试 + 清理

| 步骤 | 任务 |
|------|------|
| 8.1 | 补充 provider 工厂函数的单元测试 |
| 8.2 | 补充各 provider 结构化输出的单元测试 |
| 8.3 | 补充 Runner 独立测试 |
| 8.4 | 补充 Guardrail core 测试 |
| 8.5 | 更新所有 examples 文件 |
| 8.6 | 更新 CLAUDE.md 架构说明 |

---

## 10. 执行顺序依赖图

```
Phase 1 (Model精简)
    ↓
Phase 2 (dataclass化)  ←─ 先删 19 目录，再改 5 个 provider 到 dataclass
    ↓
Phase 3 (异步一致性 + 结构化输出)  ←─ Model 接口稳定后再改
    ↓
Phase 4 (Tool注册)  ←─ 独立，可与 Phase 3 并行
    ↓
Phase 5 (Runner拆分)  ←─ 依赖 Phase 3（结构化输出简化后 Runner 逻辑更清晰）
    ↓
Phase 6 (Guardrails统一)  ←─ 独立，可与 Phase 5 并行
    ↓
Phase 7 (__init__.py)  ←─ 所有模块稳定后最后处理
    ↓
Phase 8 (测试清理)
```

**推荐执行顺序**：1 → 2 → 3 → (4 & 5 并行) → 6 → 7 → 8

每个 Phase 完成后单独 commit，保持提交粒度细，便于 review 和回滚。

---

## 11. 预估变更统计

| 维度 | Before | After | 变化 |
|------|--------|-------|------|
| model/ 子目录数 | 24 | 5 | -19 |
| model/ 总代码行数 | ~8000+ 行 | ~2600 行 | -67% |
| guardrails/ 总代码行数 | 1253 行 | ~340 行 | -73% |
| `__init__.py` 行数 | 534 行 | ~100 行 | -81% |
| Agent 继承的 Mixin 数 | 5 个 | 4 个 | -1 (RunnerMixin → Runner) |
| Pydantic 依赖（Model 层）| 全部 | 仅 Function/FunctionCall/Message/RunResponse | 大幅减少 |

---

## 12. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 删除 provider 导致用户代码报错 | 高 | 中 | 提供向后兼容别名 + 迁移指南 |
| Model `alias="model"` 去掉导致构造参数变更 | 高 | 高 | 全局搜索 `model=` 构造调用，统一改为 `id=` |
| dataclass 不支持 `model_dump()` | 高 | 中 | 提供 `to_dict()` 方法 |
| 结构化输出 Claude/Ollama/LiteLLM 原生支持的兼容性 | 中 | 中 | 保留 `parse_structured_output()` 作为兜底 |
| Runner 拆分后 DeepAgent 适配 | 中 | 中 | DeepAgent 使用自己的 DeepRunner |
| `__init__.py` 拆分导致现有 `from agentica import X` 失败 | 高 | 高 | 分两步：先保留全部导出 + 懒加载，再逐步迁移 |
