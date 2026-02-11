# Agentica Async-First 技术升级方案

> Write async, expose async, wrap sync explicitly.

**日期**: 2026-02-10
**状态**: 设计阶段（部分实现落地，文档需对齐代码）
**版本**: v1.0

---

## 一、现状分析

### 1.1 当前架构状态（基于最新代码）

Agentica 核心逻辑已基本完成 **Async-First** 改造：核心调用链是 async-only，同步路径通过适配器薄包装。

```
核心 async 路径（默认且唯一）:
Agent.run() → _run_impl()
  → Model.response() / response_stream()   [async-only]
    → OpenAIChat.invoke() / invoke_stream() [async-only]
    → Model.run_function_calls()           [async-only, 当前串行]
      → FunctionCall.execute()             [async-only]
        → _call_func(async await / sync run_in_executor)

流式 async 路径（显式入口）:
Agent.run_stream() → _run_impl(stream=True)

同步适配器路径:
Agent.run_sync(...)        → run_sync(self.run(...))
Agent.run_stream_sync(...) → （专用 sync-stream 适配器：后台线程驱动 async iterator）
```

**关键约束**：`run(stream=True)` / `run_sync(stream=True)` 不应存在，它们会逼出 `iter_over_async()` 这类"通用转换器"，增加 API 阴影与维护成本。

**当前主线已统一 async 实现，但 API 命名与示例仍存在残留差异（详见 1.2/9 章）。**

### 1.2 具体问题清单（结合当前代码）

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| **清理 `run(stream=True)` 遗留** | 代码已删除该入口，但文档/示例/第三方集成仍可能残留旧用法，需统一替换为 `run_stream()`/`run_stream_sync()` | 高 |
| **工具串行执行** | `Model.run_function_calls()` 仍串行 for 循环，无并行 | 中 |
| **Workflow 无异步支持** | `Workflow.run()` 仍纯同步，无法在异步上下文使用 | 中 |
| **同步调用 async-only 接口** | `acp/handlers.py`、`evaluation/run.py`、`examples/model_providers/*` 等处同步调用 `agent.run()`/`model.response()` | 高 |
| **文档/示例/测试残留旧 API** | 仍出现 `arun`/`arun_stream`/`aprint_response`/`aexecute` 文案或用法 | 中 |
| **Subagent 仍同步调度** | `deep_tools.py`/`agent/team.py` 等仍使用 `run_sync()` 驱动子代理 | 中 |


要求： run_stream() / run_stream_sync() 是"正解"，而不是 `run(stream=True)`，必须扔掉 run(stream=True)的设计。
# async-first
async def run(...)
async def run_stream(...)

# sync adapters
def run_sync(...)
def run_stream_sync(...)
---
Model 层这里：async def run_function_calls(
    self,
    calls: list[FunctionCall],
    *,
    parallel: bool = True,
    max_concurrency: int | None = None,
) -> list[ToolResult]
内部：await asyncio.gather(...)，否则 streaming + tool 会被串行拖死
---
Workflow：必须 async-first
---
Event / Stream 的统一抽象（非常重要）

### 1.3 重复代码消减（现状）

- **Runner**：主路径为 async-only `_run_impl()`（唯一执行引擎）。同步入口统一 `run_sync()`；流式同步入口统一 `run_stream_sync()`。
- **Model**：`response()`/`response_stream()` async-only，`run_function_calls()` 仍保留单实现（目前串行）。
- **Tool**：`FunctionCall.execute()` async-only，`aexecute()` 已移除。

**结论**：核心重复代码已大幅消除，但 API 命名与周边生态（示例/测试/文档）仍需清理。
### 1.4 代码现状快照（与文档对齐点）

- **Tool 层**: `FunctionCall.execute()` 已 async-only，`aexecute()` 已删除，`_call_func()` 统一处理 sync/async。
- **Model 层**: `Model.response()` / `response_stream()` async-only，`run_function_calls()` 仍串行。
- **OpenAIChat**: async-only client + response 实现已落地。
- **Runner**: `_run_impl()` 是默认且唯一的 async-only 执行引擎；`run_sync()` / `run_stream_sync()` 作为同步适配器。
- **流式入口**: 已收敛为 `run_stream()` / `run_stream_sync()`，明确删除 `run(stream=True)` 这类隐式入口。
- **Workflow**: `Workflow.run()` 仍为同步接口，缺少 async 入口。
- **ACP/Evaluation/Examples**: 仍存在同步调用 async-only 接口的风险点（见第 9 章）。

---

## 二、业界对标分析

### 2.1 OpenAI Agents SDK

**架构**：Async-First，Runner 模式

```python
# OpenAI Agents SDK 设计
class Runner:
    @staticmethod
    async def run(agent, input, ...) -> RunResult:
        """唯一的核心实现 -- 纯异步"""
        ...

    @staticmethod
    def run_sync(agent, input, ...) -> RunResult:
        """同步适配器，内部调用 run()"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(Runner.run(agent, input, ...))

    @staticmethod
    async def run_streamed(agent, input, ...) -> RunResultStreaming:
        """流式输出 -- 纯异步"""
        ...
```

**核心设计原则**：
- **只写一次核心逻辑**：`Runner.run()` 是唯一的实现，所有路径都走这里
- **run_sync 是薄包装器**：复用默认事件循环，处理 KeyboardInterrupt 清理
- **工具天然异步**：`@function_tool` 支持 async def，同步工具自动包装
- **Runner 与 Agent 分离**：Agent 只是配置容器，Runner 负责执行

### 2.2 PydanticAI

**架构**：Async-First，Agent 内置运行

```python
# PydanticAI 设计
class Agent:
    async def run(self, user_prompt, ...) -> RunResult:
        """核心实现 -- 纯异步"""
        ...

    def run_sync(self, user_prompt, ...) -> RunResult:
        """同步适配器"""
        return asyncio.get_event_loop().run_until_complete(self.run(...))

    async def run_stream(self, user_prompt, ...) -> StreamedRunResult:
        """流式 -- 纯异步"""
        ...

    def run_stream_sync(self, user_prompt, ...) -> StreamedRunResult:
        """流式同步适配器"""
        ...
```

**核心设计原则**：
- **run_sync 文档明确写了**：`run_sync is a thin wrapper around loop.run_until_complete(self.run())`
- **工具原生异步**：工具函数可以是 async def 或 def，框架自动处理
- **结构化输出**：通过 Pydantic 模型约束输出类型
- **依赖注入**：通过 `deps_type` 传递上下文

### 2.3 核心共识

| 设计点 | OpenAI Agents | PydanticAI | Agentica (当前) |
|--------|--------------|------------|-----------------|
| 核心实现 | 纯 async | 纯 async | sync + async 双实现 |
| 同步入口 | `run_sync()` 薄包装 | `run_sync()` 薄包装 | `run()` 独立实现 |
| 工具执行 | async，sync 自动包装 | async，sync 自动包装 | sync/async 双实现 |
| 并行工具 | asyncio.gather | asyncio.gather | 串行执行 |
| 流式输出 | AsyncIterator | AsyncIterator | Iterator / AsyncIterator 双实现 |

---

## 三、目标架构设计

### 3.1 设计原则

```
1. Async-Native:  所有核心逻辑只有 async 实现
2. Sync-Adapter:  run_sync() 作为唯一的同步入口，薄包装 async
3. Single-Source:  消灭 sync/async 代码重复
4. Parallel-First: asyncio.gather() 并行执行工具和 subagent
5. Stream-Native:  AsyncIterator 作为流式原语，同步流通过适配器
```

### 3.2 分层架构

```
┌─────────────────────────────────────────────────┐
│                  用户 API 层                      │
│                                                   │
│  agent.run(...)         -- async, 核心实现         │
│  agent.run_sync(...)    -- sync 薄包装            │
│  agent.run_stream(...)  -- async 流式              │
│  agent.run_stream_sync()-- sync 流式薄包装         │
├─────────────────────────────────────────────────┤
│                  执行引擎层                        │
│                                                   │
│  _run_impl()            -- async only（唯一）     │
│  _execute_tools()       -- asyncio.gather 并行    │
├─────────────────────────────────────────────────┤
│                  Model 层                         │
│                                                   │
│  model.invoke(...)      -- async only             │
│  model.invoke_stream()  -- async only             │
│  model.response(...)    -- async only             │
│  model.response_stream()-- async only             │
│  model.run_function_calls() -- async only         │
├─────────────────────────────────────────────────┤
│                  Tool 层                          │
│                                                   │
│  FunctionCall.execute() -- async only             │
│    ├── async entrypoint → await directly          │
│    └── sync entrypoint  → run_in_executor()       │
└─────────────────────────────────────────────────┘
```

### 3.3 sync 适配器实现

统一的同步适配器，处理各种边界情况：

```python
# agentica/utils/async_utils.py

import asyncio
import threading
from typing import TypeVar, Coroutine

T = TypeVar("T")


def run_sync(coro: Coroutine[None, None, T]) -> T:
    """在同步上下文中运行异步协程。

    处理三种场景：
    1. 无事件循环 → asyncio.run()
    2. 在事件循环中（如 Jupyter） → 新线程 + 新事件循环
    3. 嵌套调用保护
    """
    ...
```

**重要取舍**：不提供通用 `iter_over_async()`。
- `iter_over_async()` 往往是 `run(stream=True)` 这种隐式 API 的"影子"，会把错误的 API 设计长期固化。
- `run_stream_sync()` 应作为 `Agent` 的**专用同步流式适配器**存在（通常用后台线程驱动 async iterator + 队列转发），而不是暴露一个到处可被滥用的通用转换器。

---

## 四、各层改造方案

### 4.0 代码落地进度（基于当前代码）

| 模块 | 现状 | 说明 |
|------|------|------|
| Tool (`FunctionCall`) | ✅ 已落地 | `execute()` async-only，统一 `_call_func()` |
| Model 基类 | ✅ 已落地 | `response()`/`response_stream()` async-only，`run_function_calls()` 仍串行（TODO-6 并行化） |
| OpenAIChat | ✅ 已落地 | async-only client + response 实现 |
| 其他 Model 实现 | ✅ 已落地 | Anthropic/Bedrock/Cohere/Ollama/Gemini/Together/Mistral 等均 async-only |
| Runner | ✅ 已落地 | `_run_impl()` 唯一引擎；`_run_multi_round`/`_run_single_round` 已删除 |
| 四件套 API | ✅ 已落地 | `run()`/`run_stream()` (async) + `run_sync()`/`run_stream_sync()` (sync) |
| `iter_over_async` | ✅ 已彻底删除 | `run_stream_sync()` 用线程+队列自行实现 |
| Printer | ✅ 已落地 | `print_response()` async + `print_response_sync()` sync adapter |
| CLI | ✅ 已对齐 | 使用 `run_stream_sync()` 作为主入口 |
| ACP handlers | ✅ 已适配 | 使用 `run_sync()`/`run_stream_sync()` |
| agent/team.py | ✅ 已适配 | `as_tool()`/`get_transfer_function()` 使用 `run_sync()` |
| deep_tools.py | ✅ 已适配 | `BuiltinTaskTool.task()` 使用 `run_stream_sync()` |
| Workflow | ⏳ 未完成 | `run()` 仍同步（TODO-7） |
| evaluation/run.py | ⚠️ 运行时 Bug | sync 调用 async 方法（TODO-1） |
| Examples | ⏳ 部分完成 | 大部分已适配 `run_sync()`/`run_stream_sync()`，少量残留 Bug（TODO-2/3），待改为 async-first 原生风格（TODO-10） |
| Tests | ⏳ 部分完成 | 基本适配，`test_llm.py` 待改 AsyncMock（TODO-8），待整体改用 pytest-asyncio（TODO-11） |

### 4.1 Tool 层改造

**目标**：`FunctionCall.execute()` 变为纯 async，删除同步 `execute()`

**改造前** (tools/base.py)：
```python
class FunctionCall:
    def execute(self) -> bool:        # 82行同步实现
        ...
    async def aexecute(self) -> bool:  # 105行异步实现
        ...
    def _run_sync_or_async(self, ...): # hack: ThreadPoolExecutor + asyncio.run
        ...
```

**改造后**：
```python
class FunctionCall:
    async def execute(self) -> bool:
        """唯一实现 -- 纯异步。

        - async entrypoint → await 直接调用
        - sync entrypoint → loop.run_in_executor() 避免阻塞事件循环
        """
        if self.function.entrypoint is None:
            self.error = f"No entrypoint for function: {self.function.name}"
            logger.warning(self.error)
            return False

        logger.debug(f"Running: {self.get_call_str()}")

        # Pre-hook
        await self._run_hook(self.function.pre_hook)

        # Execute entrypoint
        try:
            args = self._build_entrypoint_args()
            merged_args = {**args, **(self.arguments or {})}
            self.result = await self._call_func(self.function.entrypoint, **merged_args)
            success = True
        except ToolCallException as e:
            logger.debug(f"{e.__class__.__name__}: {e}")
            self.error = str(e)
            raise
        except Exception as e:
            logger.warning(f"Could not run function {self.get_call_str()}")
            logger.exception(e)
            self.error = str(e)
            return False

        # Post-hook
        await self._run_hook(self.function.post_hook)
        return success

    async def _call_func(self, func: Callable, **kwargs) -> Any:
        """统一调用：async 直接 await，sync 走线程池。"""
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(func, **kwargs)
            )

    async def _run_hook(self, hook: Optional[Callable]) -> None:
        """统一执行 pre/post hook。"""
        if hook is None:
            return
        hook_args = self._build_hook_args(hook)
        await self._call_func(hook, **hook_args)

    # 删除: execute() 同步版
    # 删除: aexecute()
    # 删除: _run_sync_or_async()
```

**削减代码**：~187行 → ~60行，消灭 `_run_sync_or_async()` hack

### 4.2 Model 层改造

**目标**：Model 基类只暴露 async 接口，删除所有同步方法

**改造前** (model/base.py)：
```python
class Model:
    def invoke(self, ...): raise NotImplementedError
    async def ainvoke(self, ...): raise NotImplementedError
    def invoke_stream(self, ...): raise NotImplementedError
    async def ainvoke_stream(self, ...): raise NotImplementedError
    def response(self, ...): raise NotImplementedError
    async def aresponse(self, ...): raise NotImplementedError
    def response_stream(self, ...): raise NotImplementedError
    async def aresponse_stream(self, ...): raise NotImplementedError
    def run_function_calls(self, ...): ...       # 120行
    async def arun_function_calls(self, ...): ... # 112行
    def handle_post_tool_call_messages(self, ...): ...
    async def ahandle_post_tool_call_messages(self, ...): ...
```

**改造后**：
```python
class Model:
    # --- 子类必须实现（纯 async） ---
    async def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def invoke_stream(self, *args, **kwargs) -> AsyncIterator[Any]:
        raise NotImplementedError

    async def response(self, messages: List[Message]) -> ModelResponse:
        raise NotImplementedError

    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        raise NotImplementedError

    # --- 基类通用实现（纯 async） ---
    async def run_function_calls(
        self,
        function_calls: List[FunctionCall],
        function_call_results: List[Message],
        tool_role: str = "tool",
    ) -> AsyncIterator[ModelResponse]:
        """执行工具调用 -- 支持并行。"""
        # 并行执行所有工具
        results = await asyncio.gather(
            *[fc.execute() for fc in function_calls],
            return_exceptions=True
        )
        # 处理结果...
        for fc, success in zip(function_calls, results):
            if isinstance(success, Exception):
                fc.error = str(success)
                success = False
            # ... 构建 tool message, yield ModelResponse

    async def handle_post_tool_call_messages(
        self, messages: List[Message], ...
    ) -> ModelResponse:
        return await self.response(messages=messages)

    async def handle_post_tool_call_messages_stream(
        self, messages: List[Message], ...
    ) -> AsyncIterator[ModelResponse]:
        async for resp in self.response_stream(messages=messages):
            yield resp

    # 删除: invoke() 同步版
    # 删除: ainvoke() → 改名为 invoke()
    # 删除: invoke_stream() 同步版 → ainvoke_stream() 改名为 invoke_stream()
    # 删除: response() 同步版 → aresponse() 改名为 response()
    # 删除: response_stream() 同步版 → aresponse_stream() 改名为 response_stream()
    # 删除: run_function_calls() 同步版
    # 删除: arun_function_calls() → 改名为 run_function_calls()
```

**方法名映射**：

| 改造前 | 改造后 | 说明 |
|--------|--------|------|
| `invoke()` | 删除 | |
| `ainvoke()` | `invoke()` | 异步成为默认 |
| `invoke_stream()` | 删除 | |
| `ainvoke_stream()` | `invoke_stream()` | 异步成为默认 |
| `response()` | 删除 | |
| `aresponse()` | `response()` | 异步成为默认 |
| `response_stream()` | 删除 | |
| `aresponse_stream()` | `response_stream()` | 异步成为默认 |
| `run_function_calls()` | 删除 | |
| `arun_function_calls()` | `run_function_calls()` | 异步成为默认，增加并行 |

### 4.3 OpenAIChat 改造

**改造前** (model/openai/chat.py)：
```python
class OpenAIChat(Model):
    def get_client(self): ...           # 同步客户端
    def get_async_client(self): ...     # 异步客户端
    def invoke(self, ...): ...          # 同步
    async def ainvoke(self, ...): ...   # 异步
    def response(self, ...): ...        # 同步 ~72行
    async def aresponse(self, ...): ... # 异步 ~73行（重复）
    # ... 同样的重复模式
```

**改造后**：
```python
class OpenAIChat(Model):
    def get_client(self) -> AsyncOpenAIClient:
        """只保留异步客户端。"""
        if self.async_client is None:
            client_params = self._build_client_params()
            client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
            self.async_client = AsyncOpenAIClient(**client_params)
        return self.async_client

    async def invoke(self, messages, **kwargs) -> Any:
        return await self.get_client().chat.completions.create(
            model=self.id, messages=messages, **self.request_kwargs
        )

    async def invoke_stream(self, messages, **kwargs) -> AsyncIterator[Any]:
        return await self.get_client().chat.completions.create(
            model=self.id, messages=messages, stream=True, **self.request_kwargs
        )

    async def response(self, messages: List[Message]) -> ModelResponse:
        """统一实现，消灭 response/aresponse 重复。"""
        response = await self.invoke(messages=api_messages)
        # 解析 response → ModelResponse
        # 如果有 tool_calls → await self.handle_tool_calls(...)
        # 如果有 post_tool_call → await self.handle_post_tool_call_messages(...)
        return model_response

    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """统一实现，消灭 response_stream/aresponse_stream 重复。"""
        async for chunk in await self.invoke_stream(messages=api_messages):
            yield self._parse_stream_chunk(chunk)
        # 如果有 tool_calls → async for in self.handle_stream_tool_calls(...)

    # 删除: get_client() 同步客户端
    # 删除: invoke() 同步版, ainvoke() 改名为 invoke()
    # 删除: response() 同步版, aresponse() 改名为 response()
    # 删除: response_stream() 同步版, aresponse_stream() 改名为 response_stream()
    # 删除: handle_tool_calls() 同步版, ahandle_tool_calls() 改名为 handle_tool_calls()
```

**其他 Model 实现同理**：Anthropic, ZhipuAI, DeepSeek, Ollama 均做相同改造。

### 4.4 Agent 层改造

**改造前** (agent/runner.py, 1885行)：
```python
class RunnerMixin:
    def run(self, ...): ...                    # 同步公共API ~106行
    async def arun(self, ...): ...             # 异步公共API ~110行（重复）
    def _run(self, ...): ...                   # 同步调度 ~75行
    async def _arun(self, ...): ...            # 异步调度 ~37行（功能缺失）
    def _run_single_round(self, ...): ...      # 同步单轮 ~240行
    async def _arun_single_round(self, ...): ...# 异步单轮 ~270行（重复）
    def _run_multi_round(self, ...): ...       # 同步多轮 ~320行
    async def _arun_multi_round(self, ...): ...# 异步多轮 ~310行（重复）
```

**改造后** (agent/runner.py, 预计 ~800行)：
```python
class RunnerMixin:
    # ==================== 公共 API（显式分离：run vs run_stream） ====================

    async def run(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        **kwargs,
    ) -> RunResponse:
        """非流式运行（async）。"""
        final = None
        async for response in self._run_impl(message, stream=False, **kwargs):
            final = response
        return final

    async def run_stream(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        **kwargs,
    ) -> AsyncIterator[RunResponse]:
        """流式运行（async generator）。"""
        async for response in self._run_impl(message, stream=True, **kwargs):
            yield response

    def run_sync(self: "Agent", message=None, **kwargs) -> RunResponse:
        """同步适配器：仅用于非流式。"""
        return run_sync(self.run(message, **kwargs))

    def run_stream_sync(self: "Agent", message=None, **kwargs) -> Iterator[RunResponse]:
        """同步流式适配器：后台线程驱动 async iterator，通过队列转发输出。"""
        ...

    # ==================== 核心引擎（唯一实现） ====================

    async def _run_impl(self: "Agent", message=None, *, stream: bool = False, **kwargs) -> AsyncIterator[RunResponse]:
        """唯一执行引擎：single-round + model 内建工具循环。

        - multi-round 研究范式不属于 base Agent；需要时在 `DeepAgent` 里实现。
        """
        with langfuse_trace_context(...):
            ...
```

### 4.5 Workflow 层改造

```python
class Workflow(BaseModel):
    async def run(self, *args, **kwargs) -> Optional[RunResponse]:
        """核心运行 -- 纯 async。子类重写此方法。"""
        raise NotImplementedError

    def run_sync(self, *args, **kwargs) -> Optional[RunResponse]:
        """同步适配器。"""
        from agentica.utils.async_utils import run_sync
        return run_sync(self.run(*args, **kwargs))
```

### 4.6 CLI 层改造

CLI 作为最顶层的同步入口：

```python
# agentica/cli/interactive.py

def run_interactive():
    """CLI 交互主循环 -- 保持同步（terminal I/O 本身是同步的）。"""
    agent = create_agent(...)
    while True:
        user_input = prompt(...)
        # 流式输出：使用同步流式适配器
        for response in agent.run_stream_sync(user_input):
            display_stream(response)
```

---

## 五、并行工具执行

### 5.1 当前问题

```python
# 当前: 串行执行每个工具
for function_call in function_calls:
    function_call_success = await function_call.execute()
    # ... 处理结果
```

**当前代码现状**：`model/base.py` 的 `run_function_calls()` 仍是上述串行实现，因此并行化尚未落地。

### 5.2 改造方案

```python
# model/base.py

async def run_function_calls(
    self,
    function_calls: List[FunctionCall],
    function_call_results: List[Message],
    tool_role: str = "tool",
) -> AsyncIterator[ModelResponse]:
    """并行执行工具调用。"""
    # 阶段1: 并行执行所有工具
    results = await asyncio.gather(
        *[self._execute_single_tool(fc) for fc in function_calls],
        return_exceptions=True  # 部分失败不影响其他工具
    )

    # 阶段2: 按顺序处理结果（保持消息顺序）
    for fc, result in zip(function_calls, results):
        if isinstance(result, ToolCallException):
            # ToolCallException 需要向上传播
            raise result
        elif isinstance(result, Exception):
            fc.error = str(result)
            logger.warning(f"Tool {fc.function.name} failed: {result}")

        # 构建 tool message
        tool_message = self._build_tool_message(fc, tool_role)
        function_call_results.append(tool_message)

        # yield 工具执行事件
        yield ModelResponse(
            event=ModelResponseEvent.tool_call_completed,
            tool_calls=[fc],
        )

async def _execute_single_tool(self, fc: FunctionCall) -> bool:
    """执行单个工具，带指标收集。"""
    timer = Timer()
    timer.start()
    try:
        success = await fc.execute()
        timer.stop()
        # 记录指标
        if fc.metrics is None:
            fc.metrics = {}
        fc.metrics["time"] = timer.elapsed
        return success
    except Exception as e:
        timer.stop()
        raise
```

### 5.3 Subagent 并行

```python
# 多个 subagent 并行执行
async def _run_parallel_subagents(self, subagents, inputs):
    results = await asyncio.gather(
        *[agent.run(input) for agent, input in zip(subagents, inputs)],
        return_exceptions=True
    )
    return results
```

---

## 六、方法命名规范

### 6.1 命名约定

```
异步方法（默认）:  run(), execute(), response(), invoke()
同步适配器:       run_sync(), run_stream_sync()
流式方法:         run_stream(), response_stream(), invoke_stream()
内部方法:         _run_impl()（唯一）
```

**原则**：
- 默认名字留给 async（因为 async 是主路径）
- sync 适配器统一加 `_sync` 后缀
- 不再使用 `a` 前缀 (`arun`, `aresponse`, `ainvoke`) -- 这是旧模式

**当前代码提醒**：已统一为 `run()` / `run_stream()` / `run_sync()` / `run_stream_sync()` 四件套，并**明确删除** `run(stream=True)`/`run_sync(stream=True)` 与 `iter_over_async()`。后续重点是全库（docs/examples/tests/第三方集成）清理旧用法与旧文案。
### 6.2 API 对照表

| 改造前 (旧) | 改造后 (新) | 类型 |
|-------------|------------|------|
| `agent.run()` | `agent.run()` | async (签名变了) |
| `agent.arun()` | 删除 | - |
| `agent.run(stream=True)` | `agent.run_stream()` | async |
| `agent.arun(stream=True)` | 删除 | - |
| (无) | `agent.run_sync()` | sync 适配器 |
| (无) | `agent.run_stream_sync()` | sync 适配器 |
| `agent.print_response()` | `agent.print_response()` | async |
| `agent.aprint_response()` | 删除 | - |
| (无) | `agent.print_response_sync()` | sync 适配器 |
| `model.response()` | `model.response()` | async (签名变了) |
| `model.aresponse()` | 删除 | - |
| `model.invoke()` | `model.invoke()` | async (签名变了) |
| `model.ainvoke()` | 删除 | - |
| `fc.execute()` | `fc.execute()` | async (签名变了) |
| `fc.aexecute()` | 删除 | - |

---

## 七、迁移策略

### 7.1 分阶段执行

```
Phase 1: 基础设施                     [预计 1天]  ✅ 已落地
  ├── 新增 agentica/utils/async_utils.py (run_sync，无 iter_over_async)
  └── 编写单元测试（基础已有，待 pytest-asyncio 统一）

Phase 2: Tool 层 (自底向上)            [预计 1天]  ✅ 已落地
  ├── FunctionCall.execute() → async ✅
  ├── 删除 aexecute(), _run_sync_or_async() ✅
  └── 更新所有引用 ✅

Phase 3: Model 层                     [预计 2天]  ✅ 已落地（并行化待补）
  ├── Model 基类: async-only，无 a-prefix 残留 ✅
  ├── OpenAIChat: async-only ✅
  ├── 所有 Model 实现: Anthropic/Bedrock/Cohere/Ollama/Gemini/Together/Mistral ✅
  ├── run_function_calls asyncio.gather 并行 ⏳ (TODO-6)
  └── 更新所有引用 ✅

Phase 4: Agent 层                     [预计 2天]  ✅ 已落地
  ├── runner.py: _run_impl 唯一引擎，_run_multi_round/_run_single_round 已删 ✅
  ├── 四件套 API: run/run_stream/run_sync/run_stream_sync ✅
  ├── iter_over_async 彻底删除 ✅
  ├── run_stream_sync 用线程+队列实现 ✅
  └── base.py Mixin 注册 ✅（声明需清理 TODO-5）

Phase 5: 上层模块                     [预计 1天]  ✅ 大部分落地
  ├── CLI: run_stream_sync() ✅
  ├── ACP handlers: run_sync()/run_stream_sync() ✅
  ├── deep_tools/team: run_sync()/run_stream_sync() ✅
  ├── DeepAgent: 继承 Agent，无问题 ✅
  └── Workflow: run() → async ⏳ (TODO-7)

Phase 6: 测试/示例/文档清理            [预计 2天]  ⏳ 部分完成
  ├── Examples 基本适配 run_sync()/run_stream_sync() ✅
  ├── Tests 基本适配 ✅
  ├── 少量运行时 Bug 修复 ⏳ (TODO-1/2/3/4)
  ├── Examples 改为 async-first 原生风格 ⏳ (TODO-10)
  ├── Tests 改用 pytest-asyncio ⏳ (TODO-11)
  └── 旧 API 文案清理 ⏳ (TODO-9)
```

### 7.2 破坏性变更清单

| 变更 | 影响 | 迁移指引 |
|------|------|---------|
| `agent.run()` 变为 async | 所有同步调用方需修改 | `agent.run()` → `agent.run_sync()` 或 `await agent.run()` |
| `agent.arun()` 删除 | 所有异步调用方需修改 | `await agent.arun()` → `await agent.run()` |
| `model.response()` 变为 async | Model 子类需修改 | `model.response()` → `await model.response()` |
| `fc.execute()` 变为 async | 直接调用工具的代码 | `fc.execute()` → `await fc.execute()` |
| `agent.print_response()` 变为 async | CLI/脚本代码 | 使用 `agent.print_response_sync()` |

### 7.3 向后兼容策略

不做向后兼容（遵循项目规则：直接删除替换，不兼容旧代码）。发布新大版本号。

---

## 八、预期收益

### 8.1 代码量变化

| 模块 | 改造前 | 改造后 | 削减 |
|------|--------|--------|------|
| agent/runner.py | 1885行 | ~800行 | -57% |
| model/base.py | 758行 | ~450行 | -41% |
| model/openai/chat.py | 1244行 | ~700行 | -44% |
| tools/base.py (FunctionCall) | 487行 | ~300行 | -38% |
| **合计** | **~4374行** | **~2250行** | **-49%** |

### 8.2 质量改进

| 改进点 | 说明 |
|--------|------|
| **消灭功能不对称** | async 路径获得完整的 Langfuse trace、hooks、超时处理 |
| **并行工具执行** | asyncio.gather 替代串行 for 循环，多工具场景性能数倍提升 |
| **消灭 hack 代码** | 删除 `_run_sync_or_async()` 中的 ThreadPoolExecutor + asyncio.run hack |
| **单一数据源** | 每个逻辑只写一次，修bug只需改一处 |
| **现代 Python 风格** | 与 OpenAI Agents SDK、PydanticAI 设计对齐 |

### 8.3 性能预期

| 场景 | 当前 | 改造后 |
|------|------|--------|
| 3个独立工具调用 | ~3x 耗时（串行） | ~1x 耗时（并行） |
| 并发用户请求 (FastAPI) | 同步工具阻塞事件循环 | 所有工具在线程池/async中执行 |
| subagent 并行 | 不支持 | asyncio.gather 并行执行 |

---

## 九、当前代码审查结论与 TODO（2026-02-11 更新）

> 基于最新代码全量审查，以下为已落地状态与剩余 TODO。

### 9.0 已落地确认（无需再动）

| 模块 | 状态 | 说明 |
|------|------|------|
| `iter_over_async` | ✅ **已彻底删除** | 全局搜索 0 处引用。`run_stream_sync()` 用线程+队列自行实现，不暴露通用转换器 |
| `_run_multi_round` | ✅ **已从 runner.py 删除** | runner.py 中 0 处定义，注释明确 "Multi-round NOT part of base Agent" |
| `_run_single_round` | ✅ **已重命名为 `_run_impl`** | runner.py 中仅有 `_run_impl()`（唯一执行引擎） |
| `run(stream=True)` | ✅ **公开 API 已移除** | `run()` 不接受 `stream` 参数，流式入口为独立的 `run_stream()` |
| 四件套 API | ✅ **已就位** | `run()` / `run_stream()` (async) + `run_sync()` / `run_stream_sync()` (sync adapter) |
| `run_stream_sync()` 实现 | ✅ **线程+队列模式** | 后台 daemon 线程 `asyncio.run()` 消费 async iterator → `queue.Queue` → 主线程 `yield` |
| Tool 层 | ✅ **async-only** | `FunctionCall.execute()` async-only，`aexecute()` 已删除 |
| Model 基类 | ✅ **async-only** | `invoke/invoke_stream/response/response_stream` 无 `a`-prefix 残留，无同步版本 |
| Printer | ✅ **async-only** | `print_response()` async + `print_response_sync()` sync adapter |
| CLI | ✅ **已适配** | `interactive.py` / `main.py` 使用 `run_stream_sync()` |
| ACP handlers | ✅ **已适配** | `handle_agent_execute` / `_execute_sync` / `_execute_with_streaming` 已改为 `run_sync()`/`run_stream_sync()` |
| `agent/team.py` | ✅ **已适配** | `as_tool()` / `get_transfer_function()` 均使用 `self.run_sync()` |
| `deep_tools.py` | ✅ **已适配** | `BuiltinTaskTool.task()` 使用 `subagent.run_stream_sync()` |

### 9.1 TODO：高优先级（运行时 Bug）

#### TODO-1: `evaluation/run.py` 同步调用 async-only 方法
- **问题**：`call_llm_judge()` (行84) `judge_model.response(messages)` 是 async，sync 调用返回 coroutine 对象；`evaluate_instance()` (行243) `agent.run(question)` 同样返回 coroutine。
- **修复**：
  - 方案 A（推荐）：将 `evaluate_instance()` 改为 `async def`，内部 `await agent.run()`，`call_llm_judge()` 也改为 `async def`，内部 `await judge_model.response()`。`main()` 已经是 `async def`，调用链天然支持。
  - 方案 B：直接用同步适配器 `agent.run_sync(question)`。但 `judge_model.response()` 没有 sync adapter，需要 `run_sync(judge_model.response(messages))`。

#### TODO-2: `examples/model_providers/01_openai.py` 同步调用 async 方法
- **问题**：行27 `model.response(messages)` 和行36 `for chunk in model.response_stream(messages):` 均为 sync 调用 async 方法。
- **修复**：改为 `async def main()` + `await model.response()` + `async for chunk in model.response_stream()` + `asyncio.run(main())`。

#### TODO-3: `examples/model_providers/02_deepseek.py` 同步调用 async 方法
- **问题**：行27 `model.response(messages)` 同步调用。
- **修复**：同 TODO-2。

#### TODO-4: `agentica/tools/memori_tool.py` `__main__` 同步调用
- **问题**：行393/396/399 在 sync `__main__` 中调用 `agent.print_response()`（async）。
- **修复**：改为 `agent.print_response_sync()`。

### 9.2 TODO：中优先级（代码清理 & 设计改进）

#### TODO-5: `base.py` 清理过时声明
- **问题**：`agentica/agent/base.py` 中存在 6 个已不存在方法的 Callable 声明（行1056-1061）：`_run`、`_run_single_round`、`_run_multi_round`、`_on_pre_step`、`_on_tool_call`、`_on_post_step`。同时 `deep_copy` 的 `method_fields` 集合（行889-891）也引用了这些废弃名称。
- **修复**：
  - 删除过时的 Callable 声明。
  - 补充缺失的声明：`run_stream: Callable`、`run_stream_sync: Callable`、`_run_impl: Callable`、`_consume_run: Callable`、`_run_with_timeout: Callable`、`_wrap_stream_with_timeout: Callable`。
  - 更新 `method_fields` 集合与实际 runner.py 方法对齐。

#### TODO-6: `Model.run_function_calls()` 并行化
- **问题**：`model/base.py` 行287 仍为串行 `for function_call in function_calls: await function_call.execute()`。
- **当前注释**（行285）：*"Executes tools sequentially to maintain message ordering and streaming events."*
- **修复建议**：
  ```python
  # 阶段1: asyncio.gather 并行执行所有工具
  results = await asyncio.gather(
      *[fc.execute() for fc in function_calls],
      return_exceptions=True
  )
  # 阶段2: 按原始顺序处理结果（保持消息顺序不变）
  for fc, success in zip(function_calls, results):
      ...
  ```
  并行执行不影响消息顺序——执行是并行的，结果收集后按原始顺序 yield 即可。

#### TODO-7: `Workflow` Async 化
- **问题**：`workflow.py` 行110 `def run()` 仍为纯同步，无 async 版本，无 `run_sync()` 适配器。
- **修复**：
  ```python
  class Workflow:
      async def run(self, *args, **kwargs) -> Optional[RunResponse]:
          raise NotImplementedError

      def run_sync(self, *args, **kwargs) -> Optional[RunResponse]:
          from agentica.utils.async_utils import run_sync
          return run_sync(self.run(*args, **kwargs))
  ```
  所有继承 `Workflow` 的子类（包括 examples 中的 workflow 示例）需要同步改为 `async def run()`，内部 `agent.run_sync()` → `await agent.run()`。

#### TODO-8: `tests/test_llm.py` 改用 AsyncMock
- **问题**：行30 `res = llm.response(messages)` 通过 MagicMock 测试，未覆盖 async 行为。
- **修复**：改用 `AsyncMock` + `pytest.mark.asyncio` + `await llm.response(messages)`。

### 9.3 TODO：低优先级（文案清理）

#### TODO-9: 示例/文档旧 API 文案清理
- `examples/basic/03_stream_output.py`：行67 函数名 `async_arun_stream_demo` 应改为 `async_stream_demo`；行104 注释 `# Use arun` 应改为 `# Use run`。
- `examples/tools/02_async_tool.py`：行175 docstring `"with arun"` → `"with run"`；行186 打印文字 `"Using arun"` → `"Using run"`。
- `tests/test_async_tool.py`：测试方法名 `test_async_function_aexecute` / `test_async_tool_method_aexecute` 等仍含 `aexecute`，应改为 `test_async_function_execute` 等。
- `agentica/skills/builtin/agentica-intro/SKILL.md`：行32/195 `agent.print_response()` → `agent.print_response_sync()` 或加 async 上下文。

#### TODO-10: Examples 整体改造（async-first 原生风格）
- **原则**：examples 应**原生使用 async API**，避免 `run_sync()`：
  ```python
  async def main():
      agent = Agent(...)
      result = await agent.run("hello")
      print(result.content)

  if __name__ == "__main__":
      asyncio.run(main())
  ```
- 仅保留少量示例（如 `examples/basic/sync_demo.py`）专门演示 `run_sync()` / `run_stream_sync()` 用法。
- 当前约 78 处使用 `run_sync()`、5 处使用 `run_stream_sync()` 的 examples 需逐步改为原生 async。
- 可以增删示例，重点体现 agent 各核心功能。

#### TODO-11: Tests 改造（pytest-asyncio 统一）
- 所有 test 改用 `pytest-asyncio` + `@pytest.mark.asyncio` + `async def test_xxx()`。
- 可以增删测试，重点覆盖 agent 核心功能。

### 9.4 进度总览表

| TODO | 描述 | 优先级 | 预计工作量 |
|------|------|--------|-----------|
| TODO-1 | `evaluation/run.py` async 修复 | 🔴 高 | 0.5h |
| TODO-2 | `examples/model_providers/01_openai.py` async 修复 | 🔴 高 | 0.5h |
| TODO-3 | `examples/model_providers/02_deepseek.py` async 修复 | 🔴 高 | 0.5h |
| TODO-4 | `memori_tool.py` `__main__` 修复 | 🔴 高 | 0.2h |
| TODO-5 | `base.py` 清理过时声明 | 🟡 中 | 0.5h |
| TODO-6 | `run_function_calls()` 并行化 | 🟡 中 | 1h |
| TODO-7 | `Workflow` async 化 | 🟡 中 | 2h |
| TODO-8 | `test_llm.py` AsyncMock | 🟡 中 | 0.5h |
| TODO-9 | 旧 API 文案清理 | 🟢 低 | 1h |
| TODO-10 | Examples async-first 改造 | 🟢 低 | 3h |
| TODO-11 | Tests pytest-asyncio 统一 | 🟢 低 | 2h |

---

## 十、附录

### A. 核心文件现状快照

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `agentica/utils/async_utils.py` | 50 | ✅ | 仅 `run_sync()`，无 `iter_over_async` |
| `agentica/tools/base.py` | ~360 | ✅ | `execute()` async-only |
| `agentica/model/base.py` | 586 | ✅（串行工具待并行化） | async-only，`run_function_calls` 串行 |
| `agentica/agent/runner.py` | 692 | ✅ | `_run_impl` 唯一引擎，四件套 API 就位 |
| `agentica/agent/base.py` | 1140 | ⚠️ 过时声明待清理 | Callable 声明 & method_fields 不一致 |
| `agentica/agent/printer.py` | 215 | ✅ | async + sync adapter |
| `agentica/agent/team.py` | 210 | ✅ | `run_sync()` 适配 |
| `agentica/deep_agent.py` | 540 | ✅ | 继承 Agent，无运行时问题 |
| `agentica/deep_tools.py` | 1395 | ✅ | `run_stream_sync()` 适配 |
| `agentica/workflow.py` | 348 | ⚠️ 待 async 化 | `run()` 仍同步 |
| `agentica/acp/handlers.py` | 582 | ✅ | 已用 `run_sync()`/`run_stream_sync()` |
| `agentica/cli/interactive.py` | 669 | ✅ | 已用 `run_stream_sync()` |
| `evaluation/run.py` | 443 | ⚠️ 运行时 Bug | sync 调用 async 方法 |

### B. 参考资料

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Runner.run() / run_sync() 模式
- [PydanticAI](https://ai.pydantic.dev/) - Agent.run() / run_sync() 模式
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)
- [Agentica 技术优化方案 V3](../update_tech_v3.md) - 既有技术方案
