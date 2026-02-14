# Agentica Async-First 技术升级方案

> Write async, expose async, wrap sync explicitly.

**日期**: 2026-02-10
**状态**: 设计阶段
**版本**: v1.0

---

## 一、现状分析

### 1.1 当前架构问题

Agentica 采用 **Sync/Async 双轨并行** 模式，每层都维护独立的同步和异步方法对：

```
Agent层:     run() / arun()                    -- 独立实现，~900行 x2
Model层:     response() / aresponse()          -- 独立实现，~200行 x2
             run_function_calls() / arun_function_calls()  -- 独立实现，~120行 x2
Tool层:      execute() / aexecute()            -- 独立实现，~100行 x2
```

**调用链全貌**：

```
Sync路径:
Agent.run() → _run() → _run_single_round() / _run_multi_round()
  → Model.response() / response_stream()
    → OpenAIChat.invoke() → openai.chat.completions.create()
    → Model.run_function_calls() → FunctionCall.execute()
      → _run_sync_or_async() → entrypoint()

Async路径:
Agent.arun() → _arun() → _arun_single_round() / _arun_multi_round()
  → Model.aresponse() / aresponse_stream()
    → OpenAIChat.ainvoke() → openai.chat.completions.create() [async]
    → Model.arun_function_calls() → FunctionCall.aexecute()
      → await entrypoint() 或 loop.run_in_executor(entrypoint)
```

### 1.2 具体问题清单

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| **sync/async 代码重复** | runner.py 1885行，其中~800行是async对sync的逐行复制 | 高 |
| **async路径功能缺失** | `_arun()` 缺少 Langfuse trace context；`_arun_multi_round()` 缺少 `_on_pre_step()` hook | 高 |
| **工具串行执行** | `run_function_calls` / `arun_function_calls` 逐个执行工具，无并行 | 中 |
| **同步路径阻塞事件循环** | `_run_sync_or_async()` 使用 ThreadPoolExecutor + asyncio.run 的 hack | 中 |
| **Workflow 无异步支持** | `Workflow.run()` 纯同步，无法在异步上下文中使用 | 中 |
| **Subagent 纯同步** | `BuiltinTaskTool.task()` 同步调用子代理 | 中 |

### 1.3 代码重复量化

```
runner.py 中同步/异步方法对照：
  _run()                ~75行   vs  _arun()              ~37行  (async缺少Langfuse)
  _run_single_round()   ~240行  vs  _arun_single_round() ~270行 (几乎相同)
  _run_multi_round()    ~320行  vs  _arun_multi_round()  ~310行 (几乎相同)
  run()                 ~106行  vs  arun()               ~110行 (几乎相同)

model/base.py:
  run_function_calls()  ~120行  vs  arun_function_calls() ~112行 (仅execute→aexecute)
  handle_post_tool_*    ~27行   vs  ahandle_post_tool_*   ~28行  (仅response→aresponse)

model/openai/chat.py:
  response()            ~72行   vs  aresponse()           ~73行  (仅invoke→ainvoke)
  response_stream()     ~83行   vs  aresponse_stream()    ~83行
  handle_tool_calls()   ~57行   vs  ahandle_tool_calls()  ~46行
  invoke()              ~32行   vs  ainvoke()             ~32行

tools/base.py:
  execute()             ~82行   vs  aexecute()            ~105行

合计重复代码约 1500+ 行
```

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
│  _run_single_round()    -- async only             │
│  _run_multi_round()     -- async only             │
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
from typing import TypeVar, Coroutine, AsyncIterator, Iterator

T = TypeVar("T")


def run_sync(coro: Coroutine[None, None, T]) -> T:
    """在同步上下文中运行异步协程。

    处理三种场景：
    1. 无事件循环 → asyncio.run()
    2. 在事件循环中（如 Jupyter） → 新线程 + 新事件循环
    3. 嵌套调用保护
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 场景1: 无运行中的事件循环，直接运行
        return asyncio.run(coro)

    # 场景2: 已有事件循环（Jupyter / 嵌套调用）
    # 在新线程中创建新的事件循环来运行
    result = None
    exception = None

    def _run_in_thread():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except BaseException as e:
            exception = e

    thread = threading.Thread(target=_run_in_thread)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    return result


def iter_over_async(ait: AsyncIterator[T]) -> Iterator[T]:
    """将 AsyncIterator 转换为同步 Iterator。"""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(ait.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()
```

---

## 四、各层改造方案

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
    # ==================== 公共 API ====================

    async def run(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        stream: bool = False,
        **kwargs,
    ) -> Union[RunResponse, AsyncIterator[RunResponse]]:
        """核心运行方法 -- 纯异步，唯一实现。"""
        # 结构化输出处理
        if self.response_model is not None and self.parse_response:
            return await self._run_structured(message, **kwargs)

        # 流式/非流式
        if stream and self.is_streamable:
            return self._run_impl(message, stream=True, **kwargs)
        else:
            final = None
            async for response in self._run_impl(message, stream=False, **kwargs):
                final = response
            return final

    def run_sync(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        **kwargs,
    ) -> RunResponse:
        """同步适配器 -- 唯一的同步入口。"""
        from agentica.utils.async_utils import run_sync
        return run_sync(self.run(message, stream=False, **kwargs))

    async def run_stream(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        **kwargs,
    ) -> AsyncIterator[RunResponse]:
        """流式运行的便捷方法。"""
        return await self.run(message, stream=True, **kwargs)

    def run_stream_sync(
        self: "Agent",
        message: Optional[Union[str, List, Dict, Message]] = None,
        **kwargs,
    ) -> Iterator[RunResponse]:
        """同步流式适配器。"""
        from agentica.utils.async_utils import iter_over_async
        async_iter = self._run_impl(message, stream=True, **kwargs)
        return iter_over_async(async_iter)

    # ==================== 核心引擎 ====================

    async def _run_impl(
        self: "Agent",
        message = None,
        *,
        stream: bool = False,
        **kwargs,
    ) -> AsyncIterator[RunResponse]:
        """统一执行引擎 -- 合并原 _run() + _arun()。"""
        with langfuse_trace_context(self) as trace:  # 修复: async也有Langfuse
            if self.enable_multi_round:
                async for response in self._run_multi_round(message, stream=stream, **kwargs):
                    yield response
            else:
                async for response in self._run_single_round(message, stream=stream, **kwargs):
                    yield response

    async def _run_single_round(self, ...) -> AsyncIterator[RunResponse]:
        """单轮执行 -- 唯一实现。"""
        # 合并原 _run_single_round + _arun_single_round
        # 修复: 保留所有hooks, Langfuse, 内存并行优化
        ...

    async def _run_multi_round(self, ...) -> AsyncIterator[RunResponse]:
        """多轮执行 -- 唯一实现。"""
        # 合并原 _run_multi_round + _arun_multi_round
        # 修复: 保留 _on_pre_step() hook (原async版缺失)
        # 新增: asyncio.gather 并行工具执行
        ...

    # 删除: arun()                   → run() 就是 async
    # 删除: arun_stream()            → run_stream() 就是 async
    # 删除: _arun()                  → _run_impl() 统一
    # 删除: _run_single_round() 同步 → 只保留 async 版
    # 删除: _arun_single_round()     → 合并到 _run_single_round()
    # 删除: _run_multi_round() 同步  → 只保留 async 版
    # 删除: _arun_multi_round()      → 合并到 _run_multi_round()
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
内部方法:         _run_impl(), _run_single_round(), _run_multi_round()
```

**原则**：
- 默认名字留给 async（因为 async 是主路径）
- sync 适配器统一加 `_sync` 后缀
- 不再使用 `a` 前缀 (`arun`, `aresponse`, `ainvoke`) -- 这是旧模式

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
Phase 1: 基础设施                     [预计 1天]
  ├── 新增 agentica/utils/async_utils.py (run_sync, iter_over_async)
  └── 编写单元测试

Phase 2: Tool 层 (自底向上)            [预计 1天]
  ├── FunctionCall.execute() → async
  ├── 删除 aexecute(), _run_sync_or_async()
  └── 更新所有引用

Phase 3: Model 层                     [预计 2天]
  ├── Model 基类: 删除同步方法，async 方法去掉 a 前缀
  ├── OpenAIChat: 合并同步/异步，只保留 async
  ├── 其他 Model 实现同步改造
  ├── run_function_calls 加入 asyncio.gather 并行
  └── 更新所有引用

Phase 4: Agent 层                     [预计 2天]
  ├── runner.py: 合并 sync/async 方法对，只保留 async
  ├── 新增 run_sync(), run_stream_sync()
  ├── 修复 async 缺失的 hooks 和 Langfuse
  └── 更新 base.py Mixin 注册

Phase 5: 上层模块                     [预计 1天]
  ├── Workflow: run() → async, 新增 run_sync()
  ├── CLI: 改用 run_stream_sync()
  ├── DeepAgent: 更新 hook 调用
  └── Subagent / deep_tools: 更新为 async

Phase 6: 测试与清理                   [预计 1天]
  ├── 全量测试
  ├── 更新 examples/
  └── 更新文档
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

## 九、附录

### A. 完整文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agentica/utils/async_utils.py` | 新增 | run_sync(), iter_over_async() |
| `agentica/tools/base.py` | 修改 | FunctionCall async 化 |
| `agentica/model/base.py` | 修改 | 删除同步方法，async 改名，并行工具 |
| `agentica/model/openai/chat.py` | 修改 | 合并同步/异步，删除同步客户端 |
| `agentica/model/anthropic/claude.py` | 修改 | 同上 |
| `agentica/model/zhipuai/chat.py` | 修改 | 同上 |
| `agentica/model/deepseek/chat.py` | 修改 | 同上 |
| `agentica/model/ollama/chat.py` | 修改 | 同上 |
| `agentica/agent/base.py` | 修改 | 更新方法声明和 Mixin 注册 |
| `agentica/agent/runner.py` | 修改 | 核心: 合并sync/async，~1000行削减 |
| `agentica/agent/printer.py` | 修改 | async 化 |
| `agentica/workflow.py` | 修改 | run() async 化 |
| `agentica/deep_agent.py` | 修改 | 更新 hook 调用 |
| `agentica/deep_tools.py` | 修改 | task() async 化 |
| `agentica/cli/interactive.py` | 修改 | 使用 run_stream_sync() |
| `agentica/cli/display.py` | 修改 | 适配新 API |
| `agentica/memory/` | 修改 | 合并 sync/async 内存操作 |
| `tests/` | 修改 | 全部改为 pytest-asyncio |
| `examples/` | 修改 | 更新为新 API |

### B. 参考资料

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Runner.run() / run_sync() 模式
- [PydanticAI](https://ai.pydantic.dev/) - Agent.run() / run_sync() 模式
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)
- [Agentica 技术优化方案 V3](../update_tech_v3.md) - 既有技术方案
