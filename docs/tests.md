# Agentica 单元测试重构方案

> 目标：基于 async-first 架构，重新设计测试用例，覆盖框架核心优势和全部关键能力。

**日期**: 2026-02-12
**状态**: 设计中
**基线**: 当前 tests/ 共 23 个文件，~4849 行

---

## 一、现状分析

### 1.1 当前测试覆盖情况

| 测试文件 | 行数 | 覆盖模块 | 评估 |
|----------|------|----------|------|
| `test_agent.py` | 315 | Agent 初始化、属性、超时 | 初始化覆盖良好，**缺少 run/run_stream 核心路径测试** |
| `test_async_tool.py` | 219 | FunctionCall async execute | 基本覆盖，缺少并行执行测试 |
| `test_agent_as_tool.py` | 274 | Agent.as_tool() | 覆盖良好 |
| `test_workflow.py` | 143 | Workflow 初始化和简单 run | **缺少多步编排、错误处理** |
| `test_memory.py` | 383 | AgentMemory, Memory, token 截断 | 覆盖良好 |
| `test_memory_leak.py` | 259 | weakref / GC 无泄漏 | 覆盖良好 |
| `test_workspace.py` | 267 | Workspace 文件读写、记忆 | 覆盖良好，缺少并发安全测试 |
| `test_sanitize_messages.py` | 189 | Model.sanitize_messages | 覆盖良好 |
| `test_vision_history.py` | 531 | 多模态视觉历史清理 | 覆盖良好 |
| `test_guardrails.py` | 420 | 输入/输出/工具守卫 | 覆盖良好 |
| `test_cli.py` | 130 | CLI 配置和导入 | 仅配置测试，**缺少交互流程** |
| `test_deep_agent.py` | 187 | DeepAgent 初始化 | 仅属性测试，**缺少运行时行为** |
| `test_react_agent.py` | 57 | Agent 基本 run | 过于简单 |
| `test_llm.py` | 30 | Model mock response | 过于简单 |
| `test_patch_tool.py` | 229 | PatchTool diff 应用 | 覆盖良好 |
| `test_acp.py` | 410 | ACP 协议类型和会话 | 覆盖良好 |
| `test_guardrails.py` | 420 | Guardrails 装饰器和执行 | 覆盖良好 |
| 其他（5 个） | ~590 | 外部工具/知识库 | 需要外部依赖，保留 |

### 1.2 关键缺口

| 缺口 | 优先级 | 说明 |
|------|--------|------|
| **Agent.run() 完整路径** | P0 | 缺少 mock 模型下的完整 run/run_stream/run_sync/run_stream_sync 测试 |
| **并行工具执行** | P0 | `run_function_calls` 的 `asyncio.gather` 并行执行完全未测试 |
| **流式输出** | P0 | `run_stream()` / `run_stream_sync()` 未测试 chunk 正确性 |
| **多工具调用回路** | P1 | 模型返回 tool_calls → 执行工具 → 返回结果 → 模型再响应的完整循环 |
| **Model 基类** | P1 | `response()`/`response_stream()` 抽象接口、`add_tool()`、`get_tools_for_api()` |
| **PromptsMixin** | P1 | `get_system_message()` 和 `get_messages_for_run()` 的 async 路径 |
| **SessionMixin** | P1 | `read_from_storage()`/`write_to_storage()` 的 async 路径 |
| **多模态输入** | P1 | images/audio/videos 参数传递和消息构建 |
| **RunResponse 事件流** | P1 | RunEvent 枚举在流式场景下的完整性 |
| **Workflow 多步** | P2 | 多 Agent 编排、步骤间状态传递 |
| **错误恢复** | P2 | 工具异常 (StopAgentRun/RetryAgentRun)、模型超时、网络异常 |
| **`__init__.py` 懒加载** | P2 | 线程安全懒加载、模块不存在时 fallback |
| **Prompt 模块** | P3 | PromptBuilder 组装、各模块 MD 加载 |

---

## 二、测试架构设计

### 2.1 统一规范

```python
# 所有新测试统一使用 pytest + pytest-asyncio
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# 异步测试标记
@pytest.mark.asyncio
async def test_example():
    ...

# 同步测试不需要标记
def test_sync_example():
    ...
```

**命名规范**：
- 文件：`test_{module}.py`
- 类：`Test{Module}{Feature}`（可选，用 class 分组相关测试）
- 方法：`test_{action}_{scenario}_{expected_result}`

**Mock 策略**：
- Model 层：统一使用 `AsyncMock` mock `response()` / `response_stream()`
- 工具层：真实执行简单函数，mock 外部依赖
- DB/文件 I/O：使用 `tempfile` + `shutil.rmtree` 清理

### 2.2 文件结构

```
tests/
├── conftest.py                      # 共享 fixtures（mock model、mock agent 等）
├── test_agent_core.py               # [新] Agent 四件套核心路径
├── test_agent_stream.py             # [新] 流式输出完整测试
├── test_agent_tools_integration.py  # [新] 工具调用回路（mock LLM）
├── test_agent_multimodal.py         # [新] 多模态输入（images/audio/videos）
├── test_model_base.py               # [新] Model 基类 + 并行工具执行
├── test_model_providers.py          # [新] 各 Provider 接口一致性
├── test_prompts.py                  # [新] PromptBuilder + 各模块
├── test_session.py                  # [新] SessionMixin async 持久化
├── test_run_response.py             # [新] RunResponse/RunEvent 序列化
├── test_workflow_advanced.py        # [新] Workflow 多步编排
├── test_lazy_loading.py             # [新] __init__.py 懒加载
├── test_error_handling.py           # [新] 异常和恢复路径
│
├── test_agent.py                    # [保留] Agent 初始化和属性
├── test_async_tool.py               # [保留] 工具 async execute
├── test_agent_as_tool.py            # [保留] Agent.as_tool()
├── test_memory.py                   # [保留] AgentMemory
├── test_memory_leak.py              # [保留] weakref / GC
├── test_workspace.py                # [保留] Workspace 文件操作
├── test_sanitize_messages.py        # [保留] sanitize_messages
├── test_vision_history.py           # [保留] 多模态视觉历史
├── test_guardrails.py               # [保留] 守卫系统
├── test_patch_tool.py               # [保留] PatchTool
├── test_acp.py                      # [保留] ACP 协议
├── test_cli.py                      # [保留] CLI 配置
├── test_deep_agent.py               # [保留+增强] DeepAgent
└── ...                              # 其他外部工具测试保留
```

---

## 三、新增测试用例详细设计

### 3.0 conftest.py — 共享 Fixtures

```python
# tests/conftest.py
"""Shared fixtures for all test modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentica.agent import Agent
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.run_response import RunResponse, RunEvent
from agentica.tools.base import Function, FunctionCall


@pytest.fixture
def mock_model():
    """Create a mock Model that returns a simple text response."""
    model = MagicMock(spec=Model)
    model.id = "mock-model"
    model.name = "MockModel"
    model.provider = "mock"
    model.tools = None
    model.functions = {}
    model.function_call_stack = None
    model.run_tools = True
    model.tool_call_limit = None
    model.system_prompt = None
    model.instructions = None
    model.structured_outputs = None
    model.supports_structured_outputs = False
    model.context_window = 128000
    model.max_output_tokens = None

    # Default async response
    mock_resp = ModelResponse(content="Hello from mock model!")
    model.response = AsyncMock(return_value=mock_resp)
    model.response_stream = AsyncMock()
    model.get_tools_for_api.return_value = None
    model.add_tool = MagicMock()
    model.sanitize_messages = Model.sanitize_messages  # Use real impl
    model.to_dict.return_value = {"id": "mock-model"}
    return model


@pytest.fixture
def mock_model_with_tools():
    """Create a mock Model that returns tool_calls on first response,
    then text on second response (simulating a tool-call round trip)."""
    model = MagicMock(spec=Model)
    model.id = "mock-tool-model"
    model.name = "MockToolModel"
    model.provider = "mock"
    model.tools = []
    model.functions = {}
    model.function_call_stack = None
    model.run_tools = True
    model.tool_call_limit = None
    model.system_prompt = None
    model.instructions = None
    model.structured_outputs = None
    model.supports_structured_outputs = False
    model.context_window = 128000
    model.max_output_tokens = None
    model.get_tools_for_api.return_value = []
    model.add_tool = MagicMock()
    model.sanitize_messages = Model.sanitize_messages
    model.to_dict.return_value = {"id": "mock-tool-model"}
    return model


@pytest.fixture
def simple_agent(mock_model):
    """Create a simple Agent with a mock model."""
    agent = Agent(name="TestAgent", model=mock_model)
    return agent


def make_sync_tool(name: str = "add"):
    """Create a simple sync tool function."""
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)
    add.__name__ = name
    return add


def make_async_tool(name: str = "async_search"):
    """Create a simple async tool function."""
    import asyncio
    async def async_search(query: str) -> str:
        """Search for something."""
        await asyncio.sleep(0.01)
        return f"Result for: {query}"
    async_search.__name__ = name
    return async_search
```

### 3.1 test_agent_core.py — Agent 四件套核心路径（P0）

覆盖 **async-first 架构的核心优势**：`run()` / `run_stream()` / `run_sync()` / `run_stream_sync()`。

```
TestAgentRun
├── test_run_async_returns_run_response          # await agent.run("Hello") 返回 RunResponse
├── test_run_async_content_matches_model_output   # content 与 model.response 一致
├── test_run_async_populates_run_id               # run_id 自动生成
├── test_run_async_populates_session_id           # session_id 正确传递
├── test_run_async_populates_agent_id             # agent_id 正确传递
├── test_run_async_populates_model_id             # model 字段正确设置
├── test_run_async_with_message_object            # 传入 Message 对象
├── test_run_async_with_dict_message              # 传入 dict 消息
├── test_run_async_with_messages_list             # 传入 messages 列表
├── test_run_async_no_message_no_error            # 空消息不报错
├── test_run_async_with_images                    # 带 images 参数
├── test_run_async_with_audio                     # 带 audio 参数
├── test_run_async_with_videos                    # 带 videos 参数
├── test_run_async_stores_in_memory               # 运行后 memory.runs 增加
├── test_run_async_history_messages               # add_history_to_messages=True 时历史消息传给 model

TestAgentRunSync
├── test_run_sync_returns_run_response            # run_sync 返回与 run 相同结果
├── test_run_sync_bridges_to_async_run            # run_sync 内部调用 async run
├── test_run_sync_works_without_event_loop        # 在无事件循环的同步上下文中正常工作
├── test_run_sync_with_all_params                 # run_sync 支持完整参数传递

TestAgentRunStream
├── test_run_stream_yields_multiple_chunks        # run_stream 产生多个 chunk
├── test_run_stream_first_chunk_has_event         # 每个 chunk 有正确的 event
├── test_run_stream_final_content_complete        # 最终拼接内容与完整响应一致
├── test_run_stream_intermediate_steps            # stream_intermediate_steps=True 时产生额外事件

TestAgentRunStreamSync
├── test_run_stream_sync_returns_iterator         # 返回同步 Iterator
├── test_run_stream_sync_yields_chunks            # 产生与 run_stream 相同的 chunk
├── test_run_stream_sync_in_for_loop              # 可在普通 for 循环中使用
├── test_run_stream_sync_thread_safe              # 后台线程模式安全退出

TestAgentTimeout
├── test_run_timeout_triggers_timeout_event        # [现有] 保留增强
├── test_first_token_timeout_in_stream             # [现有] 保留增强
├── test_run_timeout_in_stream                     # [现有] 保留增强
├── test_run_timeout_returns_partial_content        # 超时前的部分内容被保留
```

### 3.2 test_agent_stream.py — 流式输出完整测试（P0）

覆盖 streaming 的 **RunEvent 状态机**。

```
TestStreamEvents
├── test_stream_emits_run_started                 # 第一个 event 为 RunStarted
├── test_stream_emits_run_response_for_content    # 内容 chunk 为 RunResponse event
├── test_stream_emits_tool_call_started           # 工具开始时发出 ToolCallStarted
├── test_stream_emits_tool_call_completed         # 工具完成时发出 ToolCallCompleted
├── test_stream_emits_run_completed               # 最后一个 event 为 RunCompleted
├── test_stream_reasoning_events                  # 推理内容产生 ReasoningStarted/Step/Completed

TestStreamContent
├── test_stream_content_accumulates               # 多个 chunk 的 content 拼接后等于完整响应
├── test_stream_reasoning_content_separate        # reasoning_content 与 content 分离
├── test_stream_empty_content_skipped             # 空 content chunk 不产生输出

TestStreamToolCalls
├── test_stream_tool_call_then_text               # 工具调用后继续输出文本
├── test_stream_multiple_tool_calls               # 多个工具调用在流中正确报告
├── test_stream_tool_error_reported               # 工具失败在流中报告

TestStreamSync
├── test_stream_sync_matches_async                # run_stream_sync 输出与 run_stream 一致
├── test_stream_sync_queue_backpressure           # 后台线程产生速度 > 消费速度时不丢数据
```

### 3.3 test_agent_tools_integration.py — 工具调用回路（P0）

覆盖 Agent + Model + Tool 的**完整调用链**（mock LLM，真实工具执行）。

```
TestSingleToolCall
├── test_sync_tool_called_and_result_returned     # 同步工具正确执行，结果送回 model
├── test_async_tool_called_and_result_returned    # 异步工具正确执行
├── test_tool_result_in_final_response            # 最终 RunResponse 包含工具调用信息
├── test_tool_call_metrics_recorded               # 工具执行时间被记录

TestMultipleToolCalls
├── test_parallel_tools_executed_concurrently     # 多工具并行执行（验证时间 < 串行时间）
├── test_parallel_tools_results_ordered           # 并行执行结果按原始顺序返回
├── test_parallel_mixed_sync_async_tools          # 混合 sync/async 工具并行执行
├── test_tool_call_limit_respected                # tool_call_limit 限制生效

TestToolErrors
├── test_tool_exception_sets_error_field          # 工具抛异常时 FunctionCall.error 被设置
├── test_tool_exception_does_not_crash_agent      # 工具异常不导致 Agent 崩溃
├── test_stop_agent_run_stops_execution           # StopAgentRun 终止执行
├── test_retry_agent_run_triggers_retry           # RetryAgentRun 触发重试
├── test_partial_tool_failure_others_succeed      # 部分工具失败不影响其他工具

TestToolHooks
├── test_pre_hook_called_before_execution         # pre_hook 在工具执行前调用
├── test_post_hook_called_after_execution         # post_hook 在工具执行后调用
├── test_async_pre_hook_with_sync_tool            # async pre_hook + sync 工具
├── test_sync_pre_hook_with_async_tool            # [现有] sync pre_hook + async 工具

TestToolCallRoundTrip
├── test_model_returns_tool_call_agent_executes   # model 返回 tool_calls → agent 执行 → 结果送回
├── test_multi_round_tool_calls                   # 多轮工具调用（model 连续请求工具）
├── test_tool_call_with_streaming                 # 流式模式下的工具调用
```

### 3.4 test_model_base.py — Model 基类 + 并行工具执行（P0）

覆盖 **asyncio.gather 并行执行** 这一核心架构优势。

```
TestModelInterface
├── test_invoke_is_async                          # invoke() 是 coroutine
├── test_invoke_stream_is_async                   # invoke_stream() 是 coroutine
├── test_response_is_async                        # response() 是 coroutine
├── test_response_stream_is_async                 # response_stream() 是 coroutine
├── test_no_sync_response_method                  # 不存在同步的 response 方法
├── test_no_a_prefix_methods                      # 不存在 arun/aresponse/ainvoke 等旧方法

TestModelAddTool
├── test_add_callable_tool                        # add_tool(callable) 正确注册
├── test_add_tool_class                           # add_tool(Tool) 正确注册
├── test_add_function_object                      # add_tool(Function) 正确注册
├── test_add_dict_tool                            # add_tool(dict) 作为 raw schema
├── test_duplicate_tool_deduplicated              # 同名工具不重复注册
├── test_get_tools_for_api_format                 # get_tools_for_api() 返回 OpenAI 格式

TestRunFunctionCalls
├── test_single_tool_execution                    # 单个工具正确执行
├── test_parallel_execution_faster_than_serial    # N 个 sleep(0.1) 工具并行 < N*0.1s
├── test_parallel_execution_preserves_order       # 结果顺序与输入顺序一致
├── test_tool_started_events_emitted_first        # 先发出所有 started 事件，再执行
├── test_tool_completed_events_in_order           # completed 事件按顺序发出
├── test_tool_exception_isolated                  # 单个工具异常不取消其他工具
├── test_tool_call_exception_propagated           # ToolCallException 向上传播
├── test_function_call_stack_tracked              # function_call_stack 正确记录

TestSanitizeMessages
├── [现有 test_sanitize_messages.py 的全部用例保留]
```

### 3.5 test_agent_multimodal.py — 多模态输入（P1）

覆盖 **images / audio / videos** 参数在整个链路中的传递。

```
TestImageInput
├── test_run_with_single_image_url                # 单个 URL 图片
├── test_run_with_base64_image                    # base64 编码图片
├── test_run_with_multiple_images                 # 多张图片
├── test_image_added_to_user_message_content      # 图片被添加到 user message 的 content 列表
├── test_image_history_cleanup                    # [整合现有] 历史中 base64 被清理

TestAudioInput
├── test_run_with_audio_parameter                 # 音频参数传递
├── test_audio_message_format                     # 音频消息格式正确
├── test_audio_history_cleanup                    # 历史中音频数据被清理

TestVideoInput
├── test_run_with_video_parameter                 # 视频参数传递
├── test_video_message_format                     # 视频消息格式正确

TestMultimodalCombined
├── test_text_and_image_combined                  # 文本 + 图片组合
├── test_text_image_audio_combined                # 文本 + 图片 + 音频组合
├── test_multimodal_in_stream_mode                # 流式模式下的多模态
├── test_multimodal_history_multi_turn            # 多轮对话中的多模态历史
```

### 3.6 test_model_providers.py — Provider 接口一致性（P1）

验证各 Provider 都实现了相同的 async-only 接口。

```
TestProviderInterface
├── test_openai_chat_has_async_response           # OpenAIChat.response 是 async
├── test_openai_chat_has_async_response_stream    # OpenAIChat.response_stream 是 async
├── test_openai_chat_has_async_invoke             # OpenAIChat.invoke 是 async
├── test_openai_like_inherits_interface           # OpenAILike 继承 OpenAIChat 接口
├── test_claude_has_async_response                # Claude.response 是 async
├── test_gemini_has_async_response                # Gemini.response 是 async
├── test_ollama_has_async_response                # OllamaChat.response 是 async
├── test_groq_has_async_response                  # GroqChat.response 是 async
├── test_all_providers_no_sync_response           # 所有 provider 无同步 response 方法
├── test_all_providers_use_override_decorator     # 所有 provider 方法使用 @override
```

### 3.7 test_prompts.py — PromptBuilder + 模块（P1）

```
TestPromptBuilder
├── test_builder_default_modules                  # 默认加载所有基础模块
├── test_builder_custom_modules                   # 自定义模块列表
├── test_builder_build_produces_string            # build() 返回非空字符串
├── test_builder_includes_soul                    # 包含 soul 模块内容
├── test_builder_includes_tools_when_present      # 有工具时包含 tools 模块
├── test_builder_includes_heartbeat               # 包含 heartbeat 模块

TestPromptModules
├── test_load_prompt_reads_md_file                # load_prompt() 正确加载 .md 文件
├── test_soul_module_content                      # soul 模块返回有效内容
├── test_tools_module_content                     # tools 模块返回有效内容
├── test_task_management_module_content            # task_management 模块返回有效内容
├── test_self_verification_module_content          # self_verification 模块返回有效内容

TestGetSystemMessage
├── test_system_message_includes_instructions     # instructions 被包含在系统消息中
├── test_system_message_includes_datetime         # add_datetime_to_instructions 生效
├── test_system_message_includes_workspace        # workspace context 被包含
├── test_system_message_callable_prompt           # callable system_prompt 被调用
```

### 3.8 test_session.py — SessionMixin async 持久化（P1）

```
TestSessionPersistence
├── test_read_from_storage_no_db_noop             # 无 db 时 read_from_storage 不报错
├── test_write_to_storage_no_db_noop              # 无 db 时 write_to_storage 不报错
├── test_read_from_storage_async                  # read_from_storage 是 async
├── test_write_to_storage_async                   # write_to_storage 是 async
├── test_load_session_creates_new_if_not_exist    # load_session 创建新会话
├── test_session_state_persisted_across_runs      # session_state 跨 run 持久化
├── test_generate_session_name_async              # generate_session_name 是 async
```

### 3.9 test_run_response.py — RunResponse/RunEvent（P1）

```
TestRunResponse
├── test_run_response_creation                    # 基本创建
├── test_run_response_default_event               # 默认 event 为 RunResponse
├── test_run_response_to_json                     # to_json 序列化
├── test_run_response_to_dict                     # to_dict 序列化
├── test_run_response_with_tools                  # 包含 tools 列表
├── test_run_response_with_images                 # 包含 images
├── test_run_response_with_reasoning_content      # 包含 reasoning_content
├── test_run_response_extra_data                  # extra_data 字段

TestRunEvent
├── test_all_events_defined                       # 所有预期事件都已定义
├── test_event_values_are_strings                 # 事件值为字符串
├── test_event_names_unique                       # 事件名不重复
├── test_tool_call_events_present                 # 工具调用相关事件存在
├── test_reasoning_events_present                 # 推理相关事件存在
├── test_workflow_events_present                  # 工作流相关事件存在
```

### 3.10 test_workflow_advanced.py — Workflow 多步编排（P2）

```
TestWorkflowMultiStep
├── test_sequential_agents_workflow               # 多 Agent 顺序执行
├── test_workflow_passes_state_between_steps      # 步骤间状态传递
├── test_workflow_run_is_async                    # run() 是 async
├── test_workflow_run_sync_adapter                # run_sync() 正确包装
├── test_workflow_with_conditional_branching      # 条件分支执行
├── test_workflow_error_in_step_propagates        # 步骤异常向上传播
├── test_workflow_session_state_across_runs       # session_state 跨运行保持

TestWorkflowMemory
├── test_workflow_memory_records_runs             # memory.runs 记录每次运行
├── test_workflow_memory_custom                   # 自定义 WorkflowMemory
```

### 3.11 test_error_handling.py — 异常和恢复路径（P2）

```
TestModelErrors
├── test_model_response_exception_handled         # model.response() 抛异常被 Agent 捕获
├── test_model_stream_exception_handled           # model.response_stream() 异常被捕获
├── test_model_timeout_handled                    # model 超时返回超时响应

TestToolErrors
├── test_tool_value_error_captured                # ValueError 被捕获为 FunctionCall.error
├── test_tool_runtime_error_captured              # RuntimeError 被捕获
├── test_stop_agent_run_with_user_message         # StopAgentRun 携带 user_message
├── test_retry_agent_run_messages_injected        # RetryAgentRun 的 messages 被注入

TestCancellation
├── test_agent_cancel_during_run                  # 运行中取消
├── test_agent_cancel_during_stream               # 流式中取消
```

### 3.12 test_lazy_loading.py — `__init__.py` 懒加载（P2）

```
TestLazyLoading
├── test_core_modules_imported_eagerly            # Agent, Model, Tool 等直接可用
├── test_optional_module_lazy_loaded              # 数据库后端等按需加载
├── test_missing_optional_module_graceful_error   # 缺少可选依赖时友好报错
├── test_lazy_loading_thread_safe                 # 并发 import 不出错
├── test_all_public_names_accessible              # __all__ 中的名称都可访问
```

---

## 四、核心架构优势覆盖矩阵

将测试与 async-first 架构的核心优势对应：

| 架构优势 | 测试覆盖文件 | 关键测试用例 |
|----------|-------------|-------------|
| **Async-Native 四件套** | `test_agent_core.py` | `run`/`run_stream`/`run_sync`/`run_stream_sync` |
| **唯一执行引擎 `_run_impl`** | `test_agent_core.py` | 所有 run 变体共享相同行为 |
| **并行工具执行 asyncio.gather** | `test_model_base.py`, `test_agent_tools_integration.py` | `test_parallel_execution_faster_than_serial` |
| **流式原语 AsyncIterator** | `test_agent_stream.py` | RunEvent 状态机测试 |
| **同步适配器薄包装** | `test_agent_core.py` | `test_run_sync_bridges_to_async_run` |
| **sync 工具 run_in_executor** | `test_async_tool.py`, `test_agent_tools_integration.py` | 混合 sync/async 工具测试 |
| **weakref 无内存泄漏** | `test_memory_leak.py` | [现有] 已覆盖 |
| **Mixin 架构** | `test_prompts.py`, `test_session.py` | 各 Mixin 独立测试 |
| **多模态支持** | `test_agent_multimodal.py`, `test_vision_history.py` | images/audio/videos 全链路 |
| **Guardrails 守卫** | `test_guardrails.py` | [现有] 已覆盖 |
| **Workflow async 编排** | `test_workflow_advanced.py` | 多步编排 + 状态传递 |
| **Provider 一致性** | `test_model_providers.py` | 所有 Provider async-only 接口 |
| **懒加载线程安全** | `test_lazy_loading.py` | 并发 import 测试 |
| **Token 感知截断** | `test_memory.py` | [现有] max_tokens 测试 |
| **工具异常控制流** | `test_error_handling.py` | StopAgentRun/RetryAgentRun |
| **ACP 协议** | `test_acp.py` | [现有] 已覆盖 |

---

## 五、执行策略

### 5.1 优先级排序

```
P0（必须）: 覆盖框架核心差异化能力
  ├── conftest.py                        — 共享 fixtures
  ├── test_agent_core.py                 — 四件套核心路径
  ├── test_agent_stream.py               — 流式输出
  ├── test_agent_tools_integration.py    — 工具调用回路
  └── test_model_base.py                 — Model 基类 + 并行执行

P1（重要）: 覆盖二级核心能力
  ├── test_agent_multimodal.py           — 多模态
  ├── test_model_providers.py            — Provider 一致性
  ├── test_prompts.py                    — Prompt 构建
  ├── test_session.py                    — 会话持久化
  └── test_run_response.py              — 响应模型

P2（增强）: 边缘场景和健壮性
  ├── test_workflow_advanced.py          — Workflow 增强
  ├── test_error_handling.py             — 错误恢复
  └── test_lazy_loading.py              — 懒加载
```

### 5.2 现有测试处理

| 文件 | 处理方式 | 说明 |
|------|---------|------|
| `test_agent.py` | **保留** | 初始化和属性测试依然有效 |
| `test_async_tool.py` | **保留** | FunctionCall 测试依然有效 |
| `test_agent_as_tool.py` | **保留** | as_tool 测试依然有效 |
| `test_memory.py` | **保留** | Memory 测试依然有效 |
| `test_memory_leak.py` | **保留** | weakref 测试依然有效 |
| `test_workspace.py` | **保留** | Workspace 测试依然有效 |
| `test_sanitize_messages.py` | **保留，整合到 test_model_base.py** | 可作为 Model 测试的一部分 |
| `test_vision_history.py` | **保留** | 多模态历史测试独立 |
| `test_guardrails.py` | **保留** | 守卫测试独立 |
| `test_patch_tool.py` | **保留** | 工具测试独立 |
| `test_acp.py` | **保留** | ACP 测试独立 |
| `test_cli.py` | **保留** | CLI 配置测试独立 |
| `test_deep_agent.py` | **保留** | DeepAgent 测试独立 |
| `test_react_agent.py` | **合并到 test_agent_core.py** | 功能重叠 |
| `test_llm.py` | **合并到 test_model_base.py** | 功能重叠 |
| `test_edit_tool.py` | **删除** | 引用已重命名模块（已在 gitignore） |

### 5.3 Mock 策略总结

| 层级 | Mock 方式 | 原因 |
|------|----------|------|
| Model.response() | `AsyncMock` | 避免真实 API 调用 |
| Model.response_stream() | `AsyncMock` 返回 async generator | 控制流式输出 chunk |
| FunctionCall.execute() | **真实执行**（简单工具函数） | 验证实际 async 行为 |
| DB/Storage | `AsyncMock` 或 `tempfile` | 避免外部依赖 |
| Workspace | `tempfile` + 真实文件操作 | 验证文件 I/O |

### 5.4 运行命令

```bash
# 运行全部测试
python -m pytest tests/ -v --tb=short --ignore=tests/test_edit_tool.py

# 运行新增测试
python -m pytest tests/test_agent_core.py tests/test_agent_stream.py tests/test_agent_tools_integration.py tests/test_model_base.py -v

# 运行 P0 测试
python -m pytest tests/test_agent_core.py tests/test_agent_stream.py tests/test_agent_tools_integration.py tests/test_model_base.py -v --tb=short

# 运行并发测试（验证并行工具执行性能）
python -m pytest tests/test_model_base.py::TestRunFunctionCalls::test_parallel_execution_faster_than_serial -v -s
```

---

## 六、实际结果

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| 测试文件数 | 23 | 35 |
| 测试用例数 | ~316 | **500** |
| 代码行数 | ~4849 | **~7157** |
| Agent 核心路径覆盖 | 低 | 完整（四件套 + 流式 + 工具调用） |
| 并行执行覆盖 | 无 | 有（性能断言 < 0.25s） |
| 多模态覆盖 | 仅视觉历史 | images/audio/videos 全链路 |
| Provider 一致性覆盖 | 无 | 有（6 provider async 检查） |
| 错误处理覆盖 | 部分 | 完整（异常/超时/取消） |
| 运行时间 | ~3s | **~5s** |
| 全部通过 | ✓ | **✓（500 passed）** |
