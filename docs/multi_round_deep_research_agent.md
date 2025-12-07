# Multi-Round Deep Research Agent

本文档介绍 `Agent` 类中 `_run_multi_round` 方法的实现原理，该方法用于多轮深度搜索研究场景。

## 概述

Multi-Round 策略是一种迭代式对话方法，Agent 可以进行多轮思考和工具调用，以找到准确答案。适用于需要多步推理、信息检索和综合分析的复杂问题。

## 启用方式

```python
from agentica import Agent, OpenAIChat, JinaTool, SearchSerperTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[JinaTool(), SearchSerperTool()],
    enable_multi_round=True,  # 启用多轮策略
    max_rounds=22,            # 最大轮次
    max_tokens=40000,         # token 上限
    debug=True
)

agent.print_response("复杂的多步骤问题...")
```

## 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_multi_round` | bool | False | 是否启用多轮策略 |
| `max_rounds` | int | 10 | 最大执行轮次 |
| `max_tokens` | int | 32000 | 输入 token 上限估算 |

## 执行流程 (重构后)

```
┌─────────────────────────────────────────────────────────────┐
│                    _run_multi_round                         │
├─────────────────────────────────────────────────────────────┤
│  1. 初始化                                                   │
│     - 生成 run_id, 初始化 run_response                       │
│     - 更新模型配置, 设置 model.run_tools = False             │
│     - 从存储读取会话                                         │
├─────────────────────────────────────────────────────────────┤
│  2. 准备消息                                                 │
│     - 构建 system_message, user_messages                    │
│     - 初始化 messages_for_model                             │
├─────────────────────────────────────────────────────────────┤
│  3. 多轮迭代执行 (for turn in range(max_rounds))            │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  3.1 Token 估算检查                                  │ │
│     │      - 超出 max_tokens * 3 则提前终止                 │ │
│     ├─────────────────────────────────────────────────────┤ │
│     │  3.2 调用模型生成响应                                 │ │
│     │      model_response = self.model.response(...)       │ │
│     ├─────────────────────────────────────────────────────┤ │
│     │  3.3 检查 assistant_message.tool_calls               │ │
│     │      - 有工具调用: 执行所有工具，添加结果到消息        │ │
│     │      - 无工具调用: break 退出循环                     │ │
│     └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  4. 恢复 model.run_tools 原始设置                           │
├─────────────────────────────────────────────────────────────┤
│  5. 更新内存和存储                                           │
│     - 添加消息到 memory                                     │
│     - 创建 AgentRun 记录                                    │
│     - 写入 storage                                          │
├─────────────────────────────────────────────────────────────┤
│  6. 返回 run_response                                       │
└─────────────────────────────────────────────────────────────┘
```

## 关键机制 (重构后)

### 1. 禁用模型内部工具执行

设置 `model.run_tools = False`，由 Agent 手动控制工具执行流程：

```python
original_run_tools = self.model.run_tools
self.model.run_tools = False
try:
    # ... 循环执行 ...
finally:
    self.model.run_tools = original_run_tools
```

### 2. 工具调用作为循环条件

以工具调用存在与否作为循环终止条件，无工具调用即认为任务完成：

```python
has_tool_calls = (
    assistant_message and
    assistant_message.tool_calls and
    len(assistant_message.tool_calls) > 0
)

if has_tool_calls:
    # 执行工具...
else:
    # 无工具调用 = 任务完成
    logger.debug("No tool calls, task completed")
    break
```

### 3. 支持多工具调用

遍历 `tool_calls` 列表，支持单轮多工具并行执行：

```python
for tool_call in assistant_message.tool_calls:
    tool_call_id = tool_call.get("id", "")
    func_info = tool_call.get("function", {})
    func_name = func_info.get("name", "")
    
    function_call = get_function_call_for_tool_call(tool_call, self.get_tools())
    result = function_call.execute()
    
    tool_message = Message(
        role="tool",
        tool_call_id=tool_call_id,
        content=str(result)
    )
    messages_for_model.append(tool_message)
```

### 4. 保留 reasoning_content

确保模型的思考过程被保留并传递：

```python
if model_response.reasoning_content and not assistant_message.reasoning_content:
    assistant_message.reasoning_content = model_response.reasoning_content
```

## 适用场景

- 复杂问题需要多步信息检索
- 需要交叉验证多个信息源
- 答案需要综合分析和推理
- 初始搜索结果不足以直接回答问题

## 与普通 run 的区别

| 特性 | 普通 run | multi_round |
|------|----------|-------------|
| 执行轮次 | 单轮 (模型内部递归) | 多轮迭代 (Agent 控制) |
| 工具调用 | 模型自动处理 | Agent 手动执行 |
| 终止条件 | 模型自行决定 | 无工具调用时终止 |
| Token 管理 | 依赖模型限制 | 主动估算和控制 |

---

## 重构记录

### 重构前 vs 重构后对比

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| **循环终止条件** | `<answer>` 标签检测 + 引导提示 | 无工具调用时自动终止 |
| **工具调用处理** | 单个 `tool_call` (dict) | 多个 `tool_calls` (list) |
| **model.run_tools** | 默认 True (模型内部处理) | 设置为 False (Agent 手动控制) |
| **引导提示** | 有 (5 种轮换提示) | 无 (让模型自然决策) |
| **最大轮次处理** | 强制生成最终答案 | 直接返回最后响应 |

### 参考实现: DeepSeekAgent

```python
class DeepSeekAgent:
    def run(self, messages, tools=None, tool_map=None, max_turns=10):
        for i in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                extra_body=self.extra_body
            )
            message = response.choices[0].message
            
            # 处理 reasoning_content
            reasoning_content = getattr(message, 'reasoning_content', None)
            if reasoning_content:
                msg_dict = message.model_dump(exclude_none=True)
                msg_dict['reasoning_content'] = reasoning_content
                messages.append(msg_dict)
            else:
                messages.append(message)
            
            # 处理工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result = tool_map[func_name](**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str
                    })
            else:
                # 无工具调用则结束
                break
```

### 重构要点

1. **移除 `<answer>` 标签检测**: 以工具调用作为唯一循环条件
2. **移除引导提示机制**: 让模型自然决策何时停止
3. **支持多工具调用**: 遍历 `tool_calls` 列表
4. **手动控制工具执行**: 设置 `model.run_tools = False`
5. **保留 reasoning_content**: 在 Message 中携带
6. **同步和异步版本**: `_run_multi_round` 和 `_arun_multi_round` 同步更新
