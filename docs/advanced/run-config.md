# RunConfig

`RunConfig` 提供 per-run 级别的配置覆盖，将"每次运行可能不同"的参数从 Agent 构造中分离出来。

## 基本用法

```python
from agentica.run.config import RunConfig

result = await agent.run("分析数据", config=RunConfig(
    run_timeout=30,
    max_cost_usd=0.5,
    enabled_tools=["web_search", "read_file"],
))
```

## 全部参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `response_model` | `Type[BaseModel]` | `None` | Pydantic 模型，启用结构化输出 |
| `use_structured_outputs` | `bool` | `None` | 使用 OpenAI 严格结构化输出 |
| `tool_choice` | `str \| dict` | `None` | 工具选择策略 |
| `run_timeout` | `float` | `None` | 运行总超时（秒） |
| `first_token_timeout` | `float` | `None` | 首个 token 超时（秒） |
| `idle_timeout` | `float` | `None` | 流式 token 间隔超时（秒） |
| `save_response_to_file` | `str` | `None` | 响应保存到文件 |
| `stream_intermediate_steps` | `bool` | `False` | 流式输出中间步骤 |
| `hooks` | `RunHooks` | `None` | 运行级生命周期钩子 |
| `enabled_tools` | `List[str]` | `None` | 工具白名单（None = 使用 Agent 默认） |
| `enabled_skills` | `List[str]` | `None` | Skill 白名单 |
| `max_cost_usd` | `float` | `None` | 运行成本上限（美元） |

## 超时控制

### run_timeout

限制整个运行的总时间：

```python
result = await agent.run("复杂任务", config=RunConfig(
    run_timeout=60,  # 最多运行 60 秒
))
```

### first_token_timeout

限制等待第一个 token 的时间（检测 API 无响应）：

```python
async for chunk in agent.run_stream("Hello", config=RunConfig(
    first_token_timeout=10,  # 10 秒内必须收到第一个 token
)):
    print(chunk.content, end="")
```

### idle_timeout

检测流式响应中的"静默挂起"——连接存活但无数据流动：

```python
async for chunk in agent.run_stream("Hello", config=RunConfig(
    idle_timeout=30,  # token 之间最多间隔 30 秒
)):
    print(chunk.content, end="")
```

## 成本预算

通过 `max_cost_usd` 设置运行成本上限：

```python
result = await agent.run("分析大量数据", config=RunConfig(
    max_cost_usd=1.0,  # 最多花费 1 美元
))
```

当 `CostTracker.total_cost_usd` 超过预算时，运行自动停止并返回预算超限警告。

## 工具/Skill 白名单

运行时限制可用工具：

```python
# 只允许使用搜索和读文件
result = await agent.run("搜索并分析", config=RunConfig(
    enabled_tools=["web_search", "read_file"],
))

# 只启用特定 Skill
result = await agent.run("写代码", config=RunConfig(
    enabled_skills=["code_review"],
))
```

## 结构化输出

运行时启用结构化输出：

```python
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str
    score: float

result = await agent.run("评估这个方案", config=RunConfig(
    response_model=Report,
))
report: Report = result.content
```

## 下一步

- [Hooks](hooks.md) -- 运行级生命周期钩子
- [Context Compression](compression.md) -- 上下文压缩
- [Agent 概念](../concepts/agent.md) -- Agent 运行方式
