# Hooks

Hooks 是 Agentica 的生命周期事件系统，让你在 Agent 执行的关键节点插入自定义逻辑。

## 两级 Hooks

| 级别 | 类 | 作用域 | 设置方式 |
|------|-----|--------|----------|
| **Agent 级** | `AgentHooks` | 单个 Agent | `Agent(hooks=...)` |
| **Run 级** | `RunHooks` | 整个运行（含子 Agent） | `RunConfig(hooks=...)` |

## AgentHooks

Per-agent 生命周期钩子：

```python
from agentica.agent.hooks import AgentHooks

class LoggingHooks(AgentHooks):
    async def on_start(self, agent, **kwargs):
        print(f"{agent.name} starting")

    async def on_end(self, agent, output, **kwargs):
        print(f"{agent.name} produced: {output}")

agent = Agent(hooks=LoggingHooks())
```

| 方法 | 触发时机 |
|------|----------|
| `on_start(agent)` | Agent 开始执行 |
| `on_end(agent, output)` | Agent 完成执行 |

## RunHooks

全局 run 级生命周期钩子，观察整个运行过程：

```python
from agentica.agent.hooks import RunHooks
from agentica.run.config import RunConfig

class MetricsHooks(RunHooks):
    def __init__(self):
        self.event_counter = 0

    async def on_agent_start(self, agent, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: Agent {agent.name} started")

    async def on_llm_start(self, agent, messages, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: LLM call started")

    async def on_llm_end(self, agent, response, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: LLM call ended")

    async def on_tool_start(self, agent, tool_name, tool_call_id, tool_args, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: Tool {tool_name} started")

    async def on_tool_end(self, agent, tool_name, tool_call_id, tool_args, result, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: Tool {tool_name} ended")

    async def on_agent_transfer(self, from_agent, to_agent, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: Transfer {from_agent.name} -> {to_agent.name}")

    async def on_agent_end(self, agent, output, **kwargs):
        self.event_counter += 1
        print(f"#{self.event_counter}: Agent {agent.name} ended")

hooks = MetricsHooks()
result = await agent.run("Hello", config=RunConfig(hooks=hooks))
```

### 全部 RunHooks 事件

| 方法 | 触发时机 | 参数 |
|------|----------|------|
| `on_agent_start` | 任意 Agent 开始 | `agent` |
| `on_agent_end` | 任意 Agent 结束 | `agent, output` |
| `on_llm_start` | LLM API 调用前 | `agent, messages` |
| `on_llm_end` | LLM API 调用后 | `agent, response` |
| `on_tool_start` | 工具执行前 | `agent, tool_name, tool_call_id, tool_args` |
| `on_tool_end` | 工具执行后 | `agent, tool_name, tool_call_id, tool_args, result, is_error, elapsed` |
| `on_agent_transfer` | Agent 委派转移 | `from_agent, to_agent` |
| `on_user_prompt` | 用户输入处理前 | `agent, message` -> 返回修改后的 message 或 None |
| `on_pre_compact` | 上下文压缩前 | `agent, messages` |
| `on_post_compact` | 上下文压缩后 | `agent, messages` |

### on_user_prompt

在用户输入被处理之前拦截和修改：

```python
class InputPreprocessor(RunHooks):
    async def on_user_prompt(self, agent, message, **kwargs):
        # 返回修改后的消息，或 None 保持不变
        if "秘密" in message:
            return message.replace("秘密", "[REDACTED]")
        return None
```

### on_pre_compact / on_post_compact

在上下文压缩前后执行自定义逻辑：

```python
class CompactionLogger(RunHooks):
    async def on_pre_compact(self, agent, messages, **kwargs):
        print(f"Compacting: {len(messages)} messages")

    async def on_post_compact(self, agent, messages, **kwargs):
        print(f"After compact: {len(messages)} messages")
```

## ConversationArchiveHooks

内置 Hook，自动将对话归档到 Workspace 的每日日志：

```python
from agentica.agent.hooks import ConversationArchiveHooks
from agentica.run.config import RunConfig

hooks = ConversationArchiveHooks()
response = await agent.run("Hello", config=RunConfig(hooks=hooks))
# 对话自动保存到 workspace/users/{user_id}/memory/YYYY-MM-DD.md
```

## 组合多个 Hooks

系统内部使用 `_CompositeRunHooks` 自动组合多个 Hook 实例，按注册顺序依次执行。

## 下一步

- [RunConfig](run-config.md) -- 运行时配置
- [Context Compression](compression.md) -- 压缩相关 Hooks
- [Memory & Workspace](../concepts/memory.md) -- 对话归档
