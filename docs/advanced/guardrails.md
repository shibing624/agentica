# Guardrails

Guardrails 提供 Agent 运行时的输入输出验证机制，防止不当内容通过。支持 Agent 级和 Tool 级两层守卫。

## 四层守卫

```
用户输入 -> [InputGuardrail] -> Agent 处理 -> [OutputGuardrail] -> 返回用户
                                  |
                          工具调用参数 -> [ToolInputGuardrail]
                          工具返回值 -> [ToolOutputGuardrail]
```

## Agent 级守卫

### 输入守卫

在 Agent 处理用户消息之前验证：

```python
from agentica import Agent, InputGuardrail, GuardrailFunctionOutput

@input_guardrail
async def block_sensitive_info(text: str) -> GuardrailFunctionOutput:
    """阻止包含敏感信息的输入"""
    sensitive_keywords = ["密码", "身份证", "银行卡"]
    for kw in sensitive_keywords:
        if kw in text:
            return GuardrailFunctionOutput(
                tripwire_triggered=True,
                output_text=f"检测到敏感信息({kw}), 请勿输入个人隐私数据。",
            )
    return GuardrailFunctionOutput(tripwire_triggered=False)

agent = Agent(
    input_guardrails=[
        InputGuardrail(guardrail_function=block_sensitive_info),
    ],
)
```

### 输出守卫

在返回用户之前验证 Agent 的输出：

```python
from agentica import OutputGuardrail, GuardrailFunctionOutput

@output_guardrail
async def check_output_quality(text: str) -> GuardrailFunctionOutput:
    """检查输出质量"""
    if len(text) < 10:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_text="输出内容过短, 请提供更详细的回答。",
        )
    return GuardrailFunctionOutput(tripwire_triggered=False)

agent = Agent(
    output_guardrails=[
        OutputGuardrail(guardrail_function=check_output_quality),
    ],
)
```

## Tool 级守卫

### 工具输入守卫

在工具执行前验证参数：

```python
from agentica import ToolInputGuardrail, ToolGuardrailFunctionOutput

@tool_input_guardrail
async def validate_shell_command(
    tool_name: str, arguments: dict
) -> ToolGuardrailFunctionOutput:
    """阻止危险的 Shell 命令"""
    if tool_name == "execute":
        cmd = arguments.get("command", "")
        dangerous = ["rm -rf", "sudo", "chmod 777", "mkfs"]
        for d in dangerous:
            if d in cmd:
                return ToolGuardrailFunctionOutput(
                    tripwire_triggered=True,
                    output_text=f"阻止执行危险命令: {d}",
                )
    return ToolGuardrailFunctionOutput(tripwire_triggered=False)
```

### 工具输出守卫

在工具返回结果后过滤：

```python
@tool_output_guardrail
async def filter_sensitive_output(
    tool_name: str, result: str
) -> ToolGuardrailFunctionOutput:
    """过滤工具输出中的敏感信息"""
    import re
    cleaned = re.sub(r'sk-[a-zA-Z0-9]{32,}', 'sk-***REDACTED***', result)
    return ToolGuardrailFunctionOutput(
        tripwire_triggered=False,
        output_text=cleaned,
    )
```

## 行为模式

守卫触发时有三种行为：

| 模式 | `tripwire_triggered` | 效果 |
|------|:-------------------:|------|
| **允许** | `False` | 正常继续 |
| **拒绝并替换** | `True` + `output_text` | 用替换文本作为响应 |
| **抛出异常** | raise `GuardrailTripwireTriggered` | 中止执行 |

## 组合使用

```python
from agentica import (
    Agent,
    OpenAIChat,
    BuiltinExecuteTool,
    InputGuardrail,
    OutputGuardrail,
    ToolInputGuardrail,
    ToolOutputGuardrail,
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[BuiltinExecuteTool()],
    input_guardrails=[
        InputGuardrail(guardrail_function=block_sensitive_info),
    ],
    output_guardrails=[
        OutputGuardrail(guardrail_function=check_output_quality),
    ],
    tool_input_guardrails=[
        ToolInputGuardrail(guardrail_function=validate_shell_command),
    ],
    tool_output_guardrails=[
        ToolOutputGuardrail(guardrail_function=filter_sensitive_output),
    ],
)
```

## 下一步

- [Tools](../concepts/tools.md) -- 工具级守卫的更多细节
- [Agent 概念](../concepts/agent.md) -- Agent 整体架构
- [Hooks](hooks.md) -- 生命周期钩子
