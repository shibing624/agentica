现在让我总结其他两个问题的详细分析：

## 详细分析总结

### 1. Agent 内存泄漏修复 (weakref)

已创建 `usage_weakref.md` 文档，核心要点：

**问题**：`RunItem` 和 `RunResult` 持有 `Agent` 的强引用，多 Agent handoff 场景下形成引用链，导致内存泄漏。

**修复**（PR #2014）：
- 在 `src/agents/items.py` 的 `RunItemBase` 和 `HandoffOutputItem` 中添加 `_agent_ref` 弱引用字段
- 在 `src/agents/result.py` 的 `RunResult` 和 `RunResultStreaming` 中添加弱引用
- 提供 `release_agent()` 方法主动释放强引用
- 通过 `__getattribute__` 实现透明的弱引用访问

---

### 2. Tool Guardrails (工具守卫) 详解

**文件位置**：`src/agents/tool_guardrails.py`

**架构设计**：

```
┌─────────────────────────────────────────────────────────────┐
│                      Tool Execution Flow                     │
├─────────────────────────────────────────────────────────────┤
│  1. Tool Call Received                                       │
│         ↓                                                    │
│  2. ToolInputGuardrail.run() ──→ 检查输入是否合法            │
│         ↓                                                    │
│     ┌─ allow ──────────→ 继续执行                            │
│     ├─ reject_content ─→ 返回拒绝消息，跳过工具执行          │
│     └─ raise_exception → 抛出异常，终止运行                  │
│         ↓                                                    │
│  3. Tool Function Execution (实际执行工具)                   │
│         ↓                                                    │
│  4. ToolOutputGuardrail.run() ──→ 检查输出是否合规           │
│         ↓                                                    │
│     ┌─ allow ──────────→ 返回原始结果                        │
│     ├─ reject_content ─→ 用拒绝消息替换结果                  │
│     └─ raise_exception → 抛出异常，终止运行                  │
│         ↓                                                    │
│  5. Return Result to LLM                                     │
└─────────────────────────────────────────────────────────────┘
```

**核心类**：

| 类名 | 作用 |
|------|------|
| `ToolInputGuardrail` | 工具执行前的输入验证守卫 |
| `ToolOutputGuardrail` | 工具执行后的输出验证守卫 |
| `ToolGuardrailFunctionOutput` | 守卫函数的返回结果 |
| `ToolInputGuardrailData` | 传递给输入守卫的上下文数据 |
| `ToolOutputGuardrailData` | 传递给输出守卫的上下文数据（含工具输出） |

**三种行为模式**：

```python
# 1. allow - 允许继续执行
ToolGuardrailFunctionOutput.allow(output_info={"checked": True})

# 2. reject_content - 拒绝但继续运行，返回消息给 LLM
ToolGuardrailFunctionOutput.reject_content(
    message="This operation is not allowed",
    output_info={"reason": "security"}
)

# 3. raise_exception - 抛出异常终止运行
ToolGuardrailFunctionOutput.raise_exception(output_info={"severity": "critical"})
```

**使用示例**：

```python
from agents import function_tool, FunctionTool
from agents.tool_guardrails import (
    tool_input_guardrail,
    tool_output_guardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolGuardrailFunctionOutput,
)

# 定义输入守卫
@tool_input_guardrail
def check_file_path(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    # 检查工具输入参数
    args = data.context.tool_call.arguments
    if "/etc/" in args or "/root/" in args:
        return ToolGuardrailFunctionOutput.reject_content(
            message="Access to system directories is forbidden"
        )
    return ToolGuardrailFunctionOutput.allow()

# 定义输出守卫
@tool_output_guardrail
def sanitize_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    # 检查工具输出
    if "password" in str(data.output).lower():
        return ToolGuardrailFunctionOutput.reject_content(
            message="[REDACTED - sensitive data removed]"
        )
    return ToolGuardrailFunctionOutput.allow()

# 创建带守卫的工具
@function_tool
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

# 手动添加守卫
read_file_tool = FunctionTool(
    name="read_file",
    description="Read a file",
    params_json_schema={...},
    on_invoke_tool=...,
    tool_input_guardrails=[check_file_path],
    tool_output_guardrails=[sanitize_output],
)
```
---

---


