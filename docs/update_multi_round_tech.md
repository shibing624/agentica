
## 十三、多轮工具调用核心机制 (Multi-Round Tool Invocation)

这是 OpenCode 能"边思考边回答"的核心秘密。

### 13.1 核心架构：不是 Agentic 的 enable_multi_round，而是 Vercel AI SDK 的 streamText + maxSteps

**关键结论：**
- **不是** Agentica 那样的显式 `enable_multi_round` 配置
- **不是** 纯粹的 ReAct 框架实现
- **是** 基于 Vercel AI SDK 的 `streamText` 原生 agentic 循环

**核心代码 (`session/llm.ts`):**

```typescript
import { streamText } from "ai"

return streamText({
  // ... 配置
  tools,                    // 工具集
  maxOutputTokens,          // 输出 token 限制
  abortSignal: input.abort, // 取消信号
  messages: [...],          // 消息历史
  model: wrapLanguageModel({
    model: language,
    middleware: [...]
  }),
})
```

### 13.2 Vercel AI SDK 的 Agentic Loop 原理

Vercel AI SDK 的 `streamText` 内置了多轮工具调用支持：

```
┌─────────────────────────────────────────────────────────────────┐
│                     streamText() 内部流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 发送 messages + tools 给 LLM                                │
│                     ↓                                           │
│  2. LLM 返回 text/tool_use                                      │
│                     ↓                                           │
│  3. 如果有 tool_use:                                            │
│     - SDK 自动执行 tool.execute()                               │
│     - 将 tool_result 追加到消息                                  │
│     - 重新发送给 LLM (自动循环)                                  │
│                     ↓                                           │
│  4. 直到 LLM 返回 finishReason != "tool-calls"                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**这就是为什么 Claude/OpenCode 能自动多轮调用工具的原因 —— SDK 层面已经实现了！**

### 13.3 OpenCode 的外层循环 (`session/prompt.ts`)

OpenCode 在 SDK 的自动循环之外，还有一个外层循环处理特殊情况：

```typescript
// prompt.ts - loop 函数
export const loop = fn(Identifier.schema("session"), async (sessionID) => {
  let step = 0
  while (true) {
    // 1. 获取消息历史
    let msgs = await MessageV2.filterCompacted(MessageV2.stream(sessionID))
    
    // 2. 找到最后的用户消息和助手消息
    let lastUser, lastAssistant, lastFinished
    // ...
    
    // 3. 退出条件检查
    if (lastAssistant?.finish && 
        !["tool-calls", "unknown"].includes(lastAssistant.finish) &&
        lastUser.id < lastAssistant.id) {
      break  // 正常完成，退出
    }
    
    step++
    
    // 4. 处理特殊任务 (subtask, compaction)
    if (task?.type === "subtask") { /* ... */ continue }
    if (task?.type === "compaction") { /* ... */ continue }
    
    // 5. 上下文溢出检查
    if (lastFinished && await SessionCompaction.isOverflow(...)) {
      await SessionCompaction.create(...)
      continue
    }
    
    // 6. 正常处理：调用 processor.process()
    const result = await processor.process({
      user: lastUser,
      agent,
      system: [...],
      messages: [...],
      tools,
      model,
    })
    
    // 7. 根据结果决定下一步
    if (result === "stop") break
    if (result === "compact") {
      await SessionCompaction.create(...)
    }
  }
})
```

### 13.4 双层循环架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     OpenCode 双层循环架构                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  【外层循环】SessionPrompt.loop()                                │
│  ├── 处理 Session 级别的状态管理                                 │
│  ├── 处理 Subtask (子任务调用)                                   │
│  ├── 处理 Compaction (上下文压缩)                                │
│  ├── 检查 Max Steps 限制                                         │
│  └── 调用 processor.process() ────────────────────┐             │
│                                                    │             │
│  【内层循环】processor.process() → LLM.stream()    ↓             │
│  ├── Vercel AI SDK streamText() 内置循环          │             │
│  ├── 自动处理 tool_use → execute → tool_result    │             │
│  ├── 自动重新发送给 LLM                            │             │
│  └── 直到 finishReason != "tool-calls"           │             │
│                                                    │             │
│                        ↓ 返回                      │             │
│  result = "continue" | "stop" | "compact"         │             │
│                        ↓                           │             │
│  外层循环继续/退出/压缩 ←─────────────────────────┘             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 13.5 processor.process() 核心流程 (`session/processor.ts`)

```typescript
export function create(input) {
  const toolcalls: Record<string, MessageV2.ToolPart> = {}
  
  return {
    async process(streamInput: LLM.StreamInput) {
      while (true) {
        const stream = await LLM.stream(streamInput)
        
        for await (const value of stream.fullStream) {
          switch (value.type) {
            case "tool-call":
              // 工具开始调用
              const match = toolcalls[value.toolCallId]
              // 更新工具状态为 running
              // 检测 Doom Loop (连续 3 次相同调用)
              break
              
            case "tool-result":
              // 工具返回结果
              await Session.updatePart({
                ...match,
                state: {
                  status: "completed",
                  output: value.output.output,
                  // ...
                },
              })
              break
              
            case "finish-step":
              // 一个步骤完成 (可能还有更多)
              // 计算 token 使用量
              // 检查是否需要 compaction
              break
              
            case "text-delta":
              // 文本流式输出
              currentText.text += value.text
              await Session.updatePart({ part: currentText, delta: value.text })
              break
          }
        }
        
        // 检查退出条件
        if (needsCompaction) return "compact"
        if (blocked) return "stop"
        if (input.assistantMessage.error) return "stop"
        return "continue"
      }
    }
  }
}
```

### 13.6 为什么不是传统 ReAct？

| 维度 | 传统 ReAct | OpenCode/Claude Code |
|------|-----------|---------------------|
| **循环控制** | 显式 while 循环 + 手动解析 | SDK 内置 agentic loop |
| **工具执行** | 手动匹配 action → 执行 | SDK 自动 tool_use → execute |
| **消息管理** | 手动拼接 observation | SDK 自动追加 tool_result |
| **流式输出** | 通常不支持 | 原生 streaming |
| **并行调用** | 需要额外实现 | SDK 原生支持 |

**传统 ReAct 伪代码:**
```python
while not done:
    response = llm.generate(prompt + history)
    thought, action = parse(response)
    if action:
        observation = execute_tool(action)
        history.append(f"Observation: {observation}")
    else:
        done = True
```

**Vercel AI SDK 方式:**
```typescript
const result = await streamText({
  model,
  messages,
  tools,  // SDK 自动处理一切
})
```

### 13.7 并行工具调用的实现

Vercel AI SDK 和 Anthropic API 原生支持并行工具调用：

```typescript
// LLM 单次响应可以包含多个 tool_use
{
  "content": [
    { "type": "text", "text": "Let me search for these files..." },
    { "type": "tool_use", "id": "call_1", "name": "grep", "input": {...} },
    { "type": "tool_use", "id": "call_2", "name": "read", "input": {...} },
    { "type": "tool_use", "id": "call_3", "name": "glob", "input": {...} }
  ]
}
```

SDK 会并行执行这些工具，然后将所有结果一起返回给 LLM。

### 13.8 与 Agentica enable_multi_round 的区别

| 特性 | Agentica enable_multi_round | OpenCode (Vercel AI SDK) |
|------|---------------------------|-------------------------|
| **配置方式** | 显式开关 | 默认行为 |
| **实现层级** | 框架封装 | SDK 原生 |
| **灵活性** | 框架约束 | 完全控制 |
| **流式支持** | 可能受限 | 完美支持 |
| **底层依赖** | LangChain/自研 | Vercel AI SDK |

### 13.9 复刻建议

**方案 A: 使用 Vercel AI SDK (推荐)**
```typescript
import { streamText } from "ai"

const result = await streamText({
  model: yourModel,
  messages: history,
  tools: {
    search: tool({
      description: "...",
      parameters: z.object({...}),
      execute: async (args) => {...}
    }),
    // 更多工具
  },
  maxSteps: 50,  // 可选：限制最大步数
})

for await (const chunk of result.fullStream) {
  // 处理流式输出
}
```

**方案 B: 手动实现 (更多控制)**
```typescript
async function agentLoop(messages) {
  while (true) {
    const response = await llm.generate(messages)
    
    if (response.stop_reason !== "tool_use") {
      return response.text
    }
    
    const toolResults = await Promise.all(
      response.tool_calls.map(call => executeTool(call))
    )
    
    messages.push(
      { role: "assistant", content: response.content },
      { role: "user", content: toolResults.map(r => ({
        type: "tool_result",
        tool_use_id: r.id,
        content: r.output
      }))}
    )
  }
}
```

### 13.10 关键洞察

1. **不需要显式配置多轮** —— 现代 LLM API 和 SDK 已经内置
2. **流式 + 工具调用** —— Vercel AI SDK 完美支持
3. **外层循环仍然需要** —— 处理 session 状态、压缩、权限等
4. **Prompt 是关键** —— "keep going until solved" 驱动 LLM 继续调用工具
5. **工具设计是核心** —— 详尽的描述让 LLM 知道何时/如何调用

---

## 总结

OpenCode 的高任务完成度来源于:

1. **强制迭代**: 明确要求 Agent 必须完全解决问题才能停止
2. **结构化追踪**: TodoWrite 提供可见的任务进度
3. **验证闭环**: 强制运行 lint/typecheck/tests
4. **模型适配**: 针对不同模型特性定制 prompt
5. **上下文管理**: Session Compaction 防止上下文爆炸
6. **工具指导**: 详尽的工具使用说明和优先级
7. **多级指令**: AGENTS.md 支持项目/目录/全局继承

复刻时务必实现以上所有机制，缺一不可。
