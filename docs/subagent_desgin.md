
## 十五、Subagent 系统实现详解

### 15.1 Agent 类型定义 (`agent/agent.ts`)

```typescript
const result: Record<string, Info> = {
  // Primary Agents (主 Agent)
  build: {
    name: "build",
    description: "The default agent. Executes tools based on configured permissions.",
    mode: "primary",
    permission: PermissionNext.merge(
      defaults,
      PermissionNext.fromConfig({ question: "allow", plan_enter: "allow" }),
      user,
    ),
  },
  plan: {
    name: "plan",
    description: "Plan mode. Disallows all edit tools.",
    mode: "primary",
    permission: PermissionNext.merge(defaults, PermissionNext.fromConfig({
      edit: { "*": "deny", ... },  // 禁止编辑
    }), user),
  },
  
  // Subagents (子 Agent)
  general: {
    name: "general",
    description: `General-purpose agent for researching complex questions and 
                  executing multi-step tasks. Use this agent to execute multiple 
                  units of work in parallel.`,
    mode: "subagent",  // 关键：标记为 subagent
    permission: PermissionNext.merge(defaults, PermissionNext.fromConfig({
      todoread: "deny",
      todowrite: "deny",
    }), user),
  },
  explore: {
    name: "explore",
    description: `Fast agent specialized for exploring codebases...`,
    prompt: PROMPT_EXPLORE,  // 专用 prompt
    mode: "subagent",
    permission: PermissionNext.merge(defaults, PermissionNext.fromConfig({
      "*": "deny",           // 默认禁止所有
      grep: "allow",         // 只允许搜索相关工具
      glob: "allow",
      list: "allow",
      bash: "allow",
      read: "allow",
      webfetch: "allow",
      websearch: "allow",
      codesearch: "allow",
    }), user),
  },
  
  // Hidden Specialized Agents
  compaction: { mode: "primary", hidden: true, prompt: PROMPT_COMPACTION, ... },
  title: { mode: "primary", hidden: true, prompt: PROMPT_TITLE, ... },
  summary: { mode: "primary", hidden: true, prompt: PROMPT_SUMMARY, ... },
}
```

### 15.2 Explore Agent Prompt (`agent/prompt/explore.txt`)

```markdown
You are a file search specialist. You excel at thoroughly navigating and exploring codebases.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path you need to read
- Use Bash for file operations like copying, moving, or listing directory contents
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Do not create any files, or run bash commands that modify the user's system state

Complete the user's search request efficiently and report your findings clearly.
```

### 15.3 Task Tool 实现 (`tool/task.ts`) - 启动 Subagent 的核心

```typescript
const parameters = z.object({
  description: z.string().describe("A short (3-5 words) description of the task"),
  prompt: z.string().describe("The task for the agent to perform"),
  subagent_type: z.string().describe("The type of specialized agent to use"),
  session_id: z.string().describe("Existing Task session to continue").optional(),
  command: z.string().describe("The command that triggered this task").optional(),
})

export const TaskTool = Tool.define("task", async (ctx) => {
  // 1. 获取所有非 primary 的 agent 作为可用 subagent
  const agents = await Agent.list().then((x) => x.filter((a) => a.mode !== "primary"))
  
  return {
    description,
    parameters,
    async execute(params, ctx) {
      // 2. 获取指定的 agent 配置
      const agent = await Agent.get(params.subagent_type)
      if (!agent) throw new Error(`Unknown agent type: ${params.subagent_type}`)
      
      // 3. 创建子 Session（关键！）
      const session = await Session.create({
        parentID: ctx.sessionID,  // 关联父 session
        title: params.description + ` (@${agent.name} subagent)`,
        permission: [
          // 禁用 todo 工具（subagent 不需要）
          { permission: "todowrite", pattern: "*", action: "deny" },
          { permission: "todoread", pattern: "*", action: "deny" },
          // 如果 agent 没有 task 权限，也禁用嵌套 task
          ...(hasTaskPermission ? [] : [{ permission: "task", pattern: "*", action: "deny" }]),
        ],
      })
      
      // 4. 订阅子 session 的工具调用事件（用于 UI 显示进度）
      const unsub = Bus.subscribe(MessageV2.Event.PartUpdated, async (evt) => {
        if (evt.properties.part.sessionID !== session.id) return
        if (evt.properties.part.type !== "tool") return
        // 更新父 session 中的元数据显示子任务进度
        ctx.metadata({
          title: params.description,
          metadata: { summary: Object.values(parts), sessionId: session.id },
        })
      })
      
      // 5. 调用 SessionPrompt.prompt 执行 subagent（核心！）
      const result = await SessionPrompt.prompt({
        messageID,
        sessionID: session.id,
        model: { modelID, providerID },
        agent: agent.name,  // 使用指定的 agent 类型
        tools: {
          todowrite: false,
          todoread: false,
          ...(hasTaskPermission ? {} : { task: false }),
        },
        parts: promptParts,
      })
      
      // 6. 返回 subagent 的执行结果
      const text = result.parts.findLast((x) => x.type === "text")?.text ?? ""
      return {
        title: params.description,
        metadata: { summary, sessionId: session.id, model },
        output: text + "\n\n<task_metadata>session_id: " + session.id + "</task_metadata>",
      }
    },
  }
})
```

### 15.4 Subagent 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Subagent 执行架构                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Primary Session (build/plan)                  │   │
│   │                                                                  │   │
│   │   User Message → AI Response → Task Tool Call                   │   │
│   │                                      │                          │   │
│   │                                      ↓                          │   │
│   │   ┌──────────────────────────────────────────────────────────┐  │   │
│   │   │                    TaskTool.execute()                     │  │   │
│   │   │                                                           │  │   │
│   │   │   1. Session.create({ parentID: ctx.sessionID })         │  │   │
│   │   │   2. Agent.get(params.subagent_type)  // explore/general │  │   │
│   │   │   3. SessionPrompt.prompt({ agent: agent.name, ... })    │  │   │
│   │   │                          │                                │  │   │
│   │   │                          ↓                                │  │   │
│   │   │   ┌────────────────────────────────────────────────────┐ │  │   │
│   │   │   │              Child Session (subagent)               │ │  │   │
│   │   │   │                                                     │ │  │   │
│   │   │   │   - 独立的 session 和消息历史                       │ │  │   │
│   │   │   │   - 使用 subagent 专属 prompt (explore.txt)         │ │  │   │
│   │   │   │   - 受限的工具权限 (只读工具)                       │ │  │   │
│   │   │   │   - 调用 LLM.stream() 执行                          │ │  │   │
│   │   │   └────────────────────────────────────────────────────┘ │  │   │
│   │   │                          │                                │  │   │
│   │   │                          ↓ 返回结果                       │  │   │
│   │   │   4. 返回 { output: text, metadata: summary }            │  │   │
│   │   └──────────────────────────────────────────────────────────┘  │   │
│   │                                      │                          │   │
│   │                                      ↓                          │   │
│   │   AI 继续处理（使用 subagent 返回的结果）                       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 十六、并行 Explore Agents 实现

### 16.1 核心机制：Prompt 指导 + SDK 原生支持

**不是代码层面实现并行**，而是通过 Prompt 指导 LLM 在单条消息中发起多个 Tool Call：

**plan-reminder-anthropic.txt (第 22-31 行):**
```markdown
### Phase 1: Initial Understanding

**Goal:** Gain a comprehensive understanding of the user's request by reading 
through code and asking them questions. Critical: In this phase you should only 
use the Explore subagent type.

2. **Launch up to 3 Explore agents IN PARALLEL** (single message, multiple tool calls) 
   to efficiently explore the codebase. Each agent can focus on different aspects:
   - Example: One agent searches for existing implementations, another explores 
     related components, a third investigates testing patterns
   - Provide each agent with a specific search focus or area to explore
   - Quality over quantity - 3 agents maximum, but you should try to use the 
     minimum number of agents necessary (usually just 1)
```

**anthropic.txt (第 83-84 行) 通用并行指导:**
```markdown
- You can call multiple tools in a single response. If you intend to call multiple 
  tools and there are no dependencies between them, make all independent tool calls 
  in parallel...
- If the user specifies that they want you to run tools "in parallel", you MUST send 
  a single message with multiple tool use content blocks. For example, if you need 
  to launch multiple agents in parallel, send a single message with multiple Task 
  tool calls.
```

### 16.2 并行执行原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     并行 Subagent 执行流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. LLM 单次响应包含多个 tool_use:                                      │
│                                                                         │
│     {                                                                   │
│       "content": [                                                      │
│         { "type": "text", "text": "Let me explore the codebase..." },   │
│         { "type": "tool_use", "id": "1", "name": "task", "input": {     │
│             "subagent_type": "explore",                                 │
│             "description": "Find authentication implementations",       │
│             "prompt": "Search for auth-related code..."                 │
│         }},                                                             │
│         { "type": "tool_use", "id": "2", "name": "task", "input": {     │
│             "subagent_type": "explore",                                 │
│             "description": "Find testing patterns",                     │
│             "prompt": "Search for test files and patterns..."          │
│         }},                                                             │
│         { "type": "tool_use", "id": "3", "name": "task", "input": {     │
│             "subagent_type": "explore",                                 │
│             "description": "Find component structure",                  │
│             "prompt": "Explore the component directory..."              │
│         }}                                                              │
│       ]                                                                 │
│     }                                                                   │
│                                                                         │
│  2. Vercel AI SDK 自动并行执行这 3 个 TaskTool.execute():               │
│                                                                         │
│     Promise.all([                                                       │
│       TaskTool.execute(task1),  // → 创建 Child Session 1               │
│       TaskTool.execute(task2),  // → 创建 Child Session 2               │
│       TaskTool.execute(task3),  // → 创建 Child Session 3               │
│     ])                                                                  │
│                                                                         │
│  3. 每个 Child Session 独立执行：                                       │
│                                                                         │
│     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │
│     │ Explore Agent │  │ Explore Agent │  │ Explore Agent │            │
│     │   Session 1   │  │   Session 2   │  │   Session 3   │            │
│     │               │  │               │  │               │            │
│     │ grep/glob/    │  │ grep/glob/    │  │ grep/glob/    │            │
│     │ read tools    │  │ read tools    │  │ read tools    │            │
│     └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │
│             │                  │                  │                     │
│             └──────────────────┼──────────────────┘                     │
│                                ↓                                        │
│  4. 所有结果返回给 Primary Session 的 LLM                               │
│                                                                         │
│     [tool_result_1, tool_result_2, tool_result_3]                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 16.3 Session Loop 中的 Subtask 处理 (`session/prompt.ts`)

```typescript
export const loop = fn(Identifier.schema("session"), async (sessionID) => {
  while (true) {
    // ... 获取消息历史 ...
    
    // 收集待处理的 subtask
    const task = tasks.pop()
    
    // 处理 subtask 类型的 part
    if (task?.type === "subtask") {
      const taskTool = await TaskTool.init()
      const taskModel = task.model ? await Provider.getModel(...) : model
      
      // 创建 assistant 消息记录
      const assistantMessage = await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "assistant",
        mode: task.agent,
        agent: task.agent,
        // ...
      })
      
      // 创建工具调用 part
      let part = await Session.updatePart({
        type: "tool",
        tool: TaskTool.id,
        state: {
          status: "running",
          input: {
            prompt: task.prompt,
            description: task.description,
            subagent_type: task.agent,
          },
        },
      })
      
      // 执行 task tool
      const result = await taskTool.execute({
        prompt: task.prompt,
        description: task.description,
        subagent_type: task.agent,
      }, taskCtx)
      
      // 更新 part 状态为完成
      await Session.updatePart({
        ...part,
        state: {
          status: "completed",
          output: result.output,
          // ...
        },
      })
      
      continue  // 继续循环处理下一个任务
    }
    
    // ... 正常处理 ...
  }
})
```

### 16.4 关键洞察

| 维度 | 实现方式 |
|------|----------|
| **并行触发** | Prompt 指导 LLM 在单条消息中发起多个 tool_use |
| **并行执行** | Vercel AI SDK 自动并行执行多个 tool.execute() |
| **Session 隔离** | 每个 subagent 创建独立的 Child Session |
| **权限控制** | Explore agent 只有只读工具权限 |
| **结果聚合** | 所有 tool_result 一起返回给 LLM |

### 16.5 复刻并行 Subagent 的建议

```typescript
// 方案 A: 使用 Vercel AI SDK (自动并行)
const result = await streamText({
  model,
  messages,
  tools: {
    task: tool({
      description: "Launch a subagent for specialized tasks",
      parameters: z.object({
        subagent_type: z.enum(["explore", "general"]),
        prompt: z.string(),
        description: z.string(),
      }),
      execute: async (args) => {
        // 创建子 session，执行 subagent
        const childSession = await createChildSession(args.subagent_type)
        return await runSubagent(childSession, args.prompt)
      },
    }),
  },
})

// 关键: 通过 Prompt 指导 LLM 并行调用
const systemPrompt = `
When exploring a codebase, you can launch up to 3 explore agents IN PARALLEL 
(single message, multiple tool calls). Each agent can focus on different aspects.
`
```

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


====



## 十、Subagent 子代理系统

> OpenClaw 的子代理系统允许主 Agent 派生后台任务，实现并行处理和上下文隔离

### 10.1 核心架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Main Agent                                    │
│  Session Key: agent:<agentId>:main                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    sessions_spawn 工具                           │ │
│  │  1. 检查权限 (禁止嵌套 spawn)                                    │ │
│  │  2. 生成 childSessionKey: agent:<agentId>:subagent:<uuid>       │ │
│  │  3. 构建 subagent system prompt (精简版)                        │ │
│  │  4. 调用 Gateway: agent(lane=subagent)                          │ │
│  │  5. 注册到 SubagentRegistry                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ spawnedBy (记录父会话)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Command Queue System (并发控制)                  │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐         │
│  │   Main Lane    │ │   Cron Lane    │ │   Subagent Lane  │         │
│  │  (maxConc: 4)  │ │  (maxConc: 1)  │ │   (maxConc: 8)   │         │
│  └────────────────┘ └────────────────┘ └──────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Subagent (独立运行)                           │
│  Session Key: agent:<agentId>:subagent:<uuid>                        │
│  - 独立的 sessionId 和 transcript 文件                               │
│  - 精简的 bootstrap 文件 (只有 AGENTS.md, TOOLS.md)                  │
│  - 受限的工具策略 (无 cron, 无 message 等)                           │
│  - 专注单一任务的 system prompt                                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                │ lifecycle: end/error
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Subagent Announce Flow (结果汇报)                  │
│  1. SubagentRegistry 监听生命周期事件                                 │
│  2. 读取子代理最终输出 (readLatestAssistantReply)                    │
│  3. 构建结果通知消息 + 统计信息                                       │
│  4. 发送到主代理会话 (steer/queue/direct)                            │
│  5. 可选: 清理子代理会话 (sessions.delete)                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 Session Key 格式与隔离

**Session Key 命名规范**：
```
主会话:       agent:<agentId>:main
子代理会话:   agent:<agentId>:subagent:<uuid>
Cron 会话:    agent:<agentId>:cron:<jobId>
```

**判断是否为子代理会话**：
```typescript
// src/sessions/session-key-utils.ts
export function isSubagentSessionKey(sessionKey: string | undefined | null): boolean {
  const raw = (sessionKey ?? "").trim();
  
  // 快速路径：直接以 "subagent:" 开头
  if (raw.toLowerCase().startsWith("subagent:")) {
    return true;
  }
  
  // 解析 agent:xxx:subagent:yyy 格式
  const parsed = parseAgentSessionKey(raw);
  return Boolean((parsed?.rest ?? "").toLowerCase().startsWith("subagent:"));
}

// 解析结果示例
// parseAgentSessionKey("agent:main:subagent:abc-123")
// → { agentId: "main", rest: "subagent:abc-123" }
```

### 10.3 sessions_spawn 工具实现

```typescript
// src/agents/tools/sessions-spawn-tool.ts

export function createSessionsSpawnTool(opts: SessionsSpawnToolOpts) {
  return {
    name: "sessions_spawn",
    description: `Spawn a background sub-agent run in an isolated session.
Use this to offload research, analysis, or other tasks that can run in parallel.
The sub-agent runs independently and reports results back when done.`,
    
    parameters: z.object({
      task: z.string().describe("Clear task description for the sub-agent"),
      label: z.string().optional().describe("Short label for tracking"),
      model: z.string().optional().describe("Model override (default: same as parent)"),
      tools: z.array(z.string()).optional().describe("Tool allowlist for sub-agent"),
    }),
    
    execute: async (_toolCallId, args) => {
      const { task, label, model, tools } = args;
      const requesterSessionKey = opts.sessionKey;
      const requesterAgentId = parseAgentSessionKey(requesterSessionKey)?.agentId;
      
      // ===== 权限检查 =====
      
      // 1. 禁止嵌套 spawn：子代理不能再创建子代理
      if (isSubagentSessionKey(requesterSessionKey)) {
        return jsonResult({
          status: "forbidden",
          error: "sessions_spawn is not allowed from sub-agent sessions",
        });
      }
      
      // 2. 跨 Agent spawn 需要配置允许
      const cfg = await loadConfig();
      const allowAgents = resolveAgentConfig(cfg, requesterAgentId)
        ?.subagents?.allowAgents ?? [];
      
      // ===== 创建子代理会话 =====
      
      // 3. 生成独立的 childSessionKey
      const childSessionKey = `agent:${requesterAgentId}:subagent:${crypto.randomUUID()}`;
      
      // 4. 构建子代理专用的 system prompt
      const childSystemPrompt = buildSubagentSystemPrompt({
        requesterSessionKey,
        childSessionKey,
        taskText: task,
        parentContext: opts.parentContext,  // 可选：传递部分上下文
      });
      
      // 5. 通过 Gateway 启动子代理
      const response = await callGateway({
        method: "agent",
        params: {
          message: task,
          sessionKey: childSessionKey,
          lane: AGENT_LANE_SUBAGENT,       // 使用独立的 "subagent" 通道
          extraSystemPrompt: childSystemPrompt,
          spawnedBy: requesterSessionKey,   // 记录父会话（关键！）
          model: model ?? opts.defaultModel,
          tools: tools,                     // 工具白名单
          deliver: false,                   // 不直接发送给用户
        },
      });
      
      // 6. 注册到 subagent 注册表
      registerSubagentRun({
        runId: response.runId,
        childSessionKey,
        requesterSessionKey,
        taskLabel: label ?? task.slice(0, 50),
        startedAt: Date.now(),
      });
      
      return jsonResult({
        status: "spawned",
        runId: response.runId,
        sessionKey: childSessionKey,
        message: `Sub-agent started: "${label ?? task.slice(0, 30)}..."`,
      });
    },
  };
}
```

### 10.4 上下文隔离机制

#### 1. Bootstrap 文件过滤

```typescript
// src/agents/workspace.ts

// 子代理只能访问的 bootstrap 文件白名单
const SUBAGENT_BOOTSTRAP_ALLOWLIST = new Set([
  "AGENTS.md",   // 基础行为规范
  "TOOLS.md",    // 工具配置
  // 不包含: SOUL.md, USER.md, MEMORY.md, IDENTITY.md, HEARTBEAT.md
]);

export function filterBootstrapFilesForSession(
  files: BootstrapFile[],
  sessionKey: string | undefined
): BootstrapFile[] {
  // 主代理获取全部文件
  if (!sessionKey || !isSubagentSessionKey(sessionKey)) {
    return files;
  }
  
  // 子代理只获取精简的文件列表
  return files.filter((file) => {
    const baseName = file.name.split("/").pop() ?? file.name;
    return SUBAGENT_BOOTSTRAP_ALLOWLIST.has(baseName);
  });
}
```

**为什么要过滤**：
- **SOUL.md**：人格定义，子代理不需要"人格"，只需要完成任务
- **USER.md**：用户信息，子代理不直接与用户交互
- **MEMORY.md**：长期记忆，子代理是短期任务，不需要历史
- **HEARTBEAT.md**：心跳任务，子代理不应该自己设置定时任务

#### 2. System Prompt 精简

```typescript
// src/agents/system-prompt.ts

// 判断是否为子代理
const isSubagent = isSubagentSessionKey(params.sessionKey);

// 子代理使用 minimal 模式
const promptMode = isSubagent ? "minimal" : (params.promptMode ?? "full");

// minimal 模式跳过的 sections:
// - Skills section (技能系统)
// - Memory section (记忆搜索)
// - User Identity section (用户信息)
// - Reply Tags section (回复标签)
// - Messaging section (消息发送)
// - Silent Replies section (静默规则)
// - Heartbeats section (心跳规则)
// - Self-Update section (自更新)
```

#### 3. 工具策略隔离

```typescript
// src/gateway/tools-invoke-http.ts

// 子代理使用受限的工具策略
const subagentPolicy = isSubagentSessionKey(sessionKey)
  ? resolveSubagentToolPolicy(cfg)
  : undefined;

// 子代理默认禁用的工具:
// - cron: 不能创建定时任务
// - message: 不能直接发消息给用户
// - sessions_spawn: 不能嵌套创建子代理
// - gateway 相关: 不能访问系统级功能
```

#### 4. 子代理专用 System Prompt

```typescript
// src/agents/subagent-announce.ts

export function buildSubagentSystemPrompt(params: {
  requesterSessionKey: string;
  childSessionKey: string;
  taskText: string;
  parentContext?: string;
}) {
  return [
    "# Subagent Context",
    "",
    "You are a **subagent** spawned by the main agent for a specific task.",
    "",
    "## Your Role",
    `- You were created to handle: ${params.taskText}`,
    "- Your output will be reported back to the main agent when done",
    "",
    "## Rules",
    "1. **Stay focused** - Do your assigned task, nothing else",
    "2. **Be thorough** - Your final message is your deliverable",
    "3. **Complete the task** - Don't ask for clarification, make reasonable assumptions",
    "",
    "## What You DON'T Do",
    "- NO user conversations (that's the main agent's job)",
    "- NO external messages unless explicitly tasked",
    "- NO cron jobs or persistent state (you're ephemeral)",
    "- NO spawning other sub-agents",
    "",
    "## Session Info",
    `- Parent session: ${params.requesterSessionKey}`,
    `- Your session: ${params.childSessionKey}`,
    params.parentContext ? `\n## Context from Parent\n${params.parentContext}` : "",
  ].filter(Boolean).join("\n");
}
```

### 10.5 并发控制：Lane 系统

```typescript
// src/process/lanes.ts
export const enum CommandLane {
  Main = "main",          // 主代理通道
  Cron = "cron",          // 定时任务通道
  Subagent = "subagent",  // 子代理通道
  Nested = "nested",      // 嵌套调用通道
}

// src/config/agent-limits.ts
export const DEFAULT_AGENT_MAX_CONCURRENT = 4;      // 主代理默认并发
export const DEFAULT_SUBAGENT_MAX_CONCURRENT = 8;   // 子代理默认并发
```

**为什么子代理并发更高**：
- 子代理任务通常更轻量、更短
- 子代理之间相互独立，不会冲突
- 允许主代理同时派发多个研究任务

```typescript
// src/process/command-queue.ts

type LaneState = {
  lane: string;
  queue: QueueEntry[];
  active: number;
  maxConcurrent: number;
};

// 每个 lane 独立的队列和并发控制
const lanes: Map<string, LaneState> = new Map();

export function enqueue(params: EnqueueParams) {
  const lane = params.lane ?? CommandLane.Main;
  const state = getOrCreateLaneState(lane);
  
  state.queue.push({
    id: crypto.randomUUID(),
    params,
    priority: params.priority ?? 0,
  });
  
  // 尝试执行（如果有空闲槽位）
  void processLane(lane);
}

async function processLane(lane: string) {
  const state = lanes.get(lane);
  if (!state) return;
  
  // 检查并发限制
  while (state.active < state.maxConcurrent && state.queue.length > 0) {
    const entry = state.queue.shift()!;
    state.active++;
    
    try {
      await executeEntry(entry);
    } finally {
      state.active--;
      void processLane(lane);  // 继续处理队列
    }
  }
}
```

### 10.6 Subagent Registry：注册与追踪

```typescript
// src/agents/subagent-registry.ts

interface SubagentRunEntry {
  runId: string;
  childSessionKey: string;
  requesterSessionKey: string;
  taskLabel: string;
  startedAt: number;
  status: "running" | "completed" | "error" | "aborted";
  endedAt?: number;
  result?: string;
}

// 内存中的运行注册表
const subagentRuns: Map<string, SubagentRunEntry> = new Map();

// 注册新的子代理运行
export function registerSubagentRun(entry: Omit<SubagentRunEntry, "status">) {
  subagentRuns.set(entry.runId, {
    ...entry,
    status: "running",
  });
  
  // 持久化到文件（用于重启恢复）
  void persistSubagentRegistry();
  
  // 确保监听器已启动
  ensureListener();
}

// 监听子代理生命周期事件
let listenerStop: (() => void) | null = null;

function ensureListener() {
  if (listenerStop) return;
  
  listenerStop = onAgentEvent((evt) => {
    if (evt.stream !== "lifecycle") return;
    
    const entry = subagentRuns.get(evt.runId);
    if (!entry) return;
    
    // 子代理完成或出错
    if (evt.data?.phase === "end" || evt.data?.phase === "error") {
      entry.status = evt.data.phase === "end" ? "completed" : "error";
      entry.endedAt = Date.now();
      
      // 触发结果公告流程
      void runSubagentAnnounceFlow({
        childRunId: entry.runId,
        childSessionKey: entry.childSessionKey,
        requesterSessionKey: entry.requesterSessionKey,
        taskLabel: entry.taskLabel,
        cleanup: "delete",  // 完成后清理会话
      });
    }
  });
}

// 查询当前会话的子代理
export function getSubagentsForSession(sessionKey: string): SubagentRunEntry[] {
  return Array.from(subagentRuns.values())
    .filter((e) => e.requesterSessionKey === sessionKey);
}
```

### 10.7 结果汇报流程

```typescript
// src/agents/subagent-announce.ts

export async function runSubagentAnnounceFlow(params: {
  childRunId: string;
  childSessionKey: string;
  requesterSessionKey: string;
  taskLabel: string;
  cleanup?: "delete" | "keep";
}) {
  // 1. 等待子代理完全结束
  const waitResult = await callGateway({
    method: "agent.wait",
    params: { runId: params.childRunId },
  });
  
  // 2. 读取子代理的最终回复
  const latestReply = await readLatestAssistantReply({
    sessionKey: params.childSessionKey,
  });
  
  // 3. 构建统计信息行
  const statsLine = await buildSubagentStatsLine({
    runId: params.childRunId,
    duration: waitResult.duration,
    tokenUsage: waitResult.usage,
  });
  // 例如: "[Stats: 45s, 2.3k tokens, 3 tool calls]"
  
  // 4. 构建触发消息
  const statusLabel = waitResult.status === "ok" ? "completed" : "failed";
  const triggerMessage = [
    `📋 Background task "${params.taskLabel}" just ${statusLabel}.`,
    "",
    "**Findings:**",
    latestReply || "(no output)",
    "",
    statsLine,
    "",
    "---",
    "Summarize this naturally for the user. If there are actionable items, highlight them.",
  ].join("\n");
  
  // 5. 尝试队列或直接发送
  const queued = await maybeQueueSubagentAnnounce({
    requesterSessionKey: params.requesterSessionKey,
    triggerMessage,
    taskLabel: params.taskLabel,
  });
  
  // 6. 如果未进入队列，直接发送
  if (queued === "none") {
    await callGateway({
      method: "agent",
      params: {
        sessionKey: params.requesterSessionKey,
        message: triggerMessage,
        deliver: true,       // 将结果发送给用户
        isSystemEvent: true, // 标记为系统事件
      },
    });
  }
  
  // 7. 可选：清理子代理会话
  if (params.cleanup === "delete") {
    await callGateway({
      method: "sessions.delete",
      params: { key: params.childSessionKey },
    });
  }
}
```

### 10.8 公告队列机制

```typescript
// src/agents/subagent-announce-queue.ts

// 为什么需要队列？
// 1. 主代理可能正在处理用户消息，不能被打断
// 2. 多个子代理可能同时完成，需要有序处理
// 3. 避免消息冲突和上下文混乱

interface QueuedAnnouncement {
  id: string;
  requesterSessionKey: string;
  triggerMessage: string;
  taskLabel: string;
  queuedAt: number;
}

const announceQueues: Map<string, QueuedAnnouncement[]> = new Map();

export async function maybeQueueSubagentAnnounce(params: {
  requesterSessionKey: string;
  triggerMessage: string;
  taskLabel: string;
}): Promise<"queued" | "none"> {
  // 检查主代理是否正忙
  const isMainAgentBusy = await checkSessionBusy(params.requesterSessionKey);
  
  if (!isMainAgentBusy) {
    return "none";  // 可以直接发送
  }
  
  // 主代理正忙，加入队列
  const queue = announceQueues.get(params.requesterSessionKey) ?? [];
  queue.push({
    id: crypto.randomUUID(),
    requesterSessionKey: params.requesterSessionKey,
    triggerMessage: params.triggerMessage,
    taskLabel: params.taskLabel,
    queuedAt: Date.now(),
  });
  announceQueues.set(params.requesterSessionKey, queue);
  
  return "queued";
}

// 主代理空闲时处理队列
export async function drainAnnounceQueue(sessionKey: string) {
  const queue = announceQueues.get(sessionKey);
  if (!queue || queue.length === 0) return;
  
  // 批量处理：将多个公告合并成一条消息
  const combined = queue.map((a) => 
    `### ${a.taskLabel}\n${a.triggerMessage}`
  ).join("\n\n---\n\n");
  
  // 清空队列
  announceQueues.delete(sessionKey);
  
  // 发送合并后的公告
  await callGateway({
    method: "agent",
    params: {
      sessionKey,
      message: `Multiple background tasks completed:\n\n${combined}`,
      deliver: true,
    },
  });
}
```

### 10.9 用户命令支持

```typescript
// src/auto-reply/reply/commands-subagents.ts

// 用户可以通过 /subagents 命令管理子代理

const ACTIONS = new Set(["list", "stop", "log", "send", "info", "help"]);

export async function handleSubagentsCommand(
  ctx: CommandContext,
  args: string[]
) {
  const [action, ...rest] = args;
  
  switch (action) {
    case "list":
      // /subagents list - 列出当前会话的所有子代理
      const subagents = getSubagentsForSession(ctx.sessionKey);
      if (subagents.length === 0) {
        return "No active sub-agents.";
      }
      return subagents.map((s) => 
        `- [${s.status}] ${s.taskLabel} (${s.runId.slice(0, 8)}...)`
      ).join("\n");
    
    case "stop":
      // /subagents stop <runId> - 停止指定子代理
      const runId = rest[0];
      await abortSubagent(runId);
      return `Sub-agent ${runId} stopped.`;
    
    case "log":
      // /subagents log <runId> - 查看子代理对话日志
      const logRunId = rest[0];
      const transcript = await readSubagentTranscript(logRunId);
      return `\`\`\`\n${transcript}\n\`\`\``;
    
    case "send":
      // /subagents send <runId> <message> - 向子代理发送消息
      const [sendRunId, ...msgParts] = rest;
      const message = msgParts.join(" ");
      await sendToSubagent(sendRunId, message);
      return `Message sent to sub-agent ${sendRunId}.`;
    
    case "info":
      // /subagents info <runId> - 查看子代理详细信息
      const infoRunId = rest[0];
      const entry = subagentRuns.get(infoRunId);
      if (!entry) return "Sub-agent not found.";
      return [
        `**Task:** ${entry.taskLabel}`,
        `**Status:** ${entry.status}`,
        `**Started:** ${new Date(entry.startedAt).toISOString()}`,
        `**Session:** ${entry.childSessionKey}`,
      ].join("\n");
    
    default:
      return [
        "Usage: /subagents <action> [args]",
        "",
        "Actions:",
        "  list          - List active sub-agents",
        "  stop <id>     - Stop a sub-agent",
        "  log <id>      - View sub-agent transcript",
        "  send <id> <msg> - Send message to sub-agent",
        "  info <id>     - View sub-agent details",
      ].join("\n");
  }
}
```

### 10.10 Python 复现方案

```python
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any
from enum import Enum

# ============== 类型定义 ==============

class CommandLane(Enum):
    MAIN = "main"
    CRON = "cron"
    SUBAGENT = "subagent"

@dataclass
class SubagentRunEntry:
    run_id: str
    child_session_key: str
    requester_session_key: str
    task_label: str
    started_at: float
    status: Literal["running", "completed", "error", "aborted"] = "running"
    ended_at: Optional[float] = None
    result: Optional[str] = None

# ============== Session Key 工具 ==============

def is_subagent_session_key(session_key: Optional[str]) -> bool:
    """判断是否为子代理会话"""
    if not session_key:
        return False
    key = session_key.strip().lower()
    if key.startswith("subagent:"):
        return True
    # 解析 agent:xxx:subagent:yyy 格式
    parts = key.split(":")
    if len(parts) >= 3 and parts[2] == "subagent":
        return True
    return False

def generate_subagent_session_key(parent_agent_id: str) -> str:
    """生成子代理会话 Key"""
    return f"agent:{parent_agent_id}:subagent:{uuid.uuid4()}"

# ============== 并发控制：Lane 系统 ==============

class LaneManager:
    """命令通道管理器"""
    
    def __init__(self):
        self.lanes: dict[str, dict] = {
            CommandLane.MAIN.value: {"queue": [], "active": 0, "max_concurrent": 4},
            CommandLane.CRON.value: {"queue": [], "active": 0, "max_concurrent": 1},
            CommandLane.SUBAGENT.value: {"queue": [], "active": 0, "max_concurrent": 8},
        }
        self._locks: dict[str, asyncio.Lock] = {
            lane: asyncio.Lock() for lane in self.lanes
        }
    
    async def enqueue(
        self,
        lane: CommandLane,
        task: Callable,
        *args,
        **kwargs
    ) -> Any:
        """将任务加入指定通道队列"""
        lane_state = self.lanes[lane.value]
        lock = self._locks[lane.value]
        
        async with lock:
            # 检查是否有空闲槽位
            if lane_state["active"] < lane_state["max_concurrent"]:
                lane_state["active"] += 1
                try:
                    return await task(*args, **kwargs)
                finally:
                    lane_state["active"] -= 1
            else:
                # 加入队列等待
                future = asyncio.Future()
                lane_state["queue"].append((task, args, kwargs, future))
                return await future
    
    async def _process_queue(self, lane: CommandLane):
        """处理队列中的任务"""
        lane_state = self.lanes[lane.value]
        while lane_state["queue"] and lane_state["active"] < lane_state["max_concurrent"]:
            task, args, kwargs, future = lane_state["queue"].pop(0)
            lane_state["active"] += 1
            try:
                result = await task(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                lane_state["active"] -= 1

# ============== Subagent Registry ==============

class SubagentRegistry:
    """子代理注册表"""
    
    def __init__(self):
        self.runs: dict[str, SubagentRunEntry] = {}
        self._listeners: list[Callable] = []
    
    def register(self, entry: SubagentRunEntry):
        """注册新的子代理运行"""
        self.runs[entry.run_id] = entry
    
    def get_for_session(self, session_key: str) -> list[SubagentRunEntry]:
        """获取指定会话的所有子代理"""
        return [
            e for e in self.runs.values()
            if e.requester_session_key == session_key
        ]
    
    def update_status(
        self,
        run_id: str,
        status: Literal["completed", "error", "aborted"],
        result: Optional[str] = None
    ):
        """更新子代理状态"""
        if run_id in self.runs:
            entry = self.runs[run_id]
            entry.status = status
            entry.ended_at = asyncio.get_event_loop().time()
            entry.result = result
            # 通知监听器
            for listener in self._listeners:
                listener(entry)
    
    def on_complete(self, callback: Callable[[SubagentRunEntry], None]):
        """注册完成回调"""
        self._listeners.append(callback)

# ============== Bootstrap 文件过滤 ==============

SUBAGENT_BOOTSTRAP_ALLOWLIST = {"AGENTS.md", "TOOLS.md"}

def filter_bootstrap_files_for_session(
    files: list[dict],
    session_key: Optional[str]
) -> list[dict]:
    """根据会话类型过滤 bootstrap 文件"""
    if not session_key or not is_subagent_session_key(session_key):
        return files  # 主代理获取全部
    
    # 子代理只获取白名单中的文件
    return [
        f for f in files
        if f.get("name", "").split("/")[-1] in SUBAGENT_BOOTSTRAP_ALLOWLIST
    ]

# ============== Subagent System Prompt ==============

def build_subagent_system_prompt(
    requester_session_key: str,
    child_session_key: str,
    task_text: str,
    parent_context: Optional[str] = None
) -> str:
    """构建子代理专用的 system prompt"""
    lines = [
        "# Subagent Context",
        "",
        "You are a **subagent** spawned by the main agent for a specific task.",
        "",
        "## Your Role",
        f"- You were created to handle: {task_text}",
        "- Your output will be reported back to the main agent when done",
        "",
        "## Rules",
        "1. **Stay focused** - Do your assigned task, nothing else",
        "2. **Be thorough** - Your final message is your deliverable",
        "3. **Complete the task** - Don't ask for clarification, make reasonable assumptions",
        "",
        "## What You DON'T Do",
        "- NO user conversations (that's the main agent's job)",
        "- NO external messages unless explicitly tasked",
        "- NO cron jobs or persistent state (you're ephemeral)",
        "- NO spawning other sub-agents",
        "",
        "## Session Info",
        f"- Parent session: {requester_session_key}",
        f"- Your session: {child_session_key}",
    ]
    
    if parent_context:
        lines.extend(["", "## Context from Parent", parent_context])
    
    return "\n".join(lines)

# ============== Sessions Spawn Tool ==============

class SessionsSpawnTool:
    """sessions_spawn 工具实现"""
    
    def __init__(
        self,
        session_key: str,
        registry: SubagentRegistry,
        lane_manager: LaneManager,
        run_agent: Callable,  # 运行 agent 的函数
    ):
        self.session_key = session_key
        self.registry = registry
        self.lane_manager = lane_manager
        self.run_agent = run_agent
    
    @property
    def name(self) -> str:
        return "sessions_spawn"
    
    @property
    def description(self) -> str:
        return """Spawn a background sub-agent run in an isolated session.
Use this to offload research, analysis, or other tasks that can run in parallel.
The sub-agent runs independently and reports results back when done."""
    
    async def execute(
        self,
        task: str,
        label: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[list[str]] = None,
    ) -> dict:
        """执行 spawn 操作"""
        
        # 1. 权限检查：禁止嵌套 spawn
        if is_subagent_session_key(self.session_key):
            return {
                "status": "forbidden",
                "error": "sessions_spawn is not allowed from sub-agent sessions",
            }
        
        # 2. 解析父 agent ID
        parts = self.session_key.split(":")
        parent_agent_id = parts[1] if len(parts) >= 2 else "main"
        
        # 3. 生成子代理会话 key
        child_session_key = generate_subagent_session_key(parent_agent_id)
        run_id = str(uuid.uuid4())
        
        # 4. 构建子代理 system prompt
        child_system_prompt = build_subagent_system_prompt(
            requester_session_key=self.session_key,
            child_session_key=child_session_key,
            task_text=task,
        )
        
        # 5. 注册到 registry
        entry = SubagentRunEntry(
            run_id=run_id,
            child_session_key=child_session_key,
            requester_session_key=self.session_key,
            task_label=label or task[:50],
            started_at=asyncio.get_event_loop().time(),
        )
        self.registry.register(entry)
        
        # 6. 在 subagent lane 中启动子代理（不等待完成）
        asyncio.create_task(
            self._run_subagent(
                run_id=run_id,
                child_session_key=child_session_key,
                task=task,
                system_prompt=child_system_prompt,
                model=model,
                tools=tools,
            )
        )
        
        return {
            "status": "spawned",
            "run_id": run_id,
            "session_key": child_session_key,
            "message": f'Sub-agent started: "{label or task[:30]}..."',
        }
    
    async def _run_subagent(
        self,
        run_id: str,
        child_session_key: str,
        task: str,
        system_prompt: str,
        model: Optional[str],
        tools: Optional[list[str]],
    ):
        """在 subagent lane 中运行子代理"""
        try:
            result = await self.lane_manager.enqueue(
                CommandLane.SUBAGENT,
                self.run_agent,
                message=task,
                session_key=child_session_key,
                system_prompt=system_prompt,
                model=model,
                tools=tools,
            )
            
            # 更新状态为完成
            self.registry.update_status(
                run_id=run_id,
                status="completed",
                result=result.get("reply", ""),
            )
            
        except Exception as e:
            # 更新状态为错误
            self.registry.update_status(
                run_id=run_id,
                status="error",
                result=str(e),
            )

# ============== Subagent Announce Flow ==============

class SubagentAnnouncer:
    """子代理结果公告器"""
    
    def __init__(
        self,
        registry: SubagentRegistry,
        send_to_session: Callable,  # 发送消息到会话的函数
    ):
        self.registry = registry
        self.send_to_session = send_to_session
        self._queues: dict[str, list[dict]] = {}
        
        # 监听子代理完成事件
        registry.on_complete(self._on_subagent_complete)
    
    def _on_subagent_complete(self, entry: SubagentRunEntry):
        """子代理完成时的回调"""
        asyncio.create_task(self._announce(entry))
    
    async def _announce(self, entry: SubagentRunEntry):
        """公告子代理结果"""
        status_label = "completed" if entry.status == "completed" else "failed"
        
        # 构建通知消息
        trigger_message = f"""📋 Background task "{entry.task_label}" just {status_label}.

**Findings:**
{entry.result or "(no output)"}

---
Summarize this naturally for the user. If there are actionable items, highlight them."""
        
        # 检查主代理是否正忙
        # （简化实现：这里直接发送，生产环境应检查忙碌状态）
        await self.send_to_session(
            session_key=entry.requester_session_key,
            message=trigger_message,
            is_system_event=True,
        )

# ============== 使用示例 ==============

async def demo():
    """演示子代理系统"""
    
    # 初始化组件
    registry = SubagentRegistry()
    lane_manager = LaneManager()
    
    # 模拟 run_agent 函数
    async def mock_run_agent(message: str, session_key: str, **kwargs) -> dict:
        await asyncio.sleep(2)  # 模拟执行时间
        return {"reply": f"Analysis complete for: {message[:30]}..."}
    
    # 模拟 send_to_session 函数
    async def mock_send(session_key: str, message: str, **kwargs):
        print(f"[{session_key}] Received: {message[:100]}...")
    
    # 创建公告器
    announcer = SubagentAnnouncer(registry, mock_send)
    
    # 创建 spawn 工具
    spawn_tool = SessionsSpawnTool(
        session_key="agent:main:main",
        registry=registry,
        lane_manager=lane_manager,
        run_agent=mock_run_agent,
    )
    
    # 主代理 spawn 一个子代理
    result = await spawn_tool.execute(
        task="Research the latest trends in AI agent frameworks",
        label="AI Trends Research",
    )
    print(f"Spawn result: {result}")
    
    # 等待子代理完成
    await asyncio.sleep(3)
    
    # 查看子代理状态
    subagents = registry.get_for_session("agent:main:main")
    for sa in subagents:
        print(f"Subagent: {sa.task_label} - {sa.status}")

if __name__ == "__main__":
    asyncio.run(demo())
```

### 10.11 关键设计要点总结

| 设计点 | 目的 | 实现方式 |
|--------|------|----------|
| **Session Key 隔离** | 区分主/子会话 | `agent:xxx:subagent:uuid` 格式 |
| **禁止嵌套 spawn** | 防止无限递归 | `isSubagentSessionKey` 检查 |
| **Bootstrap 过滤** | 减少子代理上下文 | 白名单机制，只保留 AGENTS.md, TOOLS.md |
| **工具策略隔离** | 限制子代理能力 | 禁用 cron, message, spawn 等 |
| **Lane 并发控制** | 资源隔离和限流 | 主代理 4 并发，子代理 8 并发 |
| **Registry 追踪** | 生命周期管理 | 注册表 + 事件监听 |
| **Announce Queue** | 避免消息冲突 | 队列化 + 批量合并 |
| **spawnedBy 字段** | 父子关系追踪 | 用于权限和清理 |

**核心思想**：子代理是「一次性、专注、受限」的执行单元，完成任务后自动汇报并清理。


上面是openclaw的subagent设计.

