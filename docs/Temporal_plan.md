# Agentica Temporal 集成方案

## 1. 概述

将 Temporal 集成到 Agentica，支持：
- 长时间运行的 Agent 工作流
- 工作流状态持久化与故障恢复
- 并行 Agent 执行
- 工作流可观测性

## 2. 架构设计

### 2.1 Temporal 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                    Temporal 架构                             │
├─────────────────────────────────────────────────────────────┤
│  Workflow（确定性）              Activity（非确定性）         │
│  ├─ 纯编排逻辑                   ├─ LLM API 调用             │
│  ├─ 条件判断、循环               ├─ 工具执行                  │
│  ├─ 状态机控制                   ├─ 数据库/网络操作           │
│  └─ 调用 Activity                └─ 任何有副作用的操作        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 部署架构

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Temporal Server │     │     Worker       │     │     Client       │
│  (独立服务)       │◄───►│  (执行任务)       │     │  (发起/查询)      │
│                  │     │                  │     │                  │
│  - 状态持久化     │     │  - 运行 Workflow │     │  - 启动工作流     │
│  - 任务调度       │     │  - 执行 Activity │     │  - 查询状态       │
│  - 故障恢复       │     │  - 长期运行      │     │  - 获取结果       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## 3. 目录结构

```
agentica/
├── temporal/                    # Temporal 集成模块（核心库）
│   ├── __init__.py             # 导出公共接口
│   ├── activities.py           # Agent Activity 封装
│   ├── workflows.py            # Workflow 类定义
│   └── client.py               # Temporal Client 封装

examples/
├── 58_temporal_worker.py       # Worker 启动示例
└── 58_temporal_client.py       # Client 使用示例
```

## 4. 核心模块

### 4.1 Activities (`agentica/temporal/activities.py`)

Activities 封装非确定性操作（LLM 调用、工具执行）：

```python
from agentica.temporal import (
    run_agent_activity,
    AgentActivityInput,
    AgentActivityOutput,
)

# Activity 输入
input = AgentActivityInput(
    message="What is AI?",
    agent_name="assistant",
    agent_config={"instructions": "Be helpful"},
    images=["https://example.com/image.png"],  # 可选
)

# Activity 输出
output = AgentActivityOutput(
    content="AI is...",
    agent_name="assistant",
    run_id="xxx",
    metrics={...},
)
```

### 4.2 Workflows (`agentica/temporal/workflows.py`)

提供 4 种预定义 Workflow：

| Workflow | 描述 | 输入类型 |
|----------|------|----------|
| `AgentWorkflow` | 单 Agent 执行 | `WorkflowInput` |
| `SequentialAgentWorkflow` | 顺序执行多 Agent（流水线） | `WorkflowInput` |
| `ParallelAgentWorkflow` | 并行执行多 Agent | `WorkflowInput` |
| `ParallelTranslationWorkflow` | 并行翻译 + 最佳选择 | `TranslationInput` |

```python
from agentica.temporal import (
    AgentWorkflow,
    SequentialAgentWorkflow,
    ParallelAgentWorkflow,
    ParallelTranslationWorkflow,
    WorkflowInput,
    TranslationInput,
)

# 单 Agent
input = WorkflowInput(message="Hello")

# 多 Agent（顺序/并行）
input = WorkflowInput(
    message="Write about AI",
    agent_configs=[
        {"name": "researcher", "instructions": "Research the topic"},
        {"name": "writer", "instructions": "Write based on research"},
    ],
)

# 翻译
input = TranslationInput(
    text="Hello world",
    target_language="Chinese",
    num_translations=3,
)
```

### 4.3 Client (`agentica/temporal/client.py`)

简化的 Temporal Client 封装：

```python
from agentica.temporal import TemporalClient, AgentWorkflow, WorkflowInput

# 连接
client = TemporalClient(host="localhost:7233")
await client.connect()

# 启动 Workflow
workflow_id = await client.start_workflow(
    AgentWorkflow,
    WorkflowInput(message="Hello"),
)

# 获取结果
result = await client.get_result(workflow_id)
print(result.content)

# 其他操作
status = await client.get_status(workflow_id)
await client.cancel_workflow(workflow_id)
await client.terminate_workflow(workflow_id, reason="Test")
```

## 5. 使用方式

### 5.1 安装依赖

```bash
pip install temporalio
```

### 5.2 启动 Temporal Server

```bash
# 开发模式
temporal server start-dev

# 或使用 Docker
docker run -d --name temporal -p 7233:7233 temporalio/auto-setup:latest
```

### 5.3 启动 Worker

```bash
python examples/58_temporal_worker.py
```

Worker 会注册以下 Workflow 和 Activity：
- Workflows: `AgentWorkflow`, `SequentialAgentWorkflow`, `ParallelAgentWorkflow`, `ParallelTranslationWorkflow`
- Activities: `run_agent_activity`

**注意**：Worker 使用 `UnsandboxedWorkflowRunner` 禁用 Temporal 沙箱，因为 agentica 依赖的库（如 httpx）在沙箱中受限。

### 5.4 运行 Client

```bash
# 简单 Agent
python examples/58_temporal_client.py simple "What is AI?"

# 并行翻译
python examples/58_temporal_client.py translate "Hello, how are you?"

# 顺序执行（研究 -> 写作 -> 编辑）
python examples/58_temporal_client.py sequential "machine learning"

# 并行分析（多视角）
python examples/58_temporal_client.py parallel "artificial intelligence"

# 查询状态
python examples/58_temporal_client.py status <workflow_id>

# 获取结果
python examples/58_temporal_client.py result <workflow_id>
```

## 6. 与现有 Workflow 的关系

**为什么 Temporal Workflow 没有继承 `agentica.Workflow`？**

| 特性 | `agentica.Workflow` | Temporal Workflow |
|------|---------------------|-------------------|
| 基类 | Pydantic `BaseModel` | `@workflow.defn` 装饰器 |
| 执行模型 | 直接调用 Agent | 通过 Activity 间接调用 |
| 确定性要求 | 无 | 必须确定性 |
| 状态持久化 | 手动（数据库） | 自动（Temporal Server） |
| 故障恢复 | 手动实现 | 自动 |

**技术原因**：
1. Temporal Workflow 必须是确定性的，不能直接调用 I/O、随机数、LLM 等
2. Temporal 使用装饰器模式，与 Pydantic BaseModel 不兼容
3. 现有 Workflow 的 `run()` 直接执行 Agent，而 Temporal 要求通过 Activity 间接调用

**桥接方案**：用户可以将现有 Workflow 的逻辑封装为 Temporal Activity，从而复用已有代码。

## 7. 实施状态

### Phase 1: 基础集成 ✅
- [x] 创建 `agentica/temporal/` 目录
- [x] 实现 `activities.py` - Agent Activity 封装
- [x] 实现 `workflows.py` - 4 种 Workflow 类
- [x] 实现 `client.py` - Client 封装
- [x] 实现 `__init__.py` - 导出接口

### Phase 2: 示例代码 ✅
- [x] 创建 `58_temporal_worker.py` - Worker 启动
- [x] 创建 `58_temporal_client.py` - Client 使用

### Phase 3: 文档与测试
- [ ] 更新 requirements.txt
- [ ] 添加单元测试

## 8. 后续扩展

- [ ] Human-in-the-Loop (HITL) 支持
- [ ] 工作流版本管理
- [ ] 自定义重试策略
- [ ] 监控与告警集成
- [ ] 与 `agentica.Workflow` 的适配器
