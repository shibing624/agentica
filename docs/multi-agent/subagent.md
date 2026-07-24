# Subagent

Subagent 系统允许主 Agent 生成隔离的临时子任务 Agent，用于处理复杂的分步骤任务。

如果只是固定步骤流水线，优先使用 [Workflow](workflow.md)。如果只是让父 Agent 调用一个专门助手，优先使用 `Agent.as_tool()`。只有当子任务需要独立运行时状态、工具权限、嵌套深度限制或超时控制时，再使用 Subagent；完整取舍见 [编排模式决策树](choosing.md)。

## SubagentType

```python
from agentica.subagent import SubagentType

class SubagentType(str, Enum):
    EXPLORE = "explore"    # 代码库探索（只读，auxiliary 模型）
    RESEARCH = "research"  # 网页搜索和文档分析（auxiliary 模型）
    CODE = "code"          # 代码解释：怎么工作、谁调谁（auxiliary 模型）
    REVIEW = "review"      # 判断类：正确性评审、根因定位（主模型）
    CUSTOM = "custom"      # 用户自定义类型
```

所有内置类型都是只读的（不能改文件，命令只能是只读命令，详见下面的[只读 Shell](#只读-shell)），差别在**跑哪个模型**：

| 层级 | 类型 | 适合的任务 |
|------|------|-----------|
| `auxiliary`（便宜模型） | `explore` / `research` / `code` | 事实收集：X 在哪、哪些文件用到 Y、这个模块干什么 |
| `main`（父 Agent 自己的模型） | `review` | 判断：这段代码对不对、根因是什么、能不能上线 |

把判断类任务丢给便宜模型是最坏的情况——它会自信地回答"没问题"，主模型信了就不再看了。所以 `review` 强制走主模型，代价是贵，因此它的 prompt 要求调用方给出明确的文件范围和单一问题。

## SubagentConfig

每种 Subagent 类型有独立的权限配置：

```python
from agentica.subagent import SubagentConfig, SubagentType

config = SubagentConfig(
    type=SubagentType.RESEARCH,
    name="Research Agent",
    description="负责搜索和分析文档",
    system_prompt="你是一个专业的研究助手...",
    allowed_tools=["web_search", "fetch_url"],
    denied_tools=["write_file", "edit_file"],
    tool_call_limit=100,
    can_spawn_subagents=False,
    inherit_workspace=False,
    inherit_knowledge=False,
    timeout=300,  # 秒
)
```

| 参数 | 说明 |
|------|------|
| `allowed_tools` | 允许使用的工具（None = 继承父 Agent 全部） |
| `denied_tools` | 禁用的工具（优先级高于 allowed） |
| `tool_call_limit` | 最大工具调用次数 |
| `can_spawn_subagents` | 是否允许生成子 Subagent |
| `inherit_workspace` | 是否继承父 Agent 的 Workspace |
| `inherit_knowledge` | 是否继承父 Agent 的知识库 |
| `timeout` | 执行超时秒数 |
| `model_tier` | `"auxiliary"`（默认，走 `Agent.resolve_auxiliary_model("task")`）或 `"main"`（走父 Agent 的模型） |
| `execute_policy` | `"inherit"`（默认，`execute` 不受限）或 `"read_only"`（`execute` 拒绝改变状态的命令） |

## 只读 Shell

四个内置类型的 `execute_policy` 都是 `"read_only"`：它们**能跑只读命令，不能跑改变状态的命令**。这对 `review` 特别重要——评审代码却看不到 `git diff`，只能靠猜改了什么。

```python
# review subagent 可以自己看改动
await SubagentRegistry().spawn(
    parent_agent=parent,
    task="检查未提交的改动有没有正确性问题，自己跑 git diff 看改了什么",
    agent_type="review",
)
```

放行的：`git diff` / `log` / `show` / `status` / `blame` / `grep` 等查询类子命令、测试与 lint 运行器（pytest、npm test、cargo test、mypy…）、以及 `ls` / `wc` / `jq` 这类纯查询命令。

拒绝的：`git commit` / `push` / `checkout` / `reset` / `stash`、包安装与发布、任意解释器（`python script.py`、`bash x.sh`）、输出重定向（`> file`）、命令替换（`$(...)`、反引号），以及会改写文件的 runner 变体（`ruff --fix`、`cargo fmt`）。

复合命令的**每一段**都单独判定，所以 `git log && rm -rf build` 会在第二段被拒——不会因为第一个 token 无害就整条放过。

自定义类型可以按需开启：

```python
register_custom_subagent(
    name="verifier",
    description="验证改动是否真的完成",
    system_prompt="...",
    allowed_tools=["read_file", "glob", "grep", "execute"],
    execute_policy="read_only",
    model_tier="main",
)
```

Markdown 定义的 subagent 在 frontmatter 里写 `execute_policy: read_only` 即可。

> **这是 best-effort，不是安全沙箱。** 语义对齐 Cursor 的 `readonly: true`（"no file edits, no state-changing shell commands"），但 Cursor 的实际保证来自 OS 级沙箱——它拦的是"写文件和联网"这个后果，而不是命令名字。agentica 这一层是命令字符串判定，所以放行的测试运行器仍会执行项目自己的代码（`conftest.py`、npm scripts、Makefile target），那些代码是可以写文件的。把它当作防止模型顺手 `git commit` 的护栏，不要当作对抗恶意输入的边界。真正的隔离需要 Docker / seccomp 级别的沙箱。

## SubagentRegistry

`SubagentRegistry` 是 Subagent 执行的唯一入口，负责：模型克隆 + 工具继承（自动按 `BLOCKED_TOOLS` / `allowed_tools` / `denied_tools` 过滤父 Agent 工具）+ 嵌套深度限制（`MAX_DEPTH=2`）+ 注册表跟踪 + 实时事件冒泡 + usage 合并 + 超时控制。

```python
from agentica import Agent, OpenAIChat
from agentica.subagent import (
    SubagentRegistry,
    SubagentType,
    register_custom_subagent,
)

# 注册自定义 Subagent 类型（模块级，全局生效）
register_custom_subagent(
    name="data_analyst",
    description="数据分析专家",
    system_prompt="你是一个数据分析师...",
    allowed_tools=["sql_query", "python_eval"],
)

parent = Agent(model=OpenAIChat(id="gpt-4o"), tools=[...])

# 直接调用 spawn() 启动子 Agent
result = await SubagentRegistry().spawn(
    parent_agent=parent,
    task="分析销售数据趋势",
    agent_type="data_analyst",
)
# result = {
#   "status": "completed",
#   "content": "...",
#   "agent_type": "custom",
#   "subagent_name": "data_analyst",
#   "run_id": "...",
#   "tool_calls_summary": [...],
#   "tool_count": N,
#   "execution_time": 12.345,
# }
```

## SubagentRun

跟踪 Subagent 执行生命周期：

```python
@dataclass
class SubagentRun:
    run_id: str              # 唯一标识
    subagent_type: SubagentType
    parent_agent_id: str     # 父 Agent agent_id
    task_label: str          # 截短的任务标签
    task_description: str    # 完整任务描述
    started_at: datetime
    status: str              # pending / running / completed / error / cancelled
    ended_at: Optional[datetime]
    result: Optional[str]
    error: Optional[str]
    token_usage: Optional[Dict[str, int]]
```

## 与 DeepAgent 内置 task 工具的关系

`DeepAgent` 的内置 `task` 工具底层使用 Subagent 系统：

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(model=OpenAIChat(id="gpt-4o"))
# Agent 可通过 task 工具自动委派子任务
result = await agent.run("分析项目代码结构并生成文档")
```

## 下一步

- [Swarm](swarm.md) -- 自主多智能体协作
- [Workflow](workflow.md) -- 确定性工作流编排
- [Hooks](../advanced/hooks.md) -- 监控子任务执行
