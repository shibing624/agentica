# Subagent

Subagent 用于把独立、只读、可明确交接的事实收集任务放进单独上下文。代码修改、正确性审查、根因判断和发布决策由主 Agent 完成。

如果只是固定步骤流水线，优先使用 [Workflow](workflow.md)；如果只是调用一个长期存在的专门助手，优先使用 `Agent.as_tool()`。只有子任务确实需要独立上下文、工具权限和运行预算时，才使用 Subagent。

## 配置格式

Agentica 不再用封闭的 Python `Enum` 定义 agent 类型。每个 agent 是一个 `*.md` 文件：YAML frontmatter 保存结构化字段，Markdown 正文直接作为 system prompt。

```markdown
---
name: Explore Agent
description: >-
  Fast read-only agent for locating files and mapping an unfamiliar repository.
allowed_tools: [ls, read_file, glob, grep, execute]
denied_tools: [write_file, edit_file, multi_edit_file, task]
execute_policy: read_only
model_tier: auxiliary
max_turns: 200
timeout: 1800
can_spawn_subagents: false
inherit_context: false
inherit_workspace: false
inherit_knowledge: false
---
You are a file search specialist...
```

文件名是不区分大小写的 agent ID，例如 `explore.md` 对应 `agent_type="explore"`。`name` 只是展示名称。

## 目录与优先级

同名文件按以下优先级覆盖：

1. `<cwd>/.agentica/agents/*.md`
2. `~/.agentica/agents/*.md`，或 `$AGENTICA_HOME/agents/*.md`
3. `$AGENTICA_AGENT_DIR/*.md`
4. 包内默认 `agentica/agents/*.md`

包内默认目前只有 `explore`、`research`、`code`。用户目录只保存新增或覆盖定义，不复制默认文件，避免升级后被旧副本永久遮蔽。

`review` 不再是默认 subagent。主 Agent 已经拥有完整对话、diff 和实现上下文，直接审查更可靠，也避免一次昂贵且容易丢上下文的重复调用。

## 运行字段

| 字段 | 说明 |
|------|------|
| `allowed_tools` | 允许继承的工具；省略表示继承父 Agent 可用工具，`[]` 表示不提供工具 |
| `denied_tools` | 显式禁用的工具，优先于 `allowed_tools` |
| `execute_policy` | `inherit` 或 `read_only` |
| `model_tier` | `auxiliary` 或 `main` |
| `tool_call_limit` | 可选的工具调用总上限 |
| `max_turns` | 最大模型循环轮数 |
| `timeout` | 超时秒数，`0` 表示不设置超时 |
| `can_spawn_subagents` | 是否允许继续派生子任务 |
| `inherit_context` | 是否继承父 Agent 的上下文摘要 |
| `inherit_workspace` | 是否继承 Workspace |
| `inherit_knowledge` | 是否继承 Knowledge |

用户和项目文件格式错误时会记录 warning 并跳过；包内默认配置错误会直接抛出，因为这表示安装包本身损坏。

## 调用

```python
from agentica import Agent, OpenAIChat
from agentica.subagent import SubagentRegistry, get_subagent_configs

configs = get_subagent_configs()
assert "explore" in configs

parent = Agent(model=OpenAIChat(id="gpt-4o"), tools=[...])
result = await SubagentRegistry().spawn(
    parent_agent=parent,
    task="定位请求进入服务后经过的主要函数，返回 path:line 证据",
    agent_type="explore",
)
```

`DeepAgent` 的内置 `task` 工具使用同一个 registry。CLI 可通过 `/agents` 查看有效定义，通过 `/agents create <name>` 创建项目级 Markdown 文件，通过 `/agents reload` 重载。

## 只读边界

包内三个默认 agent 都使用 `execute_policy: read_only`，允许 `git diff`、`git log`、测试和 lint 等检查命令，拒绝提交、安装、重定向写文件等状态修改。

这是命令级 best-effort 约束，不是 OS 沙箱。测试代码本身仍可能写文件；需要对抗恶意输入时，应使用 Docker、seccomp 或其他系统级隔离。

## 下一步

- [编排模式选择](choosing.md)
- [Swarm](swarm.md)
- [Workflow](workflow.md)
