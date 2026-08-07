# Subagent

Subagent 用于把边界清楚的调查任务放进独立上下文，例如定位文件、追踪调用链、收集外部资料。主 Agent 负责理解结果、修改代码、正确性审查和最终决策。

如果任务只是固定步骤流水线，优先使用 [Workflow](workflow.md)；如果要把长期存在的 Agent 当作工具组合，使用 `Agent.as_tool()`。只有子任务确实需要独立的提示词、工具权限、模型和运行预算时，才使用 Subagent。

## 内置 Subagent

Agentica 内置三个只读定义：

| ID | 用途 | 是否继承主上下文 |
|----|------|------------------|
| `explore` | 定位文件、搜索源码、梳理仓库结构 | 否 |
| `research` | 网页搜索、资料收集和证据整理 | 否 |
| `code` | 解释代码、追踪调用链和数据流 | 是 |

默认不提供 `review`。代码审查依赖完整对话、diff 和实现意图，应由主 Agent 完成。

## Markdown 配置格式

每个 Subagent 是一个 `*.md` 文件：YAML frontmatter 保存运行配置，Markdown 正文就是 system prompt。文件名 stem 是调用时使用的 ID，必须为小写，只能包含字母、数字、`-` 和 `_`；`name` 只是展示名称。

下面是可直接保存为 `.agentica/agents/data-analyst.md` 的完整示例：

```markdown
---
name: Data Analyst
description: >-
  Read-only analyst for inspecting local CSV and JSON data and reporting
  reproducible findings.
allowed_tools: [ls, read_file, glob, grep, execute]
denied_tools: [write_file, edit_file, apply_patch, task]
execute_policy: read_only
model_tier: auxiliary
tool_call_limit: 40
max_turns: 80
timeout: 900
can_spawn_subagents: false
inherit_context: false
inherit_workspace: false
inherit_knowledge: false
---
You are a read-only data analyst.

Inspect the requested data, run reproducible read-only checks, and return:
1. the files and columns examined;
2. the commands or calculations used;
3. findings with concrete values;
4. uncertainties or missing data.

Do not create or modify files. Do not make product decisions for the caller.
```

`description` 和 Markdown 正文不能为空。其他字段可省略并使用默认值。

## 配置目录与覆盖顺序

同名 ID 按以下顺序覆盖，越靠前优先级越高：

1. `<cwd>/.agentica/agents/*.md`：当前项目
2. `~/.agentica/agents/*.md`：当前用户；设置 `AGENTICA_HOME` 后改为 `$AGENTICA_HOME/agents/*.md`
3. `$AGENTICA_AGENT_DIR/*.md`：额外托管目录
4. 安装包内的 `agentica/agents/*.md`：内置默认定义

高优先级文件会完整替换同名定义，不会与低优先级 frontmatter 或正文合并。

### 项目级自定义

项目级定义随仓库共享，只在该项目中生效：

```text
my-project/
└── .agentica/
    └── agents/
        └── data-analyst.md
```

在项目根目录启动 `agentica`，然后执行 `/agents reload` 即可加载新文件。

### 用户级自定义

用户级定义对所有项目生效。将同样格式的文件保存为：

```text
~/.agentica/agents/security-research.md
```

例如：

```markdown
---
name: Security Research
description: Collects security evidence without changing the repository.
allowed_tools: [read_file, ls, glob, grep, web_search, fetch_url, execute]
denied_tools: [write_file, edit_file, apply_patch, task]
execute_policy: read_only
model_tier: main
max_turns: 120
timeout: 1800
---
You investigate the requested security question and return evidence with exact
file locations or source URLs. Separate confirmed behavior from inference.
Never modify files, dependencies, Git state, or external systems.
```

### 覆盖内置定义

要定制内置 `research`，创建项目级 `.agentica/agents/research.md` 或用户级 `~/.agentica/agents/research.md`。文件必须包含完整 frontmatter 和完整 system prompt；删除覆盖文件并执行 `/agents reload` 后，低优先级定义会重新生效。

不要把安装包内的默认文件复制到用户目录作为初始化步骤，否则后续升级的默认配置会一直被旧副本遮蔽。

## 字段参考

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `name` | 文件 ID | 展示名称，不参与调用 |
| `description` | 必填 | 主 Agent 判断何时委派时看到的能力描述 |
| `allowed_tools` | 省略 | 省略表示从父 Agent 继承可用工具；`[]` 表示不提供工具 |
| `denied_tools` | 省略 | 显式禁用工具，优先于 `allowed_tools` |
| `execute_policy` | `inherit` | `inherit` 或 `read_only`；仅在 `execute` 已被允许时生效 |
| `model_tier` | `auxiliary` | `auxiliary` 使用 task 辅助模型，`main` 使用父 Agent 主模型 |
| `tool_call_limit` | 无限制 | 工具调用总上限，必须为正整数 |
| `max_turns` | `100` | 最大模型循环轮数，必须为正整数 |
| `timeout` | `1800` | 超时秒数；`0` 表示不设超时 |
| `can_spawn_subagents` | `false` | 是否允许继续委派；运行时仍限制最大深度 |
| `inherit_context` | `false` | 是否注入父 Agent 的指令摘要、近期历史等交接上下文 |
| `inherit_workspace` | `false` | 是否共享父 Agent 的 Workspace 记忆 |
| `inherit_knowledge` | `false` | 是否共享父 Agent 的 Knowledge 实例 |

默认隔离是刻意设计的：委派描述应像交给一个没有读过当前对话的同事一样自包含。只有子任务确实依赖父上下文或长期数据时，才开启对应的继承字段。

`model_tier: auxiliary` 的解析顺序是 task 专用模型、`auxiliary_model`、主模型。CLI 配置辅助模型的方法见 [OpenAI Responses API](../guides/openai-responses.md) 和 [CLI 终端指南](../getting-started/terminal.md)。

## CLI 管理

```text
/agents                         # 列出最终生效的定义、来源、模型层级和工具
/agents list                    # 同上
/agents create data-analyst     # 交互创建项目级 Markdown 文件
/agents reload                  # 重新扫描并加载磁盘配置
/agents remove data-analyst     # 删除最高优先级的用户/项目定义
```

`/agents create` 会询问 `description` 和允许的工具，并生成 `.agentica/agents/<id>.md`。随后编辑生成文件中的 system prompt 和高级字段，再执行 `/agents reload`。

## SDK 调用

`DeepAgent` 已内置 `task` 工具。主 Agent 会根据各定义的 `description` 选择是否委派，也可以在提示词中明确指定：

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("用 data-analyst subagent 检查 data/events.csv 的异常值")
print(result.content)
```

需要由应用代码直接控制时，可调用同一个 registry：

```python
from agentica import DeepAgent, OpenAIChat
from agentica.subagent import SubagentRegistry

parent = DeepAgent(model=OpenAIChat(id="gpt-4o-mini"))
result = await SubagentRegistry().spawn(
    parent_agent=parent,
    task="定位请求进入服务后经过的主要函数，返回 path:line 证据",
    agent_type="explore",
)

if result["status"] == "completed":
    print(result["content"])
```

`register_custom_subagent()` 也能在 Python 进程内注册相同字段，但它只对当前进程有效。需要跨 CLI、Gateway 和后续运行持久生效时，应使用 Markdown 文件。

## 预算中断与续跑

正常完成时，`task` 返回结果、工具调用摘要、耗时和工具次数。遇到 `timeout`、`max_turns`、`tool_call_limit` 或截断时，返回值会明确标记 `success: false`、`partial: true`，并保留已有结果、`run_id` 和建议的下一步。主 Agent 可以使用 `resume_from_run_id` 延续该次调查，不必从头开始。

配置错误的项目级或用户级文件会记录 warning 并跳过；安装包内置定义错误会直接抛出异常，因为这表示安装内容损坏。

## 只读边界

`execute_policy: read_only` 允许 `git diff`、`git log`、测试和 lint 等检查命令，并拒绝提交、安装、Shell 重定向写文件等明显的状态修改。

这是命令级 best-effort 约束，不是 OS 沙箱。测试代码本身仍可能写文件，复杂命令也不能只靠字符串分类获得强隔离。处理不可信输入时，应使用 Docker、seccomp 或其他系统级沙箱。

## 下一步

- [选择多智能体编排模式](choosing.md)
- [Swarm](swarm.md)
- [Workflow](workflow.md)
