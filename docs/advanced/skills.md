# Skills System

Skills 是 Agentica 的"提示即能力"系统——通过 Markdown 文件定义的指令，按需注入到 Agent 的 System Prompt 中，赋予 Agent 特定领域的专业知识和工作流程。

**核心理念**：Skills 是**用户自定义内容**，不是框架内置的。框架只提供加载机制，你来定义能力。对 Agentica 来说，框架负责运行时、工具链和记忆基础设施，Skill 则负责把特定 workflow 注入进去，例如代码评审、研究流程，或 `learn-from-experience` 这类 self-learn 策略。

## 核心概念

- Skill = 一个包含 `SKILL.md` 的目录
- `SKILL.md` = YAML frontmatter（元数据）+ Markdown body（指令内容）
- 运行时由 `SkillTool` 按需加载，注入到 System Prompt
- 模型无关——纯文本指令，适用于所有 LLM

## SKILL.md 格式

```markdown
---
name: code_reviewer
description: "代码审查专家，检查安全性、性能和可读性"
trigger: /review          # 可选：触发命令（用于 CLI 斜杠命令）
requires:                 # 可选：前置依赖
  - git
allowed-tools:            # 可选：限制可用工具
  - read_file
  - grep
metadata:
  emoji: "🔍"             # 可选：CLI 显示图标
---

# Code Reviewer

你是一个代码审查专家。审查代码时遵循以下原则：

## 审查重点

1. **安全性** — 检查潜在的安全漏洞（SQL 注入、XSS、硬编码密钥等）
2. **性能** — 识别性能瓶颈（N+1 查询、不必要的循环、内存泄漏）
3. **可读性** — 代码是否易于理解，命名是否清晰
4. **最佳实践** — 是否遵循语言惯用法和项目规范

## 输出格式

对每个问题提供：
- **严重程度**: Critical / Warning / Info
- **位置**: 文件名 + 行号
- **问题描述**: 具体说明问题
- **修复建议**: 附带修复代码示例
```

### Frontmatter 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | `str` | ✓ | Skill 唯一标识，小写加连字符 |
| `description` | `str` | ✓ | 简短描述（LLM 用于决策是否激活） |
| `trigger` | `str` | — | CLI 斜杠命令（如 `/review`） |
| `requires` | `List[str]` | — | 前置依赖（命令行工具等） |
| `allowed-tools` | `List[str]` | — | 限制可用工具白名单 |
| `metadata.emoji` | `str` | — | CLI 显示图标 |

## Skill 目录结构

```
.agentica/skills/              # 项目级 skills（推荐放这里）
  code-reviewer/
    SKILL.md                   # 必须
    scripts/                   # 可选：辅助脚本
      run_lint.sh
    references/                # 可选：参考文档
      style-guide.md
    assets/                    # 可选：图片、数据等

~/.agentica/skills/            # 用户级 skills（跨项目共用）
  git-commit/
    SKILL.md

/opt/agent-skills/learn-from-experience/   # 外部托管 skill（可选）
  SKILL.md
```

**搜索路径（优先级从高到低）**：

1. `.claude/skills/`（项目级）
2. `.agentica/skills/`（项目级）
3. `AGENTICA_EXTRA_SKILL_PATH` 指定的外部目录（支持用 `:` 分隔多个路径，也支持直接指向单个 skill 目录）
4. `~/.claude/skills/`（用户级）
5. `~/.agentica/skills/`（用户级）

同名 Skill，项目级覆盖用户级。

## 加载与使用

### 方式一：SkillTool（程序化）

```python
from agentica import Agent, OpenAIChat
from agentica.tools.skill_tool import SkillTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[SkillTool()],           # 自动扫描标准目录
)

# 也可指定自定义目录
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[SkillTool(custom_skill_dirs=["./my-skills/web-research"])],
)
```

也可以直接挂外部 skill 仓库，而不必复制到项目目录：

```bash
export AGENTICA_EXTRA_SKILL_PATH="/home/user/skills/learn-from-experience:/opt/shared/skills"
```

当路径本身就是一个包含 `SKILL.md` 的 skill 目录时，`SkillLoader` 会直接加载它。

### 方式二：DeepAgent 内置（CLI 推荐）

```bash
agentica --enable-skills
# Skills 自动从标准目录加载，对话中可直接使用
```

`DeepAgent` 现在默认也会自动加载当前工作目录下的 `mcp_config.json/yaml/yml`，所以 CLI 里不需要再单独把 MCP 打开，skills 和 MCP 都是开箱即用。

安装外部 skill 集合：

```bash
agentica extensions install https://github.com/obra/superpowers
```

这个命令会：
- 克隆远程仓库，或读取本地目录
- 自动发现根目录 `SKILL.md`、`skills/`、`.agentica/skills/`、`.claude/skills/` 下的 skill
- 把发现到的 skill 安装到 `~/.agentica/skills/`

如果你已经下载到本地，也可以直接安装本地 repo：

```bash
agentica extensions install /path/to/skill-repo --target-dir ~/.agentica/skills
```

如果你安装到自定义目录而不是标准搜索路径，记得把该目录加入 `AGENTICA_EXTRA_SKILL_PATH`，否则 CLI 刷新后也不会自动发现它。

如果你已经进入交互式 CLI，则直接使用 slash command，不用退出：

```text
> /extensions install https://github.com/obra/superpowers
> /extensions list
> /extensions remove learn-from-experience
> /extensions reload
```

如果你希望 Agent 带有持续学习 workflow，推荐做法是：
- 用 `DeepAgent` 作为默认运行时
- 把 `learn-from-experience` 安装到 `~/.agentica/skills/`，或通过 `AGENTICA_EXTRA_SKILL_PATH` 指向外部目录
- 确认过的常驻偏好直接写进该用户的 `~/.agentica/workspace/users/{user_id}/AGENTS.md`（CLI 为 `.../users/default/AGENTS.md`，也可通过 symlink `~/.agentica/AGENTS.md` 改同一份文件；人手写或 agent `edit_file`；不要再走 memory→AGENTS 编译）

### 方式三：按需激活（RunConfig 白名单）

```python
from agentica.run_config import RunConfig

result = await agent.run(
    "审查这段代码",
    config=RunConfig(enabled_skills=["code-reviewer"]),
)
```

### CLI 中使用

```
> /skills                           # 列出所有可用 skills
> /extensions install <repo-or-path> # 安装并自动刷新当前 session
> /extensions list                  # 查看已安装扩展
> /extensions remove <skill-name>   # 删除已安装 skill
> /extensions reload                # 重新扫描磁盘并刷新当前 session
> 用 code-reviewer skill 审查 @main.py
> /review @authentication.py        # 用 trigger 激活（如果配置了）
```

## 示例 Skills

以下是几个可直接复用的 Skill 模板，放到 `.agentica/skills/` 即可使用。

### Git Commit（规范提交）

```markdown
---
name: git-commit
description: "创建符合 Conventional Commits 规范的 git commit。自动分析变更，生成格式正确的提交信息。"
trigger: /commit
requires:
  - git
---

# Git Commit Skill

创建符合 [Conventional Commits](https://www.conventionalcommits.org/) 规范的 git commit。

## 提交信息格式

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**类型（type）**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档变更
- `refactor`: 重构（不新增功能，不修 bug）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具链变更

## 操作步骤

1. 运行 `git diff --staged` 查看已暂存的变更（如无暂存，先用 `git add`）
2. 分析变更内容，确定 type 和 scope
3. 生成简洁的 subject（不超过 50 字符，动词开头，不加句号）
4. 如变更复杂，添加 body 解释 What/Why
5. 执行 `git commit -m "type(scope): subject"`

## 规则

- Subject 用英文，动词原形开头（Add, Fix, Update，而非 Added/Fixed）
- Scope 用小写，反映变更的模块（auth, api, cli 等）
- 一个 commit 只做一件事，不要打包不相关的变更
- 不用 `-F` 或 heredoc 写多行 message，直接用 `-m`
```

### GitHub CLI（GitHub 操作）

```markdown
---
name: github
description: "使用 gh CLI 与 GitHub 交互：管理 Issues、PR、CI 运行和仓库操作。"
trigger: /github
requires:
  - gh
---

# GitHub Skill

使用 `gh` CLI 与 GitHub 交互。**始终指定 `--repo owner/repo`**（除非已在 git 目录中）。

## Pull Requests

```bash
# 查看 PR 列表
gh pr list --repo owner/repo

# 查看 PR 详情和 CI 状态
gh pr view 123 --repo owner/repo
gh pr checks 123 --repo owner/repo

# 创建 PR
gh pr create --title "feat: add feature" --body "Description" --repo owner/repo

# 合并 PR
gh pr merge 123 --squash --repo owner/repo
```

## Issues

```bash
# 创建 Issue
gh issue create --title "Bug: xxx" --body "Steps to reproduce..." --repo owner/repo

# 列出 Issues
gh issue list --state open --label bug --repo owner/repo

# 关闭 Issue
gh issue close 456 --repo owner/repo
```

## CI/CD

```bash
# 查看最近的 workflow runs
gh run list --repo owner/repo

# 查看 run 详情和日志
gh run view 789 --log --repo owner/repo

# 重新触发失败的 run
gh run rerun 789 --failed --repo owner/repo
```
```

### Code Reviewer（代码审查）

```markdown
---
name: code-reviewer
description: "专业代码审查，重点检查安全性、性能、可读性和最佳实践。适用于 PR review 和日常代码改进。"
trigger: /review
---

# Code Reviewer Skill

## 审查流程

1. 读取目标文件（`read_file`）或查看 diff（`execute("git diff HEAD~1")`）
2. 按优先级识别问题：Critical > Warning > Info
3. 生成结构化报告

## 审查维度

### 安全性（Critical 优先）
- 硬编码密钥、密码、token
- SQL 注入、命令注入风险
- 未验证的用户输入
- 敏感数据日志泄露

### 性能
- N+1 查询
- 不必要的全表扫描
- 大对象在循环中创建
- 缺少缓存的热路径

### 可读性
- 函数超过 50 行
- 变量/函数命名不清晰
- 缺少必要的注释（WHY，不是 WHAT）
- 嵌套超过 3 层

### 最佳实践
- 异常处理是否合理
- 资源是否正确释放（context manager）
- 测试覆盖是否充分

## 输出格式

```
## 审查报告：{文件名}

### Critical 🔴
- [行号] 问题描述
  修复：`具体代码`

### Warning 🟡
- [行号] 问题描述

### Info 🟢
- [行号] 建议

### 总结
整体质量评分：X/10
主要改进点：...
```
```

## 创建自己的 Skill

好的 Skill 具备三个要素：

**1. 精确的 description**（LLM 用它决定是否激活这个 skill）

```yaml
# 差：太模糊
description: "帮助写代码"

# 好：明确说明适用场景
description: "Python 异步代码专家，处理 asyncio、aiohttp、FastAPI 相关问题。包含最佳实践、常见陷阱和性能调优指南。"
```

**2. 结构化的指令**（让 LLM 知道"做什么"和"怎么做"）

```markdown
# Skill 名称

## 什么时候用
明确说明触发场景...

## 操作步骤
1. 第一步...
2. 第二步...

## 规则/约束
- 必须做...
- 不能做...

## 输出格式
期望的输出结构...
```

**3. 具体的示例**（比抽象描述更有效）

```markdown
## 示例

用户说："优化这段 SQL"
你应该：
1. 先用 EXPLAIN ANALYZE 分析执行计划
2. 识别全表扫描（Seq Scan）
3. 建议添加索引或改写查询
```

## 在代码中集成 Skill

```python
import asyncio
from agentica import Agent, OpenAIChat
from agentica.tools.skill_tool import SkillTool

async def main():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[
            SkillTool(
                # 自动扫描 .agentica/skills/ 和 ~/.agentica/skills/
                # 加上指定的自定义目录
                custom_skill_dirs=["./team-skills/data-engineering"],
            )
        ],
    )

    # Agent 可以通过 list_skills 查看可用 skills
    # 通过 get_skill_info 加载具体 skill 的指令
    result = await agent.run("用 data-engineering skill 帮我设计数据管道")
    print(result.content)

asyncio.run(main())
```

## 下一步

- [Agent 概念](../concepts/agent.md) — `enable_agentic_prompt` 与 Skills 的关系
- [CLI 终端](../getting-started/terminal.md) — CLI 中使用 Skills
- [Tools](../concepts/tools.md) — SkillTool API 详解
