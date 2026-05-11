# CLI 终端指南

Agentica CLI 是一个功能完整的 AI 编程助手终端，基于 `DeepAgent` product preset 构建。它内置了文件读写、代码执行、网页搜索、子任务委派等 20+ 工具，支持多轮对话、会话持久化、技能系统和 IDE 集成。

<img src="https://github.com/shibing624/agentica/raw/main/docs/assets/cli_snap.png" width="700" alt="CLI Screenshot" />

## 快速启动

```bash
# 交互模式（默认，开启 DeepAgent 产品预设）
agentica

# 单次查询，执行完直接退出
agentica --query "用 Python 写一个快速排序"

# 指定模型提供商和模型
agentica --model_provider zhipuai --model_name glm-4.7-flash
agentica --model_provider openai --model_name gpt-4o
agentica --model_provider deepseek --model_name deepseek-chat
agentica --model_provider ollama --model_name llama3.1
```

## 完整参数说明

```
agentica [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` / `-q` | str | -- | 单次查询，执行后退出 |
| `--model_provider` | str | `zhipuai` | 模型提供商（见下表） |
| `--model_name` | str | `glm-4.7-flash` | 模型名称 |
| `--base_url` | str | -- | 自定义 API 地址（代理/私有部署） |
| `--api_key` | str | -- | 直接传入 API Key（覆盖环境变量） |
| `--max_tokens` | int | -- | 最大输出 token 数 |
| `--temperature` | float | -- | 温度参数（0-2，越高越发散） |
| `--work_dir` | str | CWD | 工作目录（文件操作基准路径） |
| `--workspace` | str | `~/.agentica/workspace` | Workspace 持久化目录 |
| `--no-workspace` | flag | -- | 禁用 Workspace（不注入长期记忆） |
| `--tools` | list | -- | 额外启用的工具（追加到内置工具） |
| `--enable-skills` | flag | -- | 启用 Skills 系统 |
| `--debug` | int | `0` | 调试级别（1=启用，显示内部日志） |

### 支持的模型提供商

```bash
agentica --model_provider openai    --model_name gpt-4o
agentica --model_provider azure     --model_name gpt-4o
agentica --model_provider zhipuai   --model_name glm-4.7-flash    # 免费
agentica --model_provider deepseek  --model_name deepseek-chat
agentica --model_provider moonshot  --model_name moonshot-v1-128k
agentica --model_provider ark       --model_name doubao-1.5-pro-32k
agentica --model_provider ollama    --model_name llama3.1          # 本地，无需 API Key
```

## 内置工具

CLI 模式下，`DeepAgent` 自动装载以下工具（无需 `--tools` 指定）：

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件内容（支持分页，避免大文件撑爆上下文） |
| `write_file` | 创建或完整覆写文件 |
| `edit_file` | 精确字符串替换（比 write_file 更安全，适合小改动） |
| `multi_edit_file` | 批量编辑同一文件（原子操作，避免竞态） |
| `ls` | 列出目录内容 |
| `glob` | 文件模式匹配（`**/*.py`） |
| `grep` | 内容搜索（基于 ripgrep） |
| `execute` | 执行 Shell 命令（git、pytest、pip 等） |
| `web_search` | 网页搜索 |
| `fetch_url` | 抓取网页内容 |
| `write_todos` | 创建任务清单（追踪多步骤工作） |
| `task` | 启动子 Agent 处理复杂子任务 |
| `user_input` | 请求用户确认或输入（Human-in-the-loop） |
| `save_memory` | 保存记忆到 Workspace |
| `search_memory` | 检索 Workspace 中的历史记忆 |

### 追加额外工具

```bash
# 启用 DuckDuckGo 搜索（需要 pip install duckduckgo-search）
agentica --tools duckduckgo

# 启用多个工具
agentica --tools duckduckgo arxiv wikipedia

# 完整工具列表（100+ 工具）
agentica --help
```

可用工具名（`--tools` 参数值）：

```
cogvideo, cogview, dalle, image_analysis, ocr, video_analysis,
arxiv, baidu_search, dblp, duckduckgo, search_bocha, search_exa, search_serper, wikipedia,
browser, jina, newspaper, url_crawler,
calculator, code, shell, sql, weather, yfinance,
mcp, skill, ...
```

## 交互模式

启动后进入 Rich 渲染的交互式终端，支持 Markdown、代码高亮、工具调用展示。

### 文件引用：`@filename`

在消息中用 `@` 引用文件，文件内容会自动注入到上下文：

```
> @main.py 这段代码有什么性能问题？
> @README.md 把这个文档翻译成英文
> @tests/test_agent.py 为什么这个测试会失败？
> @/absolute/path/to/file.py 分析这个文件
```

支持路径补全：输入 `@` 后按 Tab 自动补全文件路径。支持相对路径和绝对路径。

### Shell 命令：`!command`

在消息开头用 `!` 直接执行 Shell 命令，结果显示在终端：

```
> !git status
> !pytest tests/ -v
> !ls -la
> !cat requirements.txt
```

### 多行输入

按 `Esc + Enter` 输入多行内容，适合粘贴代码块：

```
> def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
  这段代码有 bug 吗？
```

## 斜杠命令

在交互模式中，以 `/` 开头的输入触发内置命令：

### `/help`
显示所有可用命令列表和说明。

### `/tools`
列出当前 Agent 已装载的所有工具及其描述：
```
> /tools
  read_file    - Read a file from the filesystem
  write_file   - Write content to a file
  execute      - Execute shell commands
  ...
```

### `/skills`
列出当前加载的所有技能：
```
> /skills
  Loaded skills (3):
    code-review  - Code review skill
    paper-digest - Paper digest skill
    ...
```

### `/memory`
显示当前会话的消息历史（含工具调用摘要）：
```
> /memory
  Session: abc-123
  Messages: 12
  ...
```

### `/workspace`
显示 Workspace 状态（路径、记忆条数、用户 ID 等）：
```
> /workspace
  Path: ~/.agentica/workspace
  User: default
  Memory entries: 5
  Context files: AGENTS.md, PERSONA.md
```

### `/model [provider/model]`
查看当前模型，或切换到新模型：
```
> /model
  Current model: zhipuai/glm-4.7-flash

> /model openai/gpt-4o
  Switched to openai/gpt-4o

> /model deepseek-chat
  Switched to deepseek-chat (keep current provider)
```

### `/compact [instructions]`
手动触发上下文压缩，将当前对话历史摘要化以释放上下文空间：
```
> /compact
  Context compacted. Summary injected.

> /compact 重点保留关于 API 设计的讨论
  Context compacted with custom instructions.
```

!!! tip "自动压缩"
    上下文接近模型 context window 80% 时，CLI 会自动触发压缩，无需手动执行。

### `/newchat`
开启全新会话（清除消息历史，保留模型和工具配置）：
```
> /newchat
  Started new chat session.
```

### `/resume [session_id]`
恢复之前的会话（基于 Session Log JSONL 机制）：
```
> /resume
  Available sessions:
    abc-123  (2026-04-05 14:32, 45 messages)
    def-456  (2026-04-04 09:15, 12 messages)

> /resume abc-123
  Session resumed: abc-123 (45 messages loaded)
```

### `/clear` / `/reset`
清屏并重置当前会话（等同于 `/newchat` + 清除屏幕）。

### `/debug`
显示内部调试信息（当前 token 用量、模型配置、工具列表等），排查问题时使用。

### `/reload-skills`
从磁盘重新加载技能文件，适合开发技能时热更新：
```
> /reload-skills
  Skills reloaded: 3 skills loaded
```

### `/exit` / `/quit`
退出 CLI。等同于 `Ctrl+D`。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 中止当前响应（不退出 CLI） |
| `Ctrl+D` | 退出 CLI |
| `Tab` | 补全 `@filename` 路径 |
| `↑` / `↓` | 历史命令浏览 |
| `Ctrl+R` | 搜索历史命令 |
| `Esc + Enter` | 多行输入 |

历史命令持久化在 `~/.agentica/cli_history.txt`，跨会话保留。

## 流式输出与工具展示

CLI 实时展示 Agent 的每一个动作：

```
DeepAgent > 好的，我来分析这段代码。

  ✂️  edit_file app.py
      old: "def foo():"
      new: "def foo(x: int) -> str:"
      ✓ Done

  ⚡ execute python -c "import ast; ast.parse(open('app.py').read())"
      ✓ Syntax OK

分析完成。修改了第 42 行的函数签名...
```

- **内容流式输出** -- 打字机效果，实时显示 LLM 生成内容
- **工具调用展示** -- 显示工具名、参数摘要和执行结果
- **子任务进度** -- `task` 工具委派子 Agent 时显示进度条
- **推理过程** -- DeepSeek-R1 等推理模型的 `<think>` 内容折叠显示
- **Cost 统计** -- 每轮结束后显示 token 用量和估算费用

## Workspace 与长期记忆

CLI 启动时自动连接 Workspace（默认 `~/.agentica/workspace`），提供跨会话的记忆能力：

```bash
# 使用指定 Workspace 目录
agentica --workspace ./my-project-workspace

# 禁用 Workspace（纯无状态模式）
agentica --no-workspace
```

每次对话结束后，重要信息（用户偏好、项目上下文、反馈）通过 `save_memory` 工具持久化。
下次启动时，Agent 自动根据当前 query 检索相关记忆注入上下文。

详见 [Memory & Workspace](../concepts/memory.md)。

## Skills 系统

Skills 是 Markdown 定义的可复用指令包，可以给 Agent 注入专业领域的指导：

```bash
# 启用 Skills 系统
agentica --enable-skills
```

Skills 目录：
- `~/.agentica/skills/` -- 用户级全局 Skills
- `.agentica/skills/` -- 项目级 Skills（当前目录）
- `.claude/skills/` -- 兼容 Claude Code 的 Skills

每个 Skill 是一个包含 `SKILL.md` 的目录：

```markdown
---
name: code-review
description: 代码审查专家，专注安全性、性能和可读性
---

# Code Review Skill

你是资深代码审查专家...（详细指令）
```

在对话中通过提示词激活：
```
> 用 code-review skill 审查 @main.py
```

详见 [Skills 进阶](../advanced/skills.md)。

## 工作目录

`--work_dir` 参数设置文件操作的基准路径，影响 `read_file`、`write_file`、`execute` 等工具：

```bash
# 在项目目录下启动（推荐）
cd /path/to/my-project
agentica

# 显式指定工作目录
agentica --work_dir /path/to/my-project
```

当 `work_dir` 是 git 仓库时，System Prompt 自动注入 git 状态（当前分支、未提交变更、最近 commit），让 Agent 了解代码上下文。

## ACP 模式（IDE 集成）

启动 ACP (Agent Client Protocol) 服务器，与 Zed、JetBrains 等 IDE 集成：

```bash
agentica acp
```

IDE 插件通过 ACP 协议与 Agent 通信，实现：
- 在 IDE 侧边栏直接对话
- Agent 读取/编辑当前打开的文件
- 代码补全和重构建议

详见 [ACP 集成](../advanced/acp.md)。

## 高级用法示例

### 代码审查工作流

```bash
cd /path/to/project
agentica
> 审查最近的代码改动，重点关注安全问题
# Agent 自动执行：git diff → 分析变更 → 生成审查报告
```

### 文档生成

```bash
agentica --work_dir ./src
> 为 @agent.py 中的所有公共方法生成 docstring，风格参考 @docs/example.py
```

### 测试驱动开发

```bash
agentica
> @src/calculator.py 为这个模块写完整的单元测试，保存到 tests/test_calculator.py，然后运行确认通过
```

### 调试模式

```bash
# 开启调试日志，查看工具调用细节和 token 用量
agentica --debug 1
```

## 下一步

- [快速入门](quickstart.md) -- Agent 基础 API
- [Agent 核心概念](../concepts/agent.md) -- DeepAgent 深度解析
- [工具系统](../concepts/tools.md) -- 自定义工具开发
- [Memory & Workspace](../concepts/memory.md) -- 长期记忆机制
- [Skills 进阶](../advanced/skills.md) -- 技能系统
- [ACP 集成](../advanced/acp.md) -- IDE 集成协议
