# CLI 终端指南 (`agentica`)

`agentica` 是开箱即用的交互式 AI 编程与研究终端。它默认挂载了文件读写、代码搜索、Shell 执行、联网搜索、长期记忆与子任务委派等全套工具。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="700" alt="CLI Screenshot" />

---

## 一、快速启动

```bash
# 1. 在当前项目目录下进入交互模式（最常用）
cd /path/to/your-project
agentica

# 2. 单次提问，执行完直接退出（适合写进脚本）
agentica --query "用 Python 写一个快速排序"
agentica --query "检查当前目录的 git diff 有没有明显 bug" --print

# 3. 恢复上一次的会话（支持传入序号、会话名或 ID 前缀）
agentica resume a8c3f217

# 4. 临时指定某个已配置的模型 profile 启动
agentica --profile deepseek_pro
```

---

## 二、日常交互技巧

### 1. 用 `@文件名` 把文件带进对话

在输入框中敲 `@`，按 `Tab` 键即可自动补全当前目录下的文件路径：

```text
> @main.py 这段代码有什么性能问题？
> @tests/test_agent.py 为什么这个测试会失败？帮我修好并跑通
```

### 2. 多行输入与常用快捷键

| 快捷键 | 作用 |
|--------|------|
| `Esc + Enter` | 换行（适合直接粘贴多行代码或报错栈） |
| `Tab` | 补全 `@文件路径` 或 `/斜杠命令` |
| `Ctrl+C` | 中止当前正在生成的回复；输入框为空时连按两次退出 |
| `Ctrl+D` | 连按两次退出 CLI（第一次只在状态栏提示，0.8 秒内再按一次才离开；退出时会打印本次会话的恢复命令 `agentica resume <id>`） |
| `↑` / `↓` / `Ctrl+R` | 浏览或搜索历史输入 |

---

## 三、斜杠命令速查

在交互终端里输入 `/` 即可看到命令提示，按常用场景分类如下：

### 会话与上下文管理

| 命令 | 用途 |
|---|---|
| `/new`（或 `/clear`） | 开启一个全新会话（旧会话自动保存，随时可 `/resume`） |
| `/rename <名称>` | 给当前会话起个好记的名字（如 `/rename 登录页重构`） |
| `/resume [名称\|序号\|all]` | 恢复历史会话；`/resume all` 可跨本机所有项目查找并恢复会话 |
| `/compact` | 手动将会话切换到一个干净的新上下文窗口，并自动将当前进度整理进 `<session>.notes.md` |
| `/history [tools [轮次]]` | 查看当前会话的问答记录；加 `tools` 可在分页器中查看完整的工具调用参数与输出 |
| `/trace [轮次]` | 查看每轮对话的耗时、Token 用量与工具调用统计 |
| `/export [analysis\|messages] [路径]` | 导出当前会话的完整 JSONL 日志、分析报告或纯问答 JSON |

### 模型、权限与配置

| 命令 | 用途 |
|---|---|
| `/model` / `/model <profile>` | 查看所有模型配置，或将当前项目切换到指定 profile（`/model --clear` 恢复默认） |
| `/permissions [ask\|auto\|allow-all]` | 切换工具执行权限档位：`ask`（写文件/跑命令前询问）、`auto`（工作区内自动放行）、`allow-all`（全部放行） |
| `/config` / `/config set <k> <v>` | 查看或直接修改配置文件中的参数（如 `/config set compact_token_limit 300000`） |
| `/status` / `/usage` | 查看当前会话的模型、工作目录、日志路径与上下文 Token 占用明细 |
| `/debug [on\|off]` | 开关终端详细调试日志 |

### 长任务、多会话与后台进程

| 命令 | 用途 |
|---|---|
| `/goal <目标>` / `/subgoal <条件>` | 开启自动持续推进的长任务循环（详见 [长任务循环](../advanced/goals.md)） |
| `/steer <指令>` / `/queue <指令>` | 运行中途立刻纠偏当前轮次（`/steer`），或把新任务排进下一轮队列（`/queue`） |
| `/worktree [new\|merge\|remove] <名>` | 为任务创建独立的 Git 分支与目录，改完一键合回（详见 [Worktree](../multi-agent/worktrees.md)） |
| `/list-agents`（或 `/peers`） | 查看本机正在运行的其他 CLI 会话、所在分支与正在修改的文件 |
| `/send-message <会话名> <内容>` | 向本机另一个正在运行的终端会话发送指令（别名 `/send`） |
| `/ps` / `/stop <id\|all>` | 查看正在后台运行的命令或委托任务；用 `/stop <id>` 或 `/stop all` 停止它们 |

### 技能、子智能体与定时任务

| 命令 | 用途 |
|---|---|
| `/skills` / `/reload-skills` | 查看已加载的 `SKILL.md` 技能，或热重载磁盘上的技能文件 |
| `/extensions install <仓库或路径>` | 从 GitHub 仓库或本地目录安装外部技能包（如 `obra/superpowers`） |
| `/agents [create\|reload\|remove]` | 查看或创建自定义 Markdown Subagent（`.agentica/agents/*.md`） |
| `/cron` | 管理定时自动运行的 Agent 任务（详见 [定时任务](../guides/cron_scheduler.md)） |

---

## 四、多任务与多会话协作

在 CLI 中，你可以根据任务大小选择四种协作方式：

| 方式 | 适用场景 | 怎么触发 |
|---|---|---|
| **`task`（进程内子任务）** | 查代码、搜资料、读多个模块等**短平快的只读调查**（默认使用便宜的 `auxiliary_model`，支持一条消息并行跑多个） | Agent 自动调用，或你在话里说「用 subagent 帮我查一下…」 |
| **`delegate`（后台独立进程）** | 需要独立上下文窗口、换到另一个仓库目录、或要动手改文件的**大型独立任务** | 对 Agent 说「把 service-a 的升级 delegate 到后台跑」；用 `/ps` 查看进度、`/stop <id>` 停止 |
| **`acp_agent`（调用外部 Agent）** | 在会话里直接调用本机安装的 **Codex (`codex`)、Claude Code (`claude_code`) 或 OpenCode (`opencode`)** 帮你干活或做交叉审查 | 对 Agent 说「让 codex 审查一下这次提交」 |
| **Peer 消息（跨终端通信）** | 你自己开了两个终端窗口，想让两边互通结论或指令 | 你敲 `/send-message <对方会话名> <话>`，或让 Agent 调 `list_agents` / `send_message` |

---

## 五、系统沙箱（`--sandbox workspace-write`）

如果你希望限制 `execute` 工具能读写的目录和网络访问，可以开启系统级沙箱（macOS 基于 Seatbelt `sandbox-exec`，Linux 基于 bubblewrap `bwrap`）：

```bash
agentica --sandbox workspace-write
```

也可以写进 `~/.agentica/config.yaml` 默认开启：

```yaml
settings:
  sandbox: workspace-write
  sandbox_network_access: false   # 设为 true 则允许沙箱内命令联网
```

开启后：
- **写入限制**：只允许写当前工作目录（及同仓库的 worktree）、系统临时目录与常用包缓存目录；`.git/hooks` 与 `.git/config` 保护为只读。
- **读取限制**：自动屏蔽 `~/.ssh`、`~/.aws`、`~/.gnupg`、`.env` 等敏感路径。
- **需要临时突破沙箱时**：如果某条命令确实需要写工作区外或联网，在 `ask` / `auto` 权限档下会弹出审批卡让你确认。

---

## 六、命令行启动参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query` / `-q` | — | 单次提问，执行完直接退出 |
| `--print` | 关 | 配合 `--query` 使用，只向标准输出打印最终回答正文（方便脚本管道调用） |
| `--profile` | — | 本次会话使用 `~/.agentica/config.yaml` 中的指定 profile |
| `--model_provider` / `--model_name` | `zhipuai` / `glm-4.7-flash` | 直接指定提供商与模型名 |
| `--work_dir` | 当前目录 | 文件读写与命令执行的基准目录 |
| `--worktree <名称>` | — | 启动在独立的 `.agentica/worktrees/<名称>` 分支目录中 |
| `--permissions` | `auto` | 初始权限档位：`ask` / `auto` / `allow-all` |
| `--sandbox` | `off` | 设为 `workspace-write` 开启系统级沙箱 |
| `--compact-token-limit` | — | 设置当前会话的上下文压缩工作上限 |
| `--no-evict` / `--no-auto-compact` | — | 关闭 80% 工具结果淘汰或 95% 自动换窗 |
| `--tools` | — | 追加额外工具（如 `--tools duckduckgo arxiv wikipedia`） |
| `--enable-diagnostics` | 开 | 编辑文件后自动运行 Pyright/LSP 语法诊断（`--no-enable-diagnostics` 关闭） |
| `--debug` | `0` | 设为 `1` 打印详细调试日志 |

---

## 下一步

- [Web、桌面端与微信/IM](../advanced/gateway.md) — 在网页、桌面端或手机微信里使用 Agentica
- [长任务循环 (/goal)](../advanced/goals.md) — 让 Agent 自动迭代直到任务完成
- [统一配置文件 (config.yaml)](../guides/config.md) — 管理多个模型 Profile 与全局设置
