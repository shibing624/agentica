# CLI 终端指南

Agentica CLI 是一个功能完整的 AI 编程助手终端，基于 `DeepAgent` product preset 构建。它内置了文件读写、代码执行、网页搜索、子任务委派等工具，支持多轮对话、会话持久化、技能系统和 IDE 集成。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="700" alt="CLI Screenshot" />

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
| `--profile` | str | -- | 本次会话改用某个已保存的 profile（不写 config.yaml）；换 provider 只能用它 |
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
| `--enable-diagnostics` | flag | 开 | 编辑后把 LSP/Pyright 诊断附到 `write_file` / `apply_patch` 结果上（`--no-enable-diagnostics` 关掉） |
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

### OpenAI Responses API

Responses API 通过配置文件选择，不需要改变 CLI 的启动方式：

```yaml
# ~/.agentica/config.yaml
active_profile: responses

profiles:
  responses:
    model_provider: openai
    model_name: gpt-5.6-sol
    wire_api: responses
    reasoning: high
    max_tokens: 4096
```

`wire_api: responses` 只能与 `model_provider: openai` 配合，推理强度使用 `reasoning`，不能使用 Chat Completions 的 `reasoning_effort`。完整参数和辅助模型配置见 [OpenAI Responses API](../guides/openai-responses.md)。

### 临时换一个 profile

```bash
agentica --profile responses          # 这一次用 responses，config.yaml 不动
```

`--profile` 是**换 provider 的唯一命令行方式**：`--model_name` 只在当前 endpoint 内换模型，base_url 和 key 仍然来自当前 profile。名字不存在会直接报错并列出可用的，不会悄悄退回默认。想永久切换用会话里的 `/model <profile>`（写项目级覆盖）。

状态栏和 `/status` 显示的 profile 名以**本次会话实际用的**为准：`--profile` 显示被指定的那个（标 `flag`），而当 `--model_name` 覆盖掉了 profile 的模型时不再显示 profile 名——此时没有哪个 profile 能描述正在跑的东西。

## 内置工具

CLI 模式下，`DeepAgent` 自动装载以下工具（无需 `--tools` 指定）：

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件内容（支持分页 / `tail`；空文件返回 `File is empty: …`） |
| `write_file` | 创建或完整覆写文件（长报告可写成 HTML，用户自己打开） |
| `apply_patch` | 一次补丁新增、更新或删除多个文件（上下文精确匹配；编辑后可附 Pyright 诊断） |
| `glob` | 文件模式匹配（`**/*.py`） |
| `grep` | 内容搜索（`pattern` / `path` / `include` / `limit`，基于 ripgrep） |
| `execute` | 执行 Shell 命令（git、pytest、pip 等）；非零退出只报 exit code |
| `web_search` | 网页搜索 |
| `fetch_url` | 抓取网页内容 |
| `write_todos` | 创建任务清单（追踪多步骤工作） |
| `task` | 启动子 Agent 处理复杂子任务 |
| `ask_user_question` | 请求用户确认或输入（Human-in-the-loop） |
| `save_memory` | 保存记忆到 Workspace |
| `search_memory` | 检索 Workspace 中的历史记忆 |

### 追加额外工具

```bash
# 启用 DuckDuckGo 搜索
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
calculator, code, sql, weather, yfinance,
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

### `/agents`

查看和管理 Markdown 定义的 Subagent：

```text
> /agents
> /agents create data-analyst
> /agents reload
> /agents remove data-analyst
```

`create` 会生成项目级 `.agentica/agents/<name>.md`。编辑其中的 system prompt、工具权限、模型层级和预算后，执行 `/agents reload` 热重载。用户级 `~/.agentica/agents/*.md`、覆盖顺序及完整示例见 [Subagent 文档](../multi-agent/subagent.md)。

### `/list-agents` / `/peers`

列出本机其他正在跑的 CLI 会话（跨终端消息）。每条包含：

- addressable name（如 `nlp-5f`，`send_message` 默认用这个）
- peer id、`session_id`（也可用前缀寻址）
- cwd，以及带 hash 后缀的 `project` 存储目录（如 `-apdcephfs-...-nlp-6115aec9`）
- `session_log`（`<project>/<session_id>.jsonl`）、CLI `log_file`（如 `~/.agentica/logs/20260809-80403.log`）、`workspace` / `memory`（`MEMORY.md`）路径
- 当前 working on

消息本身仍是短纯文本；列出路径是为了需要时自行去读对方 transcript / 运行时日志 / 长期记忆。`log_file` 通常比翻 conversation 更快定位另一会话刚发生的错误与工具痕迹。
自己发消息用 `/send-message <name|id> <text>`（别名 `/send`）。
注入到对方会话时，回复地址是对方的短名字（如 `agentica-73`），不是 opaque 的 peer id；本会话自己的短名字可在 `/status` 的 `Peer:` 行确认。`/send-message`（用户转发）在收件端按「用户亲口说的」采纳；agent 发的消息即使正文自称用户决定，也不构成授权。

这个通道用来传递结论、交接信息，不是让两个 agent 讨论细节的地方。刹车不是「一段对话最多几条」这种硬计数——那会在一次正常的多轮交接中途把消息掐掉，而被拒的往往正是你刚吩咐的那句。真正不该发生的是**把同一件事再说一遍**：同一段文字（忽略大小写和空白差异）在 5 分钟内重复发给同一个对端会被拒绝；同一对端 5 分钟内超过 20 条也会被拒绝。两个限制都只按「对端」分别计算，同时和三个会话协作不受影响。

这两个限制只约束无人值守的循环，不约束你：你在本终端**敲任何一行**（包括「给 temp-30 发条消息」这类指令）都会立即清空计数，对端用 `/send-message` 转发过来的消息同样清空。所以不会出现「你让 agent 发消息、却被限制拦住」的情况。

### `/worktree`

把**当前这个** CLI 会话绑到本仓库的一个任务 worktree（目录 + `wt/<任务>` 分支），避免几个会话抢同一个 checkout。

```text
> /worktree
> /worktree use gateway-peers
> /worktree merge
> /worktree remove
```

启动时也可以 `agentica --worktree gateway-peers`。模型侧用同名 `worktree` 工具（用法见内置 `worktree` skill），不要让它 `git worktree add` 再 `cd`。详见 [worktrees](../multi-agent/worktrees.md)。

### 委托任务给另一个 CLI（`delegate`）

一个交互会话可以把**整块**工作丢给另一个 agentica 进程去做：它有自己的上下文窗口、自己的模型、自己的工作目录，做完把结论交回来。

这和进程内的 `task`（subagent）、跨终端的 peer 消息不是一回事：

| | `task` | `delegate` | peer（`send_message`） |
|---|---|---|---|
| 关系 | 父 → 子 | 父 → 子 | 对等会话 |
| 进程 | **同进程** | **新 OS 进程**（`agentica --query --print`） | 用户自己开的两个交互 CLI |
| 模型 | 默认 auxiliary（便宜） | 默认跟主会话 | 各自会话的模型 |
| 并行 | 一条消息里多个 `task` | 最多同时几个 | 开几个会话 |
| 结果 | 工具返回值立刻回来 | 后台跑完再回执 / `wait` | 对方自己决定是否行动 |
| 出现在 `list_agents` | 否 | **否**（刻意） | **是** |

```text
> 把 ../service-a 和 ../service-b 这两个仓库分别做一遍依赖升级，做完告诉我结果

  🔧 delegate
      label='upgrade service-a'
      把 ../service-a 做一遍依赖升级……（全文，不截断）
  🔧 delegate
      label='upgrade service-b'
      ……
  …主会话继续做自己的事…

✓ Delegated task "upgrade service-a" finished in 04:12 (exit 0)
```

CLI 对 `task` / `delegate` / `send_message` 的调用行会**完整展示**任务正文（含换行），不再用 `...` 省略——这是你审计「派了什么活」的依据。

要点：

- **不阻塞**。`delegate` 立刻返回，任务在后台跑；主 agent 可以继续干别的。完成后报告会自动送回这轮对话（和 `execute(background=True)` 同一套机制），需要马上拿结果就 `wait(id="term_1")`。
- **最多同时 3 个**，第 4 个会被拒绝并告诉它去 `wait` 哪一个。用 `/ps` 查看、`/stop <id>` 停掉。
- **权限跟着父会话走**：派出去的那一刻你是什么模式（`auto` / `allow-all`），子进程就是什么模式；`ask` 模式下这个工具根本不出现。
- **模型默认继承**，也可以指定：`model="zhipuai/glm-4.7-flash"`（跨 provider）或只给模型名（沿用当前 provider）。API key 不会出现在命令行上，子进程自己读 `config.yaml` / 环境变量。
- **只有一层**。被委托出去的会话拿不到 `delegate` 工具，不会再往下派。
- **不出现在 `list_agents` 里**。它是一次性 `--query` 进程，不是一个终端会话；委托是父子关系并且有返回值，peer 消息是平级会话之间说话，两者刻意分开。
- 子进程用的是 `agentica --query "..." --print`，`--print` 只把最终回答写到 stdout（没有 banner、没有日志），你也可以在脚本里直接这么用；失败时退出码非 0。
- **小活用 `task`，而且可以并行。** 读代码、搜仓库、查资料——一条消息里发多个 `task` 就会一起跑，比 `delegate` 便宜得多。不要因为「要并行」就上 `delegate`。只有工作大到值得独立 context（或必须换目录、要写文件）时才 `delegate`。

### `/ps` 与 `/stop`

`/ps` 列出后台终端命令（`execute(background=True)`、`delegate`）和后台 agent 任务（`/background`）。
`/stop` 停掉它们，**必须显式给目标**：

- `Ctrl+C` —— 中止你正在等的这一轮（空输入时连按两次退出）。当前这一轮只由它负责。
- `/stop <id>`（也接受 `pid` 或 `/ps` 里的 `#n`）—— 停掉指定的那个后台任务。
- `/stop all` —— 停掉全部后台任务。
- `/stop`（不带参数）—— 只打印用法并列出可选目标，**什么都不停**。

两条边界是刻意的：

1. **`/stop` 不碰当前这一轮。** Ctrl+C 做的事比 `Agent.cancel()` 多：唤醒卡在 `ask_user_question` 上的线程、把常驻 goal 置为 `paused`（否则轮次结束的钩子会立刻续跑一轮）、连按第二次升级为强制退出。而且 agent 正等你回答问题时，输入框里的任何一行都会被当成**答案**提交，`/stop` 根本到不了命令处理器——最需要停的时候它恰好不可用。
2. **空参数的 `/stop` 不再默认停掉全部。** 它和 `/stop <id>` 只差一个 token，又常常在别的后台任务正跑时被敲下；"全停"要用 `/stop all` 说出口。

### `/config`

显示模型、终端和工作区配置。其中 `Project Dir` 是该 cwd 在
`~/.agentica/projects/<user>/` 下的唯一 hash 目录，方便到后台定位 session 文件。

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
  Context files: AGENTS.md
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
    占用升高时会先做免费的 Layer 1 淘汰（约 70% 起）；接近窗口上限时再跑 Layer 2
    （provider-native compact 或本地 LLM summary）。`/compact` 始终强制 Layer 2。

使用 `OpenAIResponses` 且 endpoint 支持 `/responses/compact` 时，`/compact` 会优先生成 provider-native checkpoint，并保留可跨 provider 的普通 transcript。原生请求失败时回退到本地 LLM summary；失败则保持对话不变（已无 rule-based 回退）。

### `/new` / `/newchat`
开启全新会话（清除消息历史，保留模型和工具配置）。切换前会显示当前会话的
运行时间、Token 用量和可直接执行的恢复命令：

```text
> /new
Worked for 15m 05s

Token usage: total=4,300 input=4,000 (+ 800 cached) output=300
To continue this session, run agentica resume c1392649-f07d-4f05-808b-f852c3190236
```

`/newchat` 是 `/new` 的别名。

### `/rename <name>`
为当前会话设置易识别的名称。名称会持久化，进程异常退出后仍会显示在 `/resume` 列表中：
```
> /rename 前端视觉问题排查
  Renamed current session to 前端视觉问题排查
```

### `/resume [number|name|id-prefix|all]`
按序号、名称或 ID 前缀恢复之前的会话（基于 Session Log JSONL 机制）：
```
> /resume
  Available sessions:
    1. a8c3f217  2026-07-24 10:20  (48KB, 23 turns)
       前端视觉问题排查
    2. 91be026d  2026-07-23 18:05  (12KB, 6 turns)
       修复登录超时

> /resume 前端视觉问题排查
  Resumed transcript: 前端视觉问题排查 (a8c3f217...)
  Conversation view - 48 tool results (62.4K chars) collapsed - /history tools [run] for details
  ...
  Resumed session: 前端视觉问题排查 (a8c3f217...) - restored 6 runs into context;
  showing conversation only (48 tool results collapsed)
```

恢复时，完整 Session Log 仍会重建到模型上下文中；终端默认只回放用户消息、Agent
正文和每轮工具统计，成功的 tool result 不再写入 scrollback。失败结果最多显示 3 条
单行摘要。

也可以退出 CLI 后直接从 shell 恢复（ID 前缀即可）：

```bash
agentica resume c1392649
```

#### 跨目录恢复

Session Log 按项目目录（work_dir）分区存放，但 `/resume <id>` 和 `agentica resume <id>`
会先在当前项目找，找不到再搜索该用户的全部项目，所以在任何目录都能按 ID 恢复。

当会话属于另一个目录时，会询问在哪个目录继续工作：

```
? This session was started in another directory. Choose the working directory to resume it in.
    1. Use session directory (/Users/me/Codes/agentica)
    2. Use current directory (/Users/me/temp)
    3. Always use session directory
    4. Always use current directory
```

- 选 1/2 只对本次生效；选 3/4 会写入 `~/.agentica/config.yaml` 的 `settings.resume_cwd`
  （`session` / `current`），之后不再询问。想恢复询问就把它改回 `ask` 或删除该行。
- 无论选哪个，transcript 始终继续追加到它原本所在的项目目录，不会分裂成两个文件。
- 会话目录已被删除时不再询问，直接留在当前目录并提示。

`/resume all` 列出所有项目的会话（附带各自的目录），列表里的序号可直接用于随后的
`/resume <n>`：

```
> /resume all
  Sessions across all projects:
    1. a8c3f217  2026-07-24 10:20  (48KB, 23 turns)
       > 修复登录超时
       /Users/me/Codes/agentica
```

#### 换 provider 恢复

Session Log 里的 tool 轮次统一按 OpenAI 线格式存放，Anthropic 的 `/v1/messages`
无法回放这种形状。用 `anthropic/*` 恢复（或 fork）一个会话时，历史会自动降级为纯粹的
问答文本：之前的提问和回答都在，工具调用和工具结果不跟过去。同一个 provider 内恢复
不受影响，tool 历史照旧完整回放。

### `/history [tools [run-number]]`

`/history` 使用与 `/resume` 相同的紧凑对话视图。需要检查完整工具参数和结果时，
使用 `/history tools` 在 pager 中查看整个 session，或指定从 1 开始的 run 序号：

```text
> /history tools 3
  # 在 pager 中打开第 3 轮的完整 tool calls / tool results
```

完整工具记录只在 pager 中显示，不会重新灌入终端 scrollback。

### `/trace [n]`

Session JSONL 的读时分析，和 Web 对话标题旁「查看轨迹」同一套 `analyze_entries()`。

```text
> /trace
  Trace  my-session (a8c3f217-…)
    File: ~/.agentica/projects/…/a8c3f217-….jsonl  (48,102 B)
    Totals: 4 rounds · 12.3K in / 2.1K out · $0.12 · 3 tools · 45.2s
      1. 修复登录超时
           12.1s  8.2K/1.1K  $0.04  3 tools

> /trace 1
  Round 1: 修复登录超时
  [user] 修复登录超时
  [tool_call] read_file app.py
  …
```

`/trace export [path]` 等价于 `/export jsonl`。`/status` 的 **Session log** 行是这份 jsonl 的路径；**Debug log** 是进程 `~/.agentica/logs/*.log`，两回事。

### `/export` / `/save`

默认拷贝完整 session JSONL（对话 + event 轨迹）。旧的「只要对话、不要工具正文」瘦 JSON 改走 `/export messages`：

```text
> /export                         # ./<session_id>.jsonl
> /export analysis out.json       # 与 Web /traces 同一份分析 JSON
> /export messages chat.json      # 旧行为：role/content 列表
```

### `/clear` / `/reset`
清屏并重置当前会话（等同于 `/newchat` + 清除屏幕）。

### `/debug [on|off]`
运行时开关 verbose 调试日志，等价于启动时的 `--debug`：打开后 DEBUG 级日志打到终端
（文件日志本来就有），subagent 的工具输出从下一轮起切换为 verbose 形式。不带参数为
翻转当前状态。会话本身的信息（模型、token、工具数）看 `/status`。

```text
> /debug
  Debug logging: ON
> /debug off
  Debug logging: OFF
```

### `/reload-skills`
从磁盘重新加载技能文件，适合开发技能时热更新：
```
> /reload-skills
  Skills reloaded: 3 skills loaded
```

### `/exit` / `/quit`
退出 CLI。等同于 `Ctrl+D`，退出前会打印当前 session 的运行时间、Token 用量和可复制的 `agentica resume <session-id>` 恢复命令。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 中止当前响应；空输入时连续按两次退出 |
| `Ctrl+D` | 退出 CLI，并显示恢复命令 |
| `Tab` | 补全 `@filename` 路径 |
| `↑` / `↓` | 历史命令浏览 |
| `Ctrl+R` | 搜索历史命令 |
| `Esc + Enter` | 多行输入 |

历史命令持久化在 `~/.agentica/cli_history.txt`，跨会话保留。

## 流式输出与工具展示

CLI 实时展示 Agent 的每一个动作：

```
DeepAgent > 好的，我来分析这段代码。

  ✎  apply_patch app.py
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
