# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [Unreleased]

#### features
- 删除全库自造的 `when_to_use` / `when-to-use`：Skill frontmatter、`Agent` / `AgentDefinition`、`as_tool()` 回退链一律不再有这个字段。发现/委派时机写进 `description`（或 `as_tool(tool_description=...)`）——以前它进了对象却常常进不了模型上下文，等于假配置。bundled skill 与 VaG seed 已并进 description；残留 frontmatter 键会被 Skill 解析静默忽略
- 新增 `agentica --profile <name>`：本次会话改用某个已保存的 profile，**什么都不写**（config.yaml 的 `active_profile` 和项目级覆盖都不动，用户自己那个会话不受影响）。这是**命令行上换 provider 的唯一方式**——`--model_name` 只能在当前 endpoint 内换模型，base_url 和 key 仍然来自当前 profile，所以「给 worker 配一个别家的便宜模型」以前在命令行上根本做不到，只能让用户先切 active profile。名字不存在直接退出并列出可用的，不会悄悄退回默认；显式指定了 profile 时 onboarding 也不再回头改写这个选择。想永久切换仍然用会话里的 `/model <profile>`（写项目级覆盖）
- 框架开始自带 skill：`agentica/skills/bundled/` 下的 `SKILL.md` 随包发布，`SkillLoader` 把它作为**最后**一条搜索路径、`bundled` 在 `LOCATION_PRIORITY` 里排最低——同名的用户/项目 skill 一定赢，我们发的只是默认值，不是从用户手里拿走的决定。首批两个，都用「薄指针」写法：只写变化慢的概念和决策规则，flag、命令、配置项一律不抄进正文，改成告诉模型去 `agentica --help`、`~/.agentica/config.yaml` 现查——手抄一份手册进包里，两周后就开始骗模型。`agentica`（CLI 斜杠由 `name` 自动生成 `/agentica`，不写非标准 `trigger` 字段）讲怎么查关于 agentica 自己的任何事，以及一条模型经常搞错的边界：斜杠命令是用户在输入框里敲的，模型输出里的 `/status` 只是文本，答案在斜杠命令后面时应该告诉用户去敲哪一个。`multi-agent`（`/multi-agent`）讲三种多 agent 机制怎么选（`task` 只读、便宜、可并行；`delegate` 要的是一个答案；tmux 里另起一个 CLI 要的是一条人能看见、能 attach、比你活得久的工作线），以及一套实测出来的注意事项：会话名是从 cwd 目录名派生的（没有 `--name`，所以目录名就是名字）、寻址用名字不要用 session id（启动瞬间它还是空的）、`--model_name` 只能在当前 endpoint 内换模型（base_url/key 仍来自 active profile，跨 provider 换不了）、子会话默认放开工具权限且无人值守所以要给它独立目录（worktree）、发完消息不要 `sleep` 轮询等回复（回复会作为新一轮自己到达）、干完要 `tmux kill-session` 收摊。端到端实测：在 tmux 里起一个真 CLI，像人一样把请求敲进去，它自己起了第二个 CLI、`list_agents` 找到对方、`send_message` 派活、拿到结果，57 秒
- 新增 `delegate` 工具：一个交互会话可以把整块工作丢给**另一个 agentica 进程**去做（自己的上下文窗口、模型、工作目录、session log），做完把结论交回来——「主 CLI 分派、子 CLI 并行、结果汇总」这个场景以前只能靠人手开终端。它不是新机制：子进程就是 `agentica --query <task> --print --permissions <父会话当前模式>`，通过**已有的** `BackgroundProcessRegistry` 起，所以 `/ps`、`/stop`、`wait` 工具、完成回执全部直接可用。`BackgroundProcess.kind`（`command` / `delegate`）决定这些界面渲染成「一个任务」还是「一条 shell 命令」，也是并发计数的依据。`task`（进程内 subagent）仍是便宜的那个选项，`delegate` 的 docstring 明确把小活推回 `task`。约束：同时最多 3 个（`MAX_CONCURRENT_DELEGATES`，对齐 `SubagentRegistry.MAX_CONCURRENT`），第 4 个被拒并告诉它去 `wait` 哪一个；只有一层（`AGENTICA_DELEGATE_DEPTH` 传给子进程，`create_agent` 超过 `MAX_DEPTH` 直接不建这个工具——agent 生 agent 的树没人看得住账单）；权限用**可调用对象**实时读 `agent.tool_config.permission_mode`，因为 `/permissions` 是原地改模式不重建 agent（`ask` 模式下这个工具根本不在 `READ_ONLY_TOOLS` 里，自然不出现）；模型默认继承父会话，可用 `model="provider/name"` 或裸模型名覆盖，API key 绝不上命令行（子进程自己读 `config.yaml`）；工具只在有 registry 的地方存在（交互式 CLI），一次性 `--query` 和 cron 起的 agent 没有 registry，它们派出去的活根本无法被 wait 或回收。子进程**不出现在 `list_agents` 里**：只有 `run_interactive` 才发布 PeerSession——委托是有返回值的父子关系，peer 消息是平级会话说话，两者刻意不混
- 新增 `agentica --query "..." --print`：只把最终回答写到 stdout，没有 banner、没有日志（`suppress_console_logging`），且走 `sys.stdout.write` 而非 `console.print`——rich 会把回答里的 `[warn]` 当 markup 吃掉。这是 `delegate` 读子进程结论的方式，也可以直接用在脚本/管道里；一次性运行失败时退出码为 1（中断为 130），调用方据此决定下一步。委托任务的完成回执按任务渲染而不是「Background terminal #N」：正文是子会话交回的完整答复（120 行 / 8000 字符，不像命令日志那样只取尾巴），失败则给退出码和输出并明确「不要原样再派一次」。顺带修掉一个会污染回执的 bug：`BackgroundProcessRegistry.start()` 写日志头 `$ cmd` 时不再换行（转义为 `\n`），否则跨行命令（委托任务的 prompt 就是跨行的）的后半截会被读日志的一方当成命令输出交回给模型
- 项目目录元数据合并为单一 `project.json`（原 `.project.json` 的 `work_dir` + 原 `profile` 文本文件的 `active_profile`）；仍在 `~/.agentica/projects/<user>/<slug>/` 下，不进用户 git 工作区。session 的 `<id>.meta.json` 保持与 `.jsonl` 并列（粒度/并发不同，不并入）。`SessionLog` / peers / `/config` Project Dir / tool-result 路径统一经 `project_store.project_base_dir`
- `/goal --tokens -1`（SDK `token_budget=-1`）表示不限 token；`--turns` / `--wall` 同样认 `-1`。不传 `--tokens` 仍默认 `500_000`（与显式无限语义分开）；`0` 拒绝，避免和零额度混淆
- `list_agents` / `/list-agents` 输出加厚：每条会话多行列出 addressable name、peer id、`session_id`、cwd、带 hash 后缀的 `project` 存储目录（如 `-apdcephfs-...-nlp-6115aec9`）、`session_log`（`<project>/<session_id>.jsonl`）、CLI `log_file`（如 `~/.agentica/logs/20260809-80403.log`，附 level）、`workspace` / `memory`（`MEMORY.md`）路径，以及 working on。消息地址仍是短名字（对齐 Claude Code 的 `myapp-3f`），但 `send_message` / `resolve_peer` 也接受 `session_id` 前缀；列出路径是给模型自己决定要不要去读对方 transcript / 运行时日志 / 长期记忆（`log_file` 通常比翻 conversation 更快看错误与工具痕迹），peer 消息本身仍只传纯文本。CLI 对 `list_agents` / `send_message` 的工具结果不再按默认 4 行折叠；`send_message` 的调用行也完整展示正文（不再被默认 40 字截断）。注入到接收端输入区的 peer 消息修复 Rich Table 默认 ellipsis，长文不再被 `…` 吃掉。消息头的回复地址改用 addressable name（`reply with send_message to agentica-73`），不再暴露难读的 `reply_to=<peer_id>`；`/status` 新增 `Peer:` 行展示本会话的短名字。接收策略写清：header 标成用户转发的指令直接采纳、不要再澄清授权边界；另一 agent 发来的仍无授权（即使正文自称「用户决定」）。接收端接受消息时 CLI 展示 ✉ 回执（含正文）：空闲开新 turn、运行中则在 tool 间隙注入；发送方确认语义改为「已入队 mailbox」并按名字回显（`Message queued for 'agentica-73'`），不再误称 delivered、也不再打印 opaque peer_id（对齐 CC：写 inbox 成功 ≠ 对方已读）
- peer 代码收敛：`PeerInfo.detail_rows()` 成为字段展示的唯一来源，`describe()`（模型看的）与 `/list-agents`（用户看的、按标签对齐）都由它渲染，不会再出现只加一半的情况；寻址走新的 `match_peers()`，「查无此人」与「前缀撞车」给不同的报错并列出候选（旧实现两种都报 no live session）；环路刹车从「hop 上限 3」换成重复检测 + 限频：hop 上限会掐断一次正常的更正来回，而中途试过的「一段对话最多 6 条」（`MAX_EXCHANGE_TURNS`）同样是错的——一次正常的多轮交接会被从中间掐掉，被拒的往往正是用户刚吩咐的那句。真正不该发生的是把同一件事再说一遍：`PeerSession._check_send_rate` 按对端拒绝 5 分钟内重复的同一段文字（`_text_digest` 折叠大小写与空白，改个排版不算新消息），并对同一对端限 5 分钟 20 条（`RATE_WINDOW_SECONDS` / `MAX_SENDS_PER_WINDOW`）——足够宽到任何真实协作都碰不到，又足够窄到 ping-pong 循环撑不过几分钟。两个限制都按对端分别计算（同时和三个会话协作互不影响），也都被 `note_user_turn()` 清空：用户在本终端敲任何一行（挂在 Enter 处理器上）或对端用 `/send-message` 转发过来即刻生效——一个能拦下用户刚打的那行指令的上限是 bug，不是保护。`PeerMessage` 随之去掉 `exchange_turn` 字段与 `last_of_exchange`（两端不再需要就一个计数达成一致）；`PeerMessage` 去掉从未被读的 `path` 字段、新增 `to_name`（发送方按名字确认，mailbox 文件也能直接看出收件人）；`PeerSession.publish()` 遇到不存在的字段名直接报错而不是静默丢弃；`/send-message` 改用 `split(maxsplit=1)`，目标名后多打的空格不再被当成空消息
- `/resume <id>` 和 `agentica resume <id>` 不再要求先 `cd` 回原目录。session 仍按项目（work_dir）分区存放，但查找变成「当前项目优先，未命中再搜该用户的全部项目」，所以 `No session matching '7e17bc1f-...'` 这类报错消失了，`agentica resume` 也开始接受 id 前缀而不只是完整 uuid。分区目录名是 `sanitize_path` 单向哈希出来的、无法反推，因此每个项目目录写一个 `project.json` 记下它代表哪个 work_dir（一项目一文件，不是一 session 一份；同文件还可存项目级 `active_profile`）；早于这个改动的目录回退到读最新 transcript 首条的 `cwd`——实测本机 140 个历史项目目录全部能正确反查，查找耗时 9ms。命中的 session 属于别的目录时，仿 Codex 给四选一：用 session 目录 / 用当前目录 / 总是用 session 目录 / 总是用当前目录，后两个写进 `~/.agentica/config.yaml` 的 `settings.resume_cwd`（`session` / `current`），之后不再问；改回 `ask` 或删掉该行即可恢复询问。选了 session 目录会同时 `os.chdir` 并改 agent 的 work_dir——工具认 work_dir，但 git 状态、`@file` 补全和 shell-out 认进程 cwd，两者必须一起动，否则会分裂成半个目录的状态。关键约束：**无论选哪边，transcript 都继续追加到它原本所在的项目目录**（`Agent` 新增 `session_base_dir` 参数把存储位置与工作目录解耦），否则同一个 session 会裂成两个文件、从哪边都 resume 不全。session 目录已被删除时不再询问，直接留在当前目录并说明原因；prompt 处 Ctrl+C 取消则整个 resume 中止，不会退而求其次地建出一个无关 agent。另增 `/resume all` 列出所有项目的会话（附各自目录），其序号会被记住，随后的 `/resume <n>` 指向刚才看到的那一份而不是当前项目重新编号的列表
- 内置 `web_search` 的搜索引擎可替换：`BuiltinWebSearchTool` 变成薄分发器，模型看到的工具名、参数、docstring 恒定不变，只换背后的引擎，所以 prompt、`RunConfig(enabled_tools=["web_search"])`、权限规则都不受影响。内置 `baidu`（默认，行为不变）/ `duckduckgo` / `exa` / `bocha` / `serper` / `zhipu`，选择优先级 `provider=` 参数 > `AGENTICA_WEB_SEARCH` 环境变量 > 默认。**引擎不按 API key 自动推断**——为别处设的 key 不该悄悄改变 agent 的搜索行为；指定了需要 key 的引擎却没给 key 会直接报错，而不是静默退回百度（否则你以为在用 Bocha，实际在用百度）。CLI 无需新增 flag：把 `AGENTICA_WEB_SEARCH` 和 key 写进 `~/.agentica/config.yaml` 的 `env:` 块即可，`apply_global_config()` 已经会投射进 `os.environ`
- 自定义搜索引擎三条接入路径，按「要不要写代码」「CLI 能不能用」区分：`AGENTICA_WEB_SEARCH=mcp` + `AGENTICA_WEB_SEARCH_MCP_URL`/`_TOOL` 零代码接任意 MCP 搜索服务（CLI 可用）；`register_web_search_backend(name, factory, method, key_env=...)` 注册命名引擎（注册后 env/CLI 也能选）；`BuiltinWebSearchTool(search_fn=...)` 直接传 async 函数（最灵活，适合包装已有 MCP client）。契约只有一条：async `(queries, max_results) -> str`
- 新增 `McpSearchTool`：用 httpx（核心依赖）直接讲 MCP Streamable-HTTP，不经过需要可选 `mcp` 包的 `agentica.mcp`，因此能当默认 `web_search` 引擎。Exa 是随附预设——其公开端点可匿名调用（共享免费池、有限流），设 `EXA_API_KEY` 则走自己的额度；把 `url`/`tool_name` 指向别的服务就是自定义引擎
- 六个搜索后端签名统一为 async `(queries: str | list[str], max_results: int) -> str` 且全部支持多 query，分发层因此不需要每后端的参数适配分支：`SearchBochaTool` 的 `count`、`SearchExaTool` 的 `num_results` 改名 `max_results`，`ZhipuWebSearchTool` 补 `max_results`，`DuckDuckGoTool.duckduckgo_search` 补多 query
- `web_search_pro_tool.py` / `WebSearchProTool` 更名 `zhipu_web_search_tool.py` / `ZhipuWebSearchTool`（CLI `--tools web_search_pro` → `zhipu_web_search`），并从早已过时的 `/paas/v4/tools` 通用工具端点换到专用的 `/paas/v4/web_search`。原来是把搜索伪装成一次 chat 调用（`messages=[{"role":"user"...}]`），结果得靠 `data['choices'][0]['message']['tool_calls'][1]['search_result']` 这种按下标摸出来——响应结构一变就崩，而且整个 API 只能传 query，条数只能拿回来再客户端切。新端点直接收 `count`（服务端就按条数返回，不再传完再丢）、`search_domain_filter`、`search_recency_filter`、`content_size`，并暴露智谱的 4 档引擎：`search_std`（0.01 元/次）/ `search_pro`（0.03，默认）/ `search_pro_sogou`（0.05）/ `search_pro_quark`（0.05），`web_search` 分发器路径可用 `AGENTICA_ZHIPU_SEARCH_ENGINE` 切换。实测两点值得知道：一是 `count` 只是建议值（`search_pro_sogou` 向上取整到 10/20/30/40/50，其他档位在部分查询上也超发），所以 `max_results` 由客户端截断兜底——单条正文约 1000 字，要 3 条收到 10 条会白烧几千 token；二是智谱自研的 `search_std`/`search_pro` 约 40% 的查询整批返回空 `link`（同一查询要么全有要么全无），需要每条可溯源时用 sogou/quark 档（实测这两档 100% 带 link），文档已写明。顺带修了鉴权：原先 header 直接发裸 api_key，新端点要求 `Bearer`。引擎/时间范围/摘要长度的非法取值在构造时就报错并列出可选值，而不是等 API 返回一句看不懂的错误；结果里剔掉 `icon`（favicon URL）和 `refer`（角标序号）——对模型是纯 context 浪费。默认 timeout 从 300 秒收到 60 秒（一次搜索挂 5 分钟不是超时，是卡死）
- `DuckDuckGoTool` / `SearchExaTool` 改为直接用 httpx 讲 HTTP，删掉 `duckduckgo-search` 和 `exa_py` 两个可选依赖（`pyproject` 的 `ddg` / `exa` extra 清空为占位）：两者都只是 HTTP 封装，而作为可换的 `web_search` 引擎，「装了才能用」意味着切过去才发现要装包。DDG 走公开 HTML 端点（httpx + bs4 都是核心依赖），顺带过滤掉此前会混进结果的赞助位、并把 `.result__url` 的截断展示文本换成真实链接（含 `/l/?uddg=` 重定向解包）——原先的 fallback 解析器返回的 url 是不可点的。Exa 走 `POST https://api.exa.ai/search`，原生 async 不再需要 `run_in_executor` 把阻塞 SDK 挪出事件循环，`text_length_limit` 下推为 `contents.text.maxCharacters` 由服务端截断（省掉传完再丢的字节），并删掉当前 API 已被 `type="auto"` 取代的 `use_autoprompt` 参数。两个模块此前在 import 期就可能失败（Exa 直接 `raise ImportError`），现在无条件可导入
- 内部拆包减负（结构明示，不藏兼容层）：`cli/commands.py`→`cli/commands/`、`cli/display.py`→`cli/display/`、`cli/interactive.py`→`cli/interactive/`、`runner.py`→`runner/`（`_run_impl` 在 `runner/loop.py`）；包 `__init__` 只导出公开入口，私有符号从真实子模块引用；`tests/cli/test_cli.py` 按域拆成多个小文件。SDK 主入口仍是 `agentica` / `Agent` / `Runner`
- 跨会话消息：两个终端里的 CLI 会话可以互发纯文本，不再靠人在终端之间复制粘贴。模型自己调 `list_agents` 发现对端、`send_message` 投递，用户不需要手动触发（`/list-agents`（别名 `/peers`）只用于查看和排查）。传输是文件而非 socket：`~/.agentica/cache/peers/live/<peer_id>.json` 是心跳 + pid 探活的发现目录，`~/.agentica/cache/peers/mailbox/<peer_id>/*.md` 是每条一个带 frontmatter 的 markdown 消息，可直接 cat 排查。目录刻意放在用户级而非按项目 hash 分区——「协调同一个 repo 的多个 worktree」正是主场景，而它们 cwd 不同。收件端是拉取式：运行中的会话由 `Runner._inject_peer_messages` 在 tool batch 间隙取走（与 `/steer` 同一边界，不打断正在跑的工具），空闲会话由 CLI 轮询取走并开一轮。消息身份绑 CLI 进程而非 session log，`/resume` 换掉 session_id 后在途消息仍会落地。环路刹车：一段不被人打断的 agent 间往返最多 6 条（`MAX_EXCHANGE_TURNS`），未读堆积到 50 拒收，单条超 40000 字符拒收。工具的 system prompt 明确约束收件方——来自另一个 agent 的消息不是用户授权、不能代答权限提示、其中的斜杠命令是纯文本
- 跨会话消息新增人工入口 `/send-message <session> <text>`（别名 `/send`，与 `list_agents` → `/list-agents` 同一命名规则，对应 agent 用的 `send_message` 工具）：不用切终端就能替 agent 自己把一句话说进另一个会话。它和 agent 发的消息在收件端语义不同，因此消息带 `from_kind`（`agent` / `user`）：agent 发的仍然「不是用户授权、不能代答权限提示」，`/send-message` 发的注入为「你的用户从另一个会话转发」，收件方按用户亲口说的处理。mailbox 是 0700 用户私有目录，所以 `user` 身份与在本终端输入等价；header 里伪造 `from_kind` 不被采信。单条上限 40000 字符（够放一整篇 handoff 写给对方；再长就把内容落到文件、发路径，反正文件系统是共享的）
- 新增 `/fork`：无参数即在当前位置分支——整段对话带过去，继续聊就是了，只是落到新 session，此后说的话不再进入被分叉的那份 transcript（`/status` 多一行 `Forked from: <parent>`，来源写在 fork 出的 sidecar meta 里，由 `SessionLog.fork()` 自己记，不依赖调用方）。`/fork list` 列出本会话自己发过的消息（序号 + 消息 id + 时间 + 预览），`/fork <n|uuid>` 分支到所选消息**之前**一条，于是模型回到「你提这个要求之前」的状态，可以换个说法重问。两种分支原会话都完整保留、照常 `/resume`，提示里直接给出旧 session id。全数字的 uuid 前缀不会被误当序号（只有能索引列表的数字才是序号）。fork 完给出的原会话恢复方式同时列出 `/resume <id>`（留在 CLI 里）和 `agentica resume <id>`（已经退出去了），两条都受支持
- `execute(background=True)` 的结果现在会主动回灌当前会话：命令结束时除了打印通知，还把退出码、耗时、完整命令和输出尾部交给 agent——运行中经 `steer()` 落在 tool 间隙，空闲则作为下一轮，于是它自己接着往下做，不用等用户回来手动 `wait`。设 `deliver_background_results: false` 可关掉自动唤醒
- Goal 预算耗尽不再当场砍断：任一 cap 触发时循环额外给一轮收尾 turn，喂 `[Standing goal budget reached]` prompt 要求模型交接（做完了什么、还剩什么、有什么坑），明确禁止在没有预算兜底时调 `verify_completion`；这一轮跑完才落到 `budget_limited`。收尾轮期间 goal 保持 `active` 以便正常计账（因此最终会超出 cap 一轮，状态栏如实显示），`GoalState.budget_wrapup_sent` 保证只给一次，`resume()` 重置。CLI `/goal` 与 SDK `run_goal()` 共用同一条路径
- Goal 主成本闸改为默认开启的 `token_budget=500_000`（CLI `/goal` 与 SDK `run_goal` 一致）；`turn_budget` 默认 `None`（仅 `--turns` / 显式参数时生效）。`/goal status` 与状态栏在执行中显示 `tokens used/budget`（如 `goal 12.3K/500K`）
- `execute(background=True)`：长命令可立即返回，由共享 `BackgroundProcessRegistry` 托管进程组；stdout/stderr 写入 `~/.agentica/projects/<user>/.../background/` 日志。CLI 新增 `/ps` 列出后台 terminal 与 background agent，`/stop <id|pid|#n>` 可按目标停止（空参停全部）；状态栏显示正在运行的 background terminal 数量。registry 经 `create_agent` / session rebuild 路径注入，与 `/background` agent 任务共用同一套 `/ps`/`/stop` 入口
- `wait(id=...)`：等待 `execute(background=True)` 启动的后台命令，命令一退出立即返回并给出退出码、耗时和日志尾部；未结束则在超时后返回当前进度且不停止命令，单次上限 300 秒，让调用回到模型循环以便用户打断。后台命令的退出状态只报给用户，`wait` 是它回到对话里的唯一途径。它补的是「命令可能活得比一次工具调用久」这个断层：前台命令被 timeout 或被取消的一轮杀掉会丢掉全部输出，而这类任务此前只能退回 `sleep N && tail log` 的盲等。一次调用跑得完、当下就要结果的命令仍应留在前台并调高 `timeout`；真正跑几小时以上的任务则不该 `wait`，等一两次仍未结束就结束轮次，由用户收到的完成通知驱动后续
- CLI 渲染 `apply_patch` 的真实多文件 unified diff：在原子写入前解析 patch envelope 并捕获每个目标文件的原始内容，完成后展示一份合并的 old→new diff，替代 executor 的文本摘要
- CLI execute 工具调用行支持宽度感知预览：普通长命令和 heredoc 统一最多展示 3 行正文，Ctrl+O 可分别展开完整 command 和折叠 output

#### changes
- 上下文压缩从五个 stage 塌成**两层**，Stage 4（`CompressionManager.compress` 规则压缩）整个删除。让一个超窗口的请求装下只有两种操作：**免费地扔掉单个条目**（可通过重跑工具找回）和**花一次 LLM 把历史换成摘要**（不可逆）。原来的 stage 3 和 stage 4 在做同一件事的两个版本——都是「截断最旧的工具结果」，只是各有一套阈值和保护参数，于是 bug 可以出在两个地方而不是一个。现在 Layer 1 = `evict_context()`（淘汰 + 收缩 tool_call 参数），Layer 2 = `auto_compact()`（原生 compact 是它的 provider 变体，reactive 是它加 `force=True`）；此外还有一个不算压缩的 Layer 0——工具结果预算，那是「别让超大输出进上下文」的输出策略，每条结果只在产生时跑一次。随之删除：`should_compress` / `_truncate_oldest_tool_results` / `_drop_old_messages` / `_archive_dropped_messages` / `_llm_compress_old_tool_results` / `_still_over_limit` / `get_compression_ratio`，配置字段 `compress_tool_results`（`ToolConfig` 与 `CompressionManager` 两处）/ `truncate_head_chars` / `keep_recent_rounds` / `use_llm_compression` / `compress_tool_call_instructions` / `workspace`，以及随之失去调用方的 `agentica/prompts/compression/`。**「丢弃最旧的消息轮次」不再存在**：它会静默吞掉用户自己提过的问题，那正是摘要该做的事，且做得更好。`_shrink_assistant_tool_call_arguments` 不跟着删而是并入 Layer 1——Layer 1 只碰 `role="tool"`，够不到一次 `write_file` 塞进 assistant 消息的整段 payload，这是真实缺口。`_sanitize_tool_pairs` 提为模块级 `agentica/compression/tool_pairs.py::sanitize_tool_pairs`，并接到唯一真正需要它的地方：`context_overflow_threshold` 的 FIFO 丢弃是按位置删消息的，会留下没有结果的 tool_call
- **Layer 2 现在默认可用。** `ToolConfig.compress_tool_results` 默认 `False`，意味着 `compression_manager` 为 `None`，于是默认配置下原生 compact、auto-compact 和 `prompt_too_long` 之后的 reactive 补救**全都不跑**——长会话唯一的结局是被 provider 拒绝。该 flag 已删除，`Agent.__init__` 在 `compression_manager` 为 None 时总是建一个

#### fixes
- 压缩两层都改为以「一条工具结果」而不是「一条消息」为单位，**Anthropic 路径上压缩此前从来没生效过**：只有 OpenAI 系是「一条结果一条 `role="tool"` 消息」，Anthropic 把一整轮打包进单条 `role="user"` 消息的 content 列表（`{"type": "tool_result", "tool_use_id": ...}` block，见 `AnthropicClaude.format_function_call_results`）。Layer 1 只扫 `role == "tool"`，于是在 Claude 上一条都没淘汰过——不报错、不告警，静默失效，整个提供商的上下文只能靠 Layer 2 兜。`tool_result` block 不带工具名（只有 `tool_use_id`），占位符改为回查发起调用的那条 assistant 消息的 `tool_calls` 拿到名字和参数，所以两边的占位符信息量一样。同一个形态差异连带修掉两处：`sanitize_tool_pairs` 只认 `role="tool"` 形态，Anthropic transcript 在它眼里像是每个调用都没有回复，重建会给每个调用插一条占位 `role="tool"` 消息，把本来没坏的 transcript 弄坏（现在这类 transcript 原样返回）；`auto_compact` 保留「最后一条 user 消息之后的整段尾巴」，而 Anthropic 的工具轮本身就是 user 消息，从那里切会留下一批 `tool_result`、它们对应的 `tool_use` block 却在刚被摘要替换掉的 assistant 消息里——这种孤儿 block 会被 API 直接拒绝，现在判断尾巴时跳过承载工具结果的 user 消息。`Message._evicted` 语义随之收紧为「这条消息里的结果全都淘汰完了」（一轮多条结果时不能提前关门），单条结果是否已淘汰改看占位符前缀。**未覆盖**：Layer 0 的批量预算 `enforce_tool_result_budget` 仍只看 `role="tool"`，Anthropic 上不生效；单条超大结果在产生时落盘那条路径是 provider 无关的，仍然有效，所以主要保护还在
- 修掉「读了又读」死循环，并把 micro-compact 整个换成 `agentica/compression/evict.py`：模型在一轮里并行 `read_file` 六个片段，下一轮这些结果已被清成占位符，于是它把同样六个读原样再发一遍，无限重复。旧实现两个缺陷叠加：**没有压力闸门**（每轮无条件清，200k 窗口只用了 6k 也照清，省下的上下文没人要，代价是模型重跑工具），**按条数保护且当轮结果已在计数里**（`keep_recent=5` 遇上 6 个并行调用，最旧那条在模型第一次看到它之前就被清了）。新实现只有两个量，没有「保留最近 N 条」这类参数——任何固定条数都必然输给 count+1 大小的批次，调大 N 只是把复现门槛抬高：占用低于 `context_window` 的 70%（`EVICT_THRESHOLD_RATIO`）一条都不动，超过则**按最旧优先淘汰，直到降回 50%**（`EVICT_TARGET_RATIO`）就停——最近的结果自然幸存，因为根本轮不到它们；目标取得比阈值低是为了迟滞，否则清一条刚跌破阈值下一轮又超，变成每轮清一条的抖动。消息尾部那一段连续 `role="tool"`（模型还没看过的当前批次）整体排除在可淘汰集合之外，压力再大也不动它，那种情况该走摘要而不是丢掉本轮自己的证据。占位符写明是哪个调用（`read_file(file_path=..., offset=...)`）让模型能原样重发；**不再先落盘**：取回动作两边都是一次工具调用，成本一样，而对 `read_file` 落盘副本严格更差——原路径上的内容更新鲜，快照只是过期拷贝加白占磁盘（`persist_full_result()` 随之删除）。已被 tool-result budget 落过盘的结果（含 `<persisted-output>`）跳过，它携带的路径是那份过大输出唯一的抓手。**没有采用「豁免 read_file」**：`read_file` 恰恰是体积最大的消费者，永久豁免等于让上下文被文件正文填满直到触发破坏性大得多的整轮丢弃，而且它只挡读批次——6 个并行 `grep` 会以完全相同的方式空转。`Message._micro_compacted` 更名 `_evicted`，CLI 事件 `compact.micro` 更名 `compact.evict`（仍然静默）
- `glob` / `grep` 默认超时从 10s 收到 3s：NFS 大树上慢搜尽快失败好让模型收窄 path；调用仍可传更大的 `timeout`
- `/list-agents`（及 `list_agents` 工具）对本会话与其他 live session 用同一套字段：`project` / `session_log` / `log_file` / `workspace` / `memory` / `mailbox` 都会列出。对端若是旧版未发布这些路径，本机按 cwd/pid/peer_id 补全能确定的项，不再只剩 session_id+cwd+working on
- LSP 编辑诊断不再把 `agentica` 启动打崩，也不拖慢启动：CLI 默认仍开 `--enable-diagnostics`，但 language server **懒启动**（第一次改文件才 `initialize`，`create_agent` 不阻塞）。半残 pyright（只 `pip install pyright` 没 `[nodejs]`）或 NFS 超时只会在首次编辑时 warning 降级并杀掉进程；`initialize` 的约 5s 是 deadline 不是 sleep（正常几百毫秒就返回）。安装提示改为 `pip install 'pyright[nodejs]'`
- 状态栏和 `/status` 不再报一个和正在跑的模型对不上的 profile 名：两处都读 `resolve_active_profile_name()`，那回答的是「config.yaml 指向哪个 profile」，而 `agentica --model_name X` 之后跑的已经不是那个 profile 的模型了，于是状态栏出现 `venus-opus-4.8 openai/deepseek-v4-flash` 这种自相矛盾的一行。改为由真正做决定的 `resolve_model_config` 把结果记在 `agent_config` 上（`profile_name` / `profile_source`），新的 `setup.session_profile()` 是所有展示面的唯一读取口：`--profile` 指定的显示为那个名字并标 `flag`，模型被 flag 覆盖时显示「无 profile」——此时确实没有哪个 profile 能描述这个会话，报一个名字只会让人以为自己在那个 profile 上。空字符串是「明确没有」，键缺失才回退到 config 级答案，所以手搓 `agent_config` 的调用方（测试、其他入口）行为不变。`/model <profile>` 切换时同步这两个字段，`/config` 在会话 profile 与 config.yaml 不一致时多打一行说明
- 一条消息里发出的多个 `task` 现在真的并行：执行器把一批 tool call 按 `concurrency_safe` 分成两组，True 的走 `asyncio.gather`，其余在 for 循环里一个接一个跑，而 `task` 注册时漏了这个标记（`register(self.task)` 默认 False），于是三个 subagent 排队执行，第一个跑完才起第二个——它自己的 system prompt 里那句「Launch independent tasks in one message to run them in parallel」一直是空头支票，只读的 `read_file`/`glob`/`grep`/`web_search`/`search_memory`/`list_agents` 全都标了、唯独它没有。subagent 本就只读（写操作和状态变更命令会被拒），且每个都跑在自己克隆的 model、HTTP client 和 Agent 上（`_clone_parent_model` 的注释早就写明「parent 的 client 属于 parent 的事件循环，会和并发 subagent 抢」），符合 `concurrency_safe` 的语义。回归测试直接断言调度结果而不是这个标记：三个 task 必须同时在飞（`peak == 3`）。同时删掉 `task` 上的 `interrupt_behavior="block"`——这个字段只在串行分支被读取，改并行后对它永远不生效，留着就是假配置
- `task` 的并发有了上限：`spawn_batch` 一直按 `SubagentRegistry.MAX_CONCURRENT`（3）限流，但模型是一条消息里发 N 个 `task`、走的是 `spawn()`，之前没有任何闸门——标上 `concurrency_safe` 之后，8 个 task 就是 8 个 subagent 同时烧钱。`BuiltinTaskTool` 自己持一个懒初始化的 `asyncio.Semaphore(SubagentRegistry.MAX_CONCURRENT)`（按事件循环重建，跨 `run_sync` 的临时 loop 不会串），多出来的 task 排队而不是被拒；system prompt 里也写明「最多 3 个同时跑」，免得模型以为一次发 10 个能一起完成
- 系统提示里的 git 状态不再卡住整个事件循环：`get_git_context()` 用四次同步 `subprocess.run`（每次 5s 超时）拿 `rev-parse` / branch / status / log，而它挂在**每一轮**的 system prompt 构建路径上——仓库大或磁盘慢的时候，这几百毫秒到几秒里所有并发工作（并行工具、后台进程回执、peer 消息投递）全部停摆。改为 `asyncio.create_subprocess_exec`，且先 `rev-parse` 确认是仓库，再把 branch / status / log 三个读用 `asyncio.gather` 一起发出去（它们互不依赖）——实测本仓库 38ms，事件循环最大停顿 9ms
- 只读外部工具补上并行标记：搜索类（serper / exa / bocha / 百度 / DuckDuckGo / 智谱 / jina）、`wikipedia`、`arxiv`、`dblp`、`hackernews`、`weather`、`newspaper`、`yfinance` 全部只读 HTTP，`sql` 的 `list_tables` / `describe_table` 是只读 schema 查询（`run_sql_query` 仍串行，它可能是 DML/DDL），`code` 的 AST 分析与 lint、`lsp` 的 `goto_definition` / `find_references` / `hover_info` 也是只读——之前它们全部落在串行分支，一次「查三个来源」等于三次网络往返相加。同时修掉 LSP 并行后才会暴露的问题：`_send_message` 写 server stdin 没有加锁，两个查询同时写会把 Content-Length 帧交错成乱码
- guardrails 的 `run_in_parallel` 从死字段变成真并行：`InputGuardrail.run_in_parallel` 默认 True、也在文档里写着，但执行引擎 `run_guardrails_seq` 只有串行一条路，这个字段从来没人读——三个各 0.3s 的审核就是 0.9s 全加在用户面前。新的 `run_guardrails` 把连续的可并行 guardrail 归成一批 `asyncio.gather`，`run_in_parallel=False` 的自己一批：声明顺序仍然决定短路语义（串行的那个拦下来，排在它后面的根本不会启动），报告出去的也仍是**声明**在最前面的那个而不是最先跑完的那个。输出 guardrail 保持串行——答案已经生成，第一个拦下就结束，后面的都是白花的钱
- RAG 检索不再阻塞 async 路径：`knowledge.search` 从查询 embedding（HTTP）到向量库再到 reranker（HTTP）全程同步，而 `get_relevant_docs_from_knowledge` 是直接调的——开了 `add_references` 的会话，每一轮都要在事件循环里干等这一整趟往返。改为 `run_in_executor`，`get_user_message` / `search_knowledge_base` 随之变成 async
- 并行分支补上取消检查：串行分支每次执行前都看 `agent._cancelled` 并尊重 `interrupt_behavior`，并行分支完全没有这一段——Ctrl+C 之后，那一批里的每个读、以及（`task` 并行化之后）每个 subagent 照样全部启动。两个分支现在共用同一个判断：`interrupt_behavior="cancel"` 的直接跳过并回「Tool cancelled by user」，`"block"` 的（起来了就没法干净拆掉）仍然放行
- 工具参数不再被悄悄改写：`get_function_call` 此前在 `json.loads` 之前对**整段 JSON 原文**做 `"True"→"true"` / `"False"→"false"` / `"None"→"null"` 替换，本意是容忍模型吐出 Python 字面量，实际把字符串**内容**一起改了——`send_message` 里一段 `swapped = True` 的代码到了对端就成了 `swapped = true`，接收方照着报 NameError，两个 agent 于是围着一个根本不存在的 bug 争论。改为先正常 `json.loads`，只有解析失败才用 `ast.literal_eval` 兜底（只认字面量、不执行代码，且不碰字符串内部）。同时删掉紧随其后的一遍「清洗」：它对每个字符串参数 `strip()` 并把 `"none"`/`"true"` 一类的值按字面转成 `None`/`True`，与声明的类型无关——于是消息正文和文件内容的首尾空白被吃掉、一条内容恰好是 `None` 的消息变成空值。类型修正现在只由 schema 感知的 `coerce_tool_args` 负责（`"3"`→`3`、`"true"`→`True` 仍然照做，但只在参数确实声明为该类型时）。`sanitize_arguments` 开关随之删除（`Function` 字段、`@tool` 参数、`Tool.register()` 参数）：它已无任何作用，留着就是假配置
- 换 provider 后 resume/fork 不再 400（`cache_control cannot be set for empty text blocks`）：session log 无论哪家 provider 写的，tool 轮次一律按 OpenAI 线格式落盘（assistant 带 `tool_calls`、结果是 `role="tool"`，且那条 assistant 的正文是空串）。Anthropic 的 `/v1/messages` 收不了这种形状——空正文被包成 `{"type":"text","text":""}`，滚动 prompt cache 又恰好把断点打在它上面，于是整轮请求被拒。两层修：`Model` 新增 `supports_replayed_tool_history`（Claude 为 False），resume/fork 时对这类 provider 把历史降级成纯 user/assistant 文本（复用 `/model` 切换早就在走的 `strip_all_tool_artifacts`），问答记忆保留、tool 轮次不跟过去；`Claude.format_messages` 同时不再产出空 text block，整条内容为空的消息直接跳过（只带图片的 user 消息照常保留）。同 provider 的 resume 不受影响，tool 历史照旧完整回放
- 命令处理器发起的用户提问不再被看门狗秒取消：`ask_user_question_callback` 原先在轮询到空队列时以 `not agent_running` 判定「这一轮已经结束、没人会来回答了」，但斜杠命令恰恰跑在两轮之间（`agent_running` 为 False），于是 `/cron` 的确认框和新的 `/resume` 目录选择在第一次轮询就自己取消掉。改为记录提问是否在 run 中发起，只有那种才受 run 结束影响
- `/newchat` 不再继承 `agentica resume <id>` 留在 `agent_config` 里的 session：此前新会话会沿用被恢复的 session_id，继续往它本该离开的那份 transcript 里追加
- `/resume <id> at <uuid>` 现在真的 fork：此前它复用原 session_id，分支的新对话被追加进它所分叉的那份 JSONL，两条线混在一个文件里、各自都无法独立 resume。现在在 `create_agent` 这个唯一入口调已有的 `SessionLog.fork()` 生成新 session（`--resume-at-uuid` 启动参数走同一条路径），原分支保持不变；fork 点读取即消费，后续 `/model` 重建 agent 不会从同一点反复分叉
- 修复 `/steer` 在 goal 循环中丢词：agent 不在 run 中时（一轮刚结束、goal 正在判定的那几秒，以及 UI 检查到 `steer()` 之间的 TOCTOU 窗口）原先只打印一句"用 /queue"就把用户输入丢掉。现在统一降级为排队执行，并插在待跑的 continuation prompt 之前，纠偏不会被一整轮无关工作挡住；被插队的 continuation 也不会被重复排入
- `execute(background=True)` 完成后，CLI 现在会主动异步显示成功或失败、退出码、尾部输出和完整日志路径；通知由 registry 的完成事件驱动，不会唤醒 LLM。`/stop` 和 CLI 退出触发的终止不会再重复显示为任务失败，等待 `ask_user_question` 输入期间的完成事件会保留到安全时机再展示
- 修复 `execute(background=True)` 遇到以 `&` 结尾的命令时误报完成：shell 会 fork 掉真正的工作并立刻退出，registry 追踪到的是那个空壳，于是任务还在跑就宣布结束。该组合现在直接拒绝；前台的 `nohup ... &` 仍照常执行，只在结果末尾注明它未被追踪（`/ps`、`/stop` 和完成通知都看不见它，且取消或超时的一轮会连同进程组把它杀掉），并建议改用 `background=True`
- `execute` 拒绝 120 秒以上的前台起始 `sleep`：观察到的轮询写法 `sleep 330 && tail log` 会把刚被后台化释放的这一轮重新堵死。阈值对齐前台默认 timeout，短重试 `sleep 2 && curl ...` 不受影响。拒绝信息同时给出两条正路：等后台命令用 `wait(id=...)`，等没有完成事件的外部条件用 `until curl -sf ...; do sleep 5; done` 这类成功即返回的重试循环
- `execute(background=True)` 的返回值和 docstring 改为直接声明契约：退出状态报给用户而非模型，后续步骤需要它的结果就调 `wait(id=...)`，不要自行用 sleep/轮询/阻塞 tail 模拟等待。旧的「读日志查看进度」措辞正是诱导模型接前台轮询的来源
- 修复后台 terminal registry 接线导致 CLI 启动失败：`SessionState` 现在先于首次 `create_agent()` 初始化，再将同一份 `BackgroundProcessRegistry` 注入 execute 工具、`/ps`、`/stop` 和状态栏
- 修复工具取消时子进程清理不彻底：新增 `terminate_subprocess()`，在活跃 event loop 上终止并完全回收 asyncio 子进程（`communicate()` 排空管道，支持进程组 SIGTERM→SIGKILL 宽限升级），应用于 execute/grep、shell 及 goal verify 等工具，消除取消后管道传输回调泄漏到已关闭 event loop 的问题
- CLI 顶层 agent 执行错误改为结构化展示：429/限流等 provider 异常显示红色摘要、可操作 `/retry` 提示和 code/spanId 诊断字段，完整原始异常保留到 Ctrl+O 展开
- 文件工具缺失路径错误只暴露真实路径状态：`read_file`/`edit_file`/`glob`/`grep` 缺失路径时返回 resolved path 和 nearest existing parent，并提示从 `ls`/`glob`/`grep` 重新定位；不再猜测候选路径或在 `read_file` 内容尾部追加 metadata
- 文件编辑默认收敛到 `apply_patch`：`multi_edit_file` 不再注册为内置工具，复杂/多 hunk 编辑走上下文 patch；`edit_file` 保留为单个短且唯一的 literal 替换工具。`edit_file` 的 `String not found` 保持无状态重读指引；`apply_patch` 的 context mismatch 同时展示 expected context 和 actual 当前行，便于用真实当前内容重建 patch
- 修复 CLI 输出 OSC 8 超链接泄漏：Rich 为 Markdown 链接生成 OSC 8 终端超链接，prompt_toolkit 的 ANSI 解析器不识别 OSC 序列会把 payload 渲染成可见文本，渲染前剥离不支持的 OSC 8 包装（保留链接样式文本）
- 修复 Ctrl+O 分页器在输出含控制字符时停在 less 的 binary-file 确认提示：less 调用统一加 `-f` 强制打开
- `/compact` 原生压缩失败的回退提示改为 `logger.warning`，不再向终端打印打断对话流
- 拆包后清理机械复制遗留的死 import：`runner/`（compress/core/loop/persist/retry_fallback/steer/stream）与 `cli/commands/`（context/cron_cmd/goal/helpers/model_config/runtime/session/tools_skills）各文件只保留本地真实引用，删掉约 680 行从原单文件带过来的未用 import；`tests/cli/test_cli_configuration.py` 的 4 个 patch（`reset_skill_registry`/`load_skills`/`get_skill_registry`/`create_agent`）从 `cli_tools_skills` 改到 `cli_helpers`——拆包后实际调用点在 `helpers._refresh_skills_session`，原 patch 打在错模块是无副作用的 no-op，autoflake 删掉未用 import 后才暴露
- `tools/buildin_tools.py`（2154 行）拆成 `tools/builtin/` 包：`file_tool.py`（`BuiltinFileTool`+path guards+`_GLOB/_GREP_TIMEOUT`）、`execute_tool.py`（`BuiltinExecuteTool`+exit-code helpers+`_MAX_WAIT_SECONDS`）、`__init__.py`（`get_builtin_tools`+re-export 7 个工具类）；原 `task_state_tools.py`/`web_tools.py` 已在包内。`agentica/__init__.py` 与 `tools/__init__.py` 改从 `agentica.tools.builtin` 取，所有直接引用者（tests/examples/docs/agent/acp/gateway/evaluation）改到新路径，测试 patch 字符串（`_GREP_TIMEOUT`/`shutil.which`/`asyncio.create_subprocess_exec`/`terminate_subprocess`→`file_tool`；`_MAX_WAIT_SECONDS`→`execute_tool`；`_detect_python_error_hint`/`_interpret_exit_code`/`_is_blocked_device`/`_check_sensitive_write_path`→对应子模块）同步更新
- `agent/base.py`（2091 行）抽出 goal 闭环到 `agent/goal_mixin.py`（`GoalMixin`：`get_goal_manager`/`enable_goal_tool`/`run_goal`/`run_goal_step`），`Agent` 继承链加 `GoalMixin`；`base.py` 降到 1794 行，只留公开 run API 面与薄委托，不再内嵌 goal 闭环。MRO 透明，所有 `agent.run_goal()` 调用无需改动

#### docs
- `apply_patch` docstring 明确要求 Update/Delete 操作前必须先 `read_file`，禁止凭记忆构造上下文
- README News 区重构：旧版本条目折叠进 `<details>` 区块
- 终端文档更新 `/new` 命令别名说明及退出 CLI 后按 session ID 恢复的用法
- 文档与 README 补充 CLI `task` / `delegate` / peer 选型：`docs/getting-started/terminal.md`、`docs/multi-agent/choosing.md`、`docs/concepts/tools.md`、`docs/multi-agent/subagent.md`；中/英/日 README News + 协作对照表
- CLI 对 `task` / `delegate`（及结果锚点）完整展示任务正文，不再 40/80 字截断；子 agent 启动行同样全文

## [1.4.11] - 2026-08-04

### Added
- **OpenAI Responses API support.** The new top-level `OpenAIResponses` model supports sync and streaming text, reasoning summaries and encrypted reasoning-state replay, image input, function tools, parallel tool calls, and structured output. CLI and Gateway profiles select it with OpenAI-only `wire_api: responses`; profile `max_tokens` maps to `max_output_tokens`, and Responses reasoning uses `reasoning` instead of `reasoning_effort`.
- **Provider-native Responses compaction.** `OpenAIResponses` now calls `/responses/compact` before destructive local compression, replays the returned canonical window unchanged, persists opaque checkpoints across session resume, and retains a portable transcript for cross-provider fallback. Endpoint failures and `prompt_too_long` recovery use the existing local summary/rule-based pipeline; manual `/compact [instructions]` follows the same priority.
- **Markdown-configured subagents and CLI management.** Packaged `explore` / `research` / `code` definitions now live in `agentica/agents/*.md`. Projects can add or override definitions in `.agentica/agents/`, users can share definitions through `~/.agentica/agents/`, and `/agents list|create|reload|remove` manages the effective configuration. The default `review` subagent was removed so code review stays with the main agent and its full context; partial runs retain bounded tool inputs for resume.
- **Built-in `apply_patch` can update multiple files in one tool call.** One strict patch envelope may add, update, or delete several text files. Agentica validates every path and hunk before writing, reuses the existing sandbox and sensitive-path guards, and keeps `multi_edit_file` unchanged for established single-file batch edits. CLI and Gateway summaries report the aggregated file and line counts.

### Changed
- **Shell tools preserve the model's exact command string.** `execute` and `ShellTool` no longer normalize arguments, rewrite Python literals, or convert `python -c` commands into heredocs. Safety policies may block a command and secret redaction may sanitize returned output, but the command sent to the shell is otherwise unchanged.
- **CLI now warns after successful main-agent context compaction.** Automatic compaction recommends `/new` when long-session accuracy may degrade, repeated successful auto-compactions escalate the warning with a session-local count, and reactive recovery explains that compaction happened before retrying. Spinner lifecycle events, failed attempts, manual `/compact`, and subagent compactions do not affect the count.
- **Removed the `read_file` freshness/staleness machinery entirely** (codex-style simplification): `FileReadState`, `_file_read_state`, `_record_file_read`, `mark_read_context_stale`, `mark_all_read_context_stale`, `_edit_freshness_tip`, `Agent.mark_evicted_file_reads`, `Agent.append_evicted_file_read_notice`, and the `[Context maintenance]` eviction notices are gone. Edits return the absolute path + diagnostics only; a failed `edit_file` ("String not found") remains the natural signal to re-read.
- **Default model `context_window` raised from 128k to 200k** across `Model` base, OpenAIChat, LiteLLM, Ollama, and Claude, reducing premature tool-result compaction that caused repeated `read_file` calls.
- **Prompt and tool-schema token cost reduced.** File-tool guidance is gated on registered file tools (no phantom tool names); dead `# Available Tools` table generation is removed; parallel/batch call guidance is restored; `grep`/`glob` docstrings are slimmed; `task` policy lives only in the tool system prompt; `write_todos` clarifies steps vs tool calls and returns a short status ack instead of echoing the full list.
- **`write_todos` keeps per-step progress updates.** Completions are still not batched (one sync per finished step) so the CLI progress bar stays live; only the tool-result payload shrank.

### Fixed
- **CLI resume restores both model context and the visible transcript.** Resumed JSONL history is hydrated into canonical run-response messages used by later prompts, while user, assistant, tool-call, and tool-result entries are replayed in the terminal. Session summaries now print the executable `agentica resume <id>` command, and patch failures retain actionable multi-line details instead of collapsing the error to one short line.
- **CLI context usage now reflects the current session prompt instead of a token watermark.** The status bar is updated from each main-agent request's actual messages and tool schemas, re-measured after each completed turn or `/compact`, and ignores subagent/auxiliary LLM calls. Per-turn footer tokens remain cumulative API consumption across retries and tool loops, so they no longer share a misleading data path with context occupancy.
- **Quote-tolerant file edits no longer rewrite unrelated punctuation.** `edit_file` and `multi_edit_file` now use normalized quote text only to locate a match, then apply replacements against the original content so curly quotes elsewhere in the file remain unchanged.
- **Stale `model_pricing_cache.json` is no longer discarded.** Catalog loading previously treated TTL expiry as "no cache" and, when the network refresh failed, silently fell back to the hardcoded pricing table — so new models (e.g. `claude-opus-5`, 1M context) never resolved. Refresh failures now fall back to the stale-but-valid cache file.
- **CLI resize no longer leaves repeated `Enter to send` ghost lines.** The `_resize_collapsed` flag (meant to shrink the bottom frame to a single row during a terminal resize) was set but never read by the layout, so the full multi-row frame redrew on every `SIGWINCH` and multiplied ghost copies in scrollback. The collapse is now actually wired across the input prompt, queue bar, status bar, and input height, and the post-resize restore does a clean erase + absolute-cursor redraw instead of a diff `invalidate()`.
- **File tools now expose clearer recovery signals.** `edit_file` / `multi_edit_file` append one stateless "read or re-read the relevant region" action after `String not found`, without tracking session read state or blocking edits. `grep` now documents that `path` accepts either a file or directory, supports file paths in its Python fallback, and reports missing inputs as `Path not found`.
- **Learned Experiences no longer accumulate frontmatter debris.** `strip_frontmatter` parses line-based `---` delimiters (values containing `---` no longer truncate the body); bumping a card refreshes its body from the new content; injected card bodies are capped; captured tool errors keep head+tail within a fixed budget instead of parking whole tracebacks in the system prompt.

## [1.4.10] - 2026-07-24

### Added
- **Native image capability routing.** Model image support is now resolved from `models.dev` catalog metadata, with explicit `supports_images` overrides for private or aliased endpoints. Vision-capable base models receive original images directly; text-only models use the external OCR fallback.
- **Simplified session naming.** CLI `/rename <name>` replaces `/session rename`, and `/resume` accepts a session number, name, or ID prefix.

### Fixed
- **Pillow is declared as a core dependency.** The model and default provider import paths use Pillow for PIL image inputs and image format detection, so the `1.4.10` wheel now installs `Pillow>=10.0` automatically.
- **Wheel installation is validated without checkout shadowing.** CI imports both the latest PyPI release and the newly built wheel outside the repository root, and verifies that the current wheel installs and imports Pillow.

## [1.4.9] - 2026-07-21

### Added
- **Unified 3-tier tool permission model** (`ask` / `auto` / `allow-all`) across SDK, CLI, and Web/Gateway, centralized in `agentica/agent/permissions.py`. `ask` exposes only read-only tools; `auto` allows reads everywhere + writes restricted to `work_dir` (sandbox-enforced); `allow-all` is unrestricted. `ToolConfig.permission_mode` carries the mode; `Agent.set_permission_mode()` flips it at runtime without rebuilding the agent. CLI `--permissions` and `/permissions` command, Gateway per-session approval mode, and the SDK constructor all share one source of truth. Removed legacy `yolo`/`full`/`strict` naming.
- **Claude-over-OpenAI-compatible `<invoke>` tool-call compatibility** in `OpenAIChat`: when a Claude model is reached through an OpenAI-compatible proxy that leaks `antml:invoke` blocks into text content (instead of structured `tool_calls`), the model layer now buffers the turn, parses the XML, and rewrites it into standard OpenAI function calls — so the tool actually executes and neither the leaked XML nor stray preamble (e.g. `course`) enters assistant history or the CLI. Native Anthropic (`model_provider="anthropic"`) and standard OpenAI/DeepSeek/Qwen paths are unchanged.

### Changed
- **Subagents are read-only by design.** All built-in subagent types (`explore`/`research`/`code`) now deny `write_file`/`edit_file`/`multi_edit_file`/`execute`; the `task` tool's default `subagent_type` changed from `code` to `explore`, and its system prompt states subagents are read-only and the main agent does all edits. This fixes the root cause of "the LLM delegated my query to a `task` subagent and the cheap auxiliary model wrote garbage code" — subagents run on the auxiliary model and can no longer edit. User-registered custom subagents are untouched.
- **`edit_file` / `multi_edit_file` freshness checks are advisory tips, not hard blocks.** Stale/unread/externally-modified files now produce a `freshness_tip` appended to the result (or included in a `String not found` error) instead of rejecting the edit. Sensitive-path writes still hard-fail. File version tracking is lazy (content hash only when mtime+size match) and `_record_file_read` hashes in-memory content to avoid re-reading just-written files.
- **Evicted `read_file` results are surfaced to the LLM.** When compression / context-overflow evicts a `read_file` tool result, a `[Context maintenance]` user message is appended naming the affected paths and advising a re-read before editing — so the agent no longer silently edits from stale memory.

### Fixed
- **CLI `ask_user_question` freeze.** A callback-less `AskUserQuestionTool` could fall back to bare `input()` and deadlock against prompt_toolkit's stdin ownership. Added a process-wide default-callback registry (`set_default_ask_user_question_callback`) the TUI registers at startup, a watchdog that aborts/re-arms the prompt if the agent's turn ends or the request is overwritten, a `_cprint` freeze while an ask prompt is active, and a `SIGQUIT` (Ctrl+\\) hard-escape that calls `os._exit(1)`.

## [1.4.8] - 2026-07-07

### Fixed
- **TaskAnchor no longer leaks `agent.run(message)`'s first message into the system prompt every turn.** `TaskAnchor` gains a `source: Literal["message", "goal"] = "message"` field that gates `to_prompt_block()`. Only explicit goal entry points — `Agent.run_goal()`, CLI `/goal`, and an active session-log goal — produce `source="goal"` anchors that render as `## Original Task`. Ordinary `agent.run(message)` produces `source="message"` anchors that are still used as the retrieval query but stay out of the system prompt. This restores pre-1.4.0 prompt behavior for plain `agent.run()` callers (e.g. private chat seed, workflow handoff, session resume) where the "first message" is a transcript / replay / dump and pinning it system-wide was a bug. Callers that need long-task drift defense should use `Agent.run_goal()` or set `agent.task_anchor = TaskAnchor(..., source="goal")` explicitly.

### Changed
- **Claude `max_tokens` resolution** (ported from hermes-agent's `anthropic_adapter.py`):
  - Default changed from `max_tokens: int = 8192` to `max_tokens: Optional[int] = None`. When `None`, a per-model output ceiling is looked up from `_ANTHROPIC_OUTPUT_LIMITS` (Opus 4.6/4.7 → 128K, Sonnet 4.5/4.6 → 64K, 3.5 Sonnet → 8192, etc.). Previously every model was capped at 8K which starved thinking-enabled models (thinking tokens count toward the limit).
  - Resolved cap is clamped to `max(context_window - 1, 1)` for small custom endpoints whose context window is smaller than the model's native output ceiling. No-op for full-size native models.
  - Positive-finite guard rejects locally: `max_tokens=0 / -1 / 0.5 / NaN / True` no longer leak to the API and 400 — they fall back to the model ceiling.
- **Claude auto-recovery from "max_tokens too large given prompt"**: `Claude.invoke()` and `invoke_stream()` now parse the API error message for `available_tokens: N` and retry once with `max_tokens = N - 64` (safety margin). Prompt-too-long errors are NOT touched — that path still flows through `_learn_context_limit_from_error`. New module: `agentica/model/anthropic/_max_tokens.py` with `resolve_anthropic_messages_max_tokens` + `parse_available_output_tokens_from_error` (28 unit tests).

### Removed (Breaking)
- **`agentica.model.providers` module deleted** (`ProviderConfig`, `create_provider`, `list_providers`, `register_provider`, `PROVIDER_REGISTRY`). The registry indirection had a single concrete output (`OpenAILike(**config)`) so every OpenAI-compatible factory now directly constructs `OpenAIChat` with hardcoded `base_url` / `api_key_env` / `default_model` / `context_window`.
- **`agentica.OpenAILike` deleted**. Was a 22-line subclass of `OpenAIChat` whose only behavior was a placeholder-`api_key` warning. Use `OpenAIChat(id=..., api_key=..., base_url=...)` for custom OpenAI-compatible endpoints.
- **`agentica.model.openai.like` deleted**. `AzureOpenAIChat` now subclasses `OpenAIChat` directly.

### Changed (Breaking)
- Each `XxxChat` is now a thin top-level factory in `agentica/__init__.py` (e.g. `DeepSeekChat`, `ZhipuAIChat`, `QwenChat`, `ArkChat`, …). Added 5 previously-only-by-slug factories: `NvidiaChat`, `SambanovaChat`, `OpenRouterChat`, `FireworksChat`, `InternLMChat`.
- New `agentica.PROVIDER_FACTORIES: dict[str, Callable]` exposes slug → factory dispatch for gateway / multi-tenant code (replaces `PROVIDER_REGISTRY` lookups).
- `agentica.model.defaults.create_default_model()` now uses an inline env-var table + `PROVIDER_FACTORIES`.
- `agentica.gateway.services.model_factory.create_model()` dispatches via `PROVIDER_FACTORIES` instead of `create_provider`.

### Migration
```python
# Before
from agentica.model.providers import create_provider
model = create_provider("deepseek", id="deepseek-v4-pro", api_key="sk-...")

# After
from agentica import DeepSeekChat
model = DeepSeekChat(id="deepseek-v4-pro", api_key="sk-...")
```
```python
# Custom OpenAI-compatible endpoint
# Before:
from agentica import OpenAILike
model = OpenAILike(id="my-model", api_key="sk-...", base_url="https://...")
# After:
from agentica import OpenAIChat
model = OpenAIChat(id="my-model", api_key="sk-...", base_url="https://...")
```

### Added
- Standing-goal loop judge hardening (hermes-validated + beyond):
  - **Tool-call summary fed to judge**: `Agent.run_goal()` extracts `(tool_name, is_error)` pairs from each turn's `RunResponse.tool_calls` and passes them to `judge_goal`. Judge prompt now includes a `Tools used this turn: edit_file, run_pytest(error), ls` line so it can distinguish "answered with no tools" from "actually did work". Zero extra LLM calls — names + flags only. New optional `tool_calls` param on `GoalManager.evaluate_after_turn()` and `judge_goal()`.
  - **Tool-stuck auto-pause**: `GoalState.consecutive_tool_failures` counts consecutive turns where every tool call errored. After `MAX_CONSECUTIVE_TOOL_FAILURES = 3` the loop auto-pauses with `paused_reason="tool-stuck"`. Any successful tool call resets; turns with no tool calls do NOT reset (a "just thinking while stuck" turn shouldn't get a free pass).
  - **Subgoal "find evidence" rule**: when subgoals are present, judge prompt now demands concrete evidence for each criterion (file excerpt / command output / result value) and explicitly rejects vague summaries like "all requirements met". Borrowed from hermes-agent's hard-won production prompt.
  - **JSON parsing accepts weak-model output**: `_parse_judge_response` now coerces `"yes"`, `"true"`, `"1"`, `"done"`, `"y"` strings and numeric `1` to `done=true` (small chat models and some reasoning models don't always emit JSON booleans).
  - **Static prompts lifted to `agentica/prompts/base/md/`**: `goal_judge.md` (judge system prompt) and `goal_continuation.md` (continuation template) now live alongside `soul.md` / `heartbeat.md` for consistency. New module `agentica/prompts/base/goal.py` exposes `GOAL_JUDGE_SYSTEM_PROMPT`, `GOAL_CONTINUATION_PROMPT_TEMPLATE`, and `render_goal_continuation_prompt()`. The dynamic per-turn user prompt stays in `goals.py` (it's conditional logic, not a static template).
  - **Reasoning-judge guidance documented, no magic in code**: judge models that need a large output budget (DeepSeek-Reasoner, o-series, qwq) must be constructed with `max_completion_tokens` set explicitly by the caller. The prior in-place mutation helper `_ensure_judge_output_budget` was removed — it was opaque, surprising, and mutated user-owned state. See `docs/advanced/goals.md` "Reasoning judge 的特别注意" for the recipe.

- Standing-goal loop P0 + P1 (S + A tiers):
  - Ergonomic SDK surface on `Agent`:
    - `Agent.run_goal(objective, *, turn_budget=..., token_budget=..., wall_clock_budget_sec=..., attach_goal_tool=True, event_callback=...) -> GoalRunResult` — one-liner that drives the whole loop. Replaces the previous low-level `GoalManager(agent._session_log, judge_model=...)` + hand-written driver loop.
    - `Agent.get_goal_manager(...)` for power users who want to drive turns by hand without touching `SessionLog`.
    - `Agent.enable_goal_tool()` attaches `GoalTool.update_goal` so the model can self-mark `complete` / `paused`.
    - `Agent._session_log` and `Agent.goal_manager` are now formally declared dataclass fields (no `getattr` speculation).
    - New `agentica.goals.GoalRunResult(status, reason, run_response, goal, turns_used)` with `response_content` convenience property.
  - `Runner._run_impl` early-loads any persisted active `GoalState` from `SessionLog` and binds `TaskAnchor` to the goal objective — SDK paths now get goal-aware retrieval automatically, not just the CLI.
  - `GoalState` gains `token_budget` / `tokens_used` / `wall_clock_budget_sec` / `wall_clock_used_sec` and a new `budget_limited` status (semantically distinct from `paused`). Hard budget caps take precedence over tool short-circuit and judge.
  - `agentica.tools.goal_tool.GoalTool.update_goal(status, reason)`: receive-only model tool letting the agent self-mark `complete` or `paused` (cannot rewrite the objective). CLI auto-attaches on `/goal` set and detaches on goal termination.
  - `RunEventType.goal_set / goal_continuing / goal_completed / goal_paused` events emitted through an optional `GoalManager.event_callback`.
- New example `examples/cli/03_goal_loop_demo.py`: 4-scenario SDK tutorial (`run_goal()` one-liner / budgets / event_callback / manual loop) against a real LLM.

### Changed
- `GoalManager.evaluate_after_turn` now charges turn counters (`turns_used`, `tokens_used`, `wall_clock_used_sec`) BEFORE any short-circuit branch so per-turn cost is always tracked, even when a tool ends the loop. Decision priority is now: budget cap > tool signal > judge.
- `GoalRunResult` field renamed `final_response` → `run_response` (typed `Optional[RunResponse]`, was untyped `Any`) and the convenience property `final_text` → `response_content`, to align with Agentica's existing `Agent.run_response` / `RunResponse.content` terminology. `final_*` was an LLM-style modifier that didn't add information.
- `agentica.goals.DEFAULT_TURN_BUDGET` bumped 20 → 100. Rationale: with `token_budget` and `wall_clock_budget_sec` now acting as the real hard caps, `turn_budget` is the safety-net against runaway loops; aggressive values (20–50) tripped accidentally on real coding workflows. Token / wall-clock budgets still bound actual cost, so a loose default is safe.

### Changed
- Top-level lazy imports (e.g. `from agentica import Knowledge`, `Claude`, `SqliteDb`, `Swarm`, ...) no longer emit `DeprecationWarning`. They are now treated as stable v1.x public API alongside the sub-module paths. The `DEPRECATED_TOP_LEVEL` registry has been removed; the planned v2.0 forced migration is dropped.
- `SearchSerperTool`: fix misuse of `logger.warning(..., DeprecationWarning)` for the `serper_api_key` alias (the extra arg was silently ignored).

## [1.4.5] - 2026-05-13

### Fixed
- `search_memory` now falls back to recent long-term memories when keyword search has no high-confidence matches.
- Langfuse tracing now preserves Agent `user_id` and `session_id` on both root traces and OpenAI wrapper metadata.

## [1.4.2] - 2026-05-10

### Fixed
- Default model resolution now preserves OpenAI priority when `OPENAI_API_KEY` is configured, falls back to Anthropic when `ANTHROPIC_API_KEY` is configured, and then checks OpenAI-compatible provider keys.
- Agent-owned LLM tools now reuse the parent agent model before resolving a fallback provider, avoiding accidental OpenAI usage when another main provider is configured.
- Experience capture and skill upgrade LLM calls continue to follow the agent auxiliary model or main model instead of creating a separate provider.

## [1.4.0] - 2026-04-23

### Added — Gateway IM Channels
- **`agentica.gateway.channels.QQChannel`**: 接入 QQ 开放平台机器人（`qq-botpy` WebSocket，C2C 私聊 + 群 @ 消息），自动缓存最新 `msg_id` 用于回包；新增 extras `agentica[qq]`
- **`agentica.gateway.channels.WeComChannel`**: 接入企业微信智能机器人（`wecom_aibot_sdk` WSClient），按 `chat_id` 缓存入站 `frame` 用于 `reply_stream`；新增 extras `agentica[wecom]`
- **`agentica.gateway.channels.DingTalkChannel`**: 接入钉钉机器人（`dingtalk-stream` Stream 长连接 + HTTP 回包），自动管理 `accessToken` 缓存与续期；区分 1-to-1（`channel_id=staffId`）与群（`channel_id="group:<openConversationId>"`）；新增 extras `agentica[dingtalk]`
- **`agentica.gateway.channels.WeChatChannel`**: 接入个人微信（内联 `WxBotClient` 走 ilinkai 私有 HTTP 长轮询，QR 扫码登录 + token 持久化），后台线程跑阻塞 loop，跨线程 `call_soon_threadsafe` 派发到主事件循环；新增 extras `agentica[wechat]`
- **`ChannelType`**: 扩展 `QQ` 与 `WECOM` 两个枚举值
- **`Settings`**: 新增 `qq_*` / `wecom_*` / `dingtalk_*` / `wechat_*` 字段及对应环境变量加载（`QQ_APP_ID` / `WECOM_BOT_ID` / `DINGTALK_CLIENT_ID` / `WECHAT_TOKEN_FILE` …）
- **`docs/advanced/gateway.md`**: 新增 Gateway 完整文档，覆盖架构图、所有 IM 渠道的环境变量配置、HTTP API、自定义渠道、故障排查
- **34 个新单测**: `tests/test_gateway_channel_{qq,wecom,dingtalk,wechat}.py`，全部 mock 各家 SDK，无外部依赖

### Changed
- `agentica/gateway/main.py::_setup_channels()`：按需注册 4 个新渠道，凡是缺关键凭据自动跳过并打日志
- `agentica/gateway/channels/__init__.py`：re-export 新增的 4 个 Channel 类
- 版本号：`1.3.6rc1` → `1.4.0`（按 SemVer：新增公共 Channel 类 → minor bump）

### Notes
- 所有新渠道都遵循"懒加载 SDK + 缺失依赖时抛清晰 `ImportError`"的现有模式
- WeChat 渠道走的是非公开私有协议（ilinkai），仅推荐个人 / 内部场景使用

### Added (Stage 2 + Stage 3)
- **`_DEPRECATED_TOP_LEVEL` mapping** in `agentica/__init__.py`: 35+ symbols flagged for v2.0 migration
- **DeprecationWarning** emitted when accessing top-level deprecated paths like `from agentica import Knowledge` / `Claude` / `VectorDb` / `SqliteDb` / `Swarm` etc., guiding users to explicit sub-module imports
- **`agentica.workspace` package**: Split monolithic `workspace.py` (1402 lines) into a package structure for incremental modularization

### Changed (Stage 2 + Stage 3)
- `agentica/__init__.py` docstring: rewritten with v1.3.6+ recommended import style guide + backward-compat note
- `agentica/workspace.py` → `agentica/workspace/base.py` (file move, zero business code change)
- `agentica/workspace/__init__.py` re-exports `Workspace`, `WorkspaceConfig`, plus module-level constants for test mocking
- `tests/test_workspace.py`: updated 3 patch paths from `agentica.workspace.AGENTICA_HOME` → `agentica.workspace.base.AGENTICA_HOME` (reflects new package structure)
- `tests/test_skill_lazy_loading.py`: updated `importlib.reload` target from `agentica.workspace` → `agentica.workspace.base`

### Compatibility
- **100% backward compatible**: all top-level imports still work; only emit DeprecationWarning
- `from agentica.workspace import Workspace` path is unchanged for all 11 internal usages and external users

## [1.3.6] - 2026-04-18 (sdk-dev branch)

### Added
- **`pyproject.toml`**: 新打包配置，对标 agno 细粒度 extras 风格 + 超级组合 extras
- **`docs/API.md`**: Public API Tier 1/2/3 稳定度合约
- **20+ 细粒度 extras**: `agentica[rag]` / `[qdrant]` / `[chroma]` / `[gateway]` / `[mcp]` / `[acp]` / `[arxiv]` / `[yfinance]` / `[browser]` / `[ddg]` / `[exa]` 等
- **8 个超级组合 extras**: `[tools-search]` / `[tools-research]` / `[tools-finance]` / `[tools-media]` / `[tools-browser]` / `[vectordbs]` / `[storage]` / `[models]` / `[tracing]` / `[full]`
- **`agentica.model.anthropic.Claude`**: Anthropic 直接默认装（核心 provider）
- 友好 `ImportError` 提示：未安装对应 extras 时，`agentica.gateway` / `agentica.mcp` / `agentica.acp` / `agentica.db.SqliteDb` 等会抛出带 `pip install agentica[xxx]` 命令提示的清晰错误

### Changed
- **依赖瘦身**：默认 `install_requires` 从 23 个 → **19 个**（M1-核心 A+ 方案；瘦身 17%）
- **默认产品化能力保留**：Workspace / CLI / DeepAgent 内置工具（web_search, fetch_url, file, shell, todo, task）全部默认可用
- **核心新增 6 个**：`beautifulsoup4` / `lxml` / `markdownify` / `requests` / `puremagic` / `tqdm`，确保 `agentica` CLI 和 DeepAgent 默认工作
- `setup.py` → `pyproject.toml`（PEP 621 标准）
- `requirements.txt`：更新为核心 19 个依赖的参考清单，实际以 `pyproject.toml` 为准
- `agentica/__init__.py` lazy loading：增加 `_LAZY_ATTR_OVERRIDES` 修复 `LiteLLM` / `DeepSeek` / `Moonshot` 等 alias 的延迟加载（pre-existing bug）

### Fixed
- `test_lazy_loading.py::test_all_public_names_accessible`：修正对缺失 extras 时的友好 ImportError 处理，不再误报
- **CLI 默认可用性**：之前一度把 `bs4` 移到 `[crawl]` extras 导致 `agentica --query` crash；本版通过把 6 个工具依赖纳入核心保证 CLI / DeepAgent 默认开箱即用

### Removed
- 无（1.3.6 是内部收敛 + 打包优化，不删除 Public API）

### Migration Notes
- **向后兼容 100%**：装 `pip install agentica` 即可获得 v1.3.5 的"开箱即用 DeepAgent + CLI"完整体验
- **`pip install agentica[full]`** 等价于 v1.3.5 完整能力（含 RAG / Gateway / MCP / 40+ 第三方工具）
- 仍使用 `setup.py` 等旧安装方式的场景需迁移到 `pyproject.toml`（PEP 621 自 Python 3.10 标准）

## [1.3.5]

### Added
- `MemoryType` enum — four-type memory classification (`user`, `feedback`, `project`, `reference`) for workspace memory entries
- `MemoryEntry` Pydantic model — typed memory entry with `name`, `description`, `memory_type`, `file_path`, `content` fields
- `Workspace.write_memory_entry()` — write a typed memory as an individual `.md` file with YAML frontmatter, auto-updates `MEMORY.md` index
- `Workspace.get_relevant_memories()` — relevance-based recall: parses `MEMORY.md` index, scores entries by keyword overlap against current query, loads only top-k content files; supports `already_surfaced` set for session-level dedup
- `Workspace._update_memory_index()` — enforces MEMORY.md hard limits (200 lines / 25KB); FIFO eviction of oldest entries
- `Workspace._score_memory_entries()` — hybrid keyword scoring (word-level + char 2-gram) supporting both English and CJK queries
- `Workspace._strip_frontmatter()` — strips YAML frontmatter before injecting memory content into system prompt
- Memory drift-defense note — appended to all injected memory to guard against stale file/function references
- `WorkspaceMemoryConfig.max_memory_entries` — max memory entries to inject per run (default: 5); replaces removed `memory_days`
- `Agent._surfaced_memories` — session-level set tracking surfaced memory filenames, prevents cross-turn re-injection of same entries
- `Agent.get_workspace_memory_prompt(query)` — now accepts `query` parameter, passes it to `get_relevant_memories()` for query-aware recall
- `CompressionManager.auto_compact(working_memory=...)` — reuses `WorkingMemory.summary` directly when available, skipping LLM summarization call; faster and cheaper with no information loss
- `SandboxConfig.allowed_commands` — optional command whitelist for `execute` tool (prefix-matched on first token)
- `Agent._running` flag — concurrent reuse of the same Agent instance now logs a warning
- `WorkingMemory.max_messages` — soft FIFO eviction limit (default: 200) to prevent unbounded memory growth
- `Message.role` field validator — rejects invalid roles at construction time (`system`, `user`, `assistant`, `tool` only)

### Changed
- `Workspace.get_memory_prompt(days=N)` removed — replaced by `get_relevant_memories(query, limit, already_surfaced)`; full-dump memory injection is no longer the default behavior
- `WorkspaceMemoryConfig.memory_days` removed — no longer needed; relevance-based recall replaces time-window-based loading
- System prompt memory zone: both `_build_default_system_message` and `_build_enhanced_system_message` now extract `self.run_input` as query and pass it to `get_workspace_memory_prompt(query=...)`

### Fixed
- `update_model()` now clears `model.functions` and `model.tools` before each run, preventing tool accumulation on reused Agent instances
- `OpenAIChat.response()` raises `ValueError` instead of `IndexError` when `choices` is empty
- `AnthropicChat.response()` raises `ValueError` instead of crashing when `content` is empty
- `FunctionCall.execute()` generator result concatenation now uses `str(item)` to prevent `TypeError` on non-string generators
- `OpenAILike` warns at construction time when `api_key` is still the placeholder `"not-provided"`
- `_load_mcp_tools` removed redundant `if/else` branch (both branches were identical)
- `task()` recursion depth capped at 5 levels via `_task_depth` context propagation

### Added (Tests)
- `tests/test_workspace.py::test_get_memory_prompt` updated to cover `write_memory_entry()` + `get_relevant_memories()` with and without query
- `tests/test_hooks.py` — AgentHooks, RunHooks, `_CompositeRunHooks`, ConversationArchiveHooks
- `tests/test_runner.py` — empty message guard, concurrent warning, run_timeout, structured output fallback
- `tests/test_swarm.py` — parallel mode, partial failure, duplicate name detection
- `tests/test_model_validation.py` — empty choices, usage=None, Message role validator, structured output fallback

---

## [1.3.2] — 2026-03-17

### Added
- `Swarm` — multi-agent parallel autonomous collaboration (`agentica/swarm.py`)
- `ConversationArchiveHooks` — auto-archives conversations to workspace after each run
- `_CompositeRunHooks` — internal wrapper for composing multiple `RunHooks` instances
- `RunConfig.enabled_tools` / `enabled_skills` — per-run tool/skill whitelisting
- `Agent.disable_tool()` / `enable_tool()` / `disable_skill()` / `enable_skill()` — agent-level runtime control
- `Agent._load_runtime_config()` — loads tool/skill enable/disable from `.agentica/runtime_config.yaml`
- `SandboxConfig.blocked_commands` — command-level blacklist for `execute` tool
- `examples/agent_patterns/08_swarm.py` — Swarm usage example
- `examples/agent_patterns/09_runtime_config.py` — Runtime config example
- `examples/agent_patterns/10_subagent_demo.py` — SubAgent example

### Changed
- `deep_agent.py` renamed to `tools/buildin_tools.py`; `DeepAgent` now uses `BuiltinFileTool`, `BuiltinExecuteTool`, `BuiltinWebSearchTool` etc.
- `Runner._run_impl` — removed duplicate auto-archive logic; archive is now handled exclusively by `ConversationArchiveHooks`

---

## [1.3.1] — 2026-03 (v3 post-merge cleanup)

### Added
- `WebSearchAgent` with search enhancement modules (`search/orchestrator.py`, `query_decomposer.py`, `evidence_store.py`, `answer_verifier.py`)
- Extended thinking support for Claude and KimiChat models
- Kimi provider integration (`model/kimi/`)

### Fixed
- Preserve tool call messages in multi-turn conversation history
- Deduplicate Model layer, unify `RunConfig` signatures

---

## [1.3.0] — 2026-03 (v3 architecture refactor)

### Changed (Breaking — internal architecture, public API preserved)
- **Phase 1**: Removed 19 thin provider directories; unified via `model/providers.py` registry factory
- **Phase 2**: Converted `Model` hierarchy from Pydantic `BaseModel` to `@dataclass`
- **Phase 3**: Async interface consistency + structured output for all providers
- **Phase 4**: Added `@tool` decorator and global tool registry (`tools/registry.py`)
- **Phase 5**: Extracted `Runner` from `RunnerMixin`; `Agent` now delegates execution via `self._runner`
- **Phase 6**: Unified guardrails with `core.py` abstraction layer
- **Phase 7**: Simplified `__init__.py` lazy loading
- **Phase 8**: 35 new v3 tests

### Added
- `AgentHooks`, `RunHooks` lifecycle hooks system
- `RunConfig` per-run configuration overrides
- `SubAgent` for isolated ephemeral task delegation
- Skill system (`skills/`) — Markdown+YAML frontmatter skill injection
- ACP server for IDE integration (Zed, JetBrains)

---

## [1.2.x] and earlier

See git log for historical changes prior to the v3 refactor.
