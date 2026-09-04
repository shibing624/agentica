# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [Unreleased]

#### breaking
- **删除 Serply 搜索（`SearchSerplyTool` / `web_search` 的 `serply` 引擎 / extra `[serply]` / CLI `--tools search_serply`）**：厂商自己合入的 vendor 营销，Google 搜索继续用 Serper。`SERPLY_API_KEY` 和 `AGENTICA_SERPLY_SEARCH_TYPE` 不再被读取。

#### fixes
- **CLI 退出 / 回合里的 `EINTR` 不再甩堆栈**：macOS 上 Ctrl+C 会打断 `getcwd` 等系统调用，Python 抛的是 `InterruptedError`（`[Errno 4] Interrupted system call`），不是 `KeyboardInterrupt`。以前回合里当成「provider 挂了」，`finally` 里无条件 `worktree_binder.release()`（没开 `--worktree` 也跑 git/`getcwd`）再中一次就跳过告别语。现在没进过托管 worktree 的会话退出是空操作；进过的也按用户中断处理，清理失败不挡正常结束。
- **流式中途断流按 `max_api_retry` 同 turn 重发**：SSE 被掐（`incomplete chunked read`）发生在已吐 chunk 之后，`stream_with_retry` / `_call_with_retry` 都接不住。Runner 消费点回滚本轮 `model_response` 内容并重开流，预算与同模型 API retry 共用（CLI 默认 2，即再打一次）；耗尽照旧抛。已 yield 的 chunk 收不回——交互会重打一段，`--print` / delegate 的 stdout 会拼上半截。同时修了 `_call_with_retry` 分类：`RETRYABLE`（含 `incomplete chunked read`）先于 `FALLBACK_ONLY`（含 `connection`），否则这类断流永远走 fallback、从不在同模型上重试。
- **`apply_patch` 信封按 Codex/OpenCode 宽松提取**：小模型常把补丁包进 ` ```patch ` 围栏、前面加说明、漏 `*** End Patch`，或写成 `{"patch":"..."}`，以前一律报 `Patch must start with '*** Begin Patch'`。现在找出 Begin/End（或裸的 `*** Update/Add/Delete File:`）再解析；hunk 仍精确匹配，缺 File 头或行首不是空格/`-`/`+` 照旧拒绝。
- **`apply_patch` 对不上时点出那一行**：`Hunk N: context not found: '…'` 带上 hunk 里文件中不存在的那一行，不再只报一句 not found。不回 Expected/Actual 预览。
- **`apply_patch` 找回忘了前导空格的 keep 行**：无 ` `/`-`/`+` 前缀的行若与当前文件某行完全一致（或去行尾空白后只对应一种原文），当成 keep；对不上或多种原文仍报 `Malformed patch`，不做空白/缩进 fuzzy。

#### changes
- **`worktree` 用法改走内置 skill，人用 `/worktree`**：工具还在（模型要搬家仍得调它，`execute git worktree` 只多一个目录、会话还在旧 cwd）。以前每轮把整段 `<worktrees>` policy 塞进 system prompt。现在判断在 `agentica/skills/bundled/worktree/`，目录匹配才读；人用 `/worktree status|use|merge|remove`（和 `--worktree` 同一套 binder）。
- **peer 收信规则并进 `multi-agent` skill**：`list_agents` / `send_message` 仍每轮注册。以前 `PEER_MESSAGING_POLICY` 每轮塞进 system prompt（授权头、证据用路径、peer 派的活不要 `ask_user_question`）。现在只在目录对上、模型去读 skill 时才进上下文。消息头 `format_for_model` 和 `ask_user_question` 的「这个框能到谁」仍每轮在。
- **cron 用法改走内置 `cron` skill**：`cronjob` 工具还在。以前 `CronTool.get_system_prompt()` 每轮灌用法 + 按表面切换的 `daemon_hint`（CLI 写 `/cron daemon on`，gateway 写 `cron.enabled`）。现在判断在 `agentica/skills/bundled/cron/`，两种开 daemon 的办法写在同一份 skill 里按表面选；`daemon_hint` / `CLI_DAEMON_HINT` 删掉。人用 `/cron`（skill 同名 auto-command 让位给已有斜杠命令）。
- **CLI / Web 的 `apply_patch` 调用行显示工作区相对路径**：以前只报 `Edited 1 file (+8 -3)`，看不出改了哪个文件。现在和 `read_file` / `write_file` 一样带上路径（`apply_patch agentica/cli/commands/session.py - Edited 1 file (+8 -3)`）；多文件逗号并列。
- **`execute` 鼓励一条长命令**：docstring 和 `tools.md` 以前正例都是单条短命令（还写着「尽量别 `cd` / 命令里不要换行」），模型就拆成多次往返。现在推荐共享目录下用管道、`&&`、`python3 - <<'EOF'` 拼完验证/构建/启动，输出用 `| tail` / `| head` 兜住；精确改代码仍走 `apply_patch`。
- **CLI / Web 的 `write_file` 调用行显示工作区相对路径**：以前只留文件名（`session.py`），和 `read_file` 的 `agentica/cli/commands/session.py` 不一致。现在两边都走同一套缩短（cwd 下相对，工作区外保留原路径）。
- **去掉 `@agentica-ai/sdk` 的 GitHub Actions 自动发布**（删除 `.github/workflows/npm-publish.yml`）：和 PyPI 一样改成仓库里手动 `npm publish`。`v*` tag 不再二次 PUT；`1.4.15` 已经 staged 过，Actions 再推一次会 E409，而 `npm view` 看不见 staged 版本，跳过检查拦不住。

## [1.4.15] - 2026-09-01

#### breaking
- **CLI `/export` 默认导出 session JSONL，不再是对话瘦 JSON**：以前 `/export` / `/save` 把 `working_memory.messages` 存成一份没有 event、没有工具正文的 JSON。现在默认拷贝磁盘上那份 `<session_id>.jsonl`（与 Web 轨迹同一文件）。旧行为改为 `/export messages [path]`；`/export analysis [path]` 写出与 `GET /api/sessions/{id}/trace/analysis` 相同的 JSON。
- **`apply_patch` 只精确匹配上下文**：不再对空白 / 引号做 fuzz。对不上就是 `Hunk N: context not found`。死的 SDK 类 `PatchTool`（双格式 unified/V4A）删除；补丁走 `BuiltinFileTool.apply_patch`，模块只留 `apply_diff` / `parse_patch_envelope`。
- **删除 `write_html` 和 `ShellTool`**：长报告用 `write_file` 写 HTML；shell 用 `execute` / `BuiltinExecuteTool`。`get_builtin_tools` / `DeepAgent` 去掉 `include_html_report`。
- **`grep` 只留 `pattern` / `path` / `include` / `limit`**：去掉 `output_mode`、`case_insensitive`、`fixed_strings`、`context_lines` / `before_context` / `after_context`。
- **`DeepAgent` / `get_builtin_tools` 去掉 `peer_conflict_checker`**：编辑成功后不再附「别的会话也改了这个文件」提醒。

#### features
- **npm 包 `@agentica-ai/sdk`（`sdk-ts/`）**：给跑着的 `agentica-gateway` 用的 TypeScript HTTP 客户端（session / chat SSE / 审批），不是 Python `Agent` 的移植。Web 启动路径不变，仍是 `agentica-gateway`；PyPI wheel 继续打进编译好的 UI。发布走 GitHub Actions：tag `v*` / `sdk-v*`（或手动 Run workflow）`npm publish` 到 `https://registry.npmjs.org/@agentica-ai/sdk`（org `agentica-ai`，secret `NPM_TOKEN`，npmjs 账户 `shibing624-xm`）。已经发布过的 version 会跳过，避免 Python 发版 tag 把同一版再推一次。
- **Gateway Docker 镜像**：`Dockerfile` 用 Node stage 编 Web UI，再 `pip install ".[gateway]"`。仓库根 `docker-compose.yml` 一键起自托管服务；本机 `pip` / Desktop 不受影响。
- **explore / code 子代理会用只读 `execute` 管道**：`execute` 本来就在 `allowed_tools` 里（`execute_policy: read_only`），但 explore 的 prompt 只提 `glob`/`grep`/`read_file`，包装后的说明也只写 git/测试，模型就不会 `cd` 到工作区外的树去 `rg`。现在 prompt 和只读 `execute` 说明都写上 `rg`/`find`/`head` 管道。
- **`execute` 输出里出现的路径也算 grounded**：`rg` 等扫到的精确路径字符串可以直接给 `read_file` / `apply_patch`，不必再绕一次 `glob`。`read_file` 仍只 grounded 它打开的那一个文件。
- **`web_search` 新增 `serply` 引擎（`SearchSerplyTool`）**：[Serply](https://serply.io) 的 Google 搜索 API，`SERPLY_API_KEY`，无额外依赖（`pip install agentica[serply]`）。同一个 key 还覆盖 Google News / Google Scholar：SDK 传 `SearchSerplyTool(search_type="news"|"scholar")`，`web_search` 分发器路径用 `AGENTICA_SERPLY_SEARCH_TYPE` 切换，模型看到的工具名不变。CLI `--tools search_serply`。API 文档见 [serply.io/docs](https://serply.io/docs)。
- **`apply_patch` docstring 示例以 `+#` 插入注释开头**：`@@` 后第一行就是新增，避免模型先整文件空格拷贝造成空操作。空操作不再报 `Malformed patch`。首行已是 `*** Update/Add/Delete File:` 时自动补 `Begin/End Patch`（正文语法不变）；markdown 围栏和其它缺信封仍拒绝。
- **CLI 并行工具改成 Kimi 式整块 flush**：未完成的调用停在输入框上方的 live 窗口，按开始顺序等前缀都结束后才把「调用行 + 结果」一起打进 scrollback。以前 `execute` 一开始就打印调用行，并行的 `grep` / `write_file` 结果会插在调用和 `⎿` 之间，看起来像挂错工具；现在不再需要 `↳` 锚点。`--print` 不变。
- **CLI / SDK 与 Web 共用同一套 session 轨迹出口**：一份 JSONL + `SessionLog.analyze()`。SDK：`agent.session_log`（公开句柄）、`.format_trace()`、`.export()`。CLI：`/trace` 打 rounds/tokens/工具（`/trace <n>` 展开一轮），`/status` 增加 **Session log** 路径（原 `Log file` 改名为 **Debug log**，避免和 jsonl 混淆）。Gateway `/trace/analysis` 改为调用 `log.analyze()`，不再手拼一份。公开 API 见 `docs/API.md`。
- **`/goal` 默认 token 预算不限**：CLI `/goal xxx` 与 SDK `run_goal()` 不传 `token_budget` 时不再回落 500_000（`DEFAULT_TOKEN_BUDGET` 改为 `None`）。要限额度显式传 `--tokens N` / `token_budget=N`；`-1` 仍是不限。Web 目标芯片默认显示「预算不限」，点一下才打开 Token 预算输入（空=不限，支持 `500k`/`2m`），Escape 关掉输入框而不退出目标模式。

#### fixes
- **CLI 渲染工具输出 / 纯文本回答时不再把 `[/xxx]` 当成 Rich 闭合标记**：`execute` 结果里的 `[/usr/bin/cmake]`、grep 路径、agent 纯文本里的 `[/red]` 以前走 `console.print(..., markup=True)`，Rich 抛 `MarkupError`，整轮报 `Agent execution failed`。现在不受信任的正文 `markup=False`，拼进标记串的字段先 `escape()`。
- **Web 打开 CLI 会话（以及刷新后的网页对话）不再丢掉 tool call / 结果**：`hydrateSession` 以前只把 session JSONL 里的 `user` / `assistant` 正文拼成气泡，`assistant.tool_calls` 和 `type: "tool"` 行直接丢掉，所以侧栏里点开 CLI 会话只剩问答、没有 WorkGroup。现在按 harness 同一条规则重建：一轮用户提问折成一条 assistant（思考 → 工具卡 → 终答），参数和结果走与实时 SSE 相同的 `parts`。打开会话时以服务端日志为准覆盖本地缓存（正在流式的会话不覆盖）。
- **`search_memory` 搜对话归档不再 `NameError`**：按 `---` 切 block 用了 `re.split`，模块顶上没 `import re`，一点到 conversation 源就炸。
- **`apply_patch` docstring 补回「改之前先 `read_file`」**：Update/Delete hunk 必须对着当前文件原文，不能凭记忆拼上下文。
- **CLI 回答 `ask_user_question` 时能看到完整选项**：提问组件和 live 窗口抢同一块底部高度，live 的 `LIVE_MAX_ROWS=12` 把选项挤出屏幕。有未决提问/审批时收起 live；选项折行按显示宽度（`get_cwidth`）预留行数，中文不再按 `len` 少算导致后几项被裁。
- **`apply_patch` 去掉误导性 Expected/Actual 预检预览**：上下文对不上只报 `Hunk N: context not found`；缺 `*** Begin Patch`、hunk 行没以空格/`-`/`+` 开头，报 `Malformed patch`，不再包装成「preflight + Actual from line N」。原子写入（失败不改任何文件）仍在。整函数重写用 `write_file`。
- **工具结果不再塞启发式旁白**：`execute` 非零退出只报 exit code（仍按命令决定要不要 raise），不再附 Note；`write_todos` 不再 verification nudge；空文件返回 `File is empty: …` 而不是 `<system-reminder>`；`fetch_url` 去掉四条 IMPORTANT；`background=True` 成功结果只留 id / pid / log。路径 grounded 政策只写在 `tools.md`。**编辑后的 LSP/Pyright 诊断仍附在 `write_file` / `apply_patch` 结果上**（`--enable-diagnostics`），缩进对不上时能直接看到。
- **CLI live 窗口跨线程读写加锁，provider 断流也会 flush 在飞工具块**：spinner 每 120ms `compose_live` 与 turn 线程同时改 `LiveToolStore` 的 OrderedDict，迭代中增删会让 spinner 线程静默死掉、状态栏冻结。store 加锁、读侧快照，spinner 循环包异常；泛化 `except Exception` 与取消路径一样调 `abandon_live()`。
- **CLI live 窗口跟进**：并行 `task` 按 description 绑 subagent；结果 id 对不上时先打完整 call+result；剥 Rich 标签不再吃正文 `[...]`；删掉已无生产引用的 `_ToolResultSequencer`；行数上限 `LIVE_MAX_ROWS=12` 只定义一处。
- **Layer 2 空摘要不再当成压缩成功**：`auto_compact` / `_summarise_conversation` 对摘要 `strip()`，空白或抽不出正文（含把空 `resp` `str()` 成对象 repr 的路径）整段放弃，不替换 `messages`、不写 `compact_boundary`。WorkingMemory 里的空白 session summary 不再走 SM-compact，回落到真正的摘要 LLM。
- **CLI 状态栏未配置的思考强度显示 `default` 而不是 `off`**：config.yaml 没写 `reasoning_effort` / `reasoning` 时请求里根本不带这个字段，API 用它自己的内置强度；以前 `describe_thinking_mode()` 把「没写」当成关，状态栏就打出 `opus-5-openoneapi openai/claude-opus-5 off`。现在未覆盖是 `default`，只有显式 `off`/`none`/`disabled` 才显示 `off`。

---

更早的发布说明（1.4.14 及以前）在 [docs/changelog-archive.md](docs/changelog-archive.md)。
