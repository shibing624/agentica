# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [Unreleased]

#### breaking
- **删除 Serply 搜索（`SearchSerplyTool` / `web_search` 的 `serply` 引擎 / extra `[serply]` / CLI `--tools search_serply`）**：厂商自己合入的 vendor 营销，Google 搜索继续用 Serper。`SERPLY_API_KEY` 和 `AGENTICA_SERPLY_SEARCH_TYPE` 不再被读取。

#### features
- **Claude 也认 `cache_control_session_header`，值按 CLI 会话换**：聚合型代理不粘路由会打到不同后端，缓存冷、schema 400 也更多。header（如 `X-Session-Id`）注入当前 `session_id`，新开会话换路由；没有会话才回落到 `~/.agentica/cache/cache_routing.json` 里按 `base_url` 存的 id。不再把第一次（还没 session）的 fallback 冻在实例上。`get_model` 不再只给 OpenAI chat 传这个字段；setup 对 anthropic 只问粘路由 header。
- **`config.yaml` 文档补上 prompt cache 与粘性路由**：`guides/config.md` 的 Profile schema 表原来缺 `enable_cache_control` / `cache_control_session_header` / `cache_control_messages` / `cache_keepalive` / `default_headers` 五项（`extra_headers` 也没写明对 anthropic 不生效）。新增「代理网关的粘性路由」一节：账号级（`default_headers` 写死）与会话级（`cache_control_session_header` 按会话取值）的取舍、两者同配时显式值优先、以及换项目目录会重写缓存。

#### fixes
- **Anthropic 原生路径不再把可选参数全标成必填**：`Claude.get_tools()` 自己按「type 里没有 `null` 就算 required」重算了一遍，而 `get_json_schema` 恰好相反——它用「不出现在 required 里」表达 Optional（type 数组好几家 provider 不收）。于是每个参数都被判成必填：`read_file` 要求同时传 `offset`/`limit`/`tail`，`execute` 要求传 `timeout`/`parallel_safe`。现在直接用 `parameters["required"]`（签名默认值算出来的，或 `parameters_override` 给的），非 list 一律收敛成 `[]`，`properties` 里没有的名字丢掉。只影响 `model_provider: anthropic`，OpenAI 路径发的是 `Function.to_dict()`，一直是对的。
- **Anthropic 工具 schema 不再抹掉 enum / items / 嵌套 properties**：每个参数以前被压成 `{"type","description"}` 两个键，自动生成的 schema 里没有 per-param description，所以描述还都是空串。现在整份属性 schema 原样透传。
- **deferred 和不可用的工具不再进 Anthropic 的 tools**：OpenAI 走 `self.tools` / `get_tools_for_api()`，这两道闸是白拿的；Anthropic 这条路径遍历 `self.functions`，两道都漏了，`deferred=True` 的 MCP 工具会连 schema 一起发出去，`available_when` 返回 False 的也照发。deferred 仍然可执行，只是不再出现在 schema 里。
- **全可选的 `response_model` 不再让 Anthropic 请求 400**：字段全带默认值时 pydantic 的 `model_json_schema()` **不会**输出 `required` 键，合成的 `structured_output` 工具于是发出 `"required": null`，API 回 `input_schema.required: Input should be a valid list`。
- **`mcp` 2.x 仍能 import `McpTool`**：SDK 把 `streamablehttp_client` 改成了 `streamable_http_client`，并去掉 `GetSessionIdCallback`。以前 extra 一装上 2.x，`from agentica.tools.mcp_tool import McpTool` 直接 ImportError。现在认两个名字。
- **`grep` 超长行只回匹配窗口，并从左边拆 `path:line:`**：超过 2000 字不再吐整行，格式是 `file:line: col=N, line_len=M: ...window...`。以前用贪婪的最后一个 `:数字:` 拆 rg 输出，JSONL 里的 `"12:30:45"` 会把行号和窗口抢走。`rg` 加 `--column`，读到一行就裁，结果里不再攒整行。
- **`apply_patch` schema 写明按行匹配**：超长单行（JSONL）这里改不了，那种情况自己做唯一字面替换。不改匹配引擎，也不往 `tools.md` 里教脚本。
- **CLI `write_file` / `read_file` / `apply_patch` 工作区外不再只显示文件名**：以前 `_display_path` 对不上 work dir 就收成 basename，`/tmp/dupprobe/probe.py` 变成 `probe.py`，diff 头也是。工作区内仍是相对路径，区外保留调用时的路径。
- **`/model` / Claude resume 剥 tool 时留下写入摘要**：切模型仍丢掉 thinking 和 tool 的 wire 格式（否则换 provider 会 400），但会追加一条 `<elided-tools>`：散文里的「写好了 193 行」不算证据，真正跑过的 `write_file` / `apply_patch` 列在下面。以前只留问答文本，下一轮就会接着演一遍没发生过的写文件。`execute` 输出不进摘要（会漏密钥）。摘要合进结尾那条 assistant 正文，不单独占一条消息：连续同角色轮次在 Bedrock 和部分聚合代理上是 `400 roles must alternate`，而剥离本身就是为了别 400；合并同时让操作幂等，resume 之后再按 `/model` 不会叠第二份。判据是「历史里真有 tool 轮」而不是 provider 能力，纯聊天会话 resume 到 Claude 不再被塞一条「tool calls were dropped」的假话。`/resume`、`/history` 回放时不显示这个内部标记。
- **CLI 两个路径缩短器不再各说一套**：`_display_path` 在工作区外直接返回未规范化的路径，`../` 会留在绝对路径中间（`/Users/me/proj/../other/x.py`）；现在和 `_shorten_path` 一样用 `normpath` 词法折叠（不用 `resolve`，否则 macOS 把 `/tmp` 变成 `/private/tmp`）。`~` 保持写成 `~`，不再展开成带用户名的绝对路径。
- **SDK `print_response` 的工具行按 agent 的 work dir 缩短**：`printer.py` 没给 `format_tool_display` 传 `work_dir`，回落到 `os.getcwd()`；`delegate(work_dir=...)` 或 SDK 嵌入时 agent 跑在别的目录，路径就按错的根算。
- **`apply_patch` 示例补上同文件多处**：一条补丁里一个 `*** Update File` 下几个 `@@` 改几处，不要拆成多次调用，也不要写两个同路径的 Update File。以前示例只有「一文件一 hunk」，模型就把同一文件拆成两次调用。
- **CLI 输入框折行后不再把上一行卷走**：`TextArea` 按显示宽度（`get_cwidth`，中文两列）折行，prompt 只算在第一逻辑行；以前用 `len(line) // (终端宽-2)` 估高度，中文刚折到第 2 行时盒子仍是 1 行，光标把第一行顶没，再打约一行才长高回来。现在跟 prompt_toolkit 同一套折行算行数。
- **CLI `ask_user_question` 不再裁掉问题后半段**：提问组件以前自己算预留行数，把整段 `prompt` 当成一行折行，短段落被低估，第 2 问和选项出了屏。现在原文倒出来（前面加 `?`，空一行再列选项），窗口高度交给 prompt_toolkit。tool result 本身没截过。
- **内置工具 schema 不再点名别的可选工具**：`grep` 不提 `execute`，`execute` 不提 `read_file` / `apply_patch`，`task` / memory 也不再写对方的名字。`delegate` 可以提 `wait`（同一套后台 registry）。SDK 可以只装文件工具、不装 `execute`。`tools.md` 只在有文件工具时注入，同样不提 `execute`。按文件名过滤：收窄 `grep` 的 `path`。
- **`grep` 去掉 `include`**：schema 里写 `include=` 等于每轮教 GNU grep 的 `--include`，模型再抄到 `rg` 上就炸。参数只留 `pattern` / `path` / `limit`。
- **`execute` 不再改写 `rg --include`**：命令原样进 shell。过滤扩展名写 `rg -g '*.py'` / `rg -t py`，schema 也不再提 `--include`。
- **Claude 的工具结果终于落盘了**：Anthropic 用 `role="user"` + `tool_result` 块回答工具调用，落盘只认 `role="tool"`，所以每个原生 Claude 会话写下的都是 `assistant(tool_calls)` 后面什么都没有——真实工具输出一次也没进 jsonl，投影里全是孤儿 `tool_use` id。以前看不出来，是因为坏掉的 `sanitize_messages` 会注入「execution may have been interrupted」占位，那些占位恰好是 `role="tool"`，于是成了 Claude 会话唯一的 tool 行（历史日志里 `interrupted` 占 tool 行 100%）；上一条修掉占位后 tool 行直接归零。现在按 `tool_use_id` 拆成每块一行，`tool_name`/`is_error` 一并带上；in-turn flush 手里没有 FunctionCall 记录时，从发起那一轮的 `tool_calls` 取名字和参数。
- **`trajectory_skeleton` 认得 Claude 的工具回答**：`role="user"` + `tool_result` 块归一成与 `role="tool"` 相同的轨迹步（每块一个 `("tool", id)`）。以前整轮 Claude 读起来是一串普通 user 消息，配对检查根本看不见结果——专门用来抓孤儿 `tool_use` 的那条不变式，在唯一会因此报 400 的 provider 上是瞎的。
- **流式工具轮的旁白不再糊进终答**：以前 `model_response.content += chunk` 跨过整轮工具往上累加，Claude 中间 `assistant(tool_calls)` 的 `content` 又是 block 列表，落盘 `isinstance(str) else ""` 写成空串，所有旁白（包括模型自己吐的单独一行 `count`）堆在 jsonl 最后一条。现在中间旁白留在对应的 `assistant(tool_calls)` 上（抽出 text block），终答只留最后一轮；Claude/Ollama 也不再在每轮流结束 yield `\n\n`。不滤 `count`——那是模型输出。
- **`sanitize_messages` 不再把 Claude 的成功结果标成中断**：Claude 用 `role=user` + `tool_result` 块回答，以前只认 `role=tool`，每条成功调用前都插一句「execution may have been interrupted」。现在先吃掉这些块再判断缺没缺。`format_tool_results` 的缺 id 补齐落到 OpenAI/Ollama 基类，中断回合不再只靠下一跳 sanitize。
- **Claude 回合中断后不再留下孤儿 `tool_use`**：Anthropic 要求每条 `tool_use` 都有对应 `tool_result`。以前按位置 zip，中断或乱序返回就会贴错 id，后续每跳都 400。现在按 `tool_call_id` 配对，缺的补一条 interrupted 错误；这类 400 走 transcript sanitize，不再原样重发。
- **`delegate` 不再偶发选错模型**：不填 `model` 时子进程走当前会话的 profile（`session_profile`），不再按 `model_name` 扫 `config.yaml` 里第一个同名的。两个 profile 共用一个模型名时，对得上 `base_url` 才映射，对不上就带父会话的 `--base_url`，不猜。`model` 里带 `/` 的先当完整 id（代理的 `openai/glm-5`，或环境上下文的 `provider/<id>`），对不上再拆 `provider/name`。
- **`read_file` 的 `tail=0` 不再报错**：模型把 `tail=0` 当成「从头读」，以前抛 `tail must be >= 1` 白烧一轮。0 / 省略都是从头按 `offset`/`limit` 分页；`tail=N`（N>=1）才是末尾 N 行，文件比 N 短就整份返回；负的 `tail` 当成末尾 `|N|` 行。docstring / `tools.md` 写明两套分页，不要用 `tail=700` 表示「从头读 700 行」（那是 `limit=700`）。
- **`execute` 搜文本优先 `rg`，没有再 `grep`**：docstring 写 `rg -g '*.py' -n PAT -- path || grep -n PAT path`。命令不改写。
- **切模型只留问答文本，OpenAI ↔ Claude 双向可跑**：`/model`（含 `/config set` 真换了模型）把 thinking、tool call/result 都剥掉，只保留 user 问题和 assistant 回答。thinking 的 `signature` 绑签发模型，带着切就是 400；切模型本身是新一轮问答，那些内容也不值钱。能力按 wire 格式（OpenAIChat vs 原生 Claude），不按模型名猜。同会话里 `cache_control` 仍不准打在 thinking 上；Layer 1 缩过 `tool_use.input` 的旧回合同时丢掉旁边的 thinking。
- **Layer 1 不准动正在跑的那一轮工具（参数和 result 都不动）**：上下文一紧就把调用参数切成 `…[truncated]`，或把刚返回的 tool result 换成淘汰占位符，agent loop 等于吃掉自己的证据。`live_tool_round_start` 按消息位置护住未返回的调用和末批 result，不认工具名：内置 `write_file` / `execute` / `grep` / `glob`、SDK `tools=`、CLI `--tools`、Web extra、MCP 都在这一轮里。更早回合的超长参数换成 `<evicted-tool-arg chars=N>`，不是原文前缀。
- **`execute cat` 超大文件不再撑爆 live round**：以前 `communicate()` 把整份 stdout 读进内存，Layer 1 又不能淘汰刚返回的那一轮，下一跳直接超 `max context window`，CLI 收尾再甩 `Event loop is closed`。现在管道按 `max_output_length` 落盘预览（硬顶 64MiB 后杀进程），结果只留 `<persisted-output>`；读文件走 `read_file`（offset/limit/tail），不要整文件 dump。硬上限截断的头文案写 INCOMPLETE，不再说 Full output / 去 `read_file` 当全文；开了 `AGENTICA_REDACT_TOOL_OUTPUTS` 时落盘副本也脱敏。主动 SIGKILL 不当作命令失败。Layer 0 失败也截断，不再把原文送进下一跳。OpenAI 客户端跟 Claude 一样绑当前 loop，`close_client` 先摘掉再 aclose，避免关 loop 后 httpx `__del__`。
- **CLI 退出 / 回合里的 `EINTR` 不再甩堆栈**：macOS 上 Ctrl+C 会打断 `getcwd` 等系统调用，Python 抛的是 `InterruptedError`（`[Errno 4] Interrupted system call`），不是 `KeyboardInterrupt`。以前回合里当成「provider 挂了」，`finally` 里无条件 `worktree_binder.release()`（没开 `--worktree` 也跑 git/`getcwd`）再中一次就跳过告别语。现在没进过托管 worktree 的会话退出是空操作；进过的也按用户中断处理，清理失败不挡正常结束。
- **流式中途断流按 `max_api_retry` 同 turn 重发**：SSE 被掐（`incomplete chunked read`）发生在已吐 chunk 之后，`stream_with_retry` / `_call_with_retry` 都接不住。Runner 消费点回滚本轮 `model_response` 内容并重开流，预算与同模型 API retry 共用（CLI 默认 2，即再打一次）；耗尽照旧抛。已 yield 的 chunk 收不回——交互会重打一段，`--print` / delegate 的 stdout 会拼上半截。同时修了 `_call_with_retry` 分类：`RETRYABLE`（含 `incomplete chunked read`）先于 `FALLBACK_ONLY`（含 `connection`），否则这类断流永远走 fallback、从不在同模型上重试。
- **`apply_patch` 信封按 Codex/OpenCode 宽松提取**：小模型常把补丁包进 ` ```patch ` 围栏、前面加说明、漏 `*** End Patch`，或写成 `{"patch":"..."}`，以前一律报 `Patch must start with '*** Begin Patch'`。现在找出 Begin/End（或裸的 `*** Update/Add/Delete File:`）再解析；hunk 仍精确匹配，缺 File 头或行首不是空格/`-`/`+` 照旧拒绝。
- **`apply_patch` 对不上时点出那一行**：`Hunk N: context not found: '…'` 带上 hunk 里文件中不存在的那一行，不再只报一句 not found。不回 Expected/Actual 预览。
- **`apply_patch` 对不上时分清「行不存在」和「行在、但不连着」**：hunk 里每行都能在文件里找到、只是中间跳过了几行时，以前把第一行 keep 报成 `not found: '…'`。模型刚 `read_file` 过那一行，以为工具在撒谎，再读一遍。现在写 `not found as a contiguous block (line N). file: …, hunk: …`。真缺的行仍报 `not found: '…'`。
- **`apply_patch` 找回忘了前导空格的 keep 行**：无 ` `/`-`/`+` 前缀的行若与当前文件某行完全一致（或去行尾空白后只对应一种原文），当成 keep；对不上或多种原文仍报 `Malformed patch`，不做空白/缩进 fuzzy。

#### changes
- **CLI 去掉已废弃的 `_GutteredConsole` 和从未调用的 `display_tool_call`**：左侧 gutter 早就不画了，代理类和 `agentica.cli.display_tool_call` 导出只剩死代码。
- **`worktree` 用法改走内置 skill，人用 `/worktree`**：工具还在（模型要搬家仍得调它，`execute git worktree` 只多一个目录、会话还在旧 cwd）。以前每轮把整段 `<worktrees>` policy 塞进 system prompt。现在判断在 `agentica/skills/bundled/worktree/`，目录匹配才读；人用 `/worktree status|use|merge|remove`（和 `--worktree` 同一套 binder）。
- **peer 收信规则并进 `multi-agent` skill**：`list_agents` / `send_message` 仍每轮注册。以前 `PEER_MESSAGING_POLICY` 每轮塞进 system prompt（授权头、证据用路径、peer 派的活不要 `ask_user_question`）。现在只在目录对上、模型去读 skill 时才进上下文。消息头 `format_for_model` 和 `ask_user_question` 的「这个框能到谁」仍每轮在。
- **cron 用法改走内置 `cron` skill**：`cronjob` 工具还在。以前 `CronTool.get_system_prompt()` 每轮灌用法 + 按表面切换的 `daemon_hint`（CLI 写 `/cron daemon on`，gateway 写 `cron.enabled`）。现在判断在 `agentica/skills/bundled/cron/`，两种开 daemon 的办法写在同一份 skill 里按表面选；`daemon_hint` / `CLI_DAEMON_HINT` 删掉。人用 `/cron`（skill 同名 auto-command 让位给已有斜杠命令）。
- **`apply_patch` 默认形态改成一条补丁改多个文件**：docstring 示例两个 `*** Update File`；要改的文件先并行 `read_file`，再一条补丁。以前示例只有 `app.py`、文案写「先读当前文件」，模型就读一个改一个。精确匹配没变。
- **`multi-agent` skill / 文档不再把「要并行」写成 `delegate` 的理由**：`task` 一条消息里就可以并行多个（`concurrency_safe`）。对比表补上 Parallel 行；`delegate` 只为独立 context / 换目录 / 要写文件。
- **CLI / Web 的 `apply_patch` 调用行显示工作区相对路径**：以前只报 `Edited 1 file (+8 -3)`，看不出改了哪个文件。现在和 `read_file` / `write_file` 一样带上路径（`apply_patch agentica/cli/commands/session.py - Edited 1 file (+8 -3)`）；多文件逗号并列。
- **`execute` 不再规定必须一条长命令**：依赖可用管道/`&&`，互不依赖可同一轮多条 `execute(parallel_safe=True)`，怎么拆交给模型。不要用 shell 倒整份源码（`cd && cat f.py` 是 `read_file`），也不要用 `execute` 改仓库：同一替换是 `rg` 列出位点再一条多文件 `apply_patch`。路径 grounding 只卡 `read_file` / `write_file` / `apply_patch`；`execute` 可以复用已知路径、从 `.` 搜，或对候选加 `2>/dev/null`。以前写「prefer one long / 不要 N 个 grep」会把独立探查焊成一条超长脚本。
- **CLI / Web 的 `write_file` 调用行显示工作区相对路径**：以前只留文件名（`session.py`），和 `read_file` 的 `agentica/cli/commands/session.py` 不一致。现在两边都走同一套缩短（cwd 下相对，工作区外保留原路径）。
- **去掉 `@agentica-ai/sdk` 的 GitHub Actions 自动发布**（删除 `.github/workflows/npm-publish.yml`）：和 PyPI 一样改成仓库里手动 `npm publish`。`v*` tag 不再二次 PUT；`1.4.15` 已经 staged 过，Actions 再推一次会 E409，而 `npm view` 看不见 staged 版本，跳过检查拦不住。

## [1.4.15] - 2026-09-01

#### breaking
- **CLI `/export` 默认导出 session JSONL，不再是对话瘦 JSON**：以前 `/export` / `/save` 把 `working_memory.messages` 存成一份没有 event、没有工具正文的 JSON。现在默认拷贝磁盘上那份 `<session_id>.jsonl`（与 Web 轨迹同一文件）。旧行为改为 `/export messages [path]`；`/export analysis [path]` 写出与 `GET /api/sessions/{id}/trace/analysis` 相同的 JSON。
- **`apply_patch` 只精确匹配上下文**：不再对空白 / 引号做 fuzz。对不上就是 `Hunk N: context not found`。死的 SDK 类 `PatchTool`（双格式 unified/V4A）删除；补丁走 `BuiltinFileTool.apply_patch`，模块只留 `apply_diff` / `parse_patch_envelope`。
- **删除 `write_html` 和 `ShellTool`**：长报告用 `write_file` 写 HTML；shell 用 `execute` / `BuiltinExecuteTool`。`get_builtin_tools` / `DeepAgent` 去掉 `include_html_report`。
- **`grep` 只留 `pattern` / `path` / `limit`**：去掉 `include`、`output_mode`、`case_insensitive`、`fixed_strings`、`context_lines` / `before_context` / `after_context`。按文件名过滤走 `glob` 或 `execute` 的 `rg -g`。
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
