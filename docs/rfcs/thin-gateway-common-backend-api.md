# RFC: Thin Gateway — 会话事实归 core，Gateway 只做适配，Web 只做呈现

状态：已实施（分支 `wt/thin-gateway-refactor`）。本文替换此前的实习版草案；草案中被采纳、修正、拒绝的部分见 §6。

## 0. 目标与判据

一个事实只有一个 owner。判定一段逻辑该放哪一层：

1. CLI / IM / Web / SDK 是否需要同一个答案？需要 → core（`agentica/`）。
2. 答案是否来自 JSONL、sidecar、background record、agent runtime？是 → core。
3. 只在浏览器里成立（布局、动画、草稿、localStorage、断线重连）→ Web。
4. HTTP/SSE、鉴权、账户隔离、进程内 pub/sub、IM 文案排版 → Gateway。

这次允许 break change：不留兼容 shim、不留旧 import 路径、不保留旧字段一个发布周期。

非目标：不新增持久化 store（JSONL 是 transcript 事实，`*.meta.json` 是会话元数据，`background/*.json` 是 job 事实）；不搬纯 UI 状态。

## 1. 现状问题（以 `2ee88d0c` 为基线）

| # | 问题 | 证据 |
|---|------|------|
| P1 | 会话日志放在 `agentica.memory` 下。`SessionLog` / `trace` / `log_rows` / `session_index` / `session_search` 是 transcript 存储与读模型，不是「记忆」；`memory/__init__` 还导出 `SessionLog`。 | `agentica/memory/__init__.py`、85 个文件 import `agentica.memory.session_log` |
| P2 | Gateway 把 `agentica.cli.*` 当库用：`cli.prefs`、`cli.approvals`、`cli.attach`、`cli.context_usage`、`cli.usage_display`、`cli.display.status_bar`。core 的 `memory/session_log.py`、`config/project.py`、`cron/cli_runner.py` 也反向依赖 CLI。 | `rg "agentica\.cli\." gateway/src agentica --glob '!agentica/cli/**'` |
| P3 | 会话目录的可见性规则在三处各写一份：`AgentSessionsMixin.list_sessions`（Web）、`im_bot/menus.py` 的 `_collect_session_items` 与项目计数、`im_bot/controller._default_session_id`。Web 有 30 天折叠，但 gateway/IM 来源的会话被豁免；IM 不过滤 archived/年龄，于是「共有 116 个会话」。 | `agent_service/sessions.py:88-243`、`im_bot/menus.py:273-300,370-430`、`im_bot/controller.py:136-143` |
| P4 | 图片标记 `[图片: <sha>.ext]` 有三个解析器：`memory/session_log.strip_chat_image_markers`、`services/chat_images.strip_markers`、Web `splitImageMarkers`。 | `session_log.py:143-149`、`chat_images.py:38,70`、`hydrateMessages.ts:152,192` |
| P5 | Web 解析日志协议：tool 结果里折叠的 steer（`STEER_GUIDANCE_PREFIX`）、`is_error \|\| tool_call_error`、`<context_window>` 前缀判断、cron 标题回填。 | `hydrateMessages.ts:153-184,224,327,440`、`sessions.ts:355-358` |
| P6 | 回复 footer 的用量按数组位置对齐 Trace round（`roundOffset`），任何一侧多/少一条就整体错位。 | `sessions.ts:488-556` |
| P7 | SSE 回放的合并 bug：缓冲区把相邻 `content`/`thinking` delta 合并，并把 seq 改成最后一个；客户端已收到 seq=1 的 "a" 后以 `after=1` 重连，会收到 seq=2 的 "ab"，文本重复。 | `live_turn.py:265-285` |
| P8 | pulse 混入进程内 `active[]`，Web 首次 pulse 据此决定是否同步。目录接口已经带 `running`，这是同一事实的第二份。 | `live_turn.py:362-369`、`sessions.ts:284-304` |
| P9 | tool 错误标记：磁盘写 `is_error`；`_build_messages` 读 `is_error or tool_call_error`，`trajectory_stats` 也读两种，`trace.py` 只读 `is_error`。三个读者三种写法。 | `session_log.py:1412,1652`、`trace.py:1036,1147` |

## 2. 目标结构

```
agentica/session/            # 新包：会话 transcript 与会话事实（从 memory/ 迁入）
  __init__.py                # 导出 SessionLog
  log.py                     # ← memory/session_log.py
  rows.py                    # ← memory/log_rows.py；+ 图片标记、tool 结果行视图、tool_error_of
  trace.py                   # ← memory/trace.py；transcript 页带 turn 编号；round 带 replyUuid
  index.py                   # ← memory/session_index.py
  search.py                  # ← memory/session_search.py
  catalog.py                 # 新：唯一的会话目录读模型 + 可见性规则（Web/IM/CLI 共用）
  prefs.py                   # ← cli/prefs.py（sidecar/project.json 的 cli 偏好，core 也要读）
agentica/memory/             # 只剩跨会话记忆：models / summarizer / working / recall / intents / session_recall
agentica/live/attach.py      # ← cli/attach.py（live session 的 attach 协议，CLI/cron/IM 共用）
agentica/agent/approvals.py  # + visible_approval_decisions / approval_decision_from_key（从 cli/approvals 下沉）
agentica/compression/context_usage.py   # ← cli/context_usage.py
agentica/model/usage_summary.py         # ← cli/usage_display.py
agentica/utils/timer.py      # + format_turn_clock（从 cli/display/status_bar 下沉）
```

Gateway：

- `AgentService.list_sessions` 只收集本进程的 live 信息（在跑的 `LiveTurn`、agent 锁），交给 `session.catalog.list_catalog`；不再自己推导 source、标题、可见性。
- IM `/project`、`/session`、默认会话选择全部读 `list_catalog`；文案改为「显示 N 个近期会话，另有 M 个更早/归档，发送 `/session all` 查看」。
- `services/chat_images.py` 只负责图片存储（save/resolve），标记格式归 `session.rows`。
- `LiveTurn.public()` 返回已拆好的 `user_message` + `user_images`。
- pulse 只返回 `{"ver"}`。

Web：

- `hydrateMessages.ts` 读 user 行的 `row.text`/`row.images`、tool 行的 `row.error`/`row.text`/`row.steers`；删除 `splitImageMarkers`、`splitFoldedSteer`、`STEER_GUIDANCE_PREFIX`、`WINDOW_OPEN`、`is_error || tool_call_error`。
- `syncSessionRoundStats` 按 `replyUuid` 匹配，删除 `roundOffset`。
- 删除 cron 标题回填（标题来自 `display_title`）。
- pulse 只看 `ver`。

## 3. API 契约

### `GET /api/sessions?scope=recent|all&recent_days=30`

```json
{"sessions": [Row], "scope": "recent", "recent_days": 30, "count": 12, "hidden": 154}
```

`Row` 由 `agentica.session.catalog` 产出（字段名沿用现有 Web 类型）：`session_id, name, preview, user_count, first_timestamp, last_timestamp, last_read_at, unread, size_bytes, archived, pinned, work_dir, running, background_jobs, source, origin, approval_mode`。

可见性规则（`catalog.visible`，唯一实现）：

1. 丢弃：旧版 `scheduled_` 会话；0 字节且不在跑的桩文件；`include_cli=False` 时来源为 CLI/legacy 且不在跑的会话；`hide_temp_dirs` 时系统临时目录里没有用户轮次也不是 cron 的会话。
2. 在跑或有后台 job、pinned、archived（供归档页）→ 总是保留。
3. `scope=recent`：最后修改早于 `recent_days` 的会话折叠进 `hidden` 计数——**不再按 gateway/IM 来源豁免**。

### `GET /api/sessions/{id}/trace/events?turns=N`

user 行：`row = {automated, steer, text, images, window, turn}`；`turn` 是从日志开头数的 1 起编号，只有开启人类轮次的行带。tool 行：`row = {error, text, steers}`。

### `GET /api/sessions/{id}/trace/analysis?lite=1&calls=0`

每个 round 增加 `replyUuid`：该 round 最后一条 assistant/tool 行的 uuid，与 Web 回复气泡的 `uuid` 是同一个值。

### `GET /api/sessions/pulse`

`{"ver": <int>}`；只表示「有变化，去拉一次」。

### SSE 回放

缓冲区仍合并相邻 `content`/`thinking`，但记录每个被合并 delta 的切点；`replay(after)` 在 `after` 落在合并区间内部时只返回 `after` 之后的那部分文本。不改协议，客户端不用变。

## 4. 实施步骤（每步一个提交，每步后跑相关测试；全部完成后全量验证）

1. **模块归位**（纯移动 + import 改写，无行为变化）：`git mv` 迁移 §2 列出的文件；`rg` 改写所有 import（agentica、gateway、tests、scripts、examples、docs）；`memory/__init__` 不再导出 `SessionLog`，由 `agentica.session` 导出，顶层 `agentica.SessionLog` 不变（如有）。验证：全量 pytest、`scripts/check_bare_ci_imports.py`。
2. **图片标记与 tool 行归 `session.rows`**：`image_marker()`、`split_image_markers()`、`strip_image_markers()`、`tool_error_of()`、`tool_row_payload()`；`row_payload` 输出 `images` 且 `text` 去掉标记；删除 `session_log.strip_chat_image_markers` 和 `chat_images.strip_markers`；`trace.py`、`log.py` 的错误读取统一走 `tool_error_of`。测试：`tests/session/test_rows.py`。
3. **transcript 页带 `turn`，round 带 `replyUuid`**：`with_row_views(entries, first_turn=...)`；`chat_turn_page` 用全局编号；`_build_rounds` 增加 `replyUuid`。测试钉值：编号与 `older_turns` 对齐；`replyUuid` 等于最后一条 assistant 行的 uuid。
4. **`session.catalog` + Gateway/IM 改为薄调用**：`list_catalog()` 返回 `Catalog(rows, hidden)`；预览缓存移进 core；source/origin 推断（旧版 `agent:<id>:<channel>:` id）移进 core；`AgentService.list_sessions` 只组 `live` 并调用；IM 的项目计数、会话选择、默认会话全部走 catalog；新增 `/session all`。测试：同一份磁盘数据，Web 行数 == IM「显示 N」；旧的非归档会话不进 recent；pinned 旧会话仍在；`/session all` 列出全部；变异：去掉年龄过滤测试会红。
5. **Gateway 薄化收尾**：`LiveTurn.public()` 拆好图片；pulse 只返回 `ver`；SSE 回放切点修复。测试：`after` 落在合并区间中间，回放不重复。
6. **Web 去业务化**：按 §2 删除解析逻辑，改读服务端字段；加静态边界测试 `tests/web/test_web_boundaries.py`，禁止 `web/src` 出现 `STEER_GUIDANCE_PREFIX`、`<context_window>`、`tool_call_error`、`roundOffset`、`[图片: ` 解析正则。验证：`npm test`、`npm run build`。
7. **文档与 CHANGELOG**：Unreleased 记 break change（新 import 路径、目录规则变化、IM 文案）。

## 5. 风险与回滚

- **效果风险**：目录规则统一后，gateway/IM 来源的旧会话（> 30 天）在 Web「近期」里不再出现，需要切到「全部」或搜索；这是有意为之（与 IM 默认视图保持一致），`scope=all` 能找回全部，没有删除任何数据。
- **外部代码风险**：`from agentica.memory.session_log import SessionLog` 等路径失效（break change），在 CHANGELOG 里列出新路径。
- **耗时风险**：步骤 1 是机械改写，但涉及约 120 个文件；用 `rg` 加 `python -c "import ..."` 加全量测试兜底。
- **回滚**：每步独立提交，按反序 `git revert`。

## 6. 对实习版草案的处理

| 草案项 | 处理 | 理由 |
|---|---|---|
| `agentica/session_api.py` 顶层模块 | 改为 `agentica/session/catalog.py` | 会话相关读模型应和 `SessionLog` 在同一个包里，而不是再加一个顶层文件。 |
| `tool_error_of` / `usage_view` 新文件 | `tool_error_of` 并入 `session/rows.py`；不新建 `usage_view.py` | 用量已由 `trace._token_view` 产出；真正的问题是位置对齐，用 `replyUuid` 解决。 |
| `RunStateStore`（`run_state.json`） | **拒绝** | Gateway 是单进程，`LiveTurn` 就是它的运行态；跨进程运行态已经由 `agentica.live` 的 presence 记录承担。再加一份 json 是第三个 store，会和前两个漂移，违反草案自己的非目标。 |
| SSE「去掉合并、改 snapshot 事件」 | **改为切点修复** | 草案的诊断方向对，但去掉合并会让长回复很快超出 4000 条的缓冲上限、更早触发截断路径；在合并事件里记切点即可精确回放，协议不变。 |
| pulse 去掉 `active[]` | 采纳 | 目录里已经有 `running`。 |
| Web 静态边界测试 | 采纳 | 防回流。 |
| IM「116 个会话」分析 | 采纳为步骤 4 | 根因是可见性规则有三份；统一 catalog 后 IM 和 Web 同口径。 |
| goal/queue/steer 浏览器副本、后台 badge 按请求时间归属 | 本轮不做 | 前者是在途 SSE 的乐观 UI，后者是「这一问对应哪些 job」的呈现范围，都没有跨端事实被重复推导；以后发现漂移再处理。 |
