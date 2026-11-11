# 定时任务（Cron 调度器）

Agentica CLI 内置一个定时任务调度器：你可以让 agent 按计划（每天 9 点、每 30 分钟、某个具体时间）自动执行任务，比如"每天早上汇总未读邮件""每小时检查一次构建状态"。

调度器**默认关闭**——定时跑 agent 会消耗 token，所以必须显式开启。

---

## 一、两种运行模式

| 模式 | 何时跑任务 | 适用场景 |
|------|-----------|---------|
| **CLI 内嵌**（默认推荐） | 仅在交互式 CLI 开着时 | 日常使用，开着终端就顺带跑 |
| **独立 daemon** | 前台常驻进程，与 CLI 解耦 | 服务器 / 想关掉终端也持续跑 |

两种模式共用同一把**文件锁**，即使同时开着也不会重复执行同一个任务。

---

## 二、开启调度器

### 方式 1：setup 向导
```bash
agentica setup
```
走到 "Scheduled tasks (cron)" 一步时输入 `y`，并可设置检查间隔（默认 60 秒）。

### 方式 2：CLI 内命令
进入交互式 CLI 后：
```
/cron daemon on      # 开启（并写入 config.yaml，重启后仍生效）
/cron daemon off     # 关闭
/cron daemon status  # 查看当前状态
```

### 方式 3：直接改配置
编辑 `~/.agentica/config.yaml`：
```yaml
settings:
  cron.enabled: true
  cron.interval: 60      # 检查间隔（秒）
```
也可以让 agent 自己改：在对话里说"帮我开启 cron 调度器"，它会调用 `self_manage` / 配置工具完成。

---

## 三、管理定时任务（`/cron` 命令）

```
/cron                                列出所有任务
/cron add "<prompt>" <schedule>      新建任务
/cron pause <id>                     暂停
/cron resume <id>                    恢复
/cron remove <id>                    删除（需确认）
/cron runs [<id>]                    查看最近执行记录
/cron run <id>                       立即手动执行一次
/cron daemon on|off|status           控制调度器开关
```

### schedule 支持的格式
- **Cron 表达式**：`0 9 * * *`（每天 9:00）、`*/30 * * * *`（每 30 分钟）
- **间隔语法**：`every 30m`、`every 2h`
- **具体时间**：`2026-07-01T09:00:00`

### 示例
```
/cron add "用一句话总结今天的 GitHub 通知" 0 9 * * *
/cron add "检查 CI 是否有失败的构建" every 1h
/cron                # 确认任务已建
/cron run abc123     # 先手动跑一次验证效果
```

> 提示：每个任务运行时都会**新建一个独立的 agent 实例**，不会和你当前交互会话的上下文/工具状态相互污染。模型侧用 `cronjob` 工具（用法见内置 `cron` skill），不要让它手改 crontab 或 `jobs.json`。

---

## 四、独立 daemon（关掉 CLI 也能跑）

适合放在服务器、`tmux` / `screen` 里常驻。

```bash
# 前台运行，Ctrl-C 停止
agentica cron daemon

# 自定义检查间隔（秒）与日志
agentica cron daemon --interval 30 --verbose
```

daemon 会读取 `~/.agentica/config.yaml` 里**当前激活的 profile**（模型、API key 等）来构建 agent。如果还没配置过，会提示你先跑 `agentica setup`。

### 放进 tmux 后台常驻
```bash
tmux new -s agentica-cron 'agentica cron daemon --interval 60'
# 之后 Ctrl-b d 脱离；tmux attach -t agentica-cron 重新查看
```

### 关于开机自启
当前**不提供** launchd / systemd 自启集成（避免过度设计）。如确有需要，可自行把上面的 `agentica cron daemon` 命令包进一个 systemd user service 或 macOS LaunchAgent。

---

## 五、数据与排错

- 任务定义和运行记录存放在 `~/.agentica/cron/`。
- 任务**建了但不跑**？检查调度器是否开启：`/cron daemon status`，或确认 `config.yaml` 里 `settings.cron.enabled: true`。
- 想验证某个任务的效果而不等到计划时间：用 `/cron run <id>` 立即触发一次。
- 想看历史执行成败：`/cron runs`。

---

## 六、在 Web / Desktop 与 Gateway 中使用

### 每次运行都是会话里的一轮

定时任务到点后，把任务内容作为一条**自动消息**投递进一个会话，由该会话的 agent 回答。
结果就在会话里，可以接着追问；任务的「运行记录」每条都能点「查看会话」跳过去。
投递到哪个会话由任务的**执行位置**（`target`）决定：

| `target` | 说明 |
|----------|------|
| `session` | 固定在一个已有会话（在聊天里让 agent 建任务、要求结果回到这里时，agent 用它指向当前会话；页面上从会话标题栏新建也是它） |
| `new_once` | 首次运行时新建一个以任务名命名的会话，之后一直沿用（页面新建任务的默认值） |
| `new` | 每次运行都新建会话 |

会话正在回答时，定时运行会等它空闲后再开始，不会插进用户正在进行的回合。
会话绑定了 IM 聊天时，这一轮在 IM 里跑，手机收到「⏰ 定时任务」开头的回复；
会话被终端 CLI 打开时，任务内容投进终端信箱，由终端里的 agent 处理。

### 自动化任务页面

左侧栏「自动化任务」（`/automations`）：按已开启 / 已暂停 / 已结束筛选、搜索；
右侧详情可直接改名称、内容、运行时间（每天 / 工作日 / 每周 / 每月 / 每 N 分钟 / 每 N 小时 / 仅一次 / 自定义 cron）、
时区、执行位置与工作区，并查看运行记录；「立即执行」会打开任务所在会话实时显示输出。
调度器未开启时页面顶部有提示，可一键开启（等同 `settings.cron.enabled: true`，立即生效、无需重启）。

聊天里与任务的联动：
- 定时运行在会话里显示为「定时任务 · 任务名」卡片，可「打开任务」；
- 会话标题栏显示「N 个自动化任务」，点开可跳到任务或「新建自动化任务」（默认在当前会话运行）；
- 侧栏中有任务投递的会话标题旁带时钟标记。

### HTTP API

Gateway 与 CLI **共用同一套任务存储**（`~/.agentica/cron/jobs.json`）和同一个开关。

```bash
curl -X POST http://localhost:8881/api/scheduler/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "汇总昨天项目进展，用中文输出要点",
    "schedule": "0 8 * * 1-5",
    "name": "工作日晨报",
    "timezone": "Asia/Shanghai",
    "target": "new_once",
    "work_dir": "/path/to/project",
    "validate_run": true
  }'
```

`validate_run: true` 创建后立即在任务会话里跑一次，响应的 `validation_run.session_id` 是这次运行所在的会话。

| Method | Path | 说明 |
|--------|------|------|
| GET / PUT | `/api/scheduler/status` | 调度器开关（`{"enabled": bool}`） |
| GET | `/api/scheduler/jobs` | 列出本账号任务（含 `schedule_spec`、`target`、`session_id`、`running`） |
| GET | `/api/scheduler/jobs/{job_id}` | 任务详情 |
| POST | `/api/scheduler/jobs` | 创建任务 |
| PUT | `/api/scheduler/jobs/{job_id}` | 更新任务（`schedule` / `timezone` / `target` / `work_dir` 等） |
| DELETE | `/api/scheduler/jobs/{job_id}` | 删除任务（会话保留） |
| POST | `/api/scheduler/jobs/clear_completed` | 删除全部已结束任务 |
| POST | `/api/scheduler/jobs/{job_id}/pause` | 暂停 |
| POST | `/api/scheduler/jobs/{job_id}/resume` | 恢复 |
| POST | `/api/scheduler/jobs/{job_id}/trigger` | 立即运行一次，不等待结束，返回所在会话；已在运行时 409 |
| GET | `/api/scheduler/jobs/{job_id}/runs` | 运行记录（每条带 `session_id`） |

---

## 七、安全须知

- 定时任务会**自动、无人值守地**让 agent 执行你写的 prompt，并消耗 token。请确认 prompt 的副作用可控。
- 任务每次运行都使用**创建时会话的权限档位和工作目录**，不跟随宿主进程当下的 `/permissions`。通过 `cronjob` 工具创建时，模型只能把档位调得更严，不能比当前会话更宽。`permission_mode="ask"` 可以用来试跑：读和汇报照常进行，写文件、会改东西的命令和联网调用都会被拒绝（因为没人审批）。
- 到点时如果工作目录已经不在了，任务会被暂停，不会运行，`paused_reason` 里写明原因。任务自己发现无法在无人值守的情况下继续时，也会带原因暂停自己。暂停的任务不重试；CLI 启动时和 `/cron list` 会显示暂停原因，`/cron resume <id>` 恢复后原因清空。
- 删除任务需二次确认；开关状态会持久化到 `config.yaml`，重启后保持。