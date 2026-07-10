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

> 提示：每个任务运行时都会**新建一个独立的 agent 实例**，不会和你当前交互会话的上下文/工具状态相互污染。

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

## 六、在 Gateway 中使用（HTTP API）

Gateway 与 CLI **共用同一套任务存储**（`~/.agentica/cron/jobs.json`）和同一个开关。
只要 `config.yaml` 里 `settings.cron.enabled: true`，`agentica-gateway` 启动后就会
在后台按 `cron.interval` 轮询到点任务，并通过内置的 `_GatewayAgentRunner` 运行——
无需再开一个 `agentica cron daemon` 进程。

启动 Gateway 时日志会打印：

```
Cron scheduler started (60s tick)
```

未开启时则打印（正常行为）：

```
Cron scheduler disabled (set `cron.enabled: true` in ~/.agentica/config.yaml to enable)
```

除了在 CLI 里用 `/cron add` 建任务，也可以**直接通过 Gateway 的 HTTP API** 管理任务：

`POST /api/scheduler/jobs`：

```bash
curl -X POST http://localhost:8789/api/scheduler/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "汇总昨天项目进展，用中文输出要点",
    "schedule": "0 8 * * 1-5",
    "name": "工作日晨报",
    "timezone": "Asia/Shanghai",
    "timeout_seconds": 120
  }'
```

带 `validate_run: true` 可在创建后**立即跑一次**验证 prompt / 调度是否可用：

```bash
curl -X POST http://localhost:8789/api/scheduler/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ping", "schedule": "*/30 * * * *", "validate_run": true}'
```

管理接口一览：

| Method | Path | 说明 |
|--------|------|------|
| GET | `/api/scheduler/jobs` | 列出任务（`include_disabled`, `limit`） |
| GET | `/api/scheduler/jobs/{job_id}` | 任务详情 |
| POST | `/api/scheduler/jobs` | 创建任务 |
| PUT | `/api/scheduler/jobs/{job_id}` | 更新任务 |
| DELETE | `/api/scheduler/jobs/{job_id}` | 删除任务 |
| POST | `/api/scheduler/jobs/{job_id}/pause` | 暂停 |
| POST | `/api/scheduler/jobs/{job_id}/resume` | 恢复 |
| POST | `/api/scheduler/jobs/{job_id}/trigger` | 手动触发一次 |
| GET | `/api/scheduler/jobs/{job_id}/runs` | 运行历史 |

请求体字段（`CronJobCreateRequest`）：`prompt`（必填）、`schedule`（必填，三种语法见第三节）、
`name`、`timezone`（默认 `Asia/Shanghai`）、`deliver`（默认 `local`，预留字段）、
`timeout_seconds`、`max_retries`、`retry_delay_ms`、`permissions`、`validate_run`。

> ⚠️ 关于 `deliver`：当前版本 `deliver` 仅作为任务元数据被持久化，调度器**不会**自动把
> 运行结果推送到 IM 渠道。查看结果请走 Web UI / API 的运行历史，或让 Agent 在 `prompt`
> 里自行调用工具（如发消息）。

---

## 七、安全须知

- 定时任务会**自动、无人值守地**让 agent 执行你写的 prompt，并消耗 token。请确认 prompt 的副作用可控。
- 删除任务需二次确认；开关状态会持久化到 `config.yaml`，重启后保持。