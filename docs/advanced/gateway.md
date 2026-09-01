# Gateway

Gateway 是 Agentica 的"长跑服务"层：把一个 `Agent` 实例暴露为
HTTP API + WebSocket 流式接口 + 多个 IM 平台机器人 + 定时任务调度器，
全部跑在同一个 FastAPI 进程里。

适用场景：

- 把 Agent 封装成内网 / 公网服务，给前端 Web UI、CLI、移动端共享调用
- 让 Agent 同时接入多个 IM 平台（个人微信 / 飞书 / Telegram / Discord / QQ / 企业微信 / 钉钉 / Slack），跨渠道复用同一套对话上下文
- **出门在外用手机遥控本机跑着的 CLI 会话**（Peer bridge，默认开启，见下文「手机遥控本机 CLI 会话」）
- 周期性执行 Agent 任务（cron 调度）

## 安装

```bash
pip install "agentica[gateway]"
```

PyPI 的 wheel 已经打进编译好的 Web UI，运行时不需要 Node。源码在仓库 `web/`，改界面请看那边的 README（`npm run dev`）；不要把 `agentica/gateway/ui/` 提交进 git。

按需追加 IM 平台 SDK（每个 IM 都是可选 extras，不装则该渠道自动跳过）：

```bash
pip install "agentica[wechat]"     # 个人微信（微信 ClawBot / iLink 官方协议，含媒体 AES-128-ECB + CDN）
pip install "agentica[telegram]"   # python-telegram-bot
pip install "agentica[discord]"    # discord.py
pip install "agentica[qq]"         # qq-botpy（QQ 开放平台 WebSocket）
pip install "agentica[wecom]"      # wecom_aibot_sdk（企业微信 AI Bot）
pip install "agentica[dingtalk]"   # dingtalk-stream（钉钉 Stream）
```

> 飞书（Lark）SDK `lark-oapi` 已经包含在基础 `[gateway]` 里。

### Docker

PyPI wheel **仍然**打进编译好的 Web UI；Docker 镜像只是在 build 时用 Node stage 做同一件事，这样 `docker compose up` 的机器上不需要 Node / pip。本机继续 `agentica-gateway` 即可。

```bash
cp .env.docker.example .env   # OPENAI_API_KEY 等
docker compose up -d --build
```

默认把 `8881` 绑在 `127.0.0.1`。容器内进程听 `0.0.0.0`（否则 port publish 进不来），所以 compose 示例里 `GATEWAY_AUTH=false`——生成的初始网页密码不允许绑非 loopback。不要把端口发到 `0.0.0.0:8881`，除非已经 `agentica-gateway --set-password` 并打开 `GATEWAY_AUTH=true`。

镜像：`ghcr.io/shibing624/agentica`（tag 推送时构建）。数据目录 `AGENTICA_HOME=/data`，当前目录挂到 `/workspace`。

### TypeScript 客户端

外部 Node 程序打这份 REST/SSE 时用 **`@agentica-ai/sdk`**（`npm install @agentica-ai/sdk`，registry 是 `https://registry.npmjs.org/`，源码 `sdk-ts/`）。必须写这个全名，不要写成 `@agentica/sdk` / `agentica-sdk`。**不是**启动 Web 的依赖，也不替代 wheel 里的 UI。凭据是机器令牌 `Authorization: Bearer`（`runtime.json` / `AGENTICA_GATEWAY_TOKEN`），不是浏览器 cookie。

启动：

```bash
agentica-gateway
# 等价：python -m agentica.gateway.main

agentica-gateway --set-password        # 设一个网页密码（见下文「鉴权」）
agentica-gateway --host 0.0.0.0        # 也让局域网设备访问（必须先设密码）
agentica-gateway --port 0              # 让系统挑一个空闲端口，启动时打印实际端口
```

默认监听 `127.0.0.1:8881`。**`/api` 与 `/ws` 需要凭据**：没设密码时用启动日志里那条带 `?token=` 的完整地址打开一次，之后浏览器记住会话，直接开 `http://localhost:8881/chat` 即可；设了密码就走 `/login` 登录页。

启动日志会明确区分两类服务，避免与 IM 渠道混淆；并给出本进程的文件日志路径（与 CLI 同一套 `$AGENTICA_HOME/logs/YYYYMMDD-<pid>.log`，`AGENTICA_LOG_FILE=""` 可关掉）：

```
==================================================
  Agentica Gateway v1.4.14
  Workspace: /Users/me/.agentica/workspace
  Work dir:  /Users/me/project
  Model:     openai/gpt-4o
  Log File (INFO): ~/.agentica/logs/20260814-65634.log
==================================================
Web service started — http://127.0.0.1:8881/chat?token=…  # 第一次用这条完整地址打开
  Runtime (port + token): ~/.agentica/cache/gateway/runtime.json
IM channels started — wechat, wecom                       # 按配置启用的 IM 渠道
# 或：IM channels — none enabled (configure a channel to enable)
```

其中 `Work dir` 行显示当前传给 Agent 的 project 工作目录（默认即启动 `agentica-gateway` 时所在的目录，见下文"工作目录"）。

## 鉴权

`/api` 能切 profile、读任意路径、跑 `execute`，所以它默认要凭据。**有两种凭据，分工不同**：

**机器令牌** —— 证明"我在这台机器上，能读一个 `0600` 文件"。每次启动随机生成，连同实际端口写进 `$AGENTICA_CACHE_DIR/gateway/runtime.json`（`0600`，进程退出时删除）。给脚本和桌面壳用：

```bash
TOKEN=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.agentica/cache/gateway/runtime.json')))['token'])")
curl -H "Authorization: Bearer $TOKEN" http://localhost:8881/api/status
# 或 X-Agentica-Token: <token>
```

**会话** —— 证明"这个浏览器登录过一次"。落盘在 `$AGENTICA_HOME/gateway/auth.json`（`0600`，只存 sha256），装在 `agentica_session` cookie 里（HttpOnly、SameSite=Lax），有效期 7 天、最后一天内的任何请求自动续期。

两者分开是有原因的：cookie 里放的**不是**令牌本身。令牌是进程级的，一重启就变——cookie 若存令牌，每次重启 gateway 都得回终端重新捞地址；而且 cookie 泄露就等于主凭据泄露，没法只吊销一个浏览器。

**账号** —— 首次启动会 seed 一个账号，名字就是数据分区名 `default`（等于 `DEFAULT_USER_ID`），随机密码打在启动日志里，同时写一份 `0600` 的 `$AGENTICA_HOME/gateway/initial-password`（改过密码就删掉）。浏览器打开 `/chat` 会 302 到 `/login`。

**账号 id 就是数据分区 id**：它命名 `users/<id>/`，也就是那个账号的会话与记忆。所以 `kk` 登录后看到的是 kk 自己的对话，看不到别人的；也所以账号名会先规范化再当目录名（小写、空格/连字符变下划线、其它非法字符丢掉；清理后 2–32 位、字母开头、仅字母数字下划线），撞名则拒绝，且不支持改名。新增账号还会在 workspace 下建同名默认 Project 目录（就是 `<id>`，没有 `-default_project` 后缀）。管理员在网页 **用户管理**（右侧独立页，`/users`）里新增 / 修改密码 / 删除账号；新增时由管理员填写初始密码，新账号一律是用户——seed 的那一个是唯一管理员。**「用户管理」是唯一的管理员专属功能**（`/api/auth/users*` 非管理员一律 403）。模型 profile、技能、MCP、工作目录仍是这台机器的配置，任何登录账号都能改；**会话、归档、定时任务按账号隔离**——侧边栏只信服务端列表（同一个浏览器换账号不会把上一个人的 `localStorage` 会话带过来），`/api/scheduler/jobs` 只列出当前 cookie 所属账号的任务，任务跑起来也写进那个账号的分区。改自己的密码要填当前密码（内置管理员会提示初始密码打印在首次启动的终端里）；管理员改普通用户只填新密码。删除账号只是让它无法登录，`users/<id>/` 的数据仍留在磁盘上。

设密码：`agentica-gateway --set-password`（在那台机器的终端上），或网页里 设置 › 常规 › 访问控制。密码用 `hashlib.scrypt` 存储，格式 `scrypt$n$r$p$salt$hash`（参数随 hash 一起存，以后调高成本不会作废旧密码）。改密码会让**这个账号的其他浏览器退出登录**（别人的不受影响）——改密码的场景就是"我的 cookie 可能被人拿到了"。连续失败 5 次后按 1s、2s、4s…（上限 60s）退避，返回 429 并带 `Retry-After`。

**`--host` 不是 loopback 时必须先设密码，否则拒绝启动**（退出码 2）。令牌是打印在终端里、不重启改不了的东西，不适合暴露到网络；密码是人能自己轮换的凭据。`GATEWAY_AUTH=false` 不受这条约束——显式关门是明确意图，只会得到一条醒目的告警。

不需要凭据的路径，每条都有不得不豁免的理由：`/webhook/*`（飞书等第三方回调自带签名，无法携带我们的凭据）、`/health` 与 `/api/health`（就绪探针在拿到令牌之前就要跑）、`/`、`/assets/*`（编译产物，无用户数据）、`/login` 与 `/api/auth/{status,login,logout}`（门自己不能锁在门后）。CORS 只放行 loopback 来源。

CSRF 的主防线是 `SameSite=Lax`；第二道是**带 cookie 的写请求必须是 `application/json`，或者带 `X-Agentica-Client` 头**——HTML 表单只能发那三种 Content-Type，也伪造不出自定义头。`/api/upload` 确实要 multipart，所以它带这个头。用 header 递交令牌的脚本不受此约束（`curl -d` 默认就是表单类型，而表单发不出 `Authorization`）。

| 变量 | 作用 |
|---|---|
| `AGENTICA_GATEWAY_TOKEN` | 固定令牌，不再每次随机（脚本、桌面壳重启子进程时用） |
| `GATEWAY_AUTH=false` | 整道门关掉。**只在你确认这个端口只有自己能连时**才用 |

相关端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/auth/status` | 免凭据。门开着吗 / 有密码吗 / 我登录了吗 |
| POST | `/api/auth/login` | `{"username": "…", "password": "…"}` → 下发会话 cookie（`username` 省略即 `default`） |
| POST | `/api/auth/logout` | 服务端销毁会话（不只是清 cookie） |
| POST | `/api/auth/password` | 设置或修改**自己**的密码；令牌持有者可以不填旧密码 |
| GET | `/api/auth/users` | 账号列表（仅管理员） |
| POST | `/api/auth/users` | 新增账号（仅管理员）。body 含 `username` / `password`；角色固定为用户 |
| POST | `/api/auth/users/{id}/password` | 修改某账号密码（仅管理员）。改别人只传 `password`；改自己还要 `old_password` |
| DELETE | `/api/auth/users/{id}` | 删除账号，数据保留（仅管理员） |
| POST | `/api/desktop/shutdown` | 仅限令牌持有者，优雅停机（Windows 上唯一的优雅路径） |

## 桌面应用（Electron 薄壳）

`desktop/` 是一层 Electron 壳：单实例，同一 `~/.agentica` 上已经有 gateway 就直接连过去，否则拉起 `agentica-gateway`。机器上还没有这份二进制时，第一次打开会用 uv 在 Application Support 装一份托管 runtime（不进 `~/.agentica`）。窗口里就是浏览器看到的同一份 SPA，没有第二套 UI，也没有 preload / IPC；令牌先换成会话再注入 cookie，渲染进程读不到。

几处不显然的设计：

- **端口是粘的。** SPA 的会话树、当前会话、主题都存在 `localStorage`，而它按 origin 隔离，origin 就是 `http://127.0.0.1:<port>`。所以纯 `--port 0` 会让每次启动的侧边栏都是空的；壳记住上次真正绑定的端口，下次优先要它，被占了才退回 0。
- **子进程死了会重启**，退避 1s / 2s / 4s，三次放弃并弹窗；连续跑满一分钟就重置这个额度。
- **退出先请求 `POST /api/desktop/shutdown`**，再 SIGTERM，再 SIGKILL，且退出流程会等到子进程真的没了才继续——Windows 上 `kill()` 是硬 TerminateProcess，渠道不会断开、peers 记录不会清理。`--parent-pid` 是壳被强杀时的兜底：gateway 发现父进程消失会自己退出。

详见 `desktop/README.md`。

## 整体架构

```text
     HTTP 入口                      IM 入口                       定时入口
Web UI / curl / SDK       WeChat / Feishu / Telegram /         Cron Scheduler
                            Discord / QQ / WeCom /               (60s tick)
                              DingTalk / Slack
         │                              │                             │
         │ HTTP                         │ 长轮询 / WebSocket          │
         ▼                              ▼                             ▼
   FastAPI Routes                 Channel 实现                        │
         │                              │ unified Message             │
                                 ChannelManager                       │
                                        │ _handle_channel_message     │
                                  MessageRouter                       │
         │                              │                             │
         └──────────────────────────┬───┴─────────────────────────────┘
         ┌─────────────────────────────────────────────────────┐
         │                    AgentService                     │
         │  LRU Agent 缓存; chat(message, session_id, user_id) │
         └─────────────────────────────────────────────────────┘
                                    │ Agent.run
                                    ▼
                     ┌────────────────────────────┐
                     │         Agent 引擎         │
                     │ ReAct 循环（LLM <-> 工具） │
                     └────────────────────────────┘
```

核心抽象：

| 类 | 文件 | 职责 |
|----|------|------|
| `Channel` (ABC) | [`agentica/gateway/channels/base.py`](https://github.com/shibing624/agentica/blob/main/agentica/gateway/channels/base.py) | IM 渠道协议：`connect / disconnect / send` + allowlist + `split_text` |
| `Message` (dataclass) | 同上 | 跨平台统一消息格式（`channel`, `channel_id`, `sender_id`, `content`, `metadata` …） |
| `ChannelManager` | `services/channel_manager.py` | 渠道注册 / 生命周期 / 统一发送入口 |
| `MessageRouter` | `services/router.py` | 把 `Message` 路由到具体 `agent_id` + 计算稳定 `session_id` |
| `AgentService` | `services/agent_service.py` | LRU Agent 缓存 + `chat(message, session_id, user_id)` 主入口 |

每个 IM 渠道只做"把平台原生消息翻译成 `Message` + 把回复文本发回平台"两件事，
其它统一由 Gateway 层完成。

## 手机遥控本机 CLI 会话（PEER_BRIDGE）

主打场景：人在外面，用手机 IM 直接指挥本机终端里**正在跑的 agentica CLI 会话**——
看看哪些任务还在跑、把指令"敲"进指定终端、让它停下 / 换方向 / 汇报进展，
如同坐在那台机器的键盘前。

默认开启，无需任何配置；不需要时显式关闭：

```bash
PEER_BRIDGE=false agentica-gateway
```

### 用法

在 IM 里（个人微信 / 企微 / 飞书 / Telegram …任一已启用渠道）发消息：

| 你发的 | 效果 |
|--------|------|
| `@list` | 列出本机当前 live 的 CLI 会话：名字 / 忙闲 / 工作目录 / 手头任务 |
| `@nlp-f1 rerun arm 3` | 把 `rerun arm 3` 发给名为 `nlp-f1` 的会话，并置顶（pin）它为默认目标 |
| `rerun arm 4` | 其后的裸文字继续发给同一会话，不必每行都带 `@` |
| `@nlp-f1`（不带内容） | 仅置顶，先不发送 |
| `@off` | 取消置顶，消息恢复交给 Gateway 自己的 agent |

没打 `@` 的消息照旧走 Gateway 自己的 agent，开 bridge 不影响原有对话。
CLI 侧零改动：`list_agents` 里能直接看到你的手机（如 `wechat-xuming`），
会话里的 agent 用它本来就有的 `send_message` 就能把进展推到你的微信里。

### 不打 `@`：让网关 agent 自己去指挥

`@<会话名> <话>` 是**你自己**在寻址某个终端；反过来，你也可以只说一句人话，
让 Gateway 自己的 agent 去替你寻址、群发、汇总：

```text
你：  让本机所有 CLI 会话都把自己改的文件提交成 commit
网关：（list_agents 看到 3 个会话 → 分别 send_message → 回你一句谁收到了）
```

这条路径和 `@` 走的是同一个 peers 通道：每个 Gateway 会话（网页的、每个 IM 会话的）
也是一个 peer，发布成和 CLI 一样短的名字（如 `wechat-agentica-41`：渠道 + cwd
末级目录 + 两位 id），所以 CLI 那边 `list_agents` 里能同时看到你的手机和网关 agent，
**CLI 会话也可以主动 `send_message` 给 `wechat-agentica-41` 把完整结论推回你的微信**。

几点行为值得知道：

- CLI 的回信在网关 agent**正在跑**时由它自己收进上下文（会体现在它的回复里）；
  **空闲**时由后台轮询直接推到你当前这个 IM 会话，前缀是发信会话名（`payments-a1 ›`）。
- 网页 UI 的会话没有 IM 回信路径，回信会留在邮箱里，等你下一次说话时被 agent 读到。
- 网关 agent 自己**不出现在 `@list` 里**，也不能被 `@` 寻址——不打 `@` 就是在跟它说话，
  否则等于把话转发给正在回你的那个 agent，只会原样回声。
- 定时任务（cron）不参与：它每次跑都是一个用完即弃的 agent，发布邮箱没人读。
- `PEER_BRIDGE=false` 同时关掉这两条路径（同一个信任边界：网关能否往你的终端里敲字）。

### 安全前提

转发的每行话按 `from_kind="user"` 投递，接收端 CLI 把它当作**用户本人在终端里敲的字**
——这正是目的：这是个人助手 Gateway，bot 就是你的另一只手。bridge 自身不加第二道门；
想限制谁能和你的 bot 说话时，配置渠道的 `<CHANNEL>_ALLOWED_USERS`
（如 `WECHAT_ALLOWED_USERS=你的sender_id`），它在消息到达 bridge 之前就会过滤，
对 Gateway agent 与 bridge 同一生效。

bridge 还必须与 CLI 使用同一个 `AGENTICA_HOME`（peers 目录在其缓存下，默认即本机当前用户），
否则它永远看到空会话列表；启动日志与 `@list` 的空列表回复都会打出实际搜索的目录，便于排查。

### 原理：没有新协议

bridge 只是已有 peers 通道（`agentica/peers.py`）上的又一个 peer——每个 IM 用户对应一个
`PeerSession`（实现见 [`gateway/services/peer_bridge.py`](https://github.com/shibing624/agentica/blob/main/agentica/gateway/services/peer_bridge.py)）。
邮箱顺序、背压、重复/限频刹车、"在 tool 批次边界投递"等保证全部继承而非重写；
转发消息也**不进** Gateway 按会话排队的入站队列——`@session 停` 若排在 Gateway agent
当前那一轮之后，就失去了存在的意义。

网关 agent 自己的 peer 身份同理，见
[`gateway/services/agent_peers.py`](https://github.com/shibing624/agentica/blob/main/agentica/gateway/services/agent_peers.py)：
它拿到的就是 CLI 会话本来就有的那两个工具（`list_agents` / `send_message`），
没有为网关新增任何协议或工具。

## 支持的渠道一览

| 渠道 | 依赖 extras | 连接方式 | 需要公网 | 启用所需环境变量 |
|------|------------|----------|----------|------------------|
| Web 网页 | 内置 `[gateway]` | HTTP（内置 `/chat` UI） | 否（本机 `http://localhost:8881/chat`） | 无需配置，启动即开；可用 `HOST` / `PORT` 调整监听 |
| 个人微信 | `wechat` | ilinkai HTTP 长轮询 | 否 | 默认开启。token 落在 `WECHAT_TOKEN_FILE`（默认 `~/.agentica/cache/wxbot_token.json`）；网页点「配置」扫码 |
| 飞书 Lark | 内置 `[gateway]` | WebSocket 长连接 | 否 | `FEISHU_APP_ID` + `FEISHU_APP_SECRET` |
| Telegram | `telegram` | 长轮询 | 否 | `TELEGRAM_BOT_TOKEN` |
| Discord | `discord` | Gateway 长连接 | 否 | `DISCORD_BOT_TOKEN` |
| QQ | `qq` | qq-botpy WebSocket | 否 | `QQ_APP_ID` + `QQ_APP_SECRET` |
| 企业微信 | `wecom` | wecom_aibot_sdk WS | 否 | `WECOM_BOT_ID` + `WECOM_SECRET` |
| 钉钉 | `dingtalk` | dingtalk-stream | 否 | `DINGTALK_CLIENT_ID` + `DINGTALK_CLIENT_SECRET` |
| Slack | `slack` | Socket Mode WS | 否 | `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` |

> 所有渠道都**不需要公网 IP / 域名 / webhook**：飞书 / QQ / 企业微信 / Slack 走各自厂商的
> WebSocket 长连，Telegram / Discord / 个人微信走长轮询或 HTTP 轮询，内网部署即可。

### Web 网页（内置 UI）

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Gateway Web UI" />

> 这是**默认开启**的渠道：只要 `agentica-gateway` 在跑，`http://localhost:8881/chat` 就能用（首次用启动日志里带 `?token=` 的地址，或设了密码后走 `/login`）；其余 IM 渠道都是可选的叠加层。网页和桌面版里打开 **设置 › 个人助理**（或 `/assistant`）即可看到本机 Web 地址和微信 / 企微 / QQ / 飞书等的连接方法与当前状态。

## 配置（环境变量）

服务端口、鉴权、IM 渠道等通过环境变量配置（推荐写在 `.env`）；**模型配置优先读取 `~/.agentica/config.yaml` 的 `active` profile**（profile 优先于环境变量，详见下文「模型」）。

完整字段定义见 [`agentica/gateway/config.py`](https://github.com/shibing624/agentica/blob/main/agentica/gateway/config.py)。

### 服务器

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOST` | `127.0.0.1` | 监听地址。`0.0.0.0` 才对局域网开放，此时**必须先设密码**，否则拒绝启动 |
| `PORT` | `8881` | 监听端口。`0` = 由系统挑一个空闲端口并在启动日志里报出来 |
| `GATEWAY_AUTH` | `true` | `/api` + `/ws` 的凭据门，见「鉴权」 |
| `AGENTICA_GATEWAY_TOKEN` | 随机 | 固定令牌 |
| `AGENTICA_GATEWAY_PARENT_PID` | — | 该 pid 消失后自行退出（等价 `--parent-pid`） |
| `OPENAI_API_KEY` / 各家 provider key | — | 走标准 provider 配置；config.yaml profile 也自带 `api_key`，二者等价 |

### 模型（优先 config.yaml）

Gateway 是 **profile 驱动** 的服务：启动时读取 `~/.agentica/config.yaml` 的 `active` profile 作为主模型来源；只有当 profile 里没配对应字段时，才退回 `AGENTICA_MODEL_*` 环境变量，最后退回内置默认。所以**日常改模型只动 config.yaml 即可，不必碰 `.env`**。

```yaml
# ~/.agentica/config.yaml
profiles:
  default:
    model_provider: zhipuai
    model_name: glm-4.7-flash
    api_key: "your-key"
    thinking: enabled        # 网关侧始终开启思考过程；此项仅 CLI / 其它入口会读
active: default              # CLI `/model default` 也是改这个指针
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `AGENTICA_MODEL_PROVIDER` | 继承 config.yaml `active` profile（缺省 `deepseek`） | 主模型 provider；profile 已配时此变量仅作覆盖 |
| `AGENTICA_MODEL_NAME` | 继承 config.yaml `active` profile（缺省 `deepseek-v4-flash`） | 主模型名；profile 已配时此变量仅作覆盖 |
| `AGENTICA_MODEL_THINKING` | 继承 config.yaml `active` profile 的 `thinking`（缺省空） | CLI 等入口的思维链开关。**网页 / 桌面版始终启用 thinking**（设置里已无此开关），不支持的模型会忽略 |
| `AGENTICA_MODEL_BASE_URL` | 继承 profile 的 `base_url` | 自定义/兼容端点 |
| `AGENTICA_MODEL_API_KEY` | 继承 profile 的 `api_key` | 主模型 key |
| `AGENTICA_REASONING_EFFORT` | 继承 profile 的 `reasoning_effort` | low/medium/high/max |
| `AGENTICA_AUXILIARY_MODEL_PROVIDER` / `_NAME` | 继承 profile 的 `auxiliary_model` | 后台/子 agent 用的廉价模型，留空则复用主模型 |

### 工作目录（Project Work Dir）

Agent 操作的 project 根目录由 `AGENTICA_BASE_DIR` 控制，默认 = **启动 `agentica-gateway` 时所在的目录**（`os.getcwd()`），与 CLI 行为一致——不显式配置时直接对当前项目目录工作，而不会像旧版本那样落到 `$HOME`。

| 变量 | 默认 | 说明 |
|------|------|------|
| `AGENTICA_BASE_DIR` | 启动目录 `os.getcwd()` | Agent 的 project 工作目录（读写文件、执行命令的基准） |

```bash
AGENTICA_BASE_DIR=/path/to/your/project   # 可选，显式指定工作目录
agentica-gateway
```

### 最简启动（先跑起来）

零配置就能先跑：`agentica-gateway` 启动后自带 Web UI（用启动日志里那条 `http://127.0.0.1:8881/chat?token=…`），
无需任何 IM 配置即可对话，记忆落地在 `~/.agentica/workspace`。

想接个人微信：装好 `agentica[wechat]` 后启动 gateway，打开 **设置 › 个人助理**，点微信那一行的「配置」，用个人微信扫码即可。白名单留空 = 不限制：

```bash
pip install 'agentica[wechat]'   # 提供 qrcode / pycryptodome / Pillow，用于扫码与媒体收发
# 默认 token：~/.agentica/cache/wxbot_token.json ；WECHAT_ALLOWED_USERS 留空
```

> 二维码只在网页里点「配置」时生成，离开页面就消失；过期或没扫都会显示「失败」，再点配置会换一张新码。
> token 默认缓存到 `~/.agentica/cache/wxbot_token.json`，下次启动免扫码。

其余渠道只需补对应的 app 凭证，白名单**默认全部留空即可**（见下文各渠道小节），
先把机器人跑通，再按需加白名单。

### 个人微信（WeChat）

最核心也最省事的接入方式：个人微信扫码即用，不需要申请任何开放平台应用。

> 📌 **协议说明**：该渠道直连微信官方 **ClawBot / iLink** 后端（`https://ilinkai.weixin.qq.com`），
> 与腾讯开源的 `@tencent-weixin/openclaw-weixin` Node 插件是**同一套 HTTP 协议**的 Python 实现，
> 无需启动任何 Node 进程。文本与媒体（图片 / 文件 / 语音 / 视频）均支持：媒体先以
> **AES-128-ECB（PKCS7）** 加密后上传至 CDN，回包头 `x-encrypted-param` 作为 `encrypt_query_param`
> 回填进 `CDNMedia` 引用，随 `sendmessage` 下发。
>
> ⚠️ **风险提示**：iLink 协议可能随微信升级调整。仅推荐用于个人 / 内部小范围实验场景。

```bash
export WECHAT_TOKEN_FILE=~/.agentica/cache/wxbot_token.json
agentica-gateway
```

<div style="display: flex; gap: 16px; align-items: flex-start;">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat-clawbot-qr.png" alt="微信 ClawBot 扫码绑定" width="400" />
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat-clawbot-snap.jpg" alt="微信 ClawBot 直接对话 Agentica" width="150" />
</div>

> 左：终端 / 浏览器弹出的扫码二维码，个人微信扫码即完成绑定；右：扫码后直接在微信里和 Agentica 对话，无需申请任何开放平台应用。

说明：
```bash
WECHAT_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
```

图片：底模能看（`supports_images`，或底模是 Gemini）就直接给底模看像素。语音/视频：只有底模是 Gemini 时直挂，否则走 `settings.media_model`（指向 Gemini，`model_name` 省略则为 `gemini-3.6-flash`）转写/描述。未配置 media_model 时回复会说明怎么配，不会去扫其它 profile。

```yaml
# ~/.agentica/config.yaml
settings:
  media_model:
    model_provider: openai
    model_name: gemini-3.6-flash
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    api_key: ...
```

#### 启用条件

微信渠道**默认注册**。token 文件默认是 `~/.agentica/cache/wxbot_token.json`（可用 `WECHAT_TOKEN_FILE` 改路径），`WECHAT_ALLOWED_USERS` 默认留空。

| 状态 | 行为 |
|------|------|
| 已有有效 token | 启动时直接连上，网页显示「已连接」 |
| 没有 token | 启动安静，网页显示「失败」；点「配置」才拉二维码 |
| 二维码过期 / 没扫 | 显示「失败」；再进个人助理页时二维码是空的，需再点「配置」 |

点「配置」（或扫码成功）时，当前登录的网页账号会写进 token 文件的 `gateway_user_id`。之后微信进线、网页、桌面共用这个账号的会话和记忆，不再按微信 openid 另开分区。已连上时再点一次「配置」也会完成绑定。

启动时**不会**再弹出扫码窗口。CLI / token 过期后的终端重登仍走原来的 `login_qr()`。

### 飞书（Lark）

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
FEISHU_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
FEISHU_ALLOWED_GROUPS=   # 留空 = 不限制群组
```

申请：[飞书开放平台](https://open.feishu.cn) → 创建企业自建应用 → 启用"机器人"能力 →
开通"接收消息" 权限 → 配置长连接 / WebSocket。

### Telegram

```bash
TELEGRAM_BOT_TOKEN=123456:ABCDEF...
TELEGRAM_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
```

申请：在 Telegram 里和 [@BotFather](https://t.me/BotFather) 对话 → `/newbot` → 拿到 token。
渠道使用长轮询，**无需公网 webhook**。

### Discord

```bash
DISCORD_BOT_TOKEN=MTAxxxxx.xxxx.xxxx
DISCORD_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
DISCORD_ALLOWED_GUILDS=   # 留空 = 不限制服务器
```

申请：[Discord Developer Portal](https://discord.com/developers/applications) →
New Application → Bot → 开启 `MESSAGE CONTENT INTENT` → 复制 token。

### QQ（QQ 开放平台官方机器人）

```bash
QQ_APP_ID=102xxxxx
QQ_APP_SECRET=xxxxx
QQ_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
```

申请：[QQ 开放平台](https://q.qq.com) → 创建机器人 → 拿到 AppID / AppSecret。
渠道使用 `qq-botpy` 的 Intents WebSocket，**无需公网 webhook**。

行为说明：

- 同时支持 **C2C 私聊**（`channel_id = openid`）和 **群 @ 消息**（`channel_id = "group:<group_openid>"`）
- 用户的 `openid` 在 ta 第一次发消息时由 QQ 平台分配；想加白名单时观察 gateway 日志即可拿到
- 因为 QQ 主动推送 API 要求带原始 `msg_id`，渠道会自动缓存每个会话最新的 `msg_id`，外部调用 `/api/send` 时透传即可

### 企业微信（WeCom）

```bash
WECOM_BOT_ID=your_bot_id
WECOM_SECRET=your_bot_secret
WECOM_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
```

申请：企业微信管理后台 → 智能机器人 → 创建 AI Bot → 拿到 `bot_id` + `secret`。
渠道使用 `wecom_aibot_sdk` 的 WSClient，**无需公网 webhook**。

实现细节：企业微信回包必须用收到时的原始 `frame`，渠道内部维护
`{chat_id: frame}` 缓存；如果调用 `/api/send` 给一个从未发过消息的会话，
该次发送会失败并写日志（这是平台限制，不是 bug）。

### 钉钉（DingTalk）

```bash
DINGTALK_CLIENT_ID=your_app_key
DINGTALK_CLIENT_SECRET=your_app_secret
DINGTALK_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
```

申请：[钉钉开放平台](https://open.dingtalk.com) → 创建企业内部应用 → 开通"机器人"能力 →
拿到 AppKey / AppSecret（即 client_id / client_secret）。

实现细节：

- 入站使用 `dingtalk-stream` 的 Stream 长连接
- 出站走 HTTP，`accessToken` 由渠道内部缓存（带过期续期，60 秒缓冲）
- **私聊**：`channel_id = sender_staff_id`，发送到 `/v1.0/robot/oToMessages/batchSend`
- **群消息**：`channel_id = "group:<openConversationId>"`，发送到 `/v1.0/robot/groupMessages/send`
- 默认以 `sampleMarkdown` 消息卡片发送（标题 "Agent Reply"）

### Slack

```bash
SLACK_BOT_TOKEN=xoxb-xxx          # Bot User OAuth Token
SLACK_APP_TOKEN=xapp-xxx          # App-Level Token（用于 Socket Mode）
SLACK_ALLOWED_USERS=   # 留空 = 不限制，任何用户都能访问
SLACK_ALLOWED_CHANNELS=   # 留空 = 接收所有频道
```

申请：

1. [api.slack.com](https://api.slack.com/apps) → Create New App → 从 scratch 创建
2. **OAuth & Permissions** → 添加 Bot Token Scopes：`app_mentions:read`、`channels:history`、
   `chat:write`、`groups:history`、`im:history`、`im:write`、`mpim:history`
3. 安装到工作区，复制 **Bot User OAuth Token**（`xoxb-` 开头）→ `SLACK_BOT_TOKEN`
4. **Socket Mode** → 开启 → Generate an App-Level Token（`xapp-` 开头）→ `SLACK_APP_TOKEN`
5. **Event Subscriptions**（Socket Mode 下）订阅 `message.channels` / `message.groups` /
   `message.im` / `app_mention`

实现细节：

- 使用 **Socket Mode**，所有事件走 Slack 维护的 WebSocket，**无需公网 webhook / 域名**
- 入站监听在后台线程，通过 `run_coroutine_threadsafe` 派发到主事件循环
- 自动忽略机器人自己的消息、频道加入通知、消息编辑等噪音事件（`app_mention` 与 `message` 正常接收）
- `channel_id` 即 Slack 会话 id（`D...` 私聊 / `C...` 频道），可直接用于 `/api/send`
- 长文本按 3000 字符分片发送；`send(..., thread_ts=...)` 可指定线程回复

## 提供的 HTTP API

启动后访问 `http://localhost:8881/docs` 查看 OpenAPI 全文档。常用：

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` / `/api/health` | 健康检查（免凭据） |
| GET | `/chat` | 内置 Web UI（首次需 `?token=` 或登录，之后走 cookie） |
| GET | `/traces` | 当前会话的 Trace 页（从对话标题进入；与 `/chat` 同一份 SPA） |
| GET | `/login` | 登录页（设了密码时；与 `/chat` 同一份 SPA） |
| POST | `/api/auth/login` / `/logout` / `/password` | 登录、退出、改密码，见「鉴权」 |
| GET | `/api/workspace/files` | 列出工作目录下的文件（`root` + 相对 `path`） |
| GET | `/api/workspace/content` | 预览（`preview=1`，超过 256 KiB 截断）或下载（`download=1`） |
| POST | `/api/workspace/stat` | 批量确认相对路径是否存在 |
| POST | `/api/workspace/upload` | 上传到当前目录（multipart，需 `X-Agentica-Client`） |
| GET | `/api/sessions/{id}/trace/events` | 分页原始 JSONL 事件 |
| GET | `/api/sessions/{id}/trace/analysis` | 整份轨迹分析（`SessionLog.analyze()`，与 CLI `/trace`、SDK `session_log.analyze()` 同一 payload；重启后按 session id 跨 project 定位 jsonl） |
| GET / PUT | `/api/prefs` | 当前账号的 Web 偏好（主题 / 语言 / 审批档 / 上次会话 / `auto_extract_memory`），落在 `$AGENTICA_HOME/gateway/prefs/<账号>.json`；浏览器 localStorage 只是首屏缓存 |
| GET / PUT | `/api/user_agents_md` | 当前账号的用户级 `AGENTS.md`（常驻规则，进 system prompt）。PUT body：`{content}` |
| GET | `/api/memory` | 当前账号的 `MEMORY.md` 索引条目（只读）。没有 PUT——旧前端往这里写 AGENTS.md 会拿到 405 |
| GET | `/api/sessions` | 当前账号的会话列表（含尚未写出 jsonl 的进行中 run） |
| GET | `/api/sessions/{id}/usage` | 本会话 Context Window 拆分（与 CLI `/usage` 同一套 `measure_context`：system prompt / 规则 / 技能 / 工具定义 / 对话的 token 数）以及消息数、API 调用、费用 |
| POST | `/api/sessions/{id}/compact` | Web `/compact`：与 CLI 同一套 native + Layer 2 摘要 |
| POST | `/api/goal` | Web `/goal`：standing-goal 循环。body：`objective`、`session_id`、`token_budget`（默认 `-1` = 不限；不传 turns / wall） |
| POST | `/api/fs/temp` | 为新建对话创建一个临时工作目录（`$AGENTICA_HOME/tmp/web-chats`） |
| POST | `/api/chat` | 触发一轮 agent 对话（JSON body：`message`, `session_id`，可选 `images`） |
| POST | `/api/chat/stream` | 创建后台 run 并立刻订阅 SSE（便捷入口；断开不断 run） |
| POST | `/api/chat/runs` | 创建后台 run，立刻返回 `run_id` / `status` |
| GET | `/api/chat/runs/active?session_id=` | 该 session 进行中的 run（刷新后重连用） |
| GET | `/api/chat/runs/{run_id}/events?after=` | 订阅或重连 SSE（`after` 为已消费的 seq；空闲 15s keepalive；断开不取消 run） |
| POST | `/api/chat/runs/{run_id}/cancel` | 显式取消并等待 session lock 释放；已结束则幂等返回终态 |
| POST | `/api/sessions/{session_id}/approvals/{tool_call_id}` | 提交工具审批：body `decision` 为 `allow` / `allow_prefix` / `deny` / `deny_prefix`。按账号找 LiveTurn，未知 id / 别人的卡 404。审批走 chat SSE，不走 `/ws` |
| WS | `/ws` | 流式事件订阅 |
| GET | `/api/channels` | 列出已注册渠道 + 连接状态，以及网页「个人助理」用的完整 catalog（含未配置的 IM、`web_url`、监听地址） |
| POST | `/api/channels/wechat/qr` | 个人助理「配置」：生成微信登录二维码（`png` base64 + `qrcode` id + `expires_in`） |
| GET | `/api/channels/wechat/qr?id=` | 轮询该二维码：`wait` / `confirmed` / `expired` |
| POST | `/api/open` | 用系统默认程序打开本地路径（`path`）或 http(s) URL（`url`，默认浏览器） |
| POST | `/api/send` | 主动向某个 IM 渠道发送一条消息 |
| GET | `/api/jobs` 等 | Cron / 定时任务管理（详见 routes/scheduler.py） |

`/api/send` 示例：

```bash
curl -X POST http://localhost:8881/api/send \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "qq",
    "channel_id": "group:abc123",
    "message": "服务部署完成 ✅"
  }'
```

`channel` 可选值：`feishu` / `telegram` / `discord` / `qq` / `wecom` / `dingtalk` / `slack` / `wechat` / `web`。

## 工具审批（Ask for approval）

权限档 `ask` / `auto` / `allow-all` 的工具都在 schema 里。**破坏性变化**：以前 `ask` 会藏掉 `write_file` / `execute`；现在三个档都能发出这些调用，差别只在真正执行前要不要停车等人。Web 默认仍是 `auto`。

| 档 | 自动放行 | 停车 | 直接拒绝（不弹卡） |
|---|---|---|---|
| `ask` | **读**（含工作区外）、只读 `execute`、记忆、`task` / `delegate`、skill、其它内置工具 | **写文件**（含工作区内）、会改状态的 `execute`、**`web_search` / `fetch_url`**、硬不安全（`rm -rf /`、写 `/etc` / `~/.ssh`） | 项目 `deny_prefix` 授权 |
| `auto` | 读（含工作区外）、工作区内写、普通 `execute`、网络、内置 / skill / 第三方工具（不看 `action`） | 工作区外/敏感路径的**文件写**、硬不安全 execute | 项目 deny 授权 |
| `allow-all` | 全部（含硬不安全；项目 deny 不生效，只记 warning） | 无 | 无 |

### Registry 挂在 LiveTurn 上 

`ApprovalRegistry` **只住在** [`live_turn.py`](https://github.com/shibing624/agentica/blob/main/agentica/gateway/services/live_turn.py) 的当前 `LiveTurn` 上，不是 Agent LRU、也不是一份 Service 级 session map。构建 Agent 时注入的 `approve` 是闭包：每次 `wait` 查 `live_turn.active(session_id, owner)`。

- **没有 LiveTurn → 立即 `"deny"`**，工具结果是 `Tool call denied by user.`，对话不 500。IM、cron、非流式 `POST /chat` 都走这条。
- **不要用「当前 SSE 订阅者数量」当闸**：刷新瞬间订阅者是 0，误 deny。有 LiveTurn 就可以 park；断线靠 seq 回放把未决的 `approval_request` 再推一遍。LiveTurn 结束 / cancel 才 `deny_all`。
- 有未决审批的 LiveTurn **不会被 LRU 清掉**。

审批 **不走 `/ws`**，只走 chat SSE / `live_turn`。

### SSE 与 POST

流式回合里：

1. `tool_call` 事件带上 `tool_call_id`（以及 `name` / `args`），前端才能把卡挂到那一行。
2. 需要确认时再发 `approval_request`：`{tool_call_id, name, args, question, preview, similar_label, options}`。`question` 是模板（「是否允许运行以下命令？」），不另调 LLM。`similar_label` 是命令类（`rm -f`），不是文件名。复合命令、以及类短于 2 个 token 的 wrapper（`bash deploy.sh` / `python script.py`）的 `options` 只有 `allow` / `deny`。
3. 用户点卡：`POST /api/sessions/{session_id}/approvals/{tool_call_id}`，body `{decision: "allow"|"allow_prefix"|"deny"|"deny_prefix"}`。鉴权跟 chat 一样走 cookie 账号（`_account(request)`），按 `(owner, session_id)` 找 LiveTurn——不能点别人的卡。未知 id → 404。
4. 刷新 / 重连：`GET /api/chat/runs/{run_id}/events?after=` 先按 seq 回放已有事件（含 `tool_call`），再 `republish_pending_approvals()` 把仍未决的卡推一遍。

`allow` 只放行这一次（不落盘）；`allow_prefix` 把本项目后续**同类**调用自动放行，写入该 work_dir 的 `project.json` `approvals`（和 `work_dir` / `active_profile` 同一文件，按账号分区在 `~/.agentica/projects/<user>/<slug>/`）。`deny_prefix` 对称：同类下次在 **`ask` / `auto`** 下**直接拒绝**（不弹卡）；**`allow-all` 忽略项目拒绝**，只打 warning 并记一条 `approval_decision`（`reason: allow_all_ignore_deny`），命令照跑。命令按「可执行文件 + flags + 至多一个 subcommand」成类（`rm -f /tmp/a.ini` 批的是 `rm -f`，下一发 `rm -f /tmp/b.ini` 不再问；`git add` 不会放行 `git push`）。类短于 2 个 token 时不提供「允许/拒绝类似」：`bash deploy.sh` / `python script.py` 不能永久放行任意 `bash -c` / `python -c`。复合命令也不能用前缀概括，只提供允许一次 / 拒绝一次。argv 类在第一个像文件名的 token 处截断，因此 `find . -name x -exec …` 会被已批的 `find . -name` 覆盖——这是类授权的固有盲区。敏感路径只 grant 该文件本身，且不写入允许表。Deny 变成工具结果回给模型，同批其它工具不会 sibling-abort。人停在卡上超过 120s 也不会变成工具 TimeoutError——审批等待在 `wait_for` 之外。`tool_result` SSE 带 `tool_call_id`，并行工具不会把结果挂错行。`approval_decision` 在真正停过车的调用、以及项目 deny 授权静默拒绝时写入 session log。

## 消息路由

`MessageRouter` 决定每条入站消息交给哪个 `agent_id` 处理。默认所有消息路由到 `default_agent="main"`，
你可以按 `channel` / `channel_id` / `sender_id` 加规则：

```python
from agentica.gateway.services.router import RoutingRule
from agentica.gateway.channels.base import ChannelType
from agentica.gateway import deps

# 把所有 Telegram 消息交给 tg_agent
deps.message_router.add_rule(
    RoutingRule(agent_id="tg_agent", channel=ChannelType.TELEGRAM, priority=10)
)

# 把某个 QQ 群单独路由给专门的 agent
deps.message_router.add_rule(
    RoutingRule(
        agent_id="ops_agent",
        channel=ChannelType.QQ,
        channel_id="group:abc123",
        priority=20,
    )
)
```

会话 ID 由路由器统一生成：`agent:{agent_id}:{channel}:{channel_id}`，
保证跨渠道复用同一 Agent 时，每个会话拥有独立的上下文。

## 自定义渠道

继承 `Channel` 即可接入任何新平台：

```python
from agentica.gateway.channels.base import Channel, ChannelType, Message

class MyChannel(Channel):
    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WEB  # 或新增枚举值

    async def connect(self) -> bool:
        # 启动 SDK / 长轮询任务
        self._connected = True
        return True

    async def disconnect(self):
        self._connected = False

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:
        # 调用平台 SDK 发送
        for chunk in self.split_text(content, max_len=2000):
            ...
        return True
```

注册到 ChannelManager：

```python
from agentica.gateway import deps
deps.channel_manager.register(MyChannel())
await deps.channel_manager.connect_all()
```

## 定时任务（Cron）

Gateway 内建一个文件化的定时任务调度器，可让 Agent 在指定时刻自动跑 prompt，
结果落盘保存。调度器默认关闭，在 `~/.agentica/config.yaml` 的 `settings` 块打开
（与 CLI 的 `/cron daemon on` 共用同一个开关）：

```yaml
settings:
  cron.enabled: true
  cron.interval: 60        # 轮询间隔（秒），默认 60
```

打开后日志出现 `Cron scheduler started (60s tick)`。任务通过 HTTP API
（`/api/scheduler/jobs`）或 CLI（`/cron add ...`）创建，支持 cron 表达式、
自然语言间隔（`30m` / `every 2h`）和一次性 ISO datetime。网页上每个登录账号
只看到、也只能改自己的任务；调度器仍会执行所有账号到期的任务，跑的时候用该
任务上记录的 `user_id` 作为数据分区。

完整用法（字段说明、管理接口、运行结果查看）见
[定时任务（Cron）](../guides/cron_scheduler.md)。

## 故障排查

- **某个渠道启动后立刻报 "Missing xxx, skipped"**：环境变量没设，渠道被跳过；这是正常行为
- **`pip install 'agentica[xxx]'`：找不到 extras**：检查使用的是 `agentica` 包名（非旧名），且 pip 版本 ≥ 21；注意 extras 里的 `[]` 在 zsh 下需用单引号包裹，否则会被当成 glob 报错
- **WeCom `send` 一直返回 False**：该 chat 还没收到过用户消息，没有缓存到 `frame`；让用户先发一条
- **DingTalk 401 / errcode 9001**：`accessToken` 过期或 robotCode 与 ChatBot 创建时不一致，检查 `DINGTALK_CLIENT_ID`
- **WeChat 扫码后无响应**：检查日志里 `bot_id` 是否落盘到 `WECHAT_TOKEN_FILE`；如已过期把 token 文件删了重启即可重新扫码

## 下一步

- [ACP 集成](acp.md) — 把 Agent 接入 IDE
- [MCP 集成](mcp.md) — 让 Agent 能调用外部工具协议
- [Hooks](hooks.md) — 监听 Agent 事件流
