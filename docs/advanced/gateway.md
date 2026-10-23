# Web、桌面端与微信/IM (`agentica-gateway`)

`agentica-gateway` 把你的本地 Agent 变成一个常驻服务：既提供**本地网页（Web UI）**和**桌面应用（Desktop App）**，又能同时接入**个人微信、飞书、企业微信、Telegram、QQ、钉钉、Discord、Slack**，让你在手机上直接对话，或者遥控电脑上正在跑的 CLI 终端会话。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Gateway Web UI" />

---

## 一、安装与启动

### 1. 安装

如果你还没装过 Agentica，或者之前只装了 CLI，运行以下命令安装带 Web/IM 能力的完整版：

```bash
uv tool install --force "agentica[gateway]"
```

如果需要接入个人微信、企微、Telegram 等 IM 渠道，把渠道名一起写进引号里即可：

```bash
uv tool install --force "agentica[gateway,wechat]"                 # 个人微信（扫码即用）
uv tool install --force "agentica[gateway,wechat,wecom,telegram]"  # 同时安装多个渠道依赖
```

> 飞书（`feishu`）已默认包含在 `[gateway]` 中，无需额外追加。

### 2. 启动本地 Web 服务

在你的项目目录下直接运行：

```bash
agentica-gateway
```

默认监听 `http://127.0.0.1:8881`。终端启动日志会打印出：
- 首次免密打开的地址（`http://127.0.0.1:8881/chat?token=...`）
- 或者初始管理员账号 `default` 的随机登录密码（用于 `/login` 页面登录）

常用启动参数：

```bash
agentica-gateway --set-password      # 设置或修改网页登录密码
agentica-gateway --host 0.0.0.0      # 允许局域网其他设备访问（必须先设置密码）
agentica-gateway --port 8890         # 指定监听端口
```

### 3. 桌面应用（Desktop App）

如果你喜欢独立窗口应用，可以直接下载安装包（macOS `.dmg` / Windows `.exe` / Linux `.AppImage` & `.deb`），详见 [README 桌面版说明](https://github.com/shibing624/agentica#桌面版)。

桌面版与 CLI、Web 共用同一个 `~/.agentica` 配置与历史记录；如果本机还没安装 `agentica-gateway`，首次打开桌面版时会自动安装运行环境。

### 4. Docker 部署（可选）

```bash
cp .env.docker.example .env   # 填入 OPENAI_API_KEY 等模型密钥
docker compose up -d --build
```

启动后打开 `http://127.0.0.1:8881/chat`。

---

## 二、多账号与权限控制

### 1. 多账号隔离

Gateway 支持多用户登录：
- 管理员（`default`）登录后，点击网页右侧 **「用户管理」**（`/users`）即可新增普通账号、重置密码或删除账号。
- **每个账号拥有自己独立的会话列表、记忆（`MEMORY.md` / `AGENTS.md`）和定时任务**，互相不可见。

### 2. 工具审批档位（`ask` / `auto` / `allow-all`）

在网页右上角或设置中可以随时切换工具权限档位（默认 `auto`）：

| 档位 | 自动放行的操作 | 会弹出审批卡等你的操作 |
|---|---|---|
| **`ask`** | 读取文件、只读命令、搜索记忆、子任务 | **修改文件**、**会改动状态的命令**、联网搜索/抓取、高危命令 |
| **`auto`（默认）** | 读取文件、**当前工作区内的文件修改**、普通命令、联网搜索 | **工作区外或敏感路径的文件写入**、高危命令 |
| **`allow-all`** | 全部自动放行 | 无 |

当弹出审批卡时，你可以选择：
- **允许一次 / 拒绝一次**：仅对当前这次调用生效。
- **允许同类 / 拒绝同类**：记住这类命令前缀（如 `pytest`、`git status`），当前项目内后续遇到同类命令自动放行或直接拒绝。

---

## 三、用手机遥控电脑上的 CLI 会话

只要本机跑着 `agentica-gateway` 并连上了任意一个 IM 渠道（如个人微信），你出门在外就能用手机直接查看并指挥电脑终端里开着的 `agentica` 会话：

| 你在微信/IM 里发送 | 效果 |
|---|---|
| **`@list`** | 列出电脑上所有正在运行的 CLI 会话（名称、忙闲状态、所在目录、当前分支与正在做的任务） |
| **`@nlp-f1 把测试重新跑一遍`** | 把这句话直接敲进名为 `nlp-f1` 的终端会话，并将该会话置顶为默认目标 |
| **`看下报错日志`**（不带 `@`） | 置顶后，后续直接发文字就会继续发给同一个终端会话 |
| **`@off`** | 取消置顶，恢复和 Gateway 自己的个人助理对话 |

**甚至不需要记会话名**：你也可以不打 `@`，直接对微信里的助理说「帮我看看电脑上跑着的几个会话都在干嘛，让改完代码的那个把改动提交了」，网关 Agent 会自己调用 `list_agents` 和 `send_message` 替你安排并把结果汇总发回手机。

> 如果不需要手机访问本机终端会话，可在启动时设置 `PEER_BRIDGE=false agentica-gateway` 关闭。

---

## 四、接入个人微信与各 IM 平台

所有 IM 渠道均采用**长轮询或 WebSocket 长连接**，**不需要公网 IP、不需要域名、也不需要配置公网 Webhook**，在家里或公司内网电脑上直接启动即可连通。

### 1. 个人微信（扫码即用，最推荐）

无需注册任何开放平台应用：

1. 安装微信依赖并启动网关：
   ```bash
   uv tool install --force "agentica[gateway,wechat]"
   agentica-gateway
   ```
2. 打开浏览器进入 **设置 › 个人助理**（`/assistant`），找到「个人微信」点击 **「配置」**。
3. 用手机微信扫描页面弹出的二维码即可完成绑定（登录凭证保存在 `~/.agentica/cache/wxbot_token.json`，下次启动免扫码）。

<div style="display: flex; gap: 16px; align-items: flex-start;">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat-clawbot-qr.png" alt="微信扫码绑定" width="400" />
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat-clawbot-snap.jpg" alt="微信直接对话" width="150" />
</div>

> **发送语音与视频**：如果主模型本身不支持直接听语音/看视频，可以在 `~/.agentica/config.yaml` 的 `settings.media_model` 里配一个 Gemini 模型用于自动转写和描述。

### 2. 其他 IM 渠道配置一览

在 `~/.agentica/.env`（或启动环境变量）中填入对应平台的凭证即可自动启用：

| 渠道 | 安装依赖 (`uv tool install --force ...`) | 必填环境变量 | 限制可访问用户的白名单变量（可选，留空为不限） |
|---|---|---|---|
| **飞书 (Lark)** | `"agentica[gateway]"`（已内置） | `FEISHU_APP_ID`<br>`FEISHU_APP_SECRET` | `FEISHU_ALLOWED_USERS`<br>`FEISHU_ALLOWED_GROUPS` |
| **企业微信 (WeCom)** | `"agentica[gateway,wecom]"` | `WECOM_BOT_ID`<br>`WECOM_SECRET` | `WECOM_ALLOWED_USERS` |
| **Telegram** | `"agentica[gateway,telegram]"` | `TELEGRAM_BOT_TOKEN` | `TELEGRAM_ALLOWED_USERS` |
| **QQ 机器人** | `"agentica[gateway,qq]"` | `QQ_APP_ID`<br>`QQ_APP_SECRET` | `QQ_ALLOWED_USERS` |
| **钉钉 (DingTalk)** | `"agentica[gateway,dingtalk]"` | `DINGTALK_CLIENT_ID`<br>`DINGTALK_CLIENT_SECRET` | `DINGTALK_ALLOWED_USERS` |
| **Discord** | `"agentica[gateway,discord]"` | `DISCORD_BOT_TOKEN` | `DISCORD_ALLOWED_USERS`<br>`DISCORD_ALLOWED_GUILDS` |
| **Slack** | `"agentica[gateway,slack]"` | `SLACK_BOT_TOKEN` (`xoxb-...`)<br>`SLACK_APP_TOKEN` (`xapp-...`) | `SLACK_ALLOWED_USERS`<br>`SLACK_ALLOWED_CHANNELS` |

各平台创建机器人的简要入口：
- **飞书**：[飞书开放平台](https://open.feishu.cn) → 创建企业自建应用 → 开启机器人能力与「接收消息」权限 → 事件订阅选择**使用长连接接收事件**。
- **企业微信**：企业微信管理后台 → 安全与管理 / 智能机器人 → 创建 AI Bot → 获取 `bot_id` 与 `secret`。
- **Telegram**：私聊 [@BotFather](https://t.me/BotFather) 发送 `/newbot` 拿 token。
- **QQ**：[QQ 开放平台](https://q.qq.com) 创建机器人获取 AppID 与 AppSecret。
- **钉钉**：[钉钉开放平台](https://open.dingtalk.com) 创建企业内部应用 → 开启机器人能力并选用 Stream 模式。

---

## 五、通过代码调用 Gateway（TypeScript / HTTP API）

如果你有外部脚本或前端想调用正在运行的 `agentica-gateway`，可以直接使用官方 TypeScript SDK 或 HTTP 接口（网关启动后可访问 `http://127.0.0.1:8881/docs` 查看完整交互式 OpenAPI 文档）。

### TypeScript SDK (`@agentica-ai/sdk`)

```bash
npm install @agentica-ai/sdk
```

```ts
import { Agentica } from "@agentica-ai/sdk";

const agentica = new Agentica({
  baseURL: "http://127.0.0.1:8881",
  // 机器令牌可从 ~/.agentica/cache/gateway/runtime.json 读取，或通过环境变量 AGENTICA_GATEWAY_TOKEN 固定
  apiKey: process.env.AGENTICA_GATEWAY_TOKEN,
});

for await (const event of agentica.chat.stream({
  message: "帮我总结一下当前目录的 README",
  session_id: "demo-session",
})) {
  if (event.event === "content") {
    process.stdout.write(String(event.data));
  }
}
```

### 主动向 IM 渠道发送通知 (`/api/send`)

你可以在部署脚本或告警钩子里直接让 Gateway 往你的微信、飞书或 QQ 推消息：

```bash
TOKEN=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.agentica/cache/gateway/runtime.json')))['token'])")

curl -X POST http://127.0.0.1:8881/api/send \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "feishu",
    "channel_id": "ou_xxx",
    "message": "线上构建已完成 ✅"
  }'
```

---

## 下一步

- [CLI 终端指南](../getting-started/terminal.md) — 终端交互与斜杠命令
- [定时任务 (/cron)](../guides/cron_scheduler.md) — 让 Gateway 或 CLI 按计划定时执行任务
- [统一配置文件 (config.yaml)](../guides/config.md) — 配置主模型、副模型与全局开关
