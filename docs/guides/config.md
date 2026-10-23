# 统一配置文件 (`~/.agentica/config.yaml`)

`~/.agentica/config.yaml` 是 CLI、Web、Desktop 和 SDK 共用的配置文件，用来管理你的模型列表（`profiles`）、全局行为开关（`settings`）和环境变量（`env`）。

你可以运行向导交互生成它，也可以直接用编辑器修改（修改注释会被保留）：

```bash
agentica setup
```

---

## 完整配置示例

把下面需要的那几行拷进 `~/.agentica/config.yaml` 即可：

```yaml
# 当前默认使用的模型 profile
active_profile: default

profiles:
  # 主力模型（可配多个 profile，在 CLI 里用 /model <名字> 切换）
  default:
    model_provider: deepseek
    model_name: deepseek-v4-flash
    base_url: https://api.deepseek.com
    api_key: sk-...
    # 可选调优参数（不填则使用模型默认值）
    reasoning_effort: high
    max_tokens: 8192
    context_window: 1000000
    compact_token_limit: 300000    # 上下文达到 30 万 token 时提前触发换窗压缩
    temperature: 0.7

    # 可选：给子任务（task）和后台记忆提取配一个更便宜、更快的副模型
    auxiliary_model:
      model_provider: zhipuai
      model_name: glm-4.7-flash
      base_url: https://open.bigmodel.cn/api/paas/v4
      api_key: your-zhipu-key

  # 第二个 profile 示例：使用 OpenAI Responses API
  gpt_responses:
    model_provider: openai
    model_name: gpt-5.6-sol
    base_url: https://api.openai.com/v1
    api_key: sk-...
    wire_api: responses
    reasoning: high
    max_tokens: 4096

# 全局行为开关（CLI 与 Gateway 共用）
settings:
  enable_evict: true               # 窗口占用超 80% 时清理旧工具结果（默认 true）
  enable_auto_compact: true        # 窗口占用近 95% 时自动换新窗（默认 true）
  # compact_token_limit: 300000    # 全局压缩工作上限（可选）
  # num_history_turns: 20          # 限制最多重放最近 N 轮（默认不限，重放当前窗口全部轮次）
  # sandbox: workspace-write       # 开启系统级沙箱（macOS Seatbelt / Linux bwrap）
  # cron.enabled: true             # 开启定时任务调度器

# 附加环境变量（会自动注入进程环境，适合放搜索引擎等第三方 Key）
env:
  AGENTICA_WEB_SEARCH: bocha
  BOCHA_API_KEY: "sk-..."
```

---

## 配置优先级

当同一个配置在多处出现时，按从高到低生效：

1. **命令行参数**（如 `agentica --profile gpt_responses`、`--max_tokens 4096`）
2. **Shell 环境变量 / `.env` 文件**（当前项目 `.env` 或 `~/.agentica/.env`）
3. **`~/.agentica/config.yaml`**

---

## 日常怎么切换和修改

### 在交互式 CLI 里切换模型

```text
> /model                  # 列出所有已配置的 profile
> /model gpt_responses    # 将当前项目切换到 gpt_responses（记在项目目录下，不影响其他项目）
> /model --clear          # 清除当前项目的模型覆盖，恢复使用 config.yaml 的 active_profile
```

### 在启动时临时指定某次会话的 profile

```bash
agentica --profile gpt_responses
```

### 在 CLI 里直接改配置项

```text
> /config                                  # 查看当前生效的模型、目录与配置
> /config set compact_token_limit 300000   # 修改配置项
> /config env BOCHA_API_KEY sk-xxx         # 写入环境变量到 config.yaml
```

---

## Profile 字段一览

每个 `profiles.<名字>` 下可配置以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_provider` | `str` | 提供商标识，如 `openai`、`anthropic`、`deepseek`、`zhipuai`、`qwen`、`moonshot`、`ark`、`ollama` |
| `model_name` | `str` | 模型 ID，如 `gpt-4o`、`claude-sonnet-4-20250514`、`deepseek-chat`、`glm-4.7-flash` |
| `base_url` | `str` | API 接口地址（官方地址或你的代理网关地址） |
| `api_key` | `str` | API 密钥 |
| `auxiliary_model` | `dict` | 副模型配置块（支持与主模型相同的字段），用于 `task` 子任务、记忆提取、`/goal` 裁判等后台任务；不配则默认复用主模型 |
| `wire_api` | `str` | 仅 `model_provider: openai` 使用：`chat_completions`（默认）或 `responses`（启用 OpenAI Responses API） |
| `reasoning_effort` | `str` | Chat Completions 思考强度（如 `low` / `medium` / `high` / `max`），原样传给服务端 |
| `reasoning` | `str` | Responses API 思考强度（仅 `wire_api: responses` 时生效） |
| `max_tokens` | `int` | 单次回复的最大输出 token 数 |
| `context_window` | `int` | 模型上下文窗口大小（用于状态栏显示和压缩水位计算，覆盖内置目录默认值） |
| `compact_token_limit` | `int` | 压缩工作阈值（见 [上下文压缩](../advanced/compression.md)） |
| `temperature` / `top_p` | `float` | 采样温度与 top_p |
| `extra_body` | `dict` | 原样透传给请求体的额外 JSON 字段 |
| `extra_headers` | `dict` | 原样透传的 HTTP 请求头（非 `anthropic` 提供商使用） |
| `default_headers` | `dict` | 静态 HTTP 请求头（`anthropic` 提供商使用，或用于固定路由头） |
| `enable_cache_control` | `bool` | 是否注入显式缓存断点（`anthropic` 默认开启；OpenAI 兼容代理网关可按需设为 `true`） |
| `cache_control_session_header` | `str` | 按会话粘性路由的请求头名称（见下节） |

---

## 进阶：代理网关的缓存粘性路由

如果你使用的是多节点负载均衡的内部代理网关，要让 Prompt Cache 命中，需要让同一个会话（或同一个账号）的连续请求落到同一台上游机器上。

Depending on your gateway's supported header, choose one of the two modes inside your profile:

### 方式 A：账号级固定路由（跨会话共享缓存，推荐日常单项目开发）

```yaml
profiles:
  my_proxy:
    model_provider: anthropic
    model_name: claude-sonnet-4-20250514
    base_url: https://your-gateway.example.com
    api_key: sk-...
    default_headers:
      X-Sticky-Routing: "my-fixed-key"
```

同一个账号开的所有会话都发往同一台上游，新开会话也能直接命中之前写下的系统提示词缓存。

### 方式 B：会话级路由（每个会话独立分配节点，适合多会话高并发）

```yaml
profiles:
  my_proxy:
    model_provider: openai
    model_name: gpt-4o
    base_url: https://your-gateway.example.com/v1
    api_key: sk-...
    cache_control_session_header: X-Session-Id
```

只需填写请求头**名称**，Agentica 会在每次请求时自动填入当前会话的 `session_id`。

---

## 下一步

- [安装与配置](../getting-started/installation.md) — 环境变量与首次安装
- [模型提供商速查](models.md) — 各家模型的类名与默认配置
- [OpenAI Responses API](openai-responses.md) — `wire_api: responses` 的详细用法
- [架构与内部实现（开发者选读）](../introduction/architecture.md) — 了解配置解析优先级与项目级侧车文件的内部实现
