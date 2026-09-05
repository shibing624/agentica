# Unified Configuration (`~/.agentica/config.yaml`)

`~/.agentica/config.yaml`（路径受 `AGENTICA_HOME` 影响）是 SDK 与 CLI 共享的唯一配置源。YAML（用 `ruamel.yaml` 读写以保留注释），支持命名 profile、顶层 `settings` 块和自由格式的 `env` 块。写入时 `chmod 0o600`，因为 profile 含密钥。`cli_config.json` 已不存在——`config.yaml` 取代它成为 key 与模型配置的唯一存储（无向后兼容）。

## 两个模型概念：main + auxiliary

每个 profile 顶部是 main model 字段，可选一个 `auxiliary_model` 子块。auxiliary model 是便宜/快速模型，用于**所有**非用户面对的 LLM 工作：记忆抽取、上下文压缩、用户纠正分类、goal 判断、skill 升级，以及 `task` subagent 工具。省略 `auxiliary_model` 则复用 main model。CLI 只暴露 `--auxiliary_model_*`（没有 `--task_model_*`）；`cli/runtime.py::create_agent` 把 auxiliary model 同时作为 `auxiliary_model=` 和 `task_model=` 传入。

## SDK 契约

SDK 仍读纯环境变量。import 时 `agentica/config.py` 调 `apply_global_config()`（实现在 `agentica/global_config.py`），把**活动** profile 的 `api_key`（通过 `PROVIDER_API_KEY_ENV` 投射到 provider 专属 env 变量）和自由 `env` 块用 **`setdefault` 语义**注入 `os.environ`（绝不覆盖已有变量）。无 model 类或工具改动。

## 优先级（从高到低）：shell env > `.env` > `config.yaml`

`~/.agentica/.env` 仍由 dotenv 加载，用于手维护的 key / MCP 工具——`.env` 与 `config.yaml` 共存。`openai` provider 带自定义 `base_url` 时，`OPENAI_BASE_URL` 也会被注入，使自定义端点无需额外 flag。

## Profile schema

必填：`model_provider`、`model_name`、`base_url`、`api_key`。可选调优（省略则用 model/factory 默认）：

| Field | Type | Purpose |
|-------|------|---------|
| `reasoning_effort` | low/medium/high/max | 思考深度（OpenAI/DeepSeek；Claude 用独立 thinking budget，`wire_api: responses` 用 `reasoning`，均跳过此项） |
| `wire_api` | chat_completions/responses | 线协议（仅 `model_provider: openai`）；省略默认 `chat_completions`，`responses` 启用 OpenAI Responses API |
| `reasoning` | str | Responses API 的 reasoning 配置（仅 `wire_api: responses` 时生效） |
| `max_tokens` | int | 输出 token 上限 |
| `context_window` | int | 上下文上限；**覆盖** catalog 自动检测值。不发给 API——仅用于 budget/compression/status 显示 |
| `temperature` | float | 采样 |
| `top_p` | float | 采样 |
| `extra_body` | dict | 原样透传给 API 的额外请求体参数 |
| `extra_headers` | dict | 原样透传的额外 HTTP 头（**非 anthropic**；`/v1/messages` 没有 per-request header 通道，用 `default_headers`） |
| `default_headers` | dict | 写死值的静态 HTTP 头，建客户端时注入（**仅 anthropic**）。见下节 |
| `enable_cache_control` | bool | prompt cache 断点注入。Claude 默认 **开**（原生 `cache_control`）；OpenAI 兼容端点默认 **关**，显式开启后才注入 |
| `cache_control_session_header` | str | 粘性路由 header 的**名字**，值自动填当前会话 id。anthropic 与 openai 两侧都生效。见下节 |
| `cache_control_messages` | int | 最近若干条消息上打断点（默认 3；仅 `wire_api: chat_completions`，Claude 自己管断点） |
| `cache_keepalive` | bool | 空闲时定期 ping 保活缓存（默认 true；仅 `wire_api: chat_completions`） |
| `auxiliary_model` | block | 可选廉价模型（provider/name/base_url/api_key + 同上调优项）用于后台调用 + `task` subagent。同 provider 省略字段继承 main model；跨 provider 不继承 main 的 key/base_url；无论是否同 provider 都不继承 main 的 `extra_body`/`extra_headers` |

profile 之外的顶层块：`settings`（CLI 行为开关，与 model 无关，如 `num_history_turns`，经 `get_setting`/`set_setting` 读写）和 `env`（任意 key-value，注入 `os.environ`）。

## 代理网关的粘性路由（prompt cache 的前提）

prompt cache 只在**连续请求落到同一台上游**时才有意义。多数聚合型代理网关为了吞吐会把请求扇出到多个上游，缓存不会跟着走——于是每一轮都是 cache write，账单反而更高。这类网关通常允许用一个请求头把路由钉住，agentica 有两种配法，区别只在**粘性的粒度**。

### 账号级：`default_headers` 写死一个值

```yaml
default_headers: {X-Sticky-Routing: token}
```

原样注入客户端静态头。上例是按 Bearer token 粘——**同一个 key 的所有会话共用一台上游**，于是缓存跨会话共享：今天在同一个项目里开的第十个会话，命中的还是第一个会话写下的缓存。

### 会话级：`cache_control_session_header` 只给名字

```yaml
cache_control_session_header: X-Session-Id
```

只配 header 的**名字**，值由 `agentica/model/cache_routing.py::resolve_cache_session_id` 每次请求现算：有会话用 `session_id`，没有（裸 SDK）才回落到 `~/.agentica/cache/cache_routing.json` 里按 `base_url` 存的持久 id。

> 这个值**不能缓存到实例字段上**。`Model.session_id` 由 `Agent.update_model()` 每轮赋值，在更早的调用里合法地为 `None`；把第一次的答案冻住，真实会话就永远上不了线。

### 怎么选

| | 缓存命中 | 并发 | 路由稳定性 |
|---|---|---|---|
| 账号级（写死） | 跨会话共享，命中率高 | 所有会话挤一台上游，网关侧并发能力下降 | 稳定，除非改配置 |
| 会话级（按 session） | 每个新会话冷启动一次 | 天然分散到多台 | **每开一个会话重新路由一次** |

会话级那条"重新路由"有个反直觉的后果：它等于周期性地回到**未钉住**的状态。如果网关后面有一台行为异常的上游（例如对合法的 tool schema 回 `invalid_request_error`），会话级粒度会让你隔三差五撞上它一次；账号级钉住之后反而再也碰不到。反过来，真撞上了，账号级只能改配置才能换走，会话级开个新会话就换了。

经验法则：**长期在同一个项目里反复开会话，选账号级**；需要逃生能力或要避免并发挤压，选会话级。

### 两个都配时

`default_headers` 是用户写死的显式值，**优先**——注入用的是 `setdefault`（`agentica/model/anthropic/claude.py:327`），同名 header 不会被会话 id 覆盖。

### 缓存前缀什么时候会变

粘住了路由，缓存仍然按**字节精确的前缀**匹配。system prompt 里带工作目录和 AGENTS.md 注入，所以**换项目目录 = 换前缀 = 重写一次缓存**，这是预期行为。会话中途哪些内容被冻结、为什么，见 [Memory & Workspace · 会话快照与 prompt cache](../concepts/memory.md#prompt-cache)。

## Key functions（`agentica/global_config.py`）

读写：`global_config_path`、`load_global_config`/`save_global_config`、`get_profile`/`get_profiles`、`get_active_profile_name`/`set_active_profile`、`upsert_profile`/`delete_profile`、`find_profile_for_provider`、`apply_global_config`、`provider_api_key_env`、`resolve_active_profile_name`、`write_commented_template`、`get_setting`/`set_setting`。其中 9 个核心读/写 API（`global_config_path`、`load_global_config`、`save_global_config`、`get_profile`、`get_profiles`、`get_active_profile_name`、`set_active_profile`、`upsert_profile`、`apply_global_config`）从 `agentica/__init__.py` re-export；其余用 `from agentica.global_config import ...`。所有写路径都 round-trip YAML 文件以保留用户注释。

## Profile 生效顺序：project override > global active > default

`resolve_active_profile_name(work_dir)` 回答"config 层认为哪个 profile 生效"：**project override**（按项目目录记录在 project store，由 `/model <name>` 写入）> `config.yaml` 的 `active_profile` > 内置 `default`。返回 `(name, source)`，source ∈ `project`/`global`/`default`（启动时经 `--profile` 指定则为 `flag`）。

## CLI wiring

- `cli/setup.py::resolve_model_config` 按优先级解析 provider/model/base_url/api_key + 7 个调优参数（`wire_api`/`reasoning`/`reasoning_effort`/`max_tokens`/`context_window`/`temperature`/`top_p`）+ 可选 auxiliary model：CLI flag > `--profile <name>` > 生效 profile（见上节顺序，含其 `auxiliary_model` 块）> preset 默认。`--profile` 仅当前会话有效（不写盘），且指定的 profile 不存在时直接退出。`--model_name` 只替换模型名，无法离开当前端点（base_url/key 仍来自 profile）；`--model_provider` 可换 provider，但会放弃当前 profile 的调优参数，key 改由 `get_profile_api_key` 从匹配 profile 解析。它还报告会话所用 profile 为 `profile_name`/`profile_source`，存于 `agent_config`，通过 `setup.session_profile()` 处处读回；空名表示"无 profile 描述此会话"（flag 替换了 model）且不可 fallback，而缺失 key（手建 `agent_config`）才 fallback 到 `resolve_active_profile_name`。**绝不把 `resolve_active_profile_name` 的结果当成"当前会话正在跑的 profile"直接展示**：它回答的是 config 层指向什么，而非正在运行什么。onboarding wizard（`run_onboarding`）首次写注释模板（`write_commented_template`），再通过 `upsert_profile(make_active=True)` 写 profile；`_prompt_advanced_params` 收集调优；`_prompt_auxiliary_model` 可选收集 auxiliary model。`should_onboard`/`has_api_key` 把完整活动 profile（或任意含该 provider key 的 profile）视为足以跳过 onboarding。自定义 OpenAI 兼容端点用 host 后缀的 profile 名（如 `openai@my-llm.local`，由 `_profile_name_for` 生成）；`get_profile_api_key` 取代旧的 `get_saved_api_key`（key 现在 profile 里）。
- `cli/runtime.py::get_model`/`create_agent`/`_build_sibling_model` 接受并透传调优参数。`_build_sibling_model(cfg, "auxiliary")` 构建 auxiliary model；`create_agent` 把它同时作为 `auxiliary_model=` 和 `task_model=` 传入。
- `cli/main.py` 按优先级 `CLI flag > profile > None` 填充 `agent_config` 调优 + auxiliary 字段。
- `cli/commands/model_config.py`：`/model`（或旧写法 `/model profile`）列出 profiles；`/model <name>` 运行时切换——写入的是 **project-scoped override**（只动 override 指针，绝不重写 profile body，那是 `agentica setup` 的职责）；`/model --clear` 清除 override 并回落到 global default。自由格式 `/model provider/name` 已被拒绝（提示改用 `agentica setup`），旧的 `_persist_model_choice` 写回逻辑已移除。`_apply_profile` 应用完整调优集并刷新 task subagent 的 auxiliary 指针；`/config set <field> <value> [profile]` 原地编辑某个 profile 字段；跨 provider key 解析用 `get_profile_api_key`。

## 示例

```yaml
# ~/.agentica/config.yaml — 手可编辑；写入时保留注释。
active_profile: default

profiles:
  default:
    # main model（用户面对的轮次）
    model_provider: deepseek
    model_name: deepseek-v4-flash
    base_url: https://api.deepseek.com
    api_key: sk-...
    # 可选调优（省略则用默认）
    reasoning_effort: max
    max_tokens: 8192
    context_window: 1000000
    compact_token_limit: 300000  # 压缩工作阈值；不改 context_window
    temperature: 0.7
    top_p: 0.95
    # 可选 auxiliary model（后台调用 + `task` subagent）；省略则复用 main
    auxiliary_model:
      model_provider: zhipuai
      model_name: glm-4.7-flash
      base_url: https://open.bigmodel.cn/api/paas/v4
      api_key: sk-...

# CLI / gateway 行为开关（与 model profile 无关）
settings:
  num_history_turns: 20
  # enable_evict: true         # Layer 1 淘汰旧工具结果（默认开）
  # enable_auto_compact: true  # Layer 2 窗口满时自动摘要（默认开；/compact 仍可用）
  # compact_token_limit: 300000  # 可选工作阈值；不配 = 窗口×0.95 才摘要
  # gateway 入站图片/语音/视频：底模看不了时用这个 Gemini 描述/转写
  # media_model:
  #   model_provider: openai
  #   model_name: gemini-3.6-flash
  #   base_url: https://generativelanguage.googleapis.com/v1beta/openai
  #   api_key: sk-...

# 自由 env 块（shell/.env 值仍优先于此）
env:
  SERPER_API_KEY: "..."
```

> Note: gateway（web 服务）同样走统一流——`gateway/config.py::Settings.from_env` 调 `apply_global_config()` 读取活动 profile 的 main + auxiliary model（含 `wire_api`/`reasoning`），独立的 `task_model_*` 配置已移除（task model 即 auxiliary model）。gateway 自身的服务设置（端口、上传限制、飞书凭证等）仍是 env var，不进 config.yaml。

## 下一步

- [安装](../getting-started/installation.md) -- provider env 配置与多 provider 组合
- [Agent](../concepts/agent.md) -- auxiliary/fallback model 的 SDK 用法
