# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Policy

| Change type | Version bump | Example |
|-------------|-------------|---------|
| New public class, function, or protocol | **minor** | `1.3.x` → `1.4.0` |
| Bug fix, internal refactor (no API change) | **patch** | `1.3.2` → `1.3.3` |
| Breaking change to public API | **major** | `1.x.y` → `2.0.0` |

A "public API" is anything importable from `agentica` top-level `__init__.py`.

---

## [Unreleased]

### Added
- Standing-goal loop P1 (S + A tiers):
  - `Runner._run_impl` early-loads any persisted active `GoalState` from `SessionLog` and binds `TaskAnchor` to the goal objective — SDK paths now get goal-aware retrieval automatically, not just the CLI.
  - `GoalState` gains `token_budget` / `tokens_used` / `wall_clock_budget_sec` / `wall_clock_used_sec` and a new `budget_limited` status (semantically distinct from `paused`). `evaluate_after_turn(token_delta=..., elapsed_sec=...)` short-circuits the judge LLM call when the cap is hit.
  - New `agentica.tools.goal_tool.GoalTool.update_goal(status, reason)`: a receive-only model tool letting the agent self-mark `complete` or `paused` (cannot rewrite the objective). CLI auto-attaches on `/goal` set and detaches on goal termination.
  - `RunEventType.goal_set / goal_continuing / goal_completed / goal_paused` events emitted through an optional `GoalManager.event_callback`.
- New example `examples/cli/03_goal_loop_demo.py`: 4-scenario SDK tutorial (basic loop / GoalTool / token budget / event_callback) against a real LLM.

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
