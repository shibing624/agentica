# Thin Gateway / Common Backend API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把跨端共同语义从 Web/TS 与 Gateway 胶水层里收回到 Python 核心 API，让 Gateway 变薄、Web 只保留纯前端与 Web 特性，按 TDD 分阶段消除语义漂移。

**Architecture:** `agentica/memory/*`、`agentica/run/*` 和新的 `agentica/session_api.py` 负责事实 owner、结构化投影和稳定 identity；`gateway/src/agentica_gateway/**` 只做 HTTP/SSE、鉴权、账户隔离、进程/连接的适配；`web/src/**` 只消费 API 返回的 view model 并做浏览器交互。Desktop 继续加载同一份 Web；IM 继续作为另一种渲染媒介消费同一批 core/gateway API。

**Tech Stack:** Python 3.12 / FastAPI / pytest / node:test / React + TypeScript / Vite / SessionLog JSONL / SSE / sidecar `*.meta.json` / background job JSON records。

---

## 0. 背景与判定标准

会话标题问题已经证明：当一个结构化事实被各端按字段或自然语言反复推导时，会出现「请求发出去了但 Web 没起名」「定时任务显示 `Chat`」「IM/CLI/Web 对同一会话标题不一致」这类语义漂移。标题已通过 `SessionLog.display_title(...)` 和写回后 `bump_pulse(owner)` 收口；`18b2a884` 又通过 `agentica/memory/log_rows.py` 把 user row 的 `automated` / `steer` / window preamble 判断收回核心层。

本 plan 继续沿这个方向，但把边界明确到用户提出的原则：

- **common 的东西由后端 core API 做**：日志语义、session view、turn index、usage、run state、background jobs、title、search hits、queue/steer identity、goal state。
- **Gateway 是薄层**：HTTP/SSE、鉴权、owner scope、进程内必要的 LiveTurn pub/sub、IM/Desktop/Web 的传输适配、web-only 配置；不定义业务语义，不重复解析 transcript，不推导最终展示事实。
- **Web/TS 只做纯前端和 Web 特性**：布局、动画、输入、乐观 UI、localStorage 缓存、浏览器断线重连、Web 专属弹层；不解析日志协议，不计算跨端事实，不拥有权威状态。
- **Web 特性的 gateway 承接**：仅当事实确实是 Web/Desktop 浏览器通道特有（例如 DOM 标题、会话 tab、窗口焦点、本地缓存冲突）才留在 `web/` 或 gateway 的 web adapter。

判定一个逻辑是否放错层，用这三个问题：

1. CLI / IM / Web / SDK 是否需要同一个答案？需要 → core API。
2. 这个答案是否来自 JSONL、sidecar、background record 或 agent runtime？是 → core API。
3. 换一个前端框架、改成终端或 IM 后是否仍成立？是 → core API；只在浏览器成立 → Web。

非目标：

- 不新增第三个持久化 store；JSONL 仍是 transcript 事实，sidecar 仍是 session metadata，background record 仍是 job/query fact。
- 不做大爆炸重写；每个 phase 都要有可独立回滚的 API + 测试。
- 不把纯 UI 状态（sidebar 展开、输入框草稿、动画、折叠偏好、浏览器 localStorage key）强行搬后端。

---

## 1. 当前基线（以 `main` 为准）

已落地，不在本 plan 中重复实现：

- `SessionLog.display_title(...)`：`agentica/memory/session_log.py:1925`；Gateway/CLI/IM 读标题统一走它。
- `agentica/memory/log_rows.py`：`classify_user()` / `row_payload()` / `starts_turn()` 已经是 user row 的单一语义入口；`gateway/src/agentica_gateway/routes/traces.py:30-42` 通过 `with_row_views()` 把 `row` 发给 Web，`web/src/lib/hydrateMessages.ts:165-170` 现在要求服务端提供 `row`，没有就抛错，不再自己解析前言。
- CLI `/ps` 与 `/stop` 已改走 `load_session_job_views()`；后台 job 的 durable record 是跨进程任务事实源。
- 标题生成写回 sidecar 后 `live_turn.bump_pulse(owner)`，Web 侧栏通过 pulse 刷新目录。

仍需规划的核心问题：

1. Web `sessions.ts` 仍做 catalog merge、turn page fallback、usage/round 对齐、本地 running/background state 合并：`web/src/sessions.ts:49-139`、`:319-398`、`:588-633`。
2. `routes/chat.py` 与 `agent_service/sessions.py` 仍承担大量 view 组装与状态推导，Gateway 不薄。
3. pulse 把进程内版本、磁盘 mtime 和 `active[]` 混在一个响应：`gateway/src/agentica_gateway/routes/chat.py:615-622`；`LiveTurn` 是进程内事实：`gateway/src/agentica_gateway/services/live_turn.py:350-368`。
4. SSE replay 把 delta 和 cumulative snapshot 混用：`gateway/src/agentica_gateway/services/live_turn.py:265-304`。
5. `/goal`、queue/steer、background badge 等仍有浏览器本地权威副本：`web/src/store.ts:223`、`web/src/pages/ChatPage.tsx:2679-2723`、`web/src/components/BackgroundJobs.tsx:25-43`。
6. tool error 字段双拼写、token/cost 多来源、Trace round 与 Web bubble 按位置对齐，仍需要 core view 收口。

---

## 2. 文件结构与职责边界

### Core / backend API（拥有事实）

- Create: `agentica/session_api.py`
  - 统一 session catalog / transcript page / search hit / usage view / run state view 的 Python API。
  - 输入是 `(owner, session_id, query params)`，输出是稳定 JSON-ready dict；不依赖 FastAPI、不依赖浏览器状态。
- Create: `agentica/memory/log_rows.py` 的扩展（已存在，只允许增强）
  - 继续承载 row 级语义：`RowView`、`row_payload()`、`classify_user()`。
- Create: `agentica/memory/usage_view.py`
  - `turn_usage_view(entry)`：把 `token_usage` event、`assistant.metrics`、`usage` / `turn_usage` live payload 规范成一份每轮事实。
- Create: `agentica/memory/tool_result_view.py`
  - `tool_error_of(entry)`：兼容旧日志的 `is_error` / `tool_call_error`，但所有读者只调它。
- Create: `agentica/run_state.py`
  - 定义 durable `RunState` record 和 `active_runs(owner, session_id)`；用于替代 `LiveTurn` 作为跨进程 active-run 事实源。
- Modify: `agentica/memory/session_log.py`
  - 调用上述 view 函数；保留 JSONL / sidecar 读写 owner 身份。
- Modify: `agentica/memory/trace.py`
  - `chat_turn_page()` / `search_conversation()` / `analyze_entries()` 返回稳定 `turn_id` / `entry_uuid`，不再只给数字序号。
- Modify: `agentica/tools/background_processes.py`
  - 保持 job record / delivery record 为权威；提供 `job_badge_view()`，避免 Web 再按 request timestamp 算 badge。

### Gateway（薄适配层）

- Modify: `gateway/src/agentica_gateway/routes/chat.py`
  - 路由只做 request parse / auth / `session_api` 调用 / response model。
  - `/api/sessions/pulse` 只返回 `ver` 和必要 hint；`active[]` 从 `run_state` 读。
- Modify: `gateway/src/agentica_gateway/routes/traces.py`
  - 继续只暴露 `session_api` / `trace` 的结构化 view，不组装业务状态。
- Modify: `gateway/src/agentica_gateway/services/agent_service/sessions.py`
  - 逐步删掉 catalog/title/fork/usage 推导，只保留 owner-aware 的 `SessionLog` 定位与命令转发。
- Modify: `gateway/src/agentica_gateway/services/live_turn.py`
  - 退化为进程内 SSE buffer / pub/sub；`active run` 状态写入 `agentica/run_state.py`。
  - replay 明确区分 `event` 与 `snapshot`。
- Modify: `gateway/src/agentica_gateway/services/im_bot/*`
  - 只把 IM 输入转成统一 `session_api` 命令，把返回 view 格式化成 IM 文案；不另建标题、turn、job 读模型。

### Web / Desktop（纯前端 / Web 特性）

- Modify: `web/src/api.ts`
  - 只定义 typed HTTP/SSE client，不解释业务字段。
- Modify: `web/src/store.ts`
  - 保留 UI state、optimistic state、browser cache；删除跨端事实的权威字段或标记成 `derived / pending`。
- Modify: `web/src/sessions.ts`
  - 删除 `sliceTurnPageClient()`、`syncSessionRoundStats()` 的位置对齐、catalog 业务 merge；改为 `applyCatalog(rows)` 和 `applyTranscriptPage(page)`。
- Modify: `web/src/lib/hydrateMessages.ts`
  - 继续把 `row` + structured view 渲染成气泡/卡片；不解析 `<context_window>`、steer 前缀、tool error 拼写、token 事件语义。
- Modify: `web/src/lib/sessionRows.ts`
  - 搜索命中改用服务端 `turn_id` / `entry_uuid`；本地搜索只搜索已加载 view，不重新计算全局 turn 编号。
- Modify: `web/src/components/BackgroundJobs.tsx`
  - 只展示 `job_badge_view` 返回的 `liveCount` / `settledCount` / `visibleCount`，不再自行按 `currentRequestStartMs()` 推断归属。
- Modify: `web/src/pages/ChatPage.tsx`
  - SSE 只追加 delta / 应用 snapshot；goal/queue/steer 只投影 `run_state` 或 `pending_input` API，不把 `goalRuns` / `pendingSteers` 当权威事实。

### Tests / docs

- Create: `tests/core/test_session_api.py`
- Create: `tests/memory/test_usage_view.py`
- Create: `tests/memory/test_tool_result_view.py`
- Create: `tests/gateway/test_thin_routes.py`
- Create: `tests/runner/test_run_state.py`
- Create: `web/src/lib/sessionApiView.test.mjs`
- Create: `web/src/lib/backgroundJobsView.test.mjs`
- Modify: existing `tests/gateway/test_gateway_unit.py`, `tests/gateway/test_git_actions.py`, `tests/memory/test_log_rows.py`, `tests/memory/test_trace_analysis.py`, `web/src/lib/hydrateMessages.test.mjs`, `web/src/lib/sessionRows.test.mjs`, `web/src/lib/backgroundJobs.test.mjs`。
- Modify: `CHANGELOG.md` at Unreleased。

---

## 3. Phase 划分

### Phase A — Core view API 与「Gateway 只转发」

目标：先把 Web/TS 正在做的 common 逻辑定义成 Python API，Gateway 变薄，前端先并行消费，旧字段保留一个发布周期。

验收：同一 session row / transcript page / search hit / usage footer 在 CLI、Web、Trace、IM 得到相同结构化值；Gateway route 里不再出现业务推导。

### Phase B — Run state / pulse / SSE 语义

目标：把 active run、goal、queue/steer、SSE replay 从进程内/浏览器本地状态收成持久或半持久 view。

验收：重启网关或换浏览器 tab 后，running / goal status / steer receipt 仍可恢复；SSE replay 不重复文本；pulse 只表示「有变化」。

### Phase C — Web 去业务化

目标：删除 TS/TSX 中重复的业务规则，Web 只做渲染与 Web-specific interaction。

验收：`rg` 检查 Web 不再出现 `<context_window>`、`STEER_GUIDANCE_PREFIX`、`is_error` / `tool_call_error` 兼容、`older_turns` 位置计算、`sliceTurnPageClient`、按数组位置对齐 round stats、按 request timestamp 计算 badge 等业务规则。

### Phase D — 清理与防回归

目标：加静态检查，防止 common 逻辑再次回流到 Web/Gateway。

验收：新增 lint/test 对白名单路径做断言；全量 pytest、前端测试、build 通过。

---

## 4. Task 列表

### Task 1: `session_api` 的 catalog view 契约先写失败测试

**Files:**
- Create: `tests/core/test_session_api.py`
- Create: `agentica/session_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_session_api.py
from agentica.memory.session_log import SessionLog


def test_catalog_view_is_owned_by_core(tmp_path):
    from agentica import session_api

    log = SessionLog("s1", base_dir=str(tmp_path))
    log.append("user", "解释缓存命中率")
    log.set_source("gateway")

    rows = session_api.catalog_view(
        owner="default",
        base_dirs=[str(tmp_path)],
        scope="all",
        recent_days=30,
        include_cli=True,
        hide_temp_dirs=False,
        now=1_700_000_000.0,
    )

    assert rows == [{
        "session_id": "s1",
        "title": "解释缓存命中率",
        "display_title": "解释缓存命中率",
        "source": "gateway",
        "origin": "gateway",
        "work_dir": str(tmp_path),
        "running": False,
        "background_jobs": 0,
        "unread": False,
        "archived": False,
        "pinned": False,
        "size_bytes": log.path.stat().st_size,
        "first_timestamp": SessionLog._session_entry(log.path, str(tmp_path))["first_timestamp"],
        "last_timestamp": SessionLog._session_entry(log.path, str(tmp_path))["last_timestamp"],
        "approval_mode": None,
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/core/test_session_api.py::test_catalog_view_is_owned_by_core -q
```

Expected: `ModuleNotFoundError: No module named 'agentica.session_api'`

- [ ] **Step 3: Write minimal implementation**

```python
# agentica/session_api.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentica.memory.session_log import SessionLog, is_session_unread


def catalog_view(
    *,
    owner: str,
    base_dirs: List[str],
    scope: str = "recent",
    recent_days: int = 30,
    include_cli: bool = False,
    hide_temp_dirs: bool = False,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build the shared session-list projection from SessionLog facts."""
    if scope not in ("recent", "all"):
        raise ValueError(f"Unknown session scope: {scope}")
    if recent_days < 1:
        raise ValueError("recent_days must be positive")
    _ = (owner, include_cli, hide_temp_dirs, now)
    rows: List[Dict[str, Any]] = []
    for base_dir in base_dirs:
        for entry in SessionLog.list_sessions(base_dir=base_dir):
            preview = SessionLog.session_preview(str(entry["path"]), max_chars=250)
            display_title = SessionLog.display_title(
                name=entry.get("name"),
                first_user=preview.get("first_user", ""),
                cron_job_name=preview.get("cron_job_name", ""),
            )
            rows.append({
                "session_id": entry["session_id"],
                "title": display_title,
                "display_title": display_title,
                "source": entry.get("source") or "legacy",
                "origin": entry.get("origin") or entry.get("source") or "legacy",
                "work_dir": entry.get("work_dir") or base_dir,
                "running": False,
                "background_jobs": 0,
                "unread": is_session_unread(entry.get("last_timestamp"), entry.get("last_read_at")),
                "archived": bool(entry.get("archived")),
                "pinned": bool(entry.get("pinned")),
                "size_bytes": int(entry.get("size_bytes") or 0),
                "first_timestamp": entry.get("first_timestamp"),
                "last_timestamp": entry.get("last_timestamp"),
                "approval_mode": None,
            })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/core/test_session_api.py::test_catalog_view_is_owned_by_core -q
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add agentica/session_api.py tests/core/test_session_api.py
git commit -m "core: add session catalog view API"
```

---

### Task 2: Gateway `/api/sessions` 改为薄转发

**Files:**
- Modify: `gateway/src/agentica_gateway/routes/chat.py:625-640`
- Modify: `gateway/src/agentica_gateway/services/agent_service/sessions.py:88-242`
- Test: `tests/gateway/test_thin_routes.py`
- Test: `tests/gateway/test_gateway_unit.py`

- [ ] **Step 1: Write the failing route test**

```python
# tests/gateway/test_thin_routes.py
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_sessions_route_delegates_to_session_api(client: TestClient):
    with patch("agentica.session_api.catalog_view", return_value=[{"session_id": "s1"}]) as view:
        resp = client.get("/api/sessions")

    assert resp.status_code == 200
    assert resp.json() == {"sessions": [{"session_id": "s1"}]}
    assert view.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/gateway/test_thin_routes.py::test_sessions_route_delegates_to_session_api -q
```

Expected: route still calls `svc.list_sessions`, `view.call_count == 0`.

- [ ] **Step 3: Write minimal implementation**

```python
# gateway/src/agentica_gateway/routes/chat.py (relevant import and route body)
from agentica import session_api

@router.get("/api/sessions")
async def list_sessions(
    request: Request,
    scope: str = Query("recent", pattern="^(recent|all)$"),
    recent_days: int = Query(30, ge=1, le=365),
):
    account = _account(request)
    return {
        "sessions": await asyncio.to_thread(
            session_api.catalog_view,
            owner=account,
            base_dirs=[p["base_dir"] for p in SessionLog.list_projects(user_id=account)],
            scope=scope,
            recent_days=recent_days,
            include_cli=show_cli_sessions_pref(account),
            hide_temp_dirs=True,
        )
    }
```

```python
# gateway/src/agentica_gateway/services/agent_service/sessions.py
# Keep AgentSessionsMixin.list_sessions as a compatibility shim for one release:
def list_sessions(self, owner=None, scope="recent", recent_days=30, include_cli=False, hide_temp_dirs=False):
    from agentica import session_api
    return session_api.catalog_view(
        owner=self._owner(owner),
        base_dirs=[p["base_dir"] for p in SessionLog.list_projects(user_id=self._owner(owner))],
        scope=scope,
        recent_days=recent_days,
        include_cli=include_cli,
        hide_temp_dirs=hide_temp_dirs,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/gateway/test_thin_routes.py tests/gateway/test_gateway_unit.py -q
```

Expected: all pass; any old test asserting the shim's internals is updated to assert the returned view.

- [ ] **Step 5: Commit**

```bash
git add gateway/src/agentica_gateway/routes/chat.py gateway/src/agentica_gateway/services/agent_service/sessions.py tests/gateway/test_thin_routes.py tests/gateway/test_gateway_unit.py
git commit -m "gateway: delegate session catalog to core API"
```

---

### Task 3: `transcript_page` 统一 turn view，删除 Web fallback

**Files:**
- Modify: `agentica/memory/trace.py:597-745`
- Modify: `gateway/src/agentica_gateway/routes/traces.py:18-43`
- Modify: `agentica/session_api.py`
- Modify: `web/src/sessions.ts:319-398`
- Test: `tests/memory/test_trace_analysis.py`
- Test: `web/src/lib/sessionApiView.test.mjs`

- [ ] **Step 1: Write failing Python test for stable turn identity**

```python
# tests/memory/test_trace_analysis.py
from agentica.memory.log_rows import row_payload
from agentica.memory.trace import with_row_views


def test_trace_page_uses_core_row_and_stable_turn_id():
    entry = {"type": "user", "uuid": "u1", "content": "第一轮问题", "timestamp": "2026-01-01T00:00:00.000Z"}
    page = with_row_views([entry])

    assert page[0]["row"] == row_payload(entry)
    assert page[0]["turn_id"] == "u1"
    assert page[0]["turn"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/memory/test_trace_analysis.py::test_trace_page_uses_core_row_and_stable_turn_id -q
```

Expected: `KeyError: 'turn_id'`.

- [ ] **Step 3: Write minimal implementation**

```python
# agentica/memory/trace.py
def with_row_views(entries):
    out = []
    turn = 0
    for entry in entries:
        row = dict(entry)
        payload = row_payload(entry)
        if payload is not None:
            row["row"] = payload
            if payload["automated"] is False and payload["steer"] is False and not (payload["window"] and not payload["text"].strip()):
                turn += 1
                row["turn"] = turn
                row["turn_id"] = str(entry.get("uuid") or f"line-{len(out)}")
        out.append(row)
    return out
```

```python
# agentica/session_api.py
def transcript_page(*, owner: str, session_id: str, log: SessionLog, turns: int, before=None, include_turn=None):
    from agentica.memory.trace import chat_turn_page, with_row_views
    page = chat_turn_page(log.iter_raw_entries(), turns, before, include_turn)
    page["events"] = with_row_views(page["events"])
    return {"session_id": session_id, "owner": owner, **page}
```

- [ ] **Step 4: Delete Web fallback in the same change**

```typescript
// web/src/sessions.ts: replace fetchTurnPageCompat body with:
async function fetchTurnPageCompat(
  id: string,
  turns: number,
  before?: number,
  includeTurn?: number,
): Promise<{ ok: boolean; data?: api.TraceEventsPage | null }> {
  return api.fetchTraceTurnPage(id, turns, before, includeTurn);
}
```

Delete `CHAT_ENTRY_TYPES`, `sliceTurnPageClient`, `fetchAllTraceEvents` and their now-unused imports from `web/src/sessions.ts`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/memory/test_trace_analysis.py tests/gateway/test_trace_and_spa.py -q
npm --prefix web test && npm --prefix web run build
```

Expected: Python tests pass; front-end tests/build pass with no references to `sliceTurnPageClient`.

- [ ] **Step 6: Commit**

```bash
git add agentica/memory/trace.py agentica/session_api.py gateway/src/agentica_gateway/routes/traces.py web/src/sessions.ts tests/memory/test_trace_analysis.py tests/gateway/test_trace_and_spa.py web/src/lib/sessionApiView.test.mjs
git commit -m "core: serve transcript turn views; web stops paging logs"
```

---

### Task 4: 统一 tool error view

**Files:**
- Create: `agentica/memory/tool_result_view.py`
- Modify: `agentica/memory/trace.py:1087`, `:1198`
- Modify: `agentica/memory/session_log.py:1415`, `:1653`
- Modify: `agentica/runner/persist.py:386`
- Test: `tests/memory/test_tool_result_view.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_tool_result_view.py
from agentica.memory.tool_result_view import tool_error_of


def test_tool_error_accepts_both_legacy_spellings_without_masking():
    assert tool_error_of({"is_error": False, "tool_call_error": True}) is True
    assert tool_error_of({"is_error": True}) is True
    assert tool_error_of({"tool_call_error": True}) is True
    assert tool_error_of({}) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/memory/test_tool_result_view.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# agentica/memory/tool_result_view.py
from typing import Any, Mapping


def tool_error_of(entry: Mapping[str, Any]) -> bool:
    """One reader for legacy and current tool-error field names."""
    return bool(entry.get("is_error") or entry.get("tool_call_error"))
```

Replace all four call sites:

```python
# agentica/memory/trace.py
row["isError"] = tool_error_of(entry)
tool_errors = sum(1 for r in tool_results if tool_error_of(r))
```

```python
# agentica/memory/session_log.py
msg["tool_call_error"] = tool_error_of(entry)
if tool_error_of(entry):
    stats["tool_errors"] += 1
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/memory/test_tool_result_view.py tests/memory/test_trace_analysis.py tests/memory/test_session_log.py -q
```

Expected: all pass; unchanged behavior for current logs, fixed behavior for legacy mixed spellings.

- [ ] **Step 5: Commit**

```bash
git add agentica/memory/tool_result_view.py agentica/memory/trace.py agentica/memory/session_log.py agentica/runner/persist.py tests/memory/test_tool_result_view.py
git commit -m "core: one reader for tool error fields"
```

---

### Task 5: 统一 per-turn usage view，Web 不再按数组位置对齐

**Files:**
- Create: `agentica/memory/usage_view.py`
- Modify: `agentica/memory/trace.py:331-392`, `:1069-1080`
- Modify: `gateway/src/agentica_gateway/routes/chat.py:432-490`
- Modify: `web/src/sessions.ts:588-633`
- Modify: `web/src/pages/ChatPage.tsx:2160-2178`
- Test: `tests/memory/test_usage_view.py`
- Test: `web/src/lib/sessionApiView.test.mjs`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory/test_usage_view.py
from agentica.memory.usage_view import turn_usage_view


def test_turn_usage_view_normalizes_event_and_live_metric_names():
    event = {"type": "event", "name": "token_usage", "request": {"input": 90, "cache_read": 10, "cache_write": 5, "output": 20}}
    live = {"input_tokens": 90, "cache_read_tokens": 10, "cache_write_tokens": 5, "output_tokens": 20}

    assert turn_usage_view(event) == {
        "prompt": 105,
        "cacheRead": 10,
        "cacheWrite": 5,
        "output": 20,
        "cacheHitPercent": 9.5,
    }
    assert turn_usage_view(live)["prompt"] == 105
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/memory/test_usage_view.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# agentica/memory/usage_view.py
from typing import Any, Mapping


def _num(value: Any) -> float:
    return float(value or 0)


def turn_usage_view(entry: Mapping[str, Any]) -> dict:
    req = entry.get("request") if isinstance(entry.get("request"), Mapping) else entry
    prompt = _num(req.get("input", req.get("input_tokens"))) + _num(req.get("cache_read", req.get("cache_read_tokens"))) + _num(req.get("cache_write", req.get("cache_write_tokens")))
    cache_read = _num(req.get("cache_read", req.get("cache_read_tokens")))
    output = _num(req.get("output", req.get("output_tokens")))
    return {
        "prompt": int(prompt),
        "cacheRead": int(cache_read),
        "cacheWrite": int(_num(req.get("cache_write", req.get("cache_write_tokens")))),
        "output": int(output),
        "cacheHitPercent": round(cache_read / prompt * 100, 1) if prompt else 0.0,
    }
```

- [ ] **Step 4: Attach `usage_view` to transcript pages**

```python
# agentica/memory/trace.py, inside with_row_views
from agentica.memory.usage_view import turn_usage_view

if entry.get("type") == "event" and entry.get("name") == "token_usage":
    row["usage_view"] = turn_usage_view(entry)
```

- [ ] **Step 5: Replace Web positional alignment**

```typescript
// web/src/sessions.ts: syncSessionRoundStats becomes a direct application of server view.
export function applySessionView(id: string, view: api.SessionView) {
  const sess = getState().sessions[id];
  if (!sess) return;
  sess.lastInputTokens = view.usage.prompt;
  sess.contextTokens = view.usage.context_tokens;
  for (const m of sess.msgs) {
    if (!m.uuid || !view.turns[m.uuid]) continue;
    const usage = view.turns[m.uuid];
    m.tokIn = usage.prompt;
    m.tokOut = usage.output;
    m.cacheRead = usage.cacheRead;
    m.cacheWrite = usage.cacheWrite;
    m.cacheHitPercent = usage.cacheHitPercent;
  }
  saveSessions();
  bump();
}
```

Delete `roundOffset` / `settledRounds` / `assistants` positional logic from `syncSessionRoundStats`.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/memory/test_usage_view.py tests/memory/test_trace_analysis.py tests/gateway/test_trace_and_spa.py -q
npm --prefix web test && npm --prefix web run build
```

Expected: all pass; Web no longer contains `roundOffset`.

- [ ] **Step 7: Commit**

```bash
git add agentica/memory/usage_view.py agentica/memory/trace.py gateway/src/agentica_gateway/routes/chat.py web/src/sessions.ts web/src/pages/ChatPage.tsx tests/memory/test_usage_view.py web/src/lib/sessionApiView.test.mjs
git commit -m "core: serve per-turn usage views"
```

---

### Task 6: `run_state` 持久化 active run，pulse 只做刷新信号

**Files:**
- Create: `agentica/run_state.py`
- Modify: `gateway/src/agentica_gateway/services/live_turn.py:350-368`, `:443-455`
- Modify: `gateway/src/agentica_gateway/routes/chat.py:615-622`
- Modify: `gateway/src/agentica_gateway/services/agent_service/sessions.py:133-136`, `:211-241`
- Test: `tests/runner/test_run_state.py`
- Test: `tests/gateway/test_thin_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_run_state.py
from agentica.run_state import RunStateStore


def test_active_runs_survive_a_new_process_object(tmp_path):
    store = RunStateStore(root=tmp_path, owner="default")
    store.start(run_id="r1", session_id="s1", kind="chat", pid=1234)
    reloaded = RunStateStore(root=tmp_path, owner="default")

    assert reloaded.active(session_id="s1") == {
        "run_id": "r1", "session_id": "s1", "kind": "chat", "status": "running"
    }

    reloaded.finish("r1", "completed")
    assert reloaded.active(session_id="s1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/runner/test_run_state.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# agentica/run_state.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class RunStateStore:
    """Durable active-run facts; pulse remains only a refresh signal."""

    def __init__(self, *, root: Path, owner: str):
        self.path = Path(root) / owner / "run_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, rows: Dict[str, Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def start(self, *, run_id: str, session_id: str, kind: str, pid: int) -> None:
        rows = self._load()
        rows[run_id] = {"run_id": run_id, "session_id": session_id, "kind": kind, "pid": pid, "status": "running"}
        self._write(rows)

    def active(self, *, session_id: str) -> Optional[Dict[str, Any]]:
        for row in self._load().values():
            if row.get("session_id") == session_id and row.get("status") == "running":
                return {k: row[k] for k in ("run_id", "session_id", "kind", "status")}
        return None

    def finish(self, run_id: str, status: str) -> None:
        rows = self._load()
        if run_id in rows:
            rows[run_id]["status"] = status
            self._write(rows)
```

- [ ] **Step 4: Make `LiveTurn.start/finish` write `RunStateStore` and make pulse neutral**

```python
# gateway/src/agentica_gateway/services/live_turn.py
import os
from pathlib import Path

from agentica.run_state import RunStateStore

def _run_state(owner: str) -> RunStateStore:
    home = Path(os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica")))
    return RunStateStore(root=home / "run_state", owner=owner)


def start(session_id: str, owner: str, kind: str = "chat", user_message: str = "") -> LiveTurn:
    gc()
    key = _skey(session_id, owner)
    old_rid = _by_session.get(key)
    if old_rid:
        old = _by_run.get(old_rid)
        if old is not None and old.done:
            _drop_run(old_rid)
    turn = LiveTurn(session_id, owner, kind, user_message=user_message)
    _by_run[turn.run_id] = turn
    _by_session[key] = turn.run_id
    _run_state(owner).start(
        run_id=turn.run_id,
        session_id=session_id,
        kind=kind,
        pid=os.getpid(),
    )
    bump_pulse(owner)
    return turn


def _finish_state(turn: LiveTurn, status: str) -> None:
    _run_state(turn.owner).finish(turn.run_id, status)
```

In `LiveTurn.finish()`, call `_finish_state(self, status)` immediately after `self.finished_at = time.monotonic()`.

```python
# gateway/src/agentica_gateway/routes/chat.py
@router.get("/api/sessions/pulse")
async def sessions_pulse(request: Request):
    account = _account(request)
    stamp = read_activity_stamp_us(account)
    return {"ver": stamp, "active": []}
```

`active[]` is removed from the pulse contract; `/api/sessions` and `/api/chat/runs/active` read `RunStateStore`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/runner/test_run_state.py tests/gateway/test_thin_routes.py tests/gateway/test_gateway_unit.py -q
```

Expected: all pass; pulse tests no longer assert in-memory `_pulse_ver`.

- [ ] **Step 6: Commit**

```bash
git add agentica/run_state.py gateway/src/agentica_gateway/services/live_turn.py gateway/src/agentica_gateway/routes/chat.py gateway/src/agentica_gateway/services/agent_service/sessions.py tests/runner/test_run_state.py tests/gateway/test_thin_routes.py tests/gateway/test_gateway_unit.py
git commit -m "core: persist active run state; pulse stays a signal"
```

---

### Task 7: SSE replay 区分 delta 和 snapshot

**Files:**
- Modify: `gateway/src/agentica_gateway/services/live_turn.py:265-304`
- Modify: `gateway/src/agentica_gateway/routes/chat.py:86-131`
- Modify: `web/src/pages/ChatPage.tsx:1982-2085`
- Test: `tests/gateway/test_live_turn.py`
- Test: `web/src/lib/sessionApiView.test.mjs`

- [ ] **Step 1: Write the failing test**

```python
# tests/gateway/test_live_turn.py
def test_replay_returns_immutable_delta_events():
    turn = live_turn.start("s1", "default")
    turn.publish({"event": "content", "data": "a"})
    turn.publish({"event": "content", "data": "b"})

    assert [e["data"] for e in turn.replay(after=1)] == ["b"]
    assert [e["data"] for e in turn.replay(after=0)] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/gateway/test_live_turn.py::test_replay_returns_immutable_delta_events -q
```

Expected: current coalesced replay returns `["ab"]`.

- [ ] **Step 3: Write minimal implementation**

```python
# gateway/src/agentica_gateway/services/live_turn.py
_COALESCE_EVENTS = set()

def publish(self, item: Dict[str, Any]) -> None:
    if self.status == "starting":
        self.status = "running"
    self.seq += 1
    wrapped = {"seq": self.seq, "event": item.get("event"), "data": item.get("data")}
    self.events.append(wrapped)
    self._buffer_chars += _event_chars(wrapped)
    self._trim_buffer()
    for q in list(self._subs):
        try:
            q.put_nowait(wrapped)
        except asyncio.QueueFull:
            self.unsubscribe(q)
            dropped = getattr(q, "_dropped", None)
            if dropped is not None:
                dropped.set()
```

If buffer pressure requires compaction, emit an explicit event:

```python
wrapped = {
    "seq": self.seq,
    "event": "snapshot",
    "data": {"event": item.get("event"), "text": accumulated},
}
```

Web `consumeSse()` applies `snapshot.data.text` by replacement, never by append.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/gateway/test_live_turn.py -q
npm --prefix web test && npm --prefix web run build
```

Expected: all pass; no duplicate text on reconnect.

- [ ] **Step 5: Commit**

```bash
git add gateway/src/agentica_gateway/services/live_turn.py gateway/src/agentica_gateway/routes/chat.py web/src/pages/ChatPage.tsx tests/gateway/test_live_turn.py web/src/lib/sessionApiView.test.mjs
git commit -m "gateway: replay immutable SSE deltas"
```

---

### Task 8: Web 去业务化静态防回归

**Files:**
- Create: `tests/web/test_web_boundaries.py`
- Modify: `web/src/lib/sessionApiView.test.mjs`
- Modify: `web/src/lib/backgroundJobsView.test.mjs`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing boundary test**

```python
# tests/web/test_web_boundaries.py
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "web" / "src"
FORBIDDEN = [
    "<context_window>",
    "STEER_GUIDANCE_PREFIX",
    "sliceTurnPageClient",
    "roundOffset",
    "currentRequestStartMs",
    "tool_call_error",
]


def test_web_does_not_own_common_backend_semantics():
    offenders = []
    for path in WEB.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        offenders.extend((path.name, word) for word in FORBIDDEN if word in text)
    assert offenders == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/web/test_web_boundaries.py -q
```

Expected: offenders include current `hydrateMessages.ts` / `sessions.ts` / `backgroundJobs.ts`.

- [ ] **Step 3: Remove remaining Web-only copies after Tasks 3–7**

Delete or move the remaining matched logic to core APIs and typed Web view adapters. This step is intentionally last: it is green only after the earlier tasks have landed.

- [ ] **Step 4: Run full verification**

Run:

```bash
pytest -q
npm --prefix web test
npm --prefix web run build
python scripts/check_bare_ci_imports.py
```

Expected: full backend tests pass; front-end tests pass; build succeeds; bare import check passes.

- [ ] **Step 5: Update CHANGELOG**

```markdown
- **Gateway 变薄，common 语义收回 core API**：session catalog、transcript turn view、usage view、tool error view、active run state 由 Python core API 提供；Web 只消费 view model 并保留纯前端/Web 特性；pulse 退化为刷新信号，SSE replay 不再混用 delta 与 snapshot。
```

- [ ] **Step 6: Commit**

```bash
git add tests/web/test_web_boundaries.py web/src CHANGELOG.md
git commit -m "web: keep only presentation and web-specific state"
```

---

## 5. API 契约草案

### `GET /api/sessions`

返回 `session_api.catalog_view()` 的结果；字段名保持当前前端兼容：

```json
{
  "sessions": [
    {
      "session_id": "s1",
      "name": "解释缓存命中率",
      "display_title": "解释缓存命中率",
      "running": false,
      "background_jobs": 0,
      "unread": false,
      "source": "gateway",
      "approval_mode": "auto"
    }
  ]
}
```

Gateway 只允许附加 web adapter 字段（例如 `web_last_opened_at`），不得改 `name` / `running` / `display_title` 语义。

### `GET /api/sessions/{id}/trace/events?turns=N`

返回 core `transcript_page()`：

```json
{
  "session_id": "s1",
  "events": [
    {
      "type": "user",
      "uuid": "u1",
      "content": "...",
      "row": {"automated": false, "steer": false, "text": "第一轮问题", "window": null},
      "turn": 1,
      "turn_id": "u1"
    }
  ],
  "older_turns": 0,
  "has_more": false,
  "next_before": 0
}
```

Web 不再计算 `CHAT_ENTRY_TYPES`、turn starts、shadowed question、preamble。

### `GET /api/sessions/pulse`

只返回刷新信号：

```json
{"ver": 1731830400123456}
```

`active` 从 `RunStateStore` 读，走 `/api/chat/runs/active` 或 catalog `running`。

### SSE

- `event: content` / `event: thinking` 永远是 delta。
- `event: snapshot` 是明确 replacement snapshot，带 `target` 和完整文本。
- Web 对 delta append，对 snapshot replace；禁止按 `seq` 猜语义。

---

## 6. 风险与回滚

- **效果风险**：Phase A 的 catalog view 改动会触碰 Web 侧栏所有字段，必须保留旧字段一个发布周期；先让 `session_api.catalog_view()` 与 `AgentSessionsMixin.list_sessions()` 双跑对比，差异以测试钉住。
- **耗时风险**：最大块是 Task 5 usage view 和 Task 6 run_state；两者都涉及多端兼容，建议在独立 worktree 分提交，每个 task 单独跑全量 `pytest` + `npm test` + `npm run build`。
- **架构风险**：不要把 `session_api.py` 写成第二个 framework；它只做读模型和命令编排，不持有进程状态，不做 HTTP，不启动模型。
- **缓存/前缀风险**：SSE 改 immutable delta 可能增加 buffer 数量；只在 buffer 压力超过阈值时发 `snapshot`，不要恢复隐式 coalescing。
- **回滚**：每个 task 独立 commit；回滚按 task 反序 `git revert`，不回滚 `log_rows` 和 `display_title` 既有收口。

---

## 7. Self-review

- Spec coverage: 用户提出的「thin gateway / 后台做更多 / Web 只纯前端和 Web 特性 / common 后端 API」由 §0 边界、§2 文件结构、Task 1–8 覆盖。
- Placeholder scan: 未留 TBD / TODO；每个代码步骤给出具体路径和片段；命令给出预期结果。
- Type consistency: 全篇统一使用 `catalog_view`、`transcript_page`、`turn_usage_view`、`tool_error_of`、`RunStateStore`、`row_payload`、`display_title`；Web 函数名统一为 `applySessionView`。
- Non-goals: 不新增 store、不大爆炸重写、不搬纯 UI 状态，已在 §0 和 §6 写明。
