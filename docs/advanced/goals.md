# Standing Goal Loop (`/goal`)

让 Agent **持续向一个用户目标推进**，每轮结束自动判断是否完成，没完成就续跑——直到 judge 判 done、预算耗尽、或用户主动停下。CLI 用户用 `/goal xxx`，SDK 用户一行 `await agent.run_goal(...)`。

> **主成本闸是 `token_budget` / `wall_clock_budget_sec`；`turn_budget`（默认 100，不可关）只是防 runaway 的安全网，不是成本预算。** 便宜工具在 100 轮内就能烧掉大量 token，别指望 `turn_budget` 控成本——生产 / 长任务务必显式设 `token_budget`。

> 本特性已包含 P0 基础环 + P1 S/A 档（Runner 锚点、token/wall-clock 预算、`update_goal` 受限工具、`goal.*` 事件）。设计文档：`docs/learn_cc/goal.md`（内部）。

---

## SDK 用法（推荐）

### 一行起飞

```python
from agentica import Agent, DeepSeekChat

agent = Agent(
    session_id="my-task",
    # 主模型 + 便宜的 auxiliary 模型：judge / 压缩 / 记忆抽取等次要工作都走 auxiliary。
    # 不区分也能跑，但 judge 每轮都调一次，分开后能省 5-10x 成本。
    model=DeepSeekChat(id="deepseek-v4-pro"),
    auxiliary_model=DeepSeekChat(
        id="deepseek-v4-flash",
        max_completion_tokens=4096,   # judge JSON 输出预算
    ),
)

# 最简：不传任何 budget，只靠默认 100 turns 安全网兜底
result = await agent.run_goal("compute 17+9+16 and state the integer answer")

print(result.status)            # "complete" / "paused" / "budget_limited"
print(result.reason)
print(result.response_content)  # == result.run_response.content or ""
```

`agent.run_goal()` 内部会：

1. 懒建 `SessionLog` + `GoalManager`（绑到 `agent._session_log`，复用 CLI 同一套持久化路径）
2. `mgr.set(objective, ...)` 写持久化状态
3. 把 `TaskAnchor` 钉到 objective（SDK 路径 + Runner S1 自动生效）
4. 默认挂 `GoalTool`，让模型可以 `update_goal(status="complete"|"paused")` 短路 judge
5. 循环 `agent.run() → mgr.evaluate_after_turn(token_delta=..., elapsed_sec=...)`
6. 终止时返回 `GoalRunResult(status, reason, run_response, goal, turns_used)`

### `GoalRunResult`

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | `str` | `"complete"` / `"paused"` / `"budget_limited"` |
| `reason` | `str` | 人类可读的原因（judge verdict / 预算消息 / 工具理由） |
| `run_response` | `Optional[RunResponse]` | 最后一次 `agent.run()` 的完整响应，含 `content`、`cost_tracker`、`messages`、`tool_calls` |
| `goal` | `GoalState` | 终态快照：`objective` / `turns_used` / `tokens_used` / `wall_clock_used_sec` / `subgoals` / `last_verdict` |
| `turns_used` | `int` | 实际跑了多少轮 |
| `response_content` (property) | `str` | `run_response.content or ""` 的便捷访问 |

### 预算（hard caps）

三个预算**互相独立、任一触发即停**（取最严的那个先停）。`None` = 不限。

| 参数 | 默认 | 不传时 | 含义 |
|---|---|---|---|
| `turn_budget` | `DEFAULT_TURN_BUDGET = 100` | **fallback 到 100**（防 runaway 的安全网） | LLM 循环总轮数上限 |
| `token_budget` | `None` | **不限**（不计 token） | 累计输入+输出 token 上限 |
| `wall_clock_budget_sec` | `None` | **不限**（不计时） | agent wall-clock 秒数上限 |

> 注意：`turn_budget` 即便传 `None` 也会回落到 `DEFAULT_TURN_BUDGET = 100`，因为它的角色是"防 runaway 的最后一道闸"——不能真正去掉。想要更大的"实际无限"就传一个大数，例如 `turn_budget=10_000`。

**判定优先级**（在 `evaluate_after_turn` 里固定为）：

```
turn accounting → budget check → tool short-circuit → judge
```

即：即便模型自己通过 `update_goal` 标了 `complete`，只要 token / wall-clock 已超 cap，最终 status 仍是 `budget_limited`。budget 是 hard cap，模型短路改不了。

预算耗尽时 status 是独立的 `budget_limited`（不是 `paused`），语义"用户必须决定加额度或接受部分结果"。用 `mgr.resume()` 或 `/goal resume` 可从 `budget_limited` / `paused` 两种状态恢复。

### Best Practices：怎么设预算

| 场景 | `turn_budget` | `token_budget` | `wall_clock_budget_sec` |
|---|---|---|---|
| 试玩 / 调试 | `5` | 不传 | 不传 |
| 一次性短任务（算个数、写一句话） | 不传 (默认 100) | 不传 | 不传 |
| 修个 bug | 不传 | `50_000` | `600` (10 分钟) |
| 实现完整功能 + 测试 | 不传 | `200_000` | `1800` (30 分钟) |
| 长 refactor / migration | 不传 | `500_000` | `3600` (1 小时) |
| 完全放飞（仅安全网兜底） | `10_000` | 不传 | 不传 |
| 严控成本，定额执行 | 三个都传 | 按预算算 | 按 SLA 算 |

经验法则：

- **小任务不传 token/wall-clock**——多花一行参数没必要，turn_budget 的 100 已经兜得很松
- **生产 / 长任务一定传 `token_budget`**——一旦模型陷入死循环（比如反复 read 同一个大文件），按 turn 数算可能要烧很久才触发；按 token 算几秒钟就阻断
- **`wall_clock_budget_sec` 主要给 SLA 用**——例如"30 分钟内出个结果"，不在乎期间用了多少 token
- **`token_budget` 怎么估**：粗略按 `≈ avg_turn_tokens × 期望最大 turns`。DeepSeek 一个工具调用 turn 大约 1k–5k token，编码任务 30 turns 估 `100_000` 比较稳

示例：

```python
# 真实编码任务的常见组合
result = await agent.run_goal(
    "在 examples/ 下加一个 mcp_client 示例并跑通",
    token_budget=200_000,
    wall_clock_budget_sec=1800,
)
```

### Judge 鲁棒性（自动启用，无需配置）

每轮 judge 调用会自动：

1. **看 tool call 名字**——`agent.run_goal()` 会把本轮 `RunResponse.tool_calls` 的 `(tool_name, is_error)` 列表传给 judge，让它分清"啥也没干就嘴硬"和"跑了 5 个工具实际产出"。**零额外 LLM 调用**，只是名字。
2. **强制 evidence rule for subgoals**——当存在 subgoals 时，judge prompt 自动加一句"为每条验收条件找具体证据（文件片段 / 命令输出 / 结果值），不接受 'all requirements met' 这种泛泛之言"。来自 hermes 的实战经验。
3. **JSON 解析容错**——支持 `{"done": "yes"}`、`{"done": "TRUE"}`、`{"done": 1}` 这类弱 judge 模型的偏差输出，以及 markdown fence 包裹。
4. **Tool-stuck 自动 pause**——连续 N（默认 3）轮"所有 tool call 都失败"时，自动 pause 状态为 `tool-stuck`，避免烧穿整个 turn_budget。任意一次 tool 成功就重置计数；"光思考不调 tool"的轮**不重置**计数。

### Reasoning judge 的特别注意

如果你想用 reasoning 模型当 judge（DeepSeek-Reasoner、o-series、qwq、GLM-4 Reasoning 等），hidden CoT 会先烧掉一大块 output token，**必须在构造时显式给足 `max_completion_tokens`**（推荐 `≥ 4096`），否则 JSON 还没输出完就被截断，会被 GoalManager 当成 parse failure，连续 3 次后 auto-pause 为 `judge-broken`：

```python
agent = Agent(
    model=DeepSeekChat(id="deepseek-v4-pro"),
    auxiliary_model=DeepSeekChat(
        id="deepseek-reasoner",
        max_completion_tokens=4096,   # judge 必备
    ),
)
```

非 reasoning 的小 chat 模型（`deepseek-v4-flash`、`gpt-4o-mini` 等）默认输出预算就够用，不用设。

### Power-user 路径

需要更细粒度控制（例如自己写循环、自定义 logging、与 streaming 配合）时：

```python
mgr = agent.get_goal_manager(default_turn_budget=5)
agent.enable_goal_tool()
mgr.set("xxx", token_budget=1000)

while True:
    resp = await agent.run(mgr.next_continuation_prompt())
    ct = resp.cost_tracker
    delta = (ct.total_input_tokens + ct.total_output_tokens) if ct else 0
    # Recommended: pass tool-call names so the judge sees what work
    # actually happened + the tool-stuck counter advances correctly.
    tool_pairs = [(t.tool_name, bool(t.is_error)) for t in resp.tool_calls if t.tool_name]
    decision = await mgr.evaluate_after_turn(
        resp.content or "", token_delta=delta, tool_calls=tool_pairs or None,
    )
    if not decision.should_continue:
        break
```

`agent.get_goal_manager()` 返回的 `GoalManager` 是持久化到 `SessionLog` 的，下次进程启动 / `/resume` 后能拾回状态。

---

## CLI 用法

进入交互式 CLI 后：

```text
/goal 实现 xxx 功能并跑通 pytest    # 设置目标，自动塞入首轮
/goal status                       # 显示当前状态、预算条、subgoals
/goal pause                         # 暂停自动续跑
/goal resume                        # 恢复续跑（含 budget_limited 也可恢复）
/goal clear                         # 清空目标

/subgoal 必须补单测                  # 给目标加验收条件，会渲染到 continuation prompt
/subgoal list                       # 列出 subgoals
/subgoal clear                      # 清空 subgoals
/subgoal remove 2                   # 按编号删除
```

CLI 行为：

- 用户真实输入永远抢占 goal 循环（在队列里发现非续跑消息就让位）
- Ctrl+C 取消 agent 时自动 `pause(reason="user-interrupted")`，不会偷偷重启
- `/resume <session_id>` 重进旧 session 时，goal 强制改为 `paused(reason="resume-safety")`，等用户显式 `/goal resume`
- judge 连续 3 次 JSON 解析失败 → `paused(reason="judge-broken")`，避免 silent 死循环

---

## 事件（用于 tracing / 观测）

```python
from agentica.run_events import RunEventType

def on_goal(event_type: RunEventType, payload: dict) -> None:
    print(event_type.value, payload)

await agent.run_goal("xxx", event_callback=on_goal)
```

会触发的事件：`goal.set`、`goal.continuing`、`goal.completed`、`goal.paused`。Payload 包含 `session_id` / `objective` / `status` / `turns_used` / `turn_budget` / `tokens_used` / `token_budget` / `wall_clock_used_sec` / `wall_clock_budget_sec` 等。

---

## `update_goal` 模型工具

`agent.run_goal()` 默认会挂 `GoalTool`，模型可以通过它**自己标记 complete / paused**：

| 参数 | 取值 | 说明 |
|---|---|---|
| `status` | `"complete"` | 任务真正完成；evaluator 会跳过 judge，状态置为 `complete` |
| `status` | `"paused"` | 被卡住需要用户输入（缺凭据、需澄清等），停止自动续跑 |
| `reason` | `str` | 一句话理由，会显示给用户 |

**安全约束**：工具只能改 `status` 和 `reason`，**不能**修改 objective、不能 clear、不能调预算。这些仍归用户控制。

如果不想让模型有这个能力（让 external judge 唯一裁判）：

```python
result = await agent.run_goal("xxx", attach_goal_tool=False)
```

---

## 持久化与 `/resume`

`GoalState` 以 `type="goal"` 条目写在 `SessionLog`（JSONL）末尾，**不会**污染模型对话历史（`SessionLog._build_messages()` 只回放 `user/assistant/system/tool` 四类）。

- 重启进程、`/resume <session_id>` 后，`Agent.get_goal_manager()` 会自动 `load()` 最近一条 goal 条目
- 出于安全考虑，resume 时 active 状态会被强制改为 `paused(reason="resume-safety")`——避免"打开旧会话偷偷自动跑"
- `Runner._run_impl()` 在 anchor 初始化时也会 `load_goal()`，若有 active goal 则把 `TaskAnchor` 绑到 objective 上，所有 SDK 路径（gateway / ACP / 脚本）都受益

---

## 失败 / 暂停时的行为对照

| 触发条件 | 终态 | `paused_reason` | 恢复方式 |
|---|---|---|---|
| Judge 判 done | `complete` | — | — |
| 模型调 `update_goal(complete)` | `complete` | — | — |
| 用户 Ctrl+C | `paused` | `user-interrupted` | `/goal resume` |
| 用户 `/goal pause` | `paused` | `user` | `/goal resume` |
| 模型调 `update_goal(paused)` | `paused` | `agent-tool` | `/goal resume` |
| `/resume <sid>` 拾回 active goal | `paused` | `resume-safety` | `/goal resume` |
| Judge 连续 3 次 JSON 解析失败 | `paused` | `judge-broken` | `/goal resume`（先排查 judge） |
| 连续 3 轮所有 tool call 都失败 | `paused` | `tool-stuck` | 看最后几轮 tool 报错 → 修 → `/goal resume` |
| `turn_budget` 超 | `budget_limited` | `budget` | `/goal resume` |
| `token_budget` / `wall_clock_budget_sec` 超 | `budget_limited` | `budget` | `/goal resume` |

---

## 完整可运行示例

- `examples/cli/03_goal_loop_demo.py` — 4 个 SDK Example：one-liner、budgets、event_callback、手动循环（依赖 `DEEPSEEK_API_KEY`）
