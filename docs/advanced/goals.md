# Standing Goal Loop (`/goal`)

让 Agent **持续向一个用户目标推进**，每轮结束自动判断是否完成，没完成就续跑——直到 judge 判 done、预算耗尽、或用户主动停下。CLI 用户用 `/goal xxx`，SDK 用户一行 `await agent.run_goal(...)`。

> 本特性已包含 P0 基础环 + P1 S/A 档（Runner 锚点、token/wall-clock 预算、`update_goal` 受限工具、`goal.*` 事件）。设计文档：`docs/learn_cc/goal.md`（内部）。

---

## SDK 用法（推荐）

### 一行起飞

```python
from agentica import Agent, DeepSeekChat

agent = Agent(
    session_id="my-task",
    model=DeepSeekChat(),
    auxiliary_model=DeepSeekChat(),  # judge 走这个，省钱
)

result = await agent.run_goal(
    "compute 17+9+16 and state the integer answer",
    turn_budget=5,
    token_budget=2000,
    wall_clock_budget_sec=60,
)

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

| 参数 | 默认 | 用途 |
|---|---|---|
| `turn_budget` | `100` | **安全网**——防 runaway 循环。不是主预算 |
| `token_budget` | `None` | 真正的 cost cap：输入+输出 token 累计 |
| `wall_clock_budget_sec` | `None` | 真正的 wall-clock cap |

**优先级**：`budget cap > tool short-circuit > judge`。即使模型自己用 `update_goal` 标了 `complete`，只要 token 已经超 `token_budget`，最终 status 都是 `budget_limited`。

预算耗尽时 status 是独立的 `budget_limited`（不是 `paused`），语义"用户必须决定加额度或接受部分结果"。用 `mgr.resume()` 或 `/goal resume` 可从 `budget_limited` / `paused` 两种状态恢复。

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
    decision = await mgr.evaluate_after_turn(
        resp.content or "", token_delta=delta,
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
| `turn_budget` 超 | `budget_limited` | `budget` | `/goal resume` |
| `token_budget` / `wall_clock_budget_sec` 超 | `budget_limited` | `budget` | `/goal resume` |

---

## 完整可运行示例

- `examples/cli/03_goal_loop_demo.py` — 4 个 SDK Example：one-liner、budgets、event_callback、手动循环（依赖 `DEEPSEEK_API_KEY`）
