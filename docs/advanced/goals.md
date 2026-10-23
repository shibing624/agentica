# 长任务自主循环 (`/goal`)

当你有一项需要多轮探索、修改、跑测试、修报错才能完成的较长任务时，可以用 **`/goal`**（CLI / Web）或 **`agent.run_goal()`**（Python SDK）。

开启后，Agent 每跑完一轮都会自动核对目标是否真正完成：没完成就继续下一轮，直到任务达标、达到你设定的预算上限，或者被你手动暂停。

---

## 一、在 CLI 终端里使用

进入 `agentica` 交互终端后：

```text
/goal 为 auth 模块补齐单元测试并跑通 pytest         # 启动长任务循环（默认不限额度）
/goal --tokens 200000 --wall 1800 重构日志模块     # 限制最多用 20 万 token、跑 30 分钟
/goal status                                     # 查看当前进度、已用 token 与验收条件
/goal pause                                      # 暂停自动续跑
/goal resume                                     # 恢复续跑（超预算停下后也可用来继续）
/goal clear                                      # 清除当前目标
```

### 给目标加验收条件（`/subgoal`）

你可以随时追加具体的验收标准，Agent 每一轮结束时都会逐条核对是否有实际证据（如测试输出或文件内容）：

```text
/subgoal 必须包含边界条件与异常输入的测试
/subgoal 所有原有测试不能报错
/subgoal list                                    # 查看当前验收条件
/subgoal remove 1                                # 按编号删除某条条件
/subgoal clear                                   # 清空验收条件
```

### 运行中途怎么插话纠偏？

长任务跑起来之后，你不需要干等：

| 你想做的事 | 怎么操作 | 效果 |
|---|---|---|
| **马上纠正当前方向** | 直接打字回车，或 `/steer 别改公开接口签名` | 在当前这一轮的下一次模型推理前立即注入你的指令 |
| **加一条全程都要满足的验收标准** | `/subgoal <条件>` | 写入目标状态，后续每一轮续跑和裁判都会检查 |
| **插一件别的事让它先做** | `/queue 先帮我看眼报错日志` | 当前这轮结束后先处理你的新消息，再继续跑 goal |
| **随时喊停** | 按 `Ctrl+C` 或敲 `/goal pause` | 立即暂停自动续跑；想接着跑时敲 `/goal resume` |

> **安全保护**：如果重启了终端并用 `/resume` 恢复一个之前还在跑 `/goal` 的会话，它会自动处于 `paused` 暂停状态，不会在你刚打开旧会话时自己偷偷往外跑，等你敲 `/goal resume` 才会继续。

---

## 二、在 Web 与桌面端里使用

在输入框输入 `/goal`（或从斜杠菜单选择）：
1. 默认显示「预算不限」；点击预算标签可填入 Token 上限（支持 `200k`、`1m` 等写法）。
2. 输入你的目标并发送，页面会实时流式展示每一轮的思考、工具调用和进度，状态栏会显示已用 Token。

---

## 三、在 Python SDK 里使用 (`agent.run_goal`)

在代码中只需一行 `await agent.run_goal(...)`：

```python
from agentica import Agent, DeepSeekChat

agent = Agent(
    session_id="refactor-task",
    model=DeepSeekChat(id="deepseek-v4-pro"),
    # 推荐配一个便宜的 auxiliary_model 用于每轮结束时的完成度判断，节省成本
    auxiliary_model=DeepSeekChat(
        id="deepseek-v4-flash",
        max_completion_tokens=4096,
    ),
)

result = await agent.run_goal(
    "在 examples/ 下新增一个 mcp_client 示例并验证可运行",
    token_budget=200_000,         # 可选：总输入+输出 token 上限（默认 None 不限）
    wall_clock_budget_sec=1800,   # 可选：总耗时秒数上限（默认 None 不限）
    turn_budget=None,             # 可选：最大轮数上限（默认 None 不限）
)

print(result.status)              # "complete" | "paused" | "budget_limited"
print(result.reason)              # 完成或停止的原因说明
print(result.turns_used)          # 实际执行的轮数
print(result.response_content)    # 最后一轮的总结回复
```

### 预算怎么设最合适？

三个预算参数互相独立，任一达到上限就会触发收尾（让 Agent 用最后一轮把已完成的文件、剩余步骤和阻塞点交接清楚），然后停在 `budget_limited` 状态：

| 常见场景 | `token_budget`（或 `--tokens`） | `wall_clock_budget_sec`（或 `--wall`） |
|---|---|---|
| 日常修 Bug / 补单测 | `100_000` ~ `200_000` | `600`（10 分钟） |
| 完整功能开发 + 验证 | 不传（默认不限）或 `500_000` | `1800`（30 分钟） |
| 大型重构 / 迁移 | `1_000_000`+ | `3600`（1 小时） |

> **注意**：如果你把推理模型（如 `deepseek-reasoner`、`o3` 等带长思维链的模型）设为 `auxiliary_model`，请务必给它加上 `max_completion_tokens=4096`，避免思维链把默认输出额度占满导致裁判 JSON 被截断。

### 监听进度事件（可选）

```python
from agentica.run.events import RunEventType

def on_goal_event(event_type: RunEventType, payload: dict) -> None:
    print(event_type.value, payload)

await agent.run_goal("完成数据清洗脚本", event_callback=on_goal_event)
```

可收到 `goal.set`、`goal.continuing`、`goal.completed`、`goal.paused` 四类事件，包含当前轮数、已用 token 与耗时。

---

## 四、常见停止状态与恢复办法

| 状态 (`status` / `paused_reason`) | 含义 | 怎么处理 |
|---|---|---|
| `complete` | 目标已完成并通过核验 | 无需处理 |
| `budget_limited` | 达到了你设定的 Token、轮数或时间上限（已自动生成交接总结） | 提高预算后执行 `/goal resume`（或 `mgr.resume()`）继续 |
| `paused` (`user` / `user-interrupted`) | 你敲了 `/goal pause` 或按了 `Ctrl+C` | 敲 `/goal resume` 继续 |
| `paused` (`agent-tool`) | Agent 发现缺少权限、凭据或需要你拍板，主动暂停请求输入 | 回答它的问题后敲 `/goal resume` |
| `paused` (`tool-stuck`) | 连续 3 轮所有工具调用都报错，自动暂停防止空烧额度 | 看一眼报错原因，提示它换个思路后 `/goal resume` |

---

## 下一步

- [CLI 终端指南](../getting-started/terminal.md) — 更多交互终端命令
- [运行预算与控制 (RunConfig)](run-config.md) — 单次 `agent.run()` 的超时与费用控制
