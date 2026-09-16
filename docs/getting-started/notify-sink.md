# 外部通知汇（notify sink）

把 agentica 一轮跑到什么节点告诉本机的一个 app（目前是 [VPet](https://github.com/shibingtan/VPet) 桌宠）。
**只看，不问** —— 这条通道不接收回答。要让用户能在桌宠上按 y/n、输文本，
那是 [shell hooks](shell-hooks.md) 那条通道的事。

**默认关闭。** 见下面「开关」。

### 三件事，别混

这三条是这个通道的全部语义，写文档、写代码、写测试时都要分清：

1. **这个通道是单向的。** 它只报告 run 的状态，不问任何事，也不读回任何 body。
   回答走 shell hooks（`docs/getting-started/shell-hooks.md`）。
2. **桌宠不替用户做决定。** 它没有策略、没有自动批准、没有 YOLO；这条通道上
   没有任何路径能 auto-allow。桌宠只是把「agent 现在什么状态」显示出来。
3. **不设等候上限。** 这条通道上没有任何一处等人，所以也没有可以写死的时限。

推论：**「用户还没答」不是事件** —— 这条通道根本不承载「在等用户」这个状态。
（它曾经承载过：`/await` 会阻塞等用户回答。那条路径已删除，见下。）

## 它是什么

一条只连本机的汇，HTTP over Unix domain socket：

| 路径 | 语义 | 超时 | 响应 |
|---|---|---:|---|
| `POST /event` | 通知，不等回话 | 2s | 不解析 |

> 曾经还有 `POST /await`（等用户回答的阻塞路径）。**删的是 agentica 这一侧的
> 调用**（`notify/questions.py`、`notify/approvals.py`）：回答现在由用户自己
> 配置的 hook 命令承载（[shell-hooks.md](shell-hooks.md)）。删除的理由是
> 「一条回答通道，两套实现」—— 桌宠要为 agentica 单独实现一遍协议，而
> hook 命令是每个 CLI 都有的形态，桌宠不必为 agentica 写任何东西。
>
> **桌宠的 `POST /await` 端点还活着。** Claude Code 的 `PermissionRequest`
> （`blocking: true`）仍走它，挂起队列 / held-connection 也是那条活路。
> 不要因为 agentica 不再调用就去删端点或超时逻辑。

装上的都是**并挂**，不是替换：`AgentHooks` / `RunHooks` 的语义不变，hook 那条通道
也各自独立（一个挂了不影响另一个）。

## 开关

```yaml
# ~/.agentica/config.yaml
settings:
  notify:
    enabled: false                 # 要不要让桌宠知道
    socket: "~/Library/Application Support/VPet/notify.sock"
    token: ""                      # 留空则读 ~/Library/Application Support/VPet/notify.token
    events:                        # 逐事件开关，默认全开
      run.started: true
      run.completed: true
      run.failed: true
      run.cancelled: true
      tool.started: true
      tool.completed: true
```

环境变量覆盖同名项，前缀 `AGENTICA_NOTIFY_`（如 `AGENTICA_NOTIFY_ENABLED=true`、`AGENTICA_NOTIFY_SOCKET`）。

`enabled` 只管「要不要接这个通道」。关掉时**什么都不装** —— 不建队列、不起线程、
不注册回调。

> 曾经还有 `approve_from_desktop` 和 `timeout_seconds`。两个都已删除：前者是
> 「桌宠被允许决定吗」这个不该存在的概念（桌宠只是输入面），后者是这层不该有的
> 等候定数（等多久由终端语义决定）。旧配置里留着这两项**不会报错**，只是不再生效。

## 事件

| event | 触发点 | 阻塞 |
|---|---|---|
| `run.started` | 一轮开始 |
| `run.completed` | 一轮成功结束**且没有后续** |
| `run.failed` | 一轮抛错 |
| `run.cancelled` | 用户 Ctrl+C |
| `tool.started` | 工具开始执行 |
| `tool.completed` | 工具结束，含成功状态与耗时 |

tool 事件按真实执行边界发送：串行调用轮到自己时才 started，并行调用按实际完成顺序
completed；运行中取消仍会发 `ok: false` 的 completed。

`needs.approval` / `needs.input` 曾经也在这条通道上（`/await` 的请求体），
现在只在 hook 那条通道上（[shell-hooks.md](shell-hooks.md)）。

### payload 字段

run 类事件的 `payload` 只带元数据 + **用户已在自己屏幕上见过的正文**，不带工具输出、
不带文件内容（这个通道即使在本机 socket 上也保持窄）：

| 字段 | 出现于 | 含义 |
|---|---|---|
| `agent_name` | 全部 run 类 | 哪个 agent 跑的 |
| `duration_seconds` | `run.completed` / `run.failed` | 时长 |
| `had_response` | `run.completed` | 这一场是否产出过回复 |
| `prompt` | `run.started` | 这一轮的锚文本：普通轮是用户提问，goal 会话里是 **goal 目标** |
| `answer` | `run.completed` | assistant 这一轮的回复，截断见下 |
| `answered_at` | `run.completed` | 回复**产生**的时刻（unix 秒），**不是**信封 `ts` |
| `reason` | `run.cancelled` | 为什么被取消 |
| `error` | `run.failed` | 错误文本（`类型: 信息`） |
| `tool_name` / `tool_call_id` | `tool.*` | 当前工具及其稳定调用 id |
| `preview` | `tool.*` | 严格脱敏、裁剪后的命令、路径、查询或 URL；不含写入正文 |
| `ok` / `duration_seconds` / `error` | `tool.completed` | 成功状态、耗时及失败时严格脱敏、裁剪后的错误 |

`prompt` / `answer` 一律截断到 **500 字 + `…`**：桌宠是气泡不是阅读器，一条 40k 字的
回复既撑爆每条事件也不会更好读。**截断标记是可见的**，消费端能区分「就说了这么多」和
「说了 500 字还有下文」。要看全文，终端才是那个地方。

`answered_at` 与信封 `ts` **可能不同**，别混：信封是事件**发出**时刻，被 goal 扣住的
完成事件是在释放时才盖的章（可能晚很多）；`answered_at` 才是回复真正产生的时刻。要显示
「它什么时候答的」用后者；要显示「你什么时候收到通知」用 `ts`。

`payload` 是**白名单过滤**产物：loop 层可以带更多键，sink 只放行上表这几个。
例如 `run.failed` 的 loop 事件里有 `exception_type`，但**不过线**——消费端要判断
错误种类请解析 `error` 的前缀（`"ValueError: ..."`）。缺失即「没有这个信息」，
**不要推断**（没有 `duration_seconds` 不代表 0）。

回答的编码（四个 decision 词、`answer` 自由文本、认不出来的输入一律算「没答」）
现在定义在 hook 那条通道上，见 [shell-hooks.md](shell-hooks.md) 的「回话」一节。

**多个字段在补发的 `run.completed` 上会合并**：一场 goal 只补发一次，若中间发生过
多轮，`duration_seconds` 是各轮**累加**（用户关心的是「我离开这段时间它跑了多久」，
不是最后一轮），`had_response` 只要有一轮产出过就为真，`agent_name` 取首个非空值；
而 `answer` 与 `answered_at` 取**最后一轮**（你要看的是它最后说了什么，不是第一句）。

### `run.completed` 的准确含义

它是「**你可以回来看了**」，不是「一次 run 结束了」。两者只在「后面没有别的活」时
才重合：

- **goal 循环**：一个 N 轮 goal 会跑 N 次 run，每次 run 结束都自然 emit
  `run.completed`，而 CLI 的 goal hook 随后又排下一轮。若照发，桌宠会误报 N 次
  「跑完了」——两次 run 之间夹着 judge LLM 调用，间隔不可控，消费端**无法**靠防抖
  分辨。
- **排队输入**：一次粘贴多条消息时，每条各自成为一轮，同样会连发。

所以 goal 活跃时、或宿主还有排队输入时，sink **扣住** `run.completed`；等到 goal
真正结束（`decision.status` 不再是 `active`）的那一刻再补发一次，所以整场恰好一次。
`run.started` / `run.failed` / `run.cancelled` 一律照发——只有「完成」需要延后，
否则显示会一直卡在「干活中」。

**扣住和补发必须成对**，这是踩过两次的坑，方向相反：

- 只扣不发 = 桌宠**永远**停在 working。goal 停止发生在最后一轮**之后**，而停止了的
  goal 不会再排下一轮，所以没有任何 `run.completed` 会来「顺便」把它带出去。
- 每轮都发 = 一个 N 轮 goal 报 N 次完成。

补发的时机在 CLI 的 goal hook（它才知道还有没有下一轮），判断依据是
`decision.status != "active"`；**不能**用「队列里还有没有排队的 continuation」代替——
排队的 continuation 仍要经过它自己的评估才决定是否真的再来一轮。

goal 状态**从 session log 读**，不读 `agent.goal_manager`：CLI 自己持有
`state.goal_manager`，而 agent 上那份懒加载一次后就缓存，若在 goal 设定之前被创建
会永远报「没有 goal」（实测确认）。读日志永远是最新的。

不接 `goal.*`：桌宠是通用灵动岛，同时要接 Claude Code / opencode / codex，那三家没有
goal 概念，把 agentica 独有的内部事件放进通用协议是把实现细节泄漏给协议。但**不上
wire ≠ 不影响协议**——上面那条扣住规则就是 goal 对协议的影响。

## 降级阶梯

**任何一环出问题，都只是「这条通知没显示出来」** —— 它不影响 run，也不影响
hook 那条通道里的回答。

| 级 | 情况 | 行为 |
|---|---|---|
| 1 | socket 连得上 | 正常投递 |
| 2 | 连不上（桌宠没开） | 事件丢弃，run 不受影响 |
| 3 | 连上但没响应 | 投递超时 2s 后丢弃 |
| 4 | 非 2xx / body 解析不了 | 丢弃。**这条通道不解析 body**，因为没有任何回答可读 |

第 2 级最重要：**桌宠没开着的用户不该感到任何差别**。

非阻塞投递是**队列 + 专用 daemon 线程**，调用方只入队就返回：桌宠卡住只会让事件
被丢（队列 256，满则丢最旧），**绝不会拖住 run**。

## 安全

- **socket 是本机攻击面，需要 token。** 随机 32 字节 hex，`0600` 落在
  `notify.token`，请求带 `Authorization: Bearer <token>`。
  没有 token 的请求一律 401。桌宠那边看不到事件，仅此而已 —— 这条通道单向，
  所以「伪造回答」这个风险面已经不存在了。
- token 文件**只在不存在时创建，绝不覆盖** —— 两边谁先跑谁定，覆盖会把另一边
  用 401 锁在门外。
- **只连本机**（socket 路径）。不要配成 `http://远程`。
- **只发元数据**：`title` / `tool` / `question` / `preview` / `options`。
  不发完整 prompt、工具输出原文、文件内容、会话历史、模型回复全文。
  与 `docs/v2/08-budget.md` 的成本纪律同源 —— 一旦开口子，以后没有理由拒绝
  「顺手也发一下完整命令吧」。

## 不装的路径

`--print` / SDK / cron / 无人值守 POST 走 `build_noninteractive_approve`
（`get_registry` 为 `None`），**一律没有阻塞的回答路径** —— 给一个没人在场的运行装
「等用户批准」是纯粹的挂死。非阻塞的 `/event` 可以装。

这不是「桌宠被禁了」，而是**没有用户在场可等**：没有 registry 就没有「用户在等」这个
状态，答复也无处可落。

同一条道理也管**提问**（`needs.input`）：这几条路径**连工具都不装**。`create_agent` 的
`include_ask_user_question` 是必填关键字参数，交互 CLI（及其 `/model` / `/resume` /
`/fork` / `/clear` 重建）传 `True`，`--query`/`--print`、cron、`/bg` 传 `False`。
`ask_user_question` 会一直等人回答，这是它本来的设计；给了工具再靠超时或假答复兜底，
既改不了「没人可问」，又等于**替用户说话**。装配期不给，才是说了实话。

hook 那条通道同样遵守这条：`needs.*` 只在**交互 CLI** 里发起（那里确实有人在终端前），
因为 hook 的回答是「用户本人的回答」，而无人值守路径上的回答无处可落。

## 用起来

1. 装 VPet，让它把 `notify.sock` 和 `notify.token` 建起来。
2. 在 `config.yaml` 打开 `settings.notify.enabled: true`。

跑一轮就能验证：桌宠应该从 working 走到 done。

要让它还能**回答**（桌宠上按 y/n），去配 [shell hooks](shell-hooks.md) ——
那是另一条通道，两者可以同时开。
