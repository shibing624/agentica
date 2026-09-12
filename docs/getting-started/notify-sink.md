# 外部通知汇（notify sink）

把 agentica 一轮跑到什么节点告诉本机的一个 app（目前是 [VPet](https://github.com/shibingtan/VPet) 桌宠），
并且让用户**在桌宠上也能回答**（按 y/n、输入文本）—— 解的是「终端埋在窗口底下、agent 停在等你」这个场景。

**默认关闭。** 见下面「开关」。

### 三件事，别混

这三条是这个通道的全部语义，写文档、写代码、写测试时都要分清：

1. **桌宠不替用户做决定。** 它没有策略、没有自动批准、没有 YOLO，任何路径都不会
   auto-allow。没有「桌宠被允许决定吗」这种开关 —— 那等于承认它有自己的权限。
2. **用户在桌宠上的输入 = 用户在终端上的输入。** 同一个 session、同一个交互，
   在桌宠上按的 y 和自己在终端敲的 y 效力完全相同，都是**用户本人**的答复。
3. **用户自己担责。** 桌宠只是一个输入面；谁按的、按了什么，责任在用户。

推论：**「用户还没答」不是事件**。终端里用户不答 CLI 就一直等，所以桌宠这边也
不设等候上限 —— 由终端语义决定等多久，不由我们这层写一个数。

## 它是什么

一条只连本机的汇，HTTP over Unix domain socket：

| 路径 | 语义 | 超时 | 响应 |
|---|---|---:|---|
| `POST /event` | 通知，不等回话 | 2s | 不解析 |
| `POST /await` | 等用户回答 | 调用方给定 | `{"decision": ...}` / `{"answer": ...}` / `{"reject": true}` |

**阻塞与否由路径决定，不从 body 推断** —— 否则一条写错的 body 就能让 agent 挂 300 秒。

`/await` 的等候时长**由调用方传入，本层不写死**：CLI 传 `None`，因为终端 prompt 本来就
等到用户回答为止；给桌宠更短的时限就等于「在终端能慢慢想，在桌宠必须秒答」。

装上的都是**并挂**，不是替换：`AgentHooks` / `RunHooks` 的语义不变，终端 prompt 也原样保留。

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
| `run.started` | 一轮开始 | 否 |
| `run.completed` | 一轮成功结束**且没有后续** | 否 |
| `run.failed` | 一轮抛错 | 否 |
| `run.cancelled` | 用户 Ctrl+C | 否 |
| `needs.approval` | 工具调用被 park 等批准 | **是**（`/await`） |
| `needs.input` | `ask_user_question` | **是**（`/await`） |

`needs.approval` 与 `needs.input` 是**两种状态**（急切 / 平静），用 `payload.kind`
区分（`"permission"` vs `"question"`），别混。

### payload 字段

run 类事件的 `payload` 只带元数据（**没有** prompt、无工具输出、无文件内容——
这个通道即使在本机 socket 上也保持窄）：

| 字段 | 出现于 | 含义 |
|---|---|---|
| `agent_name` | 全部 run 类 | 哪个 agent 跑的 |
| `duration_seconds` | `run.completed` / `run.failed` | 时长 |
| `had_response` | `run.completed` | 这一场是否产出过回复 |
| `reason` | `run.cancelled` | 为什么被取消 |
| `error` | `run.failed` | 错误文本（`类型: 信息`） |

`payload` 是**白名单过滤**产物：loop 层可以带更多键，sink 只放行上表这几个。
例如 `run.failed` 的 loop 事件里有 `exception_type`，但**不过线**——消费端要判断
错误种类请解析 `error` 的前缀（`"ValueError: ..."`）。缺失即「没有这个信息」，
**不要推断**（没有 `duration_seconds` 不代表 0）。

`needs.*` 的 payload 在 `/await` 的请求体里，另有 `kind` 与 `options`：

- `needs.input`（`kind: "question"`）：`options` 是任意选项，回传 `answer` 是任意字符串；
- `needs.approval`（`kind: "permission"`）：回传 `decision` 只接受
  `allow` / `allow_prefix` / `deny` / `deny_prefix` 四个值，`options` 列出这次**实际
  给出**的（可能少于四个）。对应终端里的 `y` / `p` / `n` / `x`——桌宠的按钮应照着
  `options` 渲染，不要写死两个。

**多个字段在补发的 `run.completed` 上会合并**：一场 goal 只补发一次，若中间发生过
多轮，`duration_seconds` 是各轮**累加**（用户关心的是「我离开这段时间它跑了多久」，
不是最后一轮），`had_response` 只要有一轮产出过就为真，`agent_name` 取首个非空值。

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

**任何一环出问题，都回落终端 prompt。**

| 级 | 情况 | 行为 |
|---|---|---|
| 1 | socket 连得上 | 正常往返 |
| 2 | 连不上（桌宠没开） | **立刻**回落，不等超时 |
| 3 | 连上但没响应 | 等到调用方给的时限，然后回落 |
| 4 | 解析不了 / 字段不认识 | 当作「没有答复」，回落。不猜、不默认 allow |

第 2 级最重要：**桌宠没开着的用户不该感到任何差别**。connect 失败确实是 fast fail
（实测 ~0ms 返回），所以这里不能等满超时。

第 2/3/4 级是「桌宠那边没有答复」，与「用户在桌宠上还没按」不是一回事：前者要回落，
后者要继续等（跟终端里的等待一样久）。

非阻塞投递是**队列 + 专用 daemon 线程**，调用方只入队就返回：桌宠卡住只会让事件
被丢（队列 256，满则丢最旧），**绝不会拖住 run**。

## 安全

- **socket 是本机攻击面，需要 token。** 随机 32 字节 hex，`0600` 落在
  `notify.token`，请求带 `Authorization: Bearer <token>`。
  没有 token 的请求一律 401。理由是：一条任何本机进程都能连的通道，不该能让本机
  任何进程**冒充用户**回答 approve/question —— 那等于把「用户本人按的 y」交给
  任何进程伪造。
- token 文件**只在不存在时创建，绝不覆盖** —— 两边谁先跑谁定，覆盖会把另一边
  用 401 锁在门外。
- **只连本机**（socket 路径）。不要配成 `http://远程`。
- **只发元数据**：`title` / `tool` / `question` / `preview` / `options`。
  不发完整 prompt、工具输出原文、文件内容、会话历史、模型回复全文。
  与 `docs/v2/08-budget.md` 的成本纪律同源 —— 一旦开口子，以后没有理由拒绝
  「顺手也发一下完整命令吧」。

## 不装的路径

`--print` / SDK / cron / 无人值守 POST 走 `build_noninteractive_approve`
（`get_registry` 为 `None`），**一律没有阻塞 sink** —— 给一个没人在场的运行装
「等用户批准」是纯粹的挂死。非阻塞的 `/event` 可以装。

这不是「桌宠被禁了」，而是**没有用户在场可等**：没有 registry 就没有「用户在等」这个
状态，答复也无处可落。

## 用起来

1. 装 VPet，让它把 `notify.sock` 和 `notify.token` 建起来。
2. 在 `config.yaml` 打开 `settings.notify.enabled: true`。

跑一轮就能验证：桌宠应该从 working 走到 done。桌宠上应该也能直接按 y/n 或输回答 ——
和你在终端里敲是同一回事。
