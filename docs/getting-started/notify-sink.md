# 外部通知汇（notify sink）

把 agentica 一轮跑到什么节点告诉本机的一个 app（目前是 [VPet](https://github.com/shibingtan/VPet) 桌宠），
并且可选地让它**替你在终端里按 y/n** —— 解的是「终端埋在窗口底下、agent 停在等你」这个场景。

**默认关闭，两个开关分开。** 见下面「开关」。

## 它是什么

一条只连本机的汇，HTTP over Unix domain socket：

| 路径 | 语义 | 超时 | 响应 |
|---|---|---:|---|
| `POST /event` | 通知，不等回话 | 2s | 不解析 |
| `POST /await` | 阻塞，等一个决定 | 55s | `{"decision": ...}` / `{"answer": ...}` / `{"reject": true}` |

**阻塞与否由路径决定，不从 body 推断** —— 否则一条写错的 body 就能让 agent 挂 300 秒。

装上的都是**并挂**，不是替换：`AgentHooks` / `RunHooks` 的语义不变，终端 prompt 也原样保留。

## 开关

```yaml
# ~/.agentica/config.yaml
settings:
  notify:
    enabled: false                 # 要不要让桌宠知道
    socket: "~/Library/Application Support/VPet/notify.sock"
    token: ""                      # 留空则读 ~/Library/Application Support/VPet/notify.token
    approve_from_desktop: false    # 能不能替用户决定
    events:                        # 逐事件开关，默认全开
      run.started: true
      run.completed: true
      run.failed: true
      run.cancelled: true
    timeout_seconds: 55
```

环境变量覆盖同名项，前缀 `AGENTICA_NOTIFY_`（如 `AGENTICA_NOTIFY_ENABLED=true`、`AGENTICA_NOTIFY_SOCKET`）。

两个开关**风险完全不同**，所以分开：

- `enabled` 只管「让桌宠知道」。关掉时**什么都不装** —— 不建队列、不起线程、不注册回调。
- `approve_from_desktop` 管「能不能替你决定」，默认关。打开后桌宠可以回一个 `allow`，
  那等于它在终端里按了 y。误判一次的代价是替用户放行了一条危险命令，
  所以这个开关必须是人**明确知道自己在开什么**时打开的。

## 事件

| event | 触发点 | 阻塞 |
|---|---|---|
| `run.started` | 一轮开始 | 否 |
| `run.completed` | 一轮成功结束 | 否 |
| `run.failed` | 一轮抛错 | 否 |
| `run.cancelled` | 用户 Ctrl+C | 否 |
| `needs.approval` | 工具调用被 park 等批准 | **是** |
| `needs.input` | `ask_user_question` | **是** |

`needs.approval` 与 `needs.input` 是**两种状态**（急切 / 平静），用 `payload.kind`
区分（`"permission"` vs `"question"`），别混。

不接 `goal.*`：目标循环由 `GoalManager` 走自己的回调，桌宠不需要它。

## 降级阶梯

**任何一环出问题，都回落终端 prompt，绝不替用户放行。**

| 级 | 情况 | 行为 |
|---|---|---|
| 1 | socket 连得上 | 正常往返 |
| 2 | 连不上（桌宠没开） | **立刻**回落，不等超时 |
| 3 | 连上但没响应 | 55s 超时后回落 |
| 4 | 解析不了 / 字段不认识 | 当作「没有决定」，回落。不猜、不默认 allow |

第 2 级最重要：**桌宠没开着的用户不该感到任何差别**。connect 失败确实是 fast fail
（实测 ~0ms 返回），所以这里不能等满超时。

非阻塞投递是**队列 + 专用 daemon 线程**，调用方只入队就返回：桌宠卡住只会让事件
被丢（队列 256，满则丢最旧），**绝不会拖住 run**。

## 安全

- **socket 是本机攻击面，需要 token。** 随机 32 字节 hex，`0600` 落在
  `notify.token`，请求带 `Authorization: Bearer <token>`。
  没有 token 的请求一律 401。理由是：一条任何本机进程都能伪造的「等批准」通道，
  等于把审批权交给本机任何进程。
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

## 用起来

1. 装 VPet，让它把 `notify.sock` 和 `notify.token` 建起来。
2. 在 `config.yaml` 打开 `settings.notify.enabled: true`（只想看状态就到这儿）。
3. 想让它能替你按 y，再打开 `approve_from_desktop: true`。

跑一轮就能验证：桌宠应该从 working 走到 done。
