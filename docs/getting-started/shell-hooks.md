# 外部 hook 出口（shell hooks）

让 agentica 在生命周期节点上并行运行用户配置的具名消费者，把 JSON 喂给各自的
stdin，再从 stdout 收回答。VPet、Open Island、cc-beeper 可以各自订阅事件，安装和
卸载都只修改自己的条目。

**默认关闭。** 见下面「开关」。

设计取舍与逐字段契约见 `docs/rfcs/external-hook-egress.md`（那份是权威，
本文与它冲突时以它为准）。

### 两条语义，别丢

1. **不设等候上限。** 本层永远不给「等用户回答」定死时限，也**没有**
   `settings.hooks.timeout` 这种看起来能设、实际不生效的字段。一条命令可以自己
   设内部超时（桌宠的桥接就设了 55s），那是**那条命令自己的事**，而且在用户
   自己的配置里看得见。在这层写一个数，等于「在终端能慢慢想，在桌宠必须秒答」。
2. **终端与所有 hook 并行 race，首个合法答复生效。** 工具调用不会挂在某一个
   hook 进程上；终端照旧可答。赢家产生后会 kill 其余 hook 进程组，晚到是 race，
   不是错误。

推论：**「用户还没答」不是事件。** 用户在终端里不答，CLI 就一直等；hook 那边同理。

## 开关

```yaml
# ~/.agentica/config.yaml
settings:
  hooks:
    enabled: false
    consumers:
      - name: open-island
        command: ["/绝对/路径/OpenIslandHooks", "--source", "agentica"]
        enabled: true
        events:
          run.started: true
          needs.approval: true
      - name: vpet
        command: ["/绝对/路径/vpet-hook"]
```

`name` 必填且必须唯一；`events` 缺省全部开启，每个消费者独立配置。顶层
`enabled` 是总开关，消费者自己的 `enabled` 是局部开关。

环境变量覆盖同名项，前缀 `AGENTICA_HOOKS_`：

```bash
AGENTICA_HOOKS_ENABLED=1
AGENTICA_HOOKS_CONSUMERS='[{"name":"open-island","command":["/path with spaces/OpenIslandHooks","--source","agentica"]}]'
```

`AGENTICA_HOOKS_CONSUMERS` 必须是 JSON 数组；不做空白切分，也没有
`AGENTICA_HOOKS_COMMAND` 单槽变量。

### 每个 command 都是 argv 列表

```yaml
consumers:
  - name: wrapped
    command: ["/bin/sh", "-c", "exec /abs/path/notifier"]
```

- **不经过 shell。** 所以 `"$HOME/Library/…"` **不会被展开**，会被当成一个字面量路径，
  于是每次都 spawn 失败。而失败的表现是「桌宠什么都不说」，没有任何报错 ——
  属于最难查的一类。**写绝对路径**；确实需要 `$HOME` 就自己包一层 `/bin/sh -c`。
- 写成一个字符串（`command: "/abs/notifier --flag"`）**不会被切分**，而是记一条
  warning 后忽略。切分等于替你发明一套引号规则，然后跑一个你没写的 argv。
- 不必包 shell 获取进程身份；每份 payload 都带与 notify sink 一致的 `transport`。

## 事件

| event | 什么时候 | 阻塞 |
|---|---|---|
| `run.started` | 一轮开始 | 否 |
| `run.completed` | 一轮结束**且没有后续**（goal 循环会扣住，见下） | 否 |
| `run.failed` | 一轮抛错 | 否 |
| `run.cancelled` | 用户 Ctrl+C | 否 |
| `tool.started` | 工具开始执行 | 否 |
| `tool.completed` | 工具结束，含成功状态和耗时 | 否 |
| `session.started` | 交互 CLI 启动、`/new`、`/clear`、`/resume` 或 `/fork` 后的新 logical session | 否 |
| `session.ended` | 交互 CLI 退出或切换上述 logical session | 否 |
| `needs.approval` | 工具调用被 park，等用户批准 | **是**（回话） |
| `needs.input` | `ask_user_question` 被 park | **是**（回话） |
| `needs.resolved` | 待答请求已由终端、hook 或取消解决 | 否 |

`tool.*` 跟真实执行边界走：串行工具轮到自己时才 started，并行工具谁先结束谁先
completed；运行中取消也会补一条 `ok: false` 的 completed。

`goal.*` **不上这条线**：一次请求变成 N 个 run 是 agentica 的实现细节。但
**不上线 ≠ 不影响协议** —— goal 活跃时 `run.completed` 会被扣住，否则一个 5 轮
goal 会报 5 次「跑完了」。这条规则与 notify sink 共用（同一处代码，不是两份实现）。

## 喂进去：stdin

```json
{
  "hook_event_name": "needs.approval",
  "session_id": "e549ef44-4836-47e4-81e5-92fe60e561f7",
  "prompt": "把这个模块的测试补上",
  "cwd": "/Users/xuming/Documents/Codes/VPetMac",
  "run_id": "…",
  "tool_name": "execute",
  "tool_call_id": "call_00_wNzJ…",
  "request_id": "…",
  "options": ["allow", "allow_prefix", "deny", "deny_prefix"],
  "question": "允许运行 `rm -rf build`？",
  "preview": "rm -rf build",
  "similar_label": "以后同类命令",
  "transport": {"ppid": 1234, "cwd": "/Users/xuming/Documents/Codes/VPetMac", "tty": "/dev/ttys003"}
}
```

几条是**决定，不是默认值**：

- **事件名在 JSON 里，不在 argv 里。** 一份文档一个通道，读一个地方就不会两头对不上。
- **`prompt` 是 run 的锚文本**：普通轮是用户提问，**goal 会话里是 goal 目标**。
  可以显示，但**不能当「用户刚说的那句」**用（去重、对账都会错）。已截断。
- `session_id` 必发；没有持久 session 的 SDK 进程使用稳定的进程级合成 id。
- **可选字段是省略，不是 null。** 缺键和空串是两回事；要判两种的消费端最后只会判一种。
- `needs.input` 带 `question`（+可选 `options`），**不带 decision 词表**：它的回答是
  一段自由文本，与审批的四选一是**两种东西**。
- **`options` 是这次实际给出的**（`PendingApproval.options`，由工具自己决定）。
  照它渲染，**不要写死两档** —— 消费端照发来的渲染，就不会摆出终端会拒绝的按钮。
- **只发元数据**：`question` / `preview` / `tool` / `answer`，全部截断到 500 字 + `…`。
  **不发**完整 prompt、工具输出、文件内容、会话历史。这不是「有总比没有好」的地方。
- `transport` 与 notify sink 同源，含 `ppid`、`cwd`、可用时的 `tty` 和 attach endpoint。
- `session.started` 还带 `model`、`profile`、`permission_mode` 与可用时的
  `transcript_path`。

## 收回来：stdout

只写**一个 JSON 文档**，没有别的东西：

```json
{"request_id": "…", "decision": "allow"}
{"request_id": "…", "answer": "date-fns"}
```

`request_id` 必须原样回传；缺失或不匹配的答复会被丢弃。

**决定就是那份 JSON。退出码不参与裁决。** 包装进程可以打印 JSON 后继续占着管道；
已经写出合法文档的进程随后以非零退出，仍按那份文档算。没有合法 JSON（空输出、
解析不了、对不上的 `request_id`）才是「没答」，与退出码无关。

**「没有决定」绝不能写成一个 `allow`。** 这条桌宠侧踩过：Claude Code 协议里
「没有决定」的表达是回一个 `deny` 字面值，而在 agentica 这边是**什么都不输出** ——
两条路的写法不同，别把 Claude Code 那份直接复制过来。

**认不出来一律算没答**：`decision` 只认那四个词，`answer` 只认非空字符串，
其余（`{"decision": "sure"}`、散文、半个 JSON）都是「没答」，
终端 prompt 仍然是那条路。**绝不猜、绝不默认 allow。**

## 失败阶梯

**任何一环出问题，都等于「没有决定」，终端 prompt 照旧作答：**

| 情况 | 行为 |
|---|---|
| `enabled: false` | 什么都不装（不起线程、不起进程） |
| 命令不存在 / 不可执行 | 跳过 |
| spawn 失败 | 跳过 |
| stdout 为空 / JSON 解析不了 | 当没答 |
| decision 词不认识 | 当没答 |

**永不合成一个 `allow`。** 没写出合法 JSON 的崩溃不算回答，终端 prompt 照旧能答。

## 进程与安全

- **`start_new_session=True` + 按进程组 kill**：命令自己 fork 出来的子进程不会
  活下来挂着管道。桌宠的桥接进程在终端先答时就是这样被掐断的。
- notice consumer 最长运行 30 秒，之后整组终止并回收；agentica 进程退出时也会
  `atexit` 杀掉仍在跑的 notice 进程组（独立 session 的子进程不会跟着父进程一起
  没）。`needs.*` 不设回答时限，仍与终端一直竞速到某一路作答或取消。
- `tool.*` 的 preview 与失败文本在截断前经过严格凭证脱敏；完整工具参数和输出不外发。
- **没有 token，这不是漏洞。** notify sink 需要 token，是因为它 listen 在一个
  **任何本机进程都能连**的 socket 上。这里没有共享端点 —— 命令是**用户自己在配置里
  写的**，没有可以伪造请求进去的地方。这是逃生舱，不是服务。
- **命令以你的完整用户凭证运行。** 只批准你信任的命令；改一条 hook 命令等于
  给自己装了一个能在每次 run 上执行的程序。

## 回答从哪来（只在这条路上）

`needs.*` 的发起**只在交互 CLI 里**：那里确实有人在终端前，而 hook 的回答是
**用户本人的回答** —— 与在终端敲的是同一个效果、同一个授权。

无人值守路径（`--print` / SDK / cron / `/bg`）不发起 `needs.*`，但会自动安装
egress，并照常发送 `run.*` 与 `tool.*`；无需先启动一次交互 CLI。

## 用起来

```yaml
settings:
  hooks:
    enabled: true
    consumers:
      - name: vpet
        # 绝对路径，不要写 "$HOME/..."（这里不经过 shell，不会被展开）
        command: ["/Users/<you>/Library/Application Support/VPet/agent-notify/vpet-hook"]
```

跑一轮验证：桌宠说「跑完了」；再来一轮触发审批 → **在桌宠上按 y → 终端里的审批放行**
（这是唯一能证明回传真的到了的值）。再试一次终端先答 → hook 进程被掐断、
没有报错弹窗、没有僵尸进程。

观察与回答是每个消费者同一条命令的两个方向。多个消费者并行启动；`needs.*`
取第一个 request id 匹配且内容合法的答复，随即终止其余进程。只观察的消费者
对 `needs.*` 不输出即可。
