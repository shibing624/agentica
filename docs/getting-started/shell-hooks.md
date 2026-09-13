# 外部 hook 出口（shell hooks）

让 agentica 在生命周期节点上**运行你自己配置的一条命令**，把 JSON 喂给它的 stdin，
再从 stdout 收回答。这是 Claude Code / codex / opencode / Vibe Island 都有的形态 ——
于是接入一个桌宠是**写三行配置**，不是实现一套协议。

**默认关闭。** 见下面「开关」。

设计取舍与逐字段契约见 `docs/rfcs/external-hook-egress.md`（那份是权威，
本文与它冲突时以它为准）。

### 两条语义，别丢

1. **不设等候上限。** 本层永远不给「等用户回答」定死时限，也**没有**
   `settings.hooks.timeout` 这种看起来能设、实际不生效的字段。一条命令可以自己
   设内部超时（桌宠的桥接就设了 55s），那是**那条命令自己的事**，而且在用户
   自己的配置里看得见。在这层写一个数，等于「在终端能慢慢想，在桌宠必须秒答」。
2. **终端与 hook 并行 race，先答的算。** 工具调用**不会**挂在 hook 进程上：终端
   prompt 照旧在那儿等着，谁先给出答复谁生效。终端先答了，我们 **kill 掉 hook 的
   进程组**，它后来的回答晚了一步 —— 这是 race，不是错误。

推论：**「用户还没答」不是事件。** 用户在终端里不答，CLI 就一直等；hook 那边同理。

## 开关

```yaml
# ~/.agentica/config.yaml
settings:
  hooks:
    enabled: false            # 默认关：它会在每次 run 上执行你的命令
    command: ["/绝对/路径/notifier", "--from-agentica"]
    events:
      run.started: true
      run.completed: true
      run.failed: true
      run.cancelled: true
      needs.approval: true
      needs.input: true
```

环境变量覆盖同名项，前缀 `AGENTICA_HOOKS_`（如 `AGENTICA_HOOKS_ENABLED=1`、
`AGENTICA_HOOKS_COMMAND=/path/to/notifier`）。

`enabled` 与 `command` **两个都要有**才真的装上；只写 `enabled: true` 而没有命令
等于没配（没有东西可跑）。关掉时**什么都不装** —— 不起线程、不起进程，一次都不 fork。

### command 是 argv 列表，不是 shell 字符串

```yaml
command: ["/bin/sh", "-c", "exec /abs/path/notifier"]   # 需要 shell 就自己写出来
```

- **不经过 shell。** 所以 `"$HOME/Library/…"` **不会被展开**，会被当成一个字面量路径，
  于是每次都 spawn 失败。而失败的表现是「桌宠什么都不说」，没有任何报错 ——
  属于最难查的一类。**写绝对路径**；确实需要 `$HOME` 就自己包一层 `/bin/sh -c`。
- 写成一个字符串（`command: "/abs/notifier --flag"`）**不会被切分**，而是记一条
  warning 后忽略。切分等于替你发明一套引号规则，然后跑一个你没写的 argv。
- 包一层 shell 也是拿到我们没发的信息的正路：`$PPID` 就是 agentica 进程的 pid，
  `ps -o tty= -p $PPID` 是那个终端。这是进程语义，不是启发式。

## 事件（六个）

| event | 什么时候 | 阻塞 |
|---|---|---|
| `run.started` | 一轮开始 | 否 |
| `run.completed` | 一轮结束**且没有后续**（goal 循环会扣住，见下） | 否 |
| `run.failed` | 一轮抛错 | 否 |
| `run.cancelled` | 用户 Ctrl+C | 否 |
| `needs.approval` | 工具调用被 park，等用户批准 | **是**（回话） |
| `needs.input` | `ask_user_question` 被 park | **是**（回话） |

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
  "options": ["allow", "allow_prefix", "deny", "deny_prefix"],
  "question": "允许运行 `rm -rf build`？",
  "preview": "rm -rf build",
  "similar_label": "以后同类命令"
}
```

几条是**决定，不是默认值**：

- **事件名在 JSON 里，不在 argv 里。** 一份文档一个通道，读一个地方就不会两头对不上。
- **`prompt` 是 run 的锚文本**：普通轮是用户提问，**goal 会话里是 goal 目标**。
  可以显示，但**不能当「用户刚说的那句」**用（去重、对账都会错）。已截断。
- **可选字段是省略，不是 null。** 缺键和空串是两回事；要判两种的消费端最后只会判一种。
- `needs.input` 带 `question`（+可选 `options`），**不带 decision 词表**：它的回答是
  一段自由文本，与审批的四选一是**两种东西**。
- **`options` 是这次实际给出的**（`PendingApproval.options`，由工具自己决定）。
  照它渲染，**不要写死两档** —— 消费端照发来的渲染，就不会摆出终端会拒绝的按钮。
- **只发元数据**：`question` / `preview` / `tool` / `answer`，全部截断到 500 字 + `…`。
  **不发**完整 prompt、工具输出、文件内容、会话历史。这不是「有总比没有好」的地方。
- `cwd` 是工作目录。没有 `ppid` / `tty` —— 要这两个用上面那层 shell 包一下即可，
  所以它们是「便利」而不是「缺口」。

## 收回来：stdout

只写**一个 JSON 文档**，没有别的东西：

```json
{"decision": "allow"}       // allow | allow_prefix | deny | deny_prefix
{"answer": "date-fns"}      // 只在 needs.input 上
```

退出码三种含义，**三种都要真的实现**：

| 退出码 | stdout | 含义 |
|---:|---|---|
| 0 | 一个 JSON | 这就是用户的回答 |
| 0 | 空 | **没有决定** → 终端 prompt 照旧能答 |
| 非 0 | 忽略 | 同上，**没有决定** |

**「没有决定」绝不能写成一个 `allow`。** 这条桌宠侧踩过：Claude Code 协议里
「没有决定」的表达是回一个 `deny` 字面值，而在 agentica 这边是**什么都不输出** ——
两条路的写法不同，别把 Claude Code 那份直接复制过来。

**认不出来一律算没答**：`decision` 只认那四个词，`answer` 只认非空字符串，
其余（`{"decision": "sure"}`、散文、半个 JSON、非零退出）都是「没答」，
终端 prompt 仍然是那条路。**绝不猜、绝不默认 allow。**

## 失败阶梯

**任何一环出问题，都等于「没有决定」，终端 prompt 照旧作答：**

| 情况 | 行为 |
|---|---|
| `enabled: false` | 什么都不装（不起线程、不起进程） |
| 命令不存在 / 不可执行 | 跳过 |
| spawn 失败 | 跳过 |
| 非零退出 | 当没答 |
| stdout 为空 | 当没答 |
| JSON 解析不了 | 当没答 |
| decision 词不认识 | 当没答 |

**永不合成一个 `allow`。** 一个崩掉的脚本不该静默放行。

## 进程与安全

- **`start_new_session=True` + 按进程组 kill**：命令自己 fork 出来的子进程不会
  活下来挂着管道。桌宠的桥接进程在终端先答时就是这样被掐断的。
- **没有 token，这不是漏洞。** notify sink 需要 token，是因为它 listen 在一个
  **任何本机进程都能连**的 socket 上。这里没有共享端点 —— 命令是**用户自己在配置里
  写的**，没有可以伪造请求进去的地方。这是逃生舱，不是服务。
- **命令以你的完整用户凭证运行。** 只批准你信任的命令；改一条 hook 命令等于
  给自己装了一个能在每次 run 上执行的程序。

## 回答从哪来（只在这条路上）

`needs.*` 的发起**只在交互 CLI 里**：那里确实有人在终端前，而 hook 的回答是
**用户本人的回答** —— 与在终端敲的是同一个效果、同一个授权。

无人值守的路径（`--print` / SDK / cron / `/bg`）**不发起**：没有人在等，回答也无处可落。

## 用起来

```yaml
settings:
  hooks:
    enabled: true
    # ⚠️ 绝对路径，不要写 "$HOME/..."（这里不经过 shell，不会被展开）
    command: ["/Users/<you>/Library/Application Support/VPet/agent-notify/vpet-hook"]
```

跑一轮验证：桌宠说「跑完了」；再来一轮触发审批 → **在桌宠上按 y → 终端里的审批放行**
（这是唯一能证明回传真的到了的值）。再试一次终端先答 → hook 进程被掐断、
没有报错弹窗、没有僵尸进程。

观察（`run.*`）与回答（`needs.*`）是**同一个命令**的两个方向，可以只实现前者 ——
只打印不回答，什么都不输出即可。
