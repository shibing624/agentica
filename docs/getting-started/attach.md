# 外部程序往会话里发一条用户消息（attach）

让**别的进程**（桌宠、快捷键脚本、通知程序、你自己的工具）往一个**正在跑的**
agentica 会话里说一句话 —— 效果等同于**你自己在那个终端里敲**。

**默认关闭**，见下面「开关」。

### 它是 ACP 形态，不是 ACP 兼容

它**借用** [ACP](https://agentclientprotocol.com) 的形态：换行分隔的 JSON-RPC、
`session/*` 方法名、`session/prompt` 里的 content blocks。但它**不是**一个 ACP
实现，两点根本差别：

- **传输**：ACP 的 stdio 传输是**客户端新起一个 agent 子进程**；这条是**附着**到一个
  已经在跑（且 TUI 占着 stdin/stdout）的会话，所以走 unix socket —— ACP 允许自定义
  传输，但**标准 ACP 客户端（Zed 等）接不上这个 socket**。
- **方法集**：只实现附着所需的子集，没有 `session/new`；`initialize` 也不做官方的
  `authMethods` → `auth/login` 协商（这里第一条消息就要 `authToken`，见下）。

所以 `PROTOCOL_VERSION` 是**本通道自己的版本号**，不代表能对上 ACP v1。

### 它解决的是哪件事

会话的输入只有一个写者：坐在键盘前面的人。这条通道给了第二个入口，但它送进去的
仍然是**用户本人**的输入，不是另一个 agent 的话（那是 `send-message`，见
[terminal.md](terminal.md)）。

所以它走的正是你敲一行所走的那条路（`hand_to_agent`）：会话在跑就**插到下一次
tool 边界**（和 `/steer` 一样），空闲就**作为下一轮**。

## 开关

```yaml
# ~/.agentica/config.yaml
settings:
  attach_enabled: false   # 默认关：开了就等于多一个「用户输入」的入口
```

环境变量 `AGENTICA_ATTACH_ENABLED=1` 覆盖它。关掉时**不建 socket、不起线程**，
行为与此前完全一致；socket 在会话启动时创建，所以这一项**读了就是读了**，
中途改要重启会话。

## 它是什么：一条本机 socket + JSON-RPC

**一个会话一个 socket，连上它就是点名了这个会话。** 没有 `--to`、没有第二个寻址
方式：不会重名、不会有前缀歧义、也**不按 cwd 匹配**（符号链接会让同一条路径有两种
写法，那是真踩过的坑）。

TUI 自己占着 stdin/stdout，所以 ACP 标准的 stdio 传输在这里用不了；这条通道用 ACP
允许的**自定义传输**，并保持它的消息格式与生命周期 —— 也就是 JSON-RPC 2.0，
**一行一个 JSON 文档**（`\n` 分隔，文档内不能有换行）。

### 发现它

两条路，按你已经连着哪个通道选：

**1（桌面 App 用这条）：notify 事件的 `transport` 块。** 如果你本来就在收
`run.started` / `run.completed` 这些事件，那么每个 envelope 里已经带着算好的路径：

```json
{"event":"run.started","transport":{"ppid":1234,"cwd":"…","tty":"ttys004",
  "attach_socket":"/var/folders/…/agentica-501/<peer_id>.sock","peer_id":"<peer_id>"}}
```

`attach_socket` / `peer_id` 只在**真的在监听**时出现（没开 attach 就没有这两个键）。
这条路不需要知道 agentica 把东西放在哪，也不需要能 `import agentica` —— 后一点对
launchd 起的 `.app` 是硬要求：默认 `PATH` 下的 `/usr/bin/python3` 是 3.9，
`import agentica` 直接 `ModuleNotFoundError`，而**症状和「没有会话在跑」一模一样**。

**2：presence 记录。** 一个会话在跑时会在
`<AGENTICA_CACHE_DIR>/peers/live/<peer_id>.json` 写下自己的信息，其中
`attach_socket` 就是它的 socket 路径（没开 attach 时为 `null`）。

```bash
# 人类看：这个会话的 socket 在哪
#   /list-agents        （交互会话里；每个会话的 "attach" 行）
#   list_agents         （agent 侧，同一份字段）
```

**直接读记录里的路径，不要自己拼** —— 它由 uid 和 `TMPDIR` 决定，拼错的症状是
「那个会话好像没在跑」，与真实原因毫不相干。

> 注意：`attach_socket` 是 presence 记录里的字段；**不要**去 grep 进程或按 cwd 找会话。
> 也没有「列出所有 attach socket」的独立子命令 —— 会话列表就是 `live/*.json`
> 这一份数据，`/list-agents` 与 `list_agents` 读的是同一份。

**`AGENTICA_HOME` 会连带把 cache 一起搬走。** presence 记录写在
`AGENTICA_CACHE_DIR` 下，而它的默认值是 `$AGENTICA_HOME/cache`。所以一个
`AGENTICA_HOME=<临时目录>` 起来的会话，会把自己的记录写进**那个** home；如果
发现方还在默认位置找，结果就是「没有开着 attach 的会话」，而那个会话明明开着 ——
报出来的现象和真实原因毫无关系。用 `AGENTICA_HOME` 做隔离时，让发现方和会话
读**同一个** `AGENTICA_CACHE_DIR`（或干脆两边都设 `AGENTICA_CACHE_DIR`）。

## 方法

连上后**第一条消息必须先 `initialize` 并带上 token**，否则任何方法都回 -32000。
没有 token 的客户端**什么都问不出来**（连方法列表都不给）。

| 方法 | 作用 |
|---|---|
| `initialize` | 鉴权 + 版本协商。`params.authToken` 必填 |
| `session/load` | 附着到这个会话；返回 `sessionId`、`cwd`、`busy` |
| `session/prompt` | **把用户的话送进去**，等这一轮结束再回 |
| `session/cancel` | 中断当前这一轮。**必须另开一条连接发**（见下） |
| `ping` | 健康检查 |

**`session/cancel` 要另开一条连接。** 它和 `session/prompt` 是**同一条连接上的两条
请求**，而 `session/prompt` 会**阻塞到这一轮结束才返回**（一条连接同时只放一个
prompt）。所以在同一条连接上「先 prompt 再 cancel」永远不会发生：那句话排在一条
永远不空出来的连接后面。要中断，另开一条连接发 `session/cancel`，它会打断那一轮；
原来那条连接上的 `session/prompt` 随即以 `stopReason: "cancelled"` 返回。

这不是文档补充，是**读了会做错设计**的地方：按「同连接」实现出来的客户端，表现为
取消无效（其实是没发出去）。

**没有 `session/new`**：会话已经存在（就是这个 socket 的主人），在这里凭空造一个新
的会让客户端对着一个**没有终端在驱动**的对话说话。

### `initialize`

```json
{"jsonrpc":"2.0","id":1,"method":"initialize",
 "params":{"authToken":"<socket 同目录的 <peer_id>.token>","clientCapabilities":{}}}
```

返回 `protocolVersion`、`agentCapabilities`、`agentInfo`。token 在
`<socket 同目录>/<peer_id>.token`，`0600`。

### `session/prompt`

```json
{"jsonrpc":"2.0","id":2,"method":"session/prompt",
 "params":{"sessionId":"<可选>","prompt":[{"type":"text","text":"把测试跑一遍"}]}}
```

返回：

```json
{"jsonrpc":"2.0","id":2,"result":{"stopReason":"end_turn","agenticaAnswer":"…"}}
```

- **`stopReason`**：`end_turn`（这一轮跑完了）/ `cancelled`（被 `session/cancel`
  中断）/ `agentica_pending`（这一轮**已被接受但本层没等到它结束**，例如会话正在
  退出），同时带 `agenticaPending: true`。

  受超时这一档**不能复用 `end_turn`**：ACP 里 `end_turn` 就是"正常结束"，只读
  `stopReason` 的客户端会把超时读成"回答完了"。所以差异放在 `stopReason` 本身，
  `agenticaPending` 作为并列字段保留（已有消费端在读它）。
- **`agenticaAnswer`**（可选）：这一轮的最后一段回答。客户端要它就能省掉一次读取。
  和 notify 的 `run.completed` 里那条 `answer` **读的是同一个字段**
  （`agent.run_response.content`），只是取的时机不同：attach 在 `session/prompt`
  返回前取，notify 在那一轮报完成时取。所以两者**可能一个有一个没有**（例如工具轮、
  或被 goal 接管的轮没有最终文本），这不是两个来源打架，是**同一次读取的两个时刻**。
  要"一定拿到答案"就自己读 transcript；要"这一轮说了什么"用这两个都行。
- **`settle` 只服务 steered。** `steer()` 可能把话吃进最后一次 inference，随后
  `promote_late_steer` 再开一轮；这时 `session/prompt` 会再看一小段窗口（默认 1s）。
  **queued 不走这扇窗**：那一轮已经是吃下这句的一轮，再等只会让每条空闲注入多 1 秒，
  并把 goal 续跑 / 用户接着打的字收成 `agenticaAnswer`。
- **只收文本块。任何这一层送不到的块都整条拒绝**（-32602），**不是跳过**：`[文本,
  图片]` 这种混合请求若把图跳过，就等于把「描述这张图」交给 agent 而图不在 ——
  换成了另一个问题，且下游无从察觉。拒绝信息会点明**哪些块类型**没被送出。
- `sessionId` 若给出且与会话不符，**拒绝**（-32000），并告知这个 socket 服务于哪个
  会话。

### 鉴权与安全

- socket 是 **unix domain socket**，只连本机；目录 `0700`、socket `0600`、token `0600`。
- **这条通道的效力等于「用户本人输入」**，所以比只读的 notify sink 危险一档：
  每条连接都要 token，`initialize` 也不例外。
- 会话退出时 socket 与 token **一起删掉**，notify envelope 上的 `attach_socket` /
  `peer_id` **同时清掉**：留着 token 会让后来的读者对着一个已经不存在的会话通过鉴权；
  留着 envelope 里的路径会让消费端连上一个已经拆掉的 socket，症状仍是「会话没在跑」。

### 失败会怎样

**任何一环出问题都只是「这条通道不可用」，终端行为不变**：

| 情况 | 行为 |
|---|---|
| `enabled: false` | 不建 socket、不起线程 |
| socket 建不起来（路径过长、权限） | 记一条日志，没有 attach 点，会话照常 |
| 没有 token / token 不符 | -32000，连接可用但什么都做不了 |
| 报文不是 JSON | -32700，**连接继续可用** |
| 方法不认识 | -32601 |
| 注入时抛异常 | -32603，连接继续可用 |
| 客户端半路消失 | 服务端不受影响，其他客户端照常 |

## 用起来

```bash
# 1. 起一个交互会话（要开 settings.attach_enabled）
agentica

# 2. 另一个终端/程序里，读会话的 presence 记录拿到 socket 路径
python -c "
import json, glob, os
root = os.path.expanduser(os.environ.get('AGENTICA_CACHE_DIR', '~/.agentica/cache'))
for f in glob.glob(root + '/peers/live/*.json'):
    d = json.load(open(f))
    print(d['name'], d['peer_id'], d.get('attach_socket'))
"
# 交互会话里也可以直接 /list-agents，看每个会话的 attach 行

# 3. 用一行 python 发一句（真实客户端就是这样）
python - <<'PY'
import json, socket
path = "/var/folders/.../agentica-501/<peer_id>.sock"   # 从 presence 记录读
token = open(path.replace(".sock", ".token")).read().strip()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(path)
def call(method, params=None, i=1):
    s.sendall(json.dumps({"jsonrpc":"2.0","id":i,"method":method,
                          "params":params or {}}).encode() + b"\n")
    buf = b""
    while b"\n" not in buf: buf += s.recv(65536)
    print(json.loads(buf.split(b"\n")[0]))
call("initialize", {"authToken": token})
call("session/prompt", {"prompt": [{"type":"text","text":"说一句 PONG"}]}, i=2)
PY
```

那个终端里应该**先看到这句话被回显**（标注来自外部），然后看到它对这句话的回答。

验收脚本：`python scripts/verify_attach_e2e.py` —— 真 tmux 起一个会话、真 JSON-RPC
客户端连上去、断言会话真的把它当用户输入并回答了。

## 与另外两条通道的分工

| | 谁在说话 | 方向 | 用途 |
|---|---|---|---|
| **attach**（本文） | **用户**（从别的程序） | 进 | 让外部程序替用户在这个会话里说一句 |
| `shell hooks`（[shell-hooks.md](shell-hooks.md)） | agentica | 出 | 把 run 的状态告诉外部；审批/提问的回话 |
| `notify sink`（[notify-sink.md](notify-sink.md)） | agentica | 出，只看 | 把 run 的状态告诉桌宠（单向） |
| `send-message` / `/n`（[terminal.md](terminal.md)） | **另一个 agent 会话** | 双向 | agent 之间交接 |
