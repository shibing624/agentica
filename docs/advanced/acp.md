# ACP 集成

ACP（Agent Client Protocol）是 IDE 与自动化客户端调用 AI Agent 的标准协议。Agentica 作为 **ACP Agent（服务端）** 运行在 stdio 上：Zed 等编辑器、Agent Autoloop 这类自动化中台，或任何自己写的客户端都可以接入。

## 协议约定（v1）

| 项目 | 约定 |
|---|---|
| 传输 | JSON-RPC 2.0 的 NDJSON：**一行一个 JSON 帧**，不使用 LSP 的 `Content-Length` 分帧 |
| stdout | 只写协议帧；服务日志、Agent 输出和第三方 `print()` 全部走 stderr |
| 初始化 | 返回 `protocolVersion: 1`、`agentCapabilities.loadSession: true`、`agentInfo`、`authMethods` |
| 提示词 | `session/prompt` 的 `prompt` 是内容块数组：`[{"type":"text","text":"..."}]` |
| 流式输出 | `session/update` 通知，`update.sessionUpdate` 为 `agent_message_chunk`，文本在 `update.content.text` |
| 结束 | `session/prompt` 返回 `{"stopReason":"end_turn"}` |
| 会话恢复 | 新进程用同一个 `sessionId` 调 `session/load` 即可接着上一轮 |

`tools/list` 与 `tools/call` 仍提供文件读写、搜索、命令执行等能力，权限由客户端通过 ACP 的一次性选项控制。

## 启动

```bash
agentica acp
```

不需要额外依赖，也不监听端口。模型沿用本机 `agentica` 的配置（`~/.agentica`），先用 CLI 登录或配好密钥即可。

会话数据默认保存在 `~/.agentica/acp/sessions/`：

- `<sessionId>.json`：会话元数据与消息历史，供 `session/load` 校验和恢复
- `transcripts/<sessionId>.jsonl`：Agent 运行轨迹，跨进程回放靠它

用环境变量 `AGENTICA_ACP_SESSION_DIR` 换目录；嵌入第三方程序或跑测试时建议指向独立目录，避免和本机会话混在一起。

## 一轮对话

客户端按行写请求，服务端按行回响应与通知。通知不带 `id`：

```jsonl
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{},"clientInfo":{"name":"my-client","version":"1.0"}}}
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{"loadSession":true,"promptCapabilities":{"text":true,"embeddedContext":true}},"agentInfo":{"name":"agentica","version":"1.6.0"},"authMethods":[]}}
{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/path/to/project","mcpServers":[]}}
{"jsonrpc":"2.0","id":2,"result":{"sessionId":"a1b2c3d4"}}
{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"a1b2c3d4","prompt":[{"type":"text","text":"你好"}]}}
{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"a1b2c3d4","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"你好"}}}}
{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}
```

最小客户端（把每行当 JSON 解析即可，日志要看 stderr）：

```python
import json
import subprocess

proc = subprocess.Popen(
    ["agentica", "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, text=True, bufsize=1,
)


def call(method, params, call_id):
    proc.stdin.write(json.dumps({
        "jsonrpc": "2.0", "id": call_id, "method": method, "params": params,
    }) + "\n")
    proc.stdin.flush()
    text = []
    while True:
        frame = json.loads(proc.stdout.readline())
        if "method" in frame:
            update = frame["params"]["update"]
            if update.get("sessionUpdate") == "agent_message_chunk":
                text.append(update["content"]["text"])
        elif frame["id"] == call_id:
            return frame["result"], "".join(text)


call("initialize", {"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "demo", "version": "1.0"}}, 1)
session_id = call("session/new", {"cwd": ".", "mcpServers": []}, 2)[0]["sessionId"]
print(call("session/prompt", {
    "sessionId": session_id,
    "prompt": [{"type": "text", "text": "你好"}],
}, 3))
proc.stdin.close()
```

第二轮可以在**另一个进程**里先用 `session/load` 恢复同一个 `sessionId`，再发 `session/prompt`，Agent 会带着之前的上下文继续。

## IDE 配置

### Zed

编辑 `~/.config/zed/settings.json`：

```json
{
  "agent_servers": {
    "Agentica": {
      "command": "agentica",
      "args": ["acp"],
      "env": {}
    }
  }
}
```

### 其他编辑器

同样是让它启动 `agentica acp` 并按 stdio 读写 NDJSON。`env` 里按需放模型密钥；本机已配好 `~/.agentica` 时留空即可。

## 在 Agent Autoloop 中使用

```json
"agentica": {
  "name": "Agentica",
  "worker": "my-mac",
  "use": "acp",
  "command": ["agentica", "acp"]
}
```

中台每轮会起一个新的 Agent 进程，靠 `loadSession` 能力走 `session/load` 接回原会话，不需要额外写插件。

## 下一步

- [MCP 集成](mcp.md) -- Model Context Protocol
- [CLI 终端](../getting-started/terminal.md) -- 命令行使用
- [Agent 概念](../concepts/agent.md) -- Agent 核心概念
