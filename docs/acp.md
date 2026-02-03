# Agentica ACP (Agent Client Protocol)

Agentica ACP 实现让 Agentica 可以作为 ACP 兼容的 Agent 与 IDE 集成，提供类似 Claude Code / Roo Code / Kimi Code 的 AI 编程助手体验。

## 什么是 ACP？

ACP (Agent Client Protocol) 是一个标准化的协议，类似于 LSP (Language Server Protocol)，用于 IDE 与 AI Agent 之间的通信。

```
┌─────────────┐      JSON-RPC over stdio      ┌─────────────┐
│   IDE       │  ◄────────────────────────►   │  Agentica   │
│  (Client)   │                               │   ACP       │
└─────────────┘                               │   Server    │
      │                                        └──────┬──────┘
      │                                               │
      │                                        ┌──────▼──────┐
      └──────────────────────────────────────► │   Agent     │
                                               │   Engine    │
                                               └─────────────┘
```

**核心特性**:
- **通信方式**: JSON-RPC 2.0 over stdio
- **Session 管理**: 多会话支持，独立的上下文和消息历史
- **流式输出**: 支持实时进度通知
- **工具系统**: 文件操作、命令执行、代码搜索等
- **优势**: 一次实现，多处使用（支持 Zed、JetBrains、VSCode 等）

## 快速开始

### 命令行启动

```bash
# 直接启动 ACP 服务器
agentica acp

# 或者使用 Python
python -m agentica.cli acp
```

### IDE 配置

#### Zed

编辑 `~/.config/zed/settings.json`:

```json
{
  "agent_servers": {
    "Agentica": {
      "type": "custom",
      "command": "agentica",
      "args": ["acp"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### JetBrains (IntelliJ IDEA, PyCharm, WebStorm)

编辑 `~/.jetbrains/acp.json`:

```json
{
  "agent_servers": {
    "Agentica": {
      "command": "/path/to/agentica",
      "args": ["acp"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Session 管理

ACP 的核心是 Session 管理，每个用户交互都在独立的 Session 中进行：

```python
from agentica.acp import SessionManager, ACPSession

# 创建 Session 管理器
manager = SessionManager()

# 创建新 Session
session = manager.create_session(mode="chat")
print(f"Session ID: {session.id}")  # 例如: "a1b2c3d4"

# 添加消息
session.add_message("user", "Hello, can you help me refactor this code?")
session.add_message("assistant", "Of course! Let me analyze the code.")

# 更新状态
session.update_status("running")

# 获取会话信息
info = session.to_dict()
print(f"Messages: {info['message_count']}")
print(f"Status: {info['status']}")

# 清理
manager.delete_session(session.id)
```

### Session 模式

- `default` - 默认模式，通用对话
- `chat` - 聊天模式
- `plan` - 规划模式，用于复杂任务分解
- `edit` - 编辑模式，专注于代码编辑

### Session 状态

- `created` - 已创建
- `running` - 运行中
- `paused` - 已暂停
- `completed` - 已完成
- `error` - 出错
- `cancelled` - 已取消

## 可用工具

ACP 模式下暴露以下工具给 IDE：

| 工具 | 描述 | 参数 |
|------|------|------|
| `read_file` | 读取文件内容 | `file_path`, `offset`, `limit` |
| `write_file` | 写入文件 | `file_path`, `content` |
| `edit_file` | 编辑文件 | `file_path`, `old_string`, `new_string` |
| `ls` | 列出目录 | `directory` |
| `glob` | 文件模式匹配 | `pattern`, `path` |
| `grep` | 文本搜索 | `pattern`, `path`, `glob_pattern` |
| `execute` | 执行命令 | `command` |
| `web_search` | 网页搜索 | `queries`, `max_results` |

## 协议方法

### Core Methods

| 方法 | 描述 |
|------|------|
| `initialize` | 初始化连接 |
| `shutdown` | 优雅关闭 |
| `exit` | 退出服务器 |

### Tool Methods

| 方法 | 描述 |
|------|------|
| `tools/list` | 列出可用工具 |
| `tools/call` | 调用工具 |

### Session Methods (核心)

| 方法 | 描述 |
|------|------|
| `session/new` | 创建新会话 |
| `session/prompt` | 发送提示并获取响应 |
| `session/load` | 加载会话 |
| `session/cancel` | 取消会话 |
| `session/delete` | 删除会话 |
| `session/list` | 列出的有会话 |

### Notification Methods

| 方法 | 描述 |
|------|------|
| `notifications/progress` | 进度更新 |
| `notifications/complete` | 完成通知 |
| `notifications/session/update` | 会话更新 |

## 流式输出

ACP 支持流式输出，实时通知客户端进度：

```python
from agentica.acp import ACPHandlers

handlers = ACPHandlers()

# 创建会话
session = handlers._session_manager.create_session()

# 处理带流式输出的提示
result = handlers.handle_session_prompt({
    "sessionId": session.id,
    "prompt": "Refactor this code",
    "stream": True
})

# 流式输出会自动发送 notifications/session/update 通知
```

## 代码示例

### 基本用法

```python
from agentica.acp import ACPServer

# 创建并启动 ACP 服务器
server = ACPServer()
server.run()
```

### 使用自定义 Agent

```python
from agentica.acp import ACPServer
from agentica import DeepAgent, OpenAIChat

# 创建自定义 Agent
agent = DeepAgent(
    model=OpenAIChat(),
    name="MyACPAgent",
)

# 启动 ACP 服务器
server = ACPServer(agent=agent)
server.run()
```

### 使用自定义模型

```python
from agentica.acp import ACPServer
from agentica import ZhipuAI

# 使用自定义模型
model = ZhipuAI()
server = ACPServer(model=model)
server.run()
```

### Session 管理示例

```python
from agentica.acp import SessionManager, SessionStatus

# 创建管理器
manager = SessionManager(max_sessions=100)

# 创建会话
session = manager.create_session(
    mode="plan",
    initial_context={"project": "my-app", "language": "python"}
)

# 管理会话
sessions = manager.list_sessions()
stats = manager.get_stats()

# 清理旧会话
manager._cleanup_old_sessions(max_age_hours=24)
```

## 架构

```
agentica/acp/
├── __init__.py       # 模块导出
├── types.py          # ACP 数据模型 (ACPRequest, ACPResponse, ACPTool, etc.)
├── protocol.py       # JSON-RPC 协议处理器
├── handlers.py       # 方法处理器 (initialize, session/*, tools/*)
├── session.py        # Session 管理 (SessionManager, ACPSession)
├── server.py         # ACP Server 主类
```

## 实现细节

### JSON-RPC 消息格式

```
Content-Length: <length>\r\n
\r\n
<json_body>
```

### 请求格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "session/prompt",
  "params": {
    "sessionId": "abc123",
    "prompt": "Hello",
    "stream": true
  }
}
```

### 响应格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "sessionId": "abc123",
    "content": "Hi there!",
    "status": "completed"
  }
}
```

### 通知格式 (无 id)

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "sessionId": "abc123",
    "type": "content",
    "content": "Processing..."
  }
}
```

### 错误格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

## 错误码

| 错误码 | 描述 |
|--------|------|
| -32700 | Parse error |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |
| -32000 ~ -32099 | Server error |

## 测试

运行测试：

```bash
# 运行所有 ACP 测试
python -m pytest tests/test_acp.py -v

# 运行特定测试
python -m pytest tests/test_acp.py::TestACPSession -v

# 运行示例
python examples/acp_demo/test_acp.py
```

## 参考资料

- [ACP Specification](https://zed.dev/acp)
- [JetBrains ACP Docs](https://www.jetbrains.com/help/ai-assistant/acp.html)
- [OpenHands ACP Integration](https://docs.openhands.dev/openhands/usage/cli/ide/overview)
- [i-am-bee/acp](https://github.com/i-am-bee/acp) - ACP Python SDK
