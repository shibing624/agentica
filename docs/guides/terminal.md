# CLI 终端指南

Agentica 提供功能丰富的命令行界面，支持交互式对话、工具调用、文件引用等功能。

<img src="../assets/cli_snap.png" width="700" alt="CLI Screenshot" />

## 基本用法

```bash
# 交互模式（默认）
agentica

# 单次查询
agentica --query "解释什么是 RAG"

# 指定模型
agentica --model_provider zhipuai --model_name glm-4.7-flash

# 使用 OpenAI 模型
agentica --model_provider openai --model_name gpt-4o

# 使用 DeepSeek
agentica --model_provider deepseek --model_name deepseek-chat
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query`, `-q` | 单次查询内容 | — |
| `--model_provider` | 模型提供商 | `openai` |
| `--model_name` | 模型名称 | `gpt-4o-mini` |
| `--tools` | 启用的工具列表 | — |
| `--debug` | 调试模式 | `False` |

## 启用工具

```bash
# 启用搜索和 Shell 工具
agentica --tools baidu_search shell

# 启用多个工具
agentica --tools baidu_search shell wikipedia weather
```

## 交互模式功能

### 文件引用

使用 `@filename` 在对话中引用文件内容：

```
> @main.py 这段代码有什么问题？
> @README.md 帮我优化这个文档
```

### 内置命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清除对话历史 |
| `/exit` | 退出 |

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 中止当前响应（会调用 `agent.cancel()`） |
| `Ctrl+D` | 退出 |

## ACP 模式

启动 ACP (Agent Client Protocol) 服务器，与 IDE 集成：

```bash
agentica acp
```

### IDE 配置

**Zed** — 编辑 `~/.config/zed/settings.json`：

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

**JetBrains** — 编辑 `~/.jetbrains/acp.json`：

```json
{
  "agent_servers": {
    "Agentica": {
      "command": "agentica",
      "args": ["acp"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## 流式输出

CLI 默认使用流式输出，支持：

- **内容流式输出** — 打字机效果实时显示
- **思考过程** — 推理模型的思考步骤
- **工具调用展示** — 显示工具调用名称和参数
- **工具结果预览** — Claude Code 风格的结果摘要显示
- **子任务进度** — 子 Agent 执行时的进度汇报

## 下一步

- [快速入门](../quickstart.md) — 安装和基础用法
- [工具系统](tools.md) — 了解可用工具
- [Agent 概念](../concepts/agent.md) — Agent 核心概念
