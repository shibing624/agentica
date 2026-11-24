# 记忆 MCP

Agentica 自己的跨会话记忆写在 `MEMORY.md` 和 `memory/*.md` 里，常驻规则写在 `AGENTS.md` 里。这一套不需要另接服务，用法见 [记忆与常驻规则](../concepts/memory.md)。

记忆 MCP 是另一条路：把一个外部记忆服务接到 Agent 的工具列表里。桌面版、本机网页和 CLI 读的是同一份 `~/.agentica/mcp_config.json`。`DeepAgent` 启动时会加载这份文件里启用的服务，下一轮对话就能调用它提供的工具。本机的 `save_memory` / `search_memory` 仍然在，外部服务不会替换它们。

## 在桌面版或网页里添加

打开「插件」，切到 MCP，填服务名称，以及下面两种里的一种：

- **本机命令**：`command` 和参数。Agentica 用 stdio 拉起这个进程。
- **已在运行的地址**：`url`。适合 SSE 或 HTTP 的记忆服务。

保存后当前会话的 agent 会在下一轮重建，新工具从那一轮开始可用。删掉一个服务也是改同一份文件。

## 直接写配置

`~/.agentica/mcp_config.json`：

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "your-memory-mcp"],
      "env": {"MEMORY_PATH": "/path/to/store"}
    }
  }
}
```

已经有地址的服务写成 `"url": "http://127.0.0.1:8081/sse"`，不必再写 `command`。`"enable": false` 会跳过这一项，文件留着但不会加载。工具很多、不想每次都把完整 schema 放进上下文时，给这一项加 `"defer_schema": true`：模型先通过 `use_capability` 查看和调用，真正执行时仍走原来的审批。

项目目录里如果有 `mcp_config.json`（或 `.yaml` / `.yml`），优先用项目这份，而不是家目录那份。

## 和本机记忆怎么选

| 你要的 | 用哪个 |
|---|---|
| 记住这个人的偏好、项目里代码看不出来的事实、每次都要遵守的规矩 | 本机记忆和 `AGENTS.md` |
| 让 Agent 读写另一个记忆库、笔记库或你们自己的服务 | 记忆 MCP |

外部服务返回的内容和本机 `MEMORY.md` 是两套数据。要让两边一致，得在那个服务自己的工具里读写，Agentica 不会自动把 `save_memory` 同步过去。
