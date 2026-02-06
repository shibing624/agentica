---
name: agentica-intro
description: "Agentica 框架介绍。当用户询问 Agentica 的功能、如何创建 Agent、使用工具、配置模型或构建 AI 应用时使用此技能。"
trigger: /agentica-intro
metadata:
  emoji: "🤖"
---

# Agentica Framework

**GitHub**: https://github.com/shibing624/agentica

Agentica 是一个轻量级、功能强大的 Python 框架，用于构建、管理和部署自主 AI 智能体。

## 安装

```bash
pip install -U agentica
```

## 快速开始

```python
from agentica import Agent, ZhipuAI, WeatherTool

agent = Agent(
    model=ZhipuAI(),  # 或 OpenAIChat, DeepSeek, Moonshot 等
    tools=[WeatherTool()],
    add_datetime_to_instructions=True
)

agent.print_response("明天北京天气怎么样？")
```

## 核心组件

### 1. Agent 类型

| Agent | 用途 |
|-------|------|
| `Agent` | 基础智能体，支持工具调用和记忆 |
| `DeepAgent` | 增强智能体，内置文件、Shell、Web 等工具 |

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(
    model=OpenAIChat(id="gpt-4o"),
    work_dir="./project",  # 工作目录
    debug_mode=True,       # 调试模式
)
```

### 2. 支持的模型

```python
from agentica import (
    OpenAIChat,    # GPT-4o, GPT-4o-mini
    ZhipuAI,       # GLM-4-Flash (免费), GLM-4
    DeepSeek,      # DeepSeek-Chat, DeepSeek-Reasoner
    Moonshot,      # Moonshot-v1-8k
    Claude,        # Claude-3.5-Sonnet
)

# 配置 API Key
model = OpenAIChat(
    id="gpt-4o",
    api_key="sk-...",  # 或使用环境变量 OPENAI_API_KEY
)
```

### 3. 内置工具

```python
from agentica.tools import (
    ShellTool,        # Shell 命令执行
    CalculatorTool,   # 数学计算
    WikipediaTool,    # Wikipedia 搜索
    BrowserTool,      # 网页浏览
    DalleTool,        # 图像生成
    WeatherTool,      # 天气查询
    BaiduSearchTool,  # 百度搜索
)

agent = Agent(model=model, tools=[ShellTool(), CalculatorTool()])
```

### 4. Workspace 与 Memory

```python
from agentica import DeepAgent
from agentica.workspace import Workspace

# 创建 Workspace 持久化记忆
workspace = Workspace(user_id="alice")
workspace.initialize()

agent = DeepAgent(model=model, workspace=workspace)
# Agent 自动保存/加载记忆
```

Memory 结构：
```
~/.agentica/workspace/
├── AGENT.md      # Agent 指令
├── skills/       # 自定义技能
└── users/
    └── {user_id}/
        ├── USER.md      # 用户信息
        ├── MEMORY.md    # 长期记忆
        └── memory/      # 每日记忆
```

### 5. 技能系统 (Skills)

Skills 是基于 Prompt Engineering 的能力扩展，通过 SKILL.md 注入指令到 System Prompt。

```python
from agentica.skills import load_skills

registry = load_skills()
# 技能自动注入到 Agent
```

技能目录：
- `{project_root}/.agentica/skills/` - 项目级
- `~/.agentica/skills/` - 用户级
- 内置技能：`agentica-intro`, `commit`, `github`, `skill-creator`

## CLI 使用

```bash
# 交互模式
agentica

# 指定模型
agentica --model_provider deepseek --model_name deepseek-chat

# 添加工具
agentica --tools calculator shell wikipedia

# 单次查询
agentica --query "Python 最佳实践是什么？"
```

CLI 快捷键：
- `Enter` - 提交消息
- `Ctrl+X` - 切换 Shell 模式
- `Ctrl+D` - 退出
- `@file` - 引用文件
- `/help` - 显示命令

## 高级功能

### 多 Agent 协作

```python
from agentica import DeepAgent
from agentica.subagent import Subagent

# 创建专业子 Agent
researcher = Subagent(
    name="researcher",
    description="Research specialist",
    model=model,
)

# 主 Agent 通过 'task' 工具委派任务
main_agent = DeepAgent(
    model=model,
    description="Coordinator",
)
```

### 工作流编排

```python
from agentica.workflow import Workflow

workflow = Workflow(
    agents=[agent1, agent2, agent3],
    description="Research and report workflow",
)
workflow.run("Generate market analysis report")
```

### RAG 知识库

```python
from agentica import Agent
from agentica.knowledge import PDFKnowledgeBase

kb = PDFKnowledgeBase(path="./docs")
agent = Agent(model=model, knowledge=kb)
agent.print_response("文档中关于 X 的内容是什么？")
```

### MCP 协议支持

CLI 自动加载 `mcp_config.json` 配置文件（搜索顺序：当前目录 → 父目录 → `~/.agentica/`）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "timeout": 60.0,
      "enable": true
    },
    "disabled-server": {
      "command": "some-command",
      "enable": false
    }
  }
}
```

- `enable`: 是否启用该 MCP 服务器，默认 `true`，设为 `false` 禁用

代码中使用：
```python
from agentica.tools.mcp_tool import McpTool

# 从配置文件加载（自动过滤 enable=false 的服务器）
mcp_tool = McpTool.from_config()

# 或直接指定
from agentica.mcp import MCPClient
client = MCPClient("stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem"])
agent = Agent(model=model, mcp_clients=[client])
```

## 环境变量

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ZHIPUAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# Workspace 路径（可选）
export AGENTICA_WORKSPACE_DIR="~/.agentica/workspace"
```

## 最佳实践

1. **使用 DeepAgent** - 大多数任务推荐，内置必要工具
2. **设置 work_dir** - 控制文件操作范围
3. **启用 debug_mode** - 开发时查看工具调用
4. **使用 Workspace** - 持久化跨会话记忆
5. **创建 Skills** - 复杂重复工作流封装为技能

## 相关链接

- **GitHub**: https://github.com/shibing624/agentica
- **PyPI**: https://pypi.org/project/agentica/
- **文档**: https://github.com/shibing624/agentica/tree/main/docs
- **示例**: https://github.com/shibing624/agentica/tree/main/examples
- **Web UI**: https://github.com/shibing624/ChatPilot
