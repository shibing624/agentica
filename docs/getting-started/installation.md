# Installation

## 环境要求

- **Python >= 3.10**（推荐 3.12）
- 至少一个 LLM 提供商的 API Key

## 安装

### 从 PyPI 安装（推荐）

```bash
pip install -U agentica
```

### 从源码安装（开发模式）

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install -e .
```

开发模式下，代码修改立即生效，无需重新安装。

### 可选依赖

Agentica 的核心功能不需要额外依赖，部分工具和功能需要单独安装：

```bash
# 搜索工具
pip install duckduckgo-search       # DuckDuckGoTool
pip install exa-py                  # SearchExaTool

# 浏览器工具
pip install playwright              # BrowserTool
playwright install chromium

# RAG / 向量数据库
pip install lancedb                 # LanceDb（推荐本地向量存储）
pip install qdrant-client           # QdrantVectorDb
pip install chromadb                # ChromaDb

# MCP 协议
pip install mcp                     # McpTool（Model Context Protocol）

# 本地模型
# Ollama 无需 pip，直接下载安装：https://ollama.ai

# 文档解析
pip install pypdf                   # PDF 解析
pip install python-docx             # Word 文档

# 评测
pip install agentica[dev]           # 开发工具 + 测试依赖
```

## 配置 API Key

### 第一步：选一个 Provider，导出对应环境变量

90% 的用户只需要 **一个 provider** 的 API key 就够了。每个 provider 用各自专属的环境变量名：

| Provider | 推荐场景 | 环境变量 | 备注 |
|---|---|---|---|
| 智谱 ZhipuAI | **零成本起步**（glm-4.7-flash 免费、128k、支持工具调用） | `ZAI_API_KEY` | 也接受 `ZHIPUAI_API_KEY` |
| OpenAI | 生态最完整 | `OPENAI_API_KEY` | |
| Anthropic Claude | 长上下文 / 推理 | `ANTHROPIC_API_KEY` | |
| DeepSeek | 性价比 | `DEEPSEEK_API_KEY` | |
| Moonshot Kimi | 中文长文本 | `MOONSHOT_API_KEY` | |
| 通义 Qwen / DashScope | 阿里云 | `DASHSCOPE_API_KEY` | |
| 火山引擎 Ark（豆包系列） | 字节家 | `ARK_API_KEY` | 模型 ID 形如 `doubao-1.5-pro-32k` |
| xAI Grok | | `XAI_API_KEY` | |
| OpenRouter（聚合多家） | | `OPENROUTER_API_KEY` | |

```bash
# 选你需要的那一个就够了
export ZAI_API_KEY="your-api-key"          # 推荐：智谱免费 glm-4.7-flash
# export OPENAI_API_KEY="sk-xxx"
# export ANTHROPIC_API_KEY="sk-ant-xxx"
# export DEEPSEEK_API_KEY="your-api-key"
# export ARK_API_KEY="your-api-key"        # 火山引擎，跑豆包模型
```

完整 provider 列表见 `agentica/model/providers.py`（每条 `api_key_env`）。

### 进阶：多 Provider 组合（不需要新 env，只用 Python）

`auxiliary_model` 和 `fallback_models` 是**对象传参**，不是 env，所以多 provider 协作时只需各自 export 自己的 key，构造时分别注入：

**A. Auxiliary Model — 用便宜小模型跑副任务**（context 压缩、记忆抽取、用户纠正分类等）

```python
from agentica import Agent, OpenAIChat
from agentica.model.providers import create_provider

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),                   # 主流程读 OPENAI_API_KEY
    auxiliary_model=create_provider(                 # 副任务读 DEEPSEEK_API_KEY
        "deepseek",
        id="deepseek-v4-flash",
        api_key="sk-xxx",                            # 不传则从环境变量读
        base_url="https://api.deepseek.com",         # 可选
    ),
)
# export OPENAI_API_KEY=...   # 主流程
# export DEEPSEEK_API_KEY=... # auxiliary
```

**B. Fallback Models — 生产高可用**（content_filter / 5xx / 429 / timeout 自动跳到下一个）

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    fallback_models=[
        create_provider("deepseek", id="deepseek-v4-flash", api_key="sk-xxx"),
        create_provider("zhipuai",  id="glm-4.7-flash",     api_key="sk-xxx"),
    ],
)
# 三家各 export 一份；RunResponse.model 反映实际应答的 provider
```

**C. main + auxiliary + fallback 全开**

```python
agent = Agent(
    model=create_provider("openai",   id="gpt-4o",            api_key="sk-xxx"),
    auxiliary_model=create_provider("deepseek", id="deepseek-v4-flash", api_key="sk-xxx"),
    fallback_models=[
        create_provider("zhipuai", id="glm-4.7-flash", api_key="sk-xxx"),
    ],
)
```

> `create_provider(slug, id=..., api_key=..., base_url=...)` 的 4 个参数：`slug` 选 provider，`id` 指定具体模型名，`api_key` 不传则按 provider 默认环境变量读，`base_url` 仅在私有部署 / 代理时覆盖。完整 slug 列表见 `agentica/model/providers.py`。

> **同 provider 复用**：如果 main / auxiliary / fallback 都在同一家（比如全用智谱不同 size），只需一份 env，所有 Model 实例共享。

### `.env` 文件（替代 shell export）

在项目目录或 `~/.agentica/` 放 `.env`，启动时自动加载：

```ini
# ~/.agentica/.env
ZAI_API_KEY=your-api-key
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=your-api-key
```

### 代码内直接传 `api_key`（最显式）

```python
from agentica import Agent, OpenAIChat

agent = Agent(
    model=OpenAIChat(
        id="gpt-4o",
        api_key="sk-xxx",
        base_url="https://...",    # 代理 / 私有部署
    )
)
```

## 验证安装

```bash
# 检查版本
python -c "import agentica; print(agentica.__version__)"

# 运行 CLI（需要配置 API Key）
agentica --query "你好"
```

## 免费快速入门（零成本）

智谱 AI 的 `glm-4.7-flash` 模型免费，支持工具调用和 128k 上下文，适合快速体验：

```bash
# 1. 注册并获取免费 API Key：https://open.bigmodel.cn/
export ZAI_API_KEY="your-free-key"

# 2. 运行
agentica --model_provider zhipuai --model_name glm-4.7-flash
```

## 使用 Ollama 本地模型（无需 API Key）

```bash
# 1. 安装 Ollama：https://ollama.ai
# 2. 下载模型
ollama pull llama3.1
# 3. 运行
agentica --model_provider ollama --model_name llama3.1
```

代码中使用：

```python
from agentica import Agent
from agentica.model.ollama import OllamaChat

agent = Agent(model=OllamaChat(id="llama3.1"))
result = agent.run_sync("你好")
print(result.content)
```

## 下一步

- [快速入门](quickstart.md) -- 5 分钟上手第一个 Agent
- [CLI 终端](terminal.md) -- 命令行交互模式全功能介绍
- [模型提供商](../guides/models.md) -- 20+ 模型配置指南
