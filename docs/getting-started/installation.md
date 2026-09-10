# Installation

## 环境要求

- [`uv`](https://docs.astral.sh/uv/guides/tools/)（产品安装走它；开发本仓库也可以只用 Python >= 3.10）
- 至少一个 LLM 提供商的 API Key

没有 uv 时用独立安装器（二进制进 `~/.local/bin`，不绑任何一个 Python）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS：brew install uv
# Windows：powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

不要 `pip3 install uv`：uv 自己会被焊到当前 `pip3` 的解释器上，换 conda / 卸那个 Python，`uv` 就没了。`uv tool install` 自己带一份隔离的 Python，不占用系统 / conda。装完命令不在 PATH 时跑一次 `uv tool update-shell`。

## 装产品（推荐）

`agentica` 是 CLI，`agentica-gateway` 是 Web / Desktop 的后端，两个都是命令行产品，用 `uv tool` 装进隔离环境。不要用系统自带的 `pip install agentica` 把它们焊到某一个 Python 上。

```bash
# CLI
uv tool install agentica

# Web / Desktop（同一套隔离环境，多出 agentica-gateway）
uv tool install "agentica[gateway]"
```

已经装过 CLI、再补 Web：

```bash
uv tool install --force "agentica[gateway]"
```

升级 / 卸载：

```bash
uv tool upgrade agentica
uv tool uninstall agentica
```

CLI 里的 `/upgrade` 仍是对**当前解释器**跑 `pip install -U`。`uv tool` 装的请用上面的 `uv tool upgrade`，不要用 `/upgrade` 去碰系统 pip。

IM 等 extras 写进同一个 spec（zsh 下括号要加引号）：

```bash
uv tool install --force "agentica[gateway,wechat,telegram]"
```

Desktop 安装包第一次打开时，如果本机还没有 `agentica-gateway`，会自己用 uv 装一份托管 runtime。已经 `uv tool install` 过的继续用你原来的。

## 当库用（自己的项目）

把 Agentica 嵌进另一个 Python 项目，用项目里的 `uv add`，不要 `pip install` 进系统 Python：

```bash
uv add agentica
uv add "agentica[rag]"          # 可选 extras
```

## 开发本仓库

改 Agentica 源码才把包装进当前解释器：

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install -e .
# 或：uv pip install -e ".[dev,gateway]"
```

开发模式下，代码修改立即生效，无需重新安装。

### 可选依赖

产品安装用 extras（见上）。开发 / 当库用时再按需加：

```bash
# 浏览器工具
uv add playwright
playwright install chromium

# RAG / 向量数据库
uv add lancedb                  # LanceDb（推荐本地向量存储）
uv add qdrant-client            # QdrantVectorDb
uv add chromadb                 # ChromaDb

# MCP 协议
uv add mcp                      # McpTool（Model Context Protocol）

# 本地模型
# Ollama 无需 Python 包：https://ollama.ai

# 文档解析
uv add pypdf
uv add python-docx

# 评测
uv pip install -e ".[dev]"
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

完整 provider 列表见 `agentica/__init__.py`（顶层 `XxxChat` 工厂函数）。

### 进阶：多 Provider 组合（不需要新 env，只用 Python）

`auxiliary_model` 和 `fallback_models` 是**对象传参**，不是 env，所以多 provider 协作时只需各自 export 自己的 key，构造时分别注入：

**A. Auxiliary Model — 用便宜小模型跑副任务**（context 压缩、记忆抽取、用户纠正分类等）

```python
from agentica import Agent, OpenAIChat, DeepSeekChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),                          # 主流程读 OPENAI_API_KEY
    auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),   # 副任务读 DEEPSEEK_API_KEY
)
# export OPENAI_API_KEY=...   # 主流程
# export DEEPSEEK_API_KEY=... # auxiliary
```

**B. Fallback Models — 生产高可用**（content_filter / 5xx / 429 / timeout 自动跳到下一个）

```python
from agentica import Agent, OpenAIChat, DeepSeekChat, ZhipuAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    fallback_models=[
        DeepSeekChat(id="deepseek-v4-flash"),
        ZhipuAIChat(id="glm-4.7-flash"),
    ],
)
# 三家各 export 一份；RunResponse.model 反映实际应答的 provider
```

**C. main + auxiliary + fallback 全开**

```python
from agentica import Agent, OpenAIChat, DeepSeekChat, ZhipuAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),
    fallback_models=[ZhipuAIChat(id="glm-4.7-flash")],
)
```

> 每个工厂内部硬编码了 `base_url` 与默认 env 名（如 `DeepSeekChat` 读 `DEEPSEEK_API_KEY`）。私有部署 / 代理传 `base_url=` 显式覆盖即可。完整工厂列表见 `agentica/__init__.py`。

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
# 产品（uv tool 装的走这条，不要用当前 python -c import）
agentica --version
agentica --query "你好"          # 需要配置 API Key

# 当库 / 开发本仓库
python -c "import agentica; print(agentica.__version__)"
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
- [模型提供商](../guides/models.md) -- 模型配置指南
