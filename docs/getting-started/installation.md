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

从 1.5.1 起，Web 后端是单独的发行包 `agentica-gateway`（含 Web UI、IM 渠道、cron），和 `agentica` 同版本发布。`agentica[gateway]` 就是「`agentica` + 同版本的 `agentica-gateway[feishu]`」，上面的命令不用改。几点区别：

- `uv tool install agentica-gateway` 也能装，但只会暴露 `agentica-gateway` 一个命令；要 CLI 和 Web 两个命令，用 `"agentica[gateway]"`。
- 不用 uv 时：`pip install agentica-gateway`（加飞书：`"agentica-gateway[feishu]"`，全部渠道：`"agentica-gateway[all]"`）。
- 只装 `agentica` 时运行 `agentica-gateway`，会提示你装哪个包，而不是报 `command not found`。

Desktop 安装包自带 Python 和同版本的 `agentica`、`agentica-gateway`，打开即用，不需要另装 Python、Node、uv，也不用联网。下载和第一次被系统拦住时的处理见 [下载桌面版](https://shibing624.github.io/agentica/desktop/)。它和 `uv tool install` 装的 CLI 共用 `~/.agentica` 数据，但各用各的可执行包。

把 Agentica 嵌进自己的 Python 项目，见 [5 分钟上手 (SDK)](quickstart.md)。改这个仓库的源码，见文末。

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

副模型、备用模型和审查模型写在 `~/.agentica/config.yaml` 的 profile 里，见 [配置文件](../guides/config.md)。同一家厂商的主模型和副模型共用一把 key。

在 Python 里构造 `Agent` 时再传模型对象，见 [模型与提供商](../concepts/model.md)。

### `.env` 文件（替代 shell export）

在项目目录或 `~/.agentica/` 放 `.env`，启动时自动加载：

```ini
# ~/.agentica/.env
ZAI_API_KEY=your-api-key
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=your-api-key
```

## 验证安装

```bash
agentica --version
agentica --query "你好"          # 需要配置 API Key
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

## 改这个仓库

只在改 Agentica 源码时把包装进当前解释器：

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install -e .                         # CLI
pip install -e ./gateway                 # 再加 Web 后端，改 gateway / Web 时需要
# 或：uv pip install -e ".[dev]" -e "./gateway[all]"
cd web && npm ci && npm run build        # Web UI → web/dist，网关直接读这里
```

代码改完不用重装；前端改完重新 `npm run build` 再刷新。不要用 `pip install -e ".[gateway]"`，那个 extra 会去 PyPI 装 `agentica-gateway`，装不到你本地的网关代码。跑测试再装 `uv pip install -e ".[dev]"`。

仓库是 `agentica/`（CLI）、`gateway/`（Web 后端）、`web/`、`desktop/`、`sdk-ts/`。版本由 `python scripts/versions.py set X.Y.Z` 一次改齐。

## 下一步

- [CLI 终端](terminal.md) — 命令行交互模式
- [Web 与桌面端](../advanced/gateway.md) — 本地网页与桌面应用
- [配置文件](../guides/config.md) — `~/.agentica/config.yaml`
- [5 分钟上手 (SDK)](quickstart.md) — 嵌进自己的 Python 项目（在「开发」里）
