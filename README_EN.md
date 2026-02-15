[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: Build AI Agents
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

**Agentica** is a lightweight Python framework for building AI agents. Async-First architecture with support for tool calling, RAG, multi-agent teams, workflow orchestration, and MCP protocol.

## Installation

```bash
pip install -U agentica
```

## Quick Start

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(model=ZhipuAI())
    result = await agent.run("Describe Beijing in one sentence")
    print(result.content)

asyncio.run(main())
```

```
Beijing is the capital of China, a historic city with over 3,000 years of history, and the nation's political, cultural, and international exchange center.
```

Set up your API keys first:

```bash
export ZHIPUAI_API_KEY="your-api-key"      # ZhipuAI (glm-4.7-flash is free)
export OPENAI_API_KEY="sk-xxx"              # OpenAI
export DEEPSEEK_API_KEY="your-api-key"      # DeepSeek
```

## Features

- **Async-First** — Native async API, `asyncio.gather()` parallel tool execution, sync adapter included
- **20+ Models** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM and more
- **40+ Built-in Tools** — Search, code execution, file operations, browser, OCR, image generation
- **RAG** — Knowledge base management, hybrid retrieval, Rerank, LangChain / LlamaIndex integration
- **Multi-Agent** — Team (dynamic delegation) and Workflow (deterministic orchestration)
- **Guardrails** — Input / output / tool-level guardrails
- **MCP / ACP** — Model Context Protocol and Agent Communication Protocol support
- **Skill System** — Markdown-based skill injection, model-agnostic
- **Multi-Modal** — Text, image, audio, video understanding

## CLI

```bash
agentica --model_provider zhipuai --model_name glm-4.7-flash
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

## Web UI

[agentica-gateway](https://github.com/shibing624/agentica-gateway) provides a web interface, and also supports Feishu App and WeCom direct integration with Agentica.

## Examples

See [examples/](https://github.com/shibing624/agentica/tree/main/examples) for full examples, covering:

| Category | Content |
|----------|---------|
| **Basics** | Hello World, streaming, structured output, multi-turn, multi-modal |
| **Tools** | Custom tools, async tools, search, code execution, file operations |
| **Agent Patterns** | Agent-as-tool, parallel execution, team collaboration, debate |
| **RAG** | PDF Q&A, advanced RAG, LangChain / LlamaIndex integration |
| **Workflows** | Data pipeline, investment research, news reporting, code review |
| **MCP** | Stdio / SSE / HTTP transport, JSON config |
| **Applications** | LLM OS, deep research, customer service |

[→ View full examples directory](https://github.com/shibing624/agentica/blob/main/examples/README.md)

## Documentation

Full documentation: **https://shibing624.github.io/agentica**

## Community & Support

- **GitHub Issues** — [Open an issue](https://github.com/shibing624/agentica/issues)
- **WeChat Group** — Add `xuming624` on WeChat, mention "llm" to join the developer group

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## Citation

If you use Agentica in your research, please cite:

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

## License

[Apache License 2.0](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
