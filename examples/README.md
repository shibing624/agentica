# Agentica Examples

本目录包含 Agentica 的各种示例，按功能模块组织，便于学习和参考。

## 快速开始

### 安装

```bash
pip install agentica
```

### 运行第一个示例

```bash
python examples/basic/01_hello_world.py
```

## 示例目录

### 🚀 基础用法 (`basic/`)

从这里开始学习 Agentica 的核心概念。

| 示例 | 描述 | 关键概念 |
|------|------|----------|
| [01_hello_world.py](basic/01_hello_world.py) | 创建并运行最简单的 Agent | `Agent`, `run()` |
| [02_custom_prompt.py](basic/02_custom_prompt.py) | 自定义系统提示词和用户消息 | `instructions`, `messages` |
| [03_stream_output.py](basic/03_stream_output.py) | 流式输出 | `stream=True` |
| [04_structured_output.py](basic/04_structured_output.py) | 结构化输出 (Pydantic) | `response_model` |
| [05_multi_turn.py](basic/05_multi_turn.py) | 多轮对话 | `add_history_to_messages` |
| [06_vision.py](basic/06_vision.py) | 视觉理解 | `images` |

### 🔧 工具系统 (`tools/`)

学习如何为 Agent 添加各种能力。

| 示例 | 描述 |
|------|------|
| [01_custom_tool.py](tools/01_custom_tool.py) | 自定义工具（函数和类） |
| [02_builtin_tools.py](tools/02_builtin_tools.py) | 内置工具概览 |
| [03_web_search.py](tools/03_web_search.py) | 网页搜索工具 |
| [04_code_execution.py](tools/04_code_execution.py) | 代码执行工具 |
| [05_file_operations.py](tools/05_file_operations.py) | 文件操作工具 |
| [06_browser.py](tools/06_browser.py) | 浏览器工具 |

### 🎯 Agent 设计模式 (`agent_patterns/`)

常见的 Agent 架构模式和最佳实践。

| 示例 | 描述 |
|------|------|
| [01_agent_as_tool.py](agent_patterns/01_agent_as_tool.py) | Agent 作为工具 |
| [02_parallelization.py](agent_patterns/02_parallelization.py) | 并行执行 |
| [03_team_collaboration.py](agent_patterns/03_team_collaboration.py) | 团队协作 |
| [04_debate.py](agent_patterns/04_debate.py) | 多Agent辩论 |
| [05_context_passing.py](agent_patterns/05_context_passing.py) | 上下文传递 |

### 🛡️ 安全护栏 (`guardrails/`)

输入/输出验证和安全检查。

| 示例 | 描述 |
|------|------|
| [01_input_guardrail.py](guardrails/01_input_guardrail.py) | 输入检查 |
| [02_output_guardrail.py](guardrails/02_output_guardrail.py) | 输出检查 |
| [03_tool_guardrail.py](guardrails/03_tool_guardrail.py) | 工具护栏 |

### 🧠 记忆系统 (`memory/`)

会话记忆、长期记忆和上下文压缩。

| 示例 | 描述 |
|------|------|
| [01_session_memory.py](memory/01_session_memory.py) | 会话记忆 |
| [02_long_term_memory.py](memory/02_long_term_memory.py) | 长期记忆 (SqliteDb) |
| [03_compression.py](memory/03_compression.py) | Token 压缩 |

### 📚 RAG 检索增强 (`rag/`)

基于文档的问答和知识库。

| 示例 | 描述 |
|------|------|
| [01_naive_rag.py](rag/01_naive_rag.py) | 基础 RAG |
| [02_advanced_rag.py](rag/02_advanced_rag.py) | 高级 RAG (rerank) |
| [03_chat_pdf.py](rag/03_chat_pdf.py) | PDF 对话应用 |
| [04_langchain_integration.py](rag/04_langchain_integration.py) | LangChain 集成 |
| [05_llamaindex_integration.py](rag/05_llamaindex_integration.py) | LlamaIndex 集成 |

### ⚙️ 工作流编排 (`workflow/`)

多步骤任务的编排和执行。

| 示例 | 描述 |
|------|------|
| [01_simple_workflow.py](workflow/01_simple_workflow.py) | 简单工作流入门 |
| [02_investment.py](workflow/02_investment.py) | 投资研究工作流 |
| [03_news_article.py](workflow/03_news_article.py) | 新闻报道生成工作流 |
| [04_novel_writing.py](workflow/04_novel_writing.py) | 小说写作工作流 |

### 🔌 MCP 协议 (`mcp/`)

Model Context Protocol 集成。

| 示例 | 描述 |
|------|------|
| [01_stdio.py](mcp/01_stdio.py) | Stdio 传输 |
| [02_sse_server.py](mcp/02_sse_server.py) | SSE Server |
| [02_sse_client.py](mcp/02_sse_client.py) | SSE Client |
| [03_http_server.py](mcp/03_http_server.py) | HTTP Server |
| [03_http_client.py](mcp/03_http_client.py) | HTTP Client |
| [04_json_config.py](mcp/04_json_config.py) | JSON 配置加载 |

### 🤖 模型提供商 (`model_providers/`)

支持多种 LLM 提供商。

| 示例 | 描述 |
|------|------|
| [01_openai.py](model_providers/01_openai.py) | OpenAI |
| [02_deepseek.py](model_providers/02_deepseek.py) | DeepSeek |
| [03_zhipuai.py](model_providers/03_zhipuai.py) | 智谱 AI |
| [04_custom_endpoint.py](model_providers/04_custom_endpoint.py) | 自定义端点 |
| [05_litellm.py](model_providers/05_litellm.py) | LiteLLM 统一接口 |

### 🎨 技能系统 (`skills/`)

基于 SKILL.md 的能力扩展。

| 示例 | 描述 |
|------|------|
| [01_skill_basics.py](skills/01_skill_basics.py) | 技能基础 |
| [02_web_research.py](skills/02_web_research.py) | 网络研究技能 |
| [03_custom_skill.py](skills/03_custom_skill.py) | 自定义技能 |

### ⏱️ 分布式工作流 (`temporal/`)

Temporal 集成，支持持久化执行。

| 示例 | 描述 |
|------|------|
| [01_worker.py](temporal/01_worker.py) | Worker 启动 |
| [02_client.py](temporal/02_client.py) | Client 使用 |
| [03_parallel_workflow.py](temporal/03_parallel_workflow.py) | 并行工作流 |

### 💪 DeepAgent (`deep_agent/`)

内置工具的增强版 Agent。

| 示例 | 描述 |
|------|------|
| [01_basic.py](deep_agent/01_basic.py) | 基础用法 |
| [02_file_operations.py](deep_agent/02_file_operations.py) | 文件操作 |
| [03_code_assistant.py](deep_agent/03_code_assistant.py) | 代码助手 |
| [04_research_assistant.py](deep_agent/04_research_assistant.py) | 研究助手 |

### 📊 可观测性 (`observability/`)

监控、追踪和调试。

| 示例 | 描述 |
|------|------|
| [01_langfuse.py](observability/01_langfuse.py) | Langfuse 集成 |
| [02_token_tracking.py](observability/02_token_tracking.py) | Token 追踪 |

### 🖥️ 命令行工具 (`cli/`)

交互式命令行界面。

| 示例 | 描述 |
|------|------|
| [01_cli_demo.py](cli/01_cli_demo.py) | CLI 演示 |

### 🏢 完整应用 (`applications/`)

端到端的应用示例。

| 示例 | 描述 |
|------|------|
| [llm_os/main.py](applications/llm_os/main.py) | LLM OS - 综合AI助手 |
| [research_bot/main.py](applications/research_bot/main.py) | 研究机器人 |
| [customer_service/main.py](applications/customer_service/main.py) | 客服系统 |

---

## 学习路径

### 入门级

1. `basic/01_hello_world.py` - 第一个 Agent
2. `basic/02_custom_prompt.py` - 自定义提示词
3. `tools/01_custom_tool.py` - 添加工具
4. `memory/01_session_memory.py` - 会话记忆

### 进阶级

1. `agent_patterns/01_agent_as_tool.py` - Agent 组合
2. `agent_patterns/02_parallelization.py` - 并行执行
3. `guardrails/01_input_guardrail.py` - 安全护栏
4. `rag/02_advanced_rag.py` - 高级 RAG

### 高级

1. `workflow/02_investment.py` - 复杂工作流
2. `temporal/01_worker.py` - 分布式执行
3. `applications/llm_os/main.py` - 完整应用

---

## LLM OS

Let's build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

### The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

- LLMs are the kernel process of an emerging operating system.
- This process (LLM) can solve problems by coordinating other resources (memory, computation tools).
- The LLM OS:
  - [x] Can read/generate text
  - [x] Has more knowledge than any single human about all subjects
  - [x] Can browse the internet
  - [x] Can use existing software infra (calculator, python, mouse/keyboard)
  - [x] Can see and generate images and video
  - [x] Can hear and speak, and generate music
  - [x] Can think for a long time using a system 2
  - [x] Can "self-improve" in domains
  - [x] Can be customized and fine-tuned for specific tasks
  - [x] Can communicate with other LLMs


## Running the LLM OS:

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Install libraries

```shell
pip install agentica streamlit text2vec sqlalchemy lancedb pyarrow
```

### 3. Export credentials

- Our initial implementation uses GPT-4o, so export your OpenAI API Key in the `../.env` file

```shell
OPENAI_API_KEY=***
```

### 4. Run the LLM OS App

```shell
cd examples/applications/llm_os
streamlit run main.py
```

![llm_os](https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png)

- Open [localhost:8501](http://localhost:8501) to view your LLM OS.
- Add a blog URL to knowledge base: https://blog.samaltman.com/gpt-4o
- Ask: What is gpt-4o?
- `Web search`: 北京今天天气?
- `Code execution`: 帮我计算下 [168, 151, 171, 105, 124, 159, 153, 132, 112.2] 的平均值
- `File operations`: 列出当前目录下的所有文件
