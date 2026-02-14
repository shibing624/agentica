# Agentica Examples

Agentica 是一个 **Async-First** 的 Python AI Agent 框架。所有核心方法原生 async，同步调用通过 `_sync()` 适配器。

```
核心 API 四件套:
  await agent.run(...)           # async 非流式
  async for chunk in agent.run_stream(...)  # async 流式
  agent.run_sync(...)            # sync 适配器
  for chunk in agent.run_stream_sync(...)   # sync 流式适配器
```

## 快速开始

### 安装

```bash
pip install agentica
```

### 运行第一个示例

```bash
# async 原生（推荐）
python examples/basic/01_hello_world.py

# 5 行代码启动一个 Agent
python -c "
import asyncio
from agentica import Agent

asyncio.run(Agent().run('一句话介绍北京').then(print))
"
```

---

## 示例目录

### 1. 基础用法 (`basic/`)

从这里开始学习 Agentica 的核心概念。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [hello_world.py](basic/01_hello_world.py) | 最简单的 Agent：`await agent.run()` | `Agent`, `run()`, `asyncio.run()` |
| 02 | [custom_prompt.py](basic/02_custom_prompt.py) | 自定义系统提示词、消息列表输入 | `instructions`, `description`, `messages` |
| 03 | [stream_output.py](basic/03_stream_output.py) | 五种流式输出方式对比 | `run_stream()`, `run_stream_sync()`, `print_response_stream()` |
| 04 | [structured_output.py](basic/04_structured_output.py) | Pydantic 结构化输出 | `response_model`, `BaseModel` |
| 05 | [multi_turn.py](basic/05_multi_turn.py) | 多轮对话与历史记忆 | `add_history_to_messages` |
| 06 | [vision.py](basic/06_vision.py) | 多模态：图片理解（URL / Base64） | `images`, 多模态输入 |
| 07 | [function_calling_auto_demo.py](basic/07_function_calling_auto_demo.py) | 手动 Loop vs Agent 自动 Loop 对比 | Function Calling, 工具调用循环 |
| 08 | [cli_app.py](basic/08_cli_app.py) | 内置交互式 CLI | `agent.cli_app()` |

### 2. 工具系统 (`tools/`)

为 Agent 添加外部能力。工具可以是 sync 或 async 函数，框架自动处理。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [custom_tool.py](tools/01_custom_tool.py) | 自定义函数工具 + 类工具 | `tools=[func]`, `Tool` 基类 |
| 02 | [async_tool.py](tools/02_async_tool.py) | async 工具（原生 `await`，零阻塞） | `async def` 工具, `run_in_executor` 自动包装 |
| 03 | [web_search.py](tools/03_web_search.py) | 网页搜索工具 | `BaiduSearchTool`, `SearchSerperTool` |
| 04 | [code_execution.py](tools/04_code_execution.py) | 代码执行工具 | `ShellTool`, `CodeTool` |
| 05 | [file_operations.py](tools/05_file_operations.py) | 文件操作工具 | `PatchTool`, `ShellTool` |
| 06 | [browser.py](tools/06_browser.py) | 浏览器工具 | `BrowserTool` |

### 3. Agent 设计模式 (`agent_patterns/`)

核心架构模式，充分发挥 async 并行优势。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [agent_as_tool.py](agent_patterns/01_agent_as_tool.py) | Agent 作为工具嵌套调用 | `agent.as_tool()`, 编排器模式 |
| 02 | [parallelization.py](agent_patterns/02_parallelization.py) | **asyncio.gather 并行执行**（含耗时对比） | `asyncio.gather()`, 并行 vs 串行 |
| 03 | [team_collaboration.py](agent_patterns/03_team_collaboration.py) | 多 Agent 团队协作 | `team=[...]`, 自动委派 |
| 04 | [debate.py](agent_patterns/04_debate.py) | 多 Agent 辩论 | 多角色对抗, 主持人总结 |
| 05 | [context_passing.py](agent_patterns/05_context_passing.py) | 上下文传递与共享 | `context={}`, `add_context=True` |

### 4. 安全护栏 (`guardrails/`)

输入/输出验证和安全检查。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [input_guardrail.py](guardrails/01_input_guardrail.py) | 输入检查（关键词过滤、长度限制） | `@input_guardrail`, `GuardrailFunctionOutput` |
| 02 | [output_guardrail.py](guardrails/02_output_guardrail.py) | 输出检查 | `@output_guardrail` |
| 03 | [tool_guardrail.py](guardrails/03_tool_guardrail.py) | 工具输入/输出护栏 | `@tool_input_guardrail`, `@tool_output_guardrail` |

### 5. 记忆系统 (`memory/`)

会话记忆、长期记忆、自动记忆和上下文压缩。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [session_history.py](memory/01_session_history.py) | 会话历史：无历史 vs 多轮对话 vs 会话摘要 | `add_history_to_messages`, `AgentMemory.with_summary()` |
| 02 | [agent_session.py](memory/02_agent_session.py) | Agent-as-Session：隔离、共享记忆、窗口控制 | `AgentMemory`, `num_history_responses`, 共享 memory |
| 03 | [compression.py](memory/03_compression.py) | Token 压缩（对长对话自动截断） | `CompressionManager`, `ToolConfig` |
| 04 | [workspace_memory.py](memory/04_workspace_memory.py) | Workspace 文件记忆：存取、多用户隔离、搜索 | `Workspace`, `MEMORY.md`, `save_memory()` |
| 05 | [auto_memory.py](memory/05_auto_memory.py) | **LLM 自动记忆**：模型自主判断保存 → 新会话自动加载 | `BuiltinMemoryTool`, LLM 自主 tool call |

### 6. 工作区 (`workspace/`)

文件级持久化记忆 + 技能系统。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [basic_workspace.py](workspace/01_basic_workspace.py) | 工作区基础 | `Workspace`, `AGENT.md`, `MEMORY.md` |
| 02 | [skills_usage.py](workspace/02_skills_usage.py) | 技能加载与使用 | `SkillTool`, `SKILL.md` |
| 03 | [memory_search.py](workspace/03_memory_search.py) | 记忆搜索 | `search_memory()` |

### 7. RAG 检索增强 (`rag/`)

基于文档的问答和知识库。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [naive_rag.py](rag/01_naive_rag.py) | 基础 RAG（PDF 知识库） | `Knowledge`, `LanceDb`, `Text2VecEmb` |
| 02 | [advanced_rag.py](rag/02_advanced_rag.py) | 高级 RAG（rerank + 混合检索） | `search_knowledge=True`, agentic RAG |
| 03 | [chat_pdf.py](rag/03_chat_pdf.py) | PDF 对话应用 | 端到端 RAG 应用 |
| 04 | [langchain_integration.py](rag/04_langchain_integration.py) | LangChain 集成 | `LangChainKnowledge` |
| 05 | [llamaindex_integration.py](rag/05_llamaindex_integration.py) | LlamaIndex 集成 | `LlamaIndexKnowledge` |

### 8. 工作流编排 (`workflow/`)

确定性多步骤任务编排。Workflow 是 async-first 的：`run()` 为 async，`run_sync()` 为同步适配器。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [data_pipeline.py](workflow/01_data_pipeline.py) | ETL 数据管道（提取 -> 验证 -> 分析） | `Workflow`, 混合 LLM + Python 步骤 |
| 02 | [investment.py](workflow/02_investment.py) | 投资研究工作流 | 多 Agent 协作工作流 |
| 03 | [news_report.py](workflow/03_news_report.py) | 新闻报道生成 | 搜索 -> 分析 -> 写作 |
| 04 | [code_review.py](workflow/04_code_review.py) | 代码审查工作流 | 多步骤审查 |

### 9. MCP 协议 (`mcp/`)

Model Context Protocol 集成。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [stdio.py](mcp/01_stdio.py) | Stdio 传输 | `MCPConfig`, stdio |
| 02 | [sse_server.py](mcp/02_sse_server.py) / [sse_client.py](mcp/02_sse_client.py) | SSE 传输 | Server / Client |
| 03 | [http_server.py](mcp/03_http_server.py) / [http_client.py](mcp/03_http_client.py) | HTTP 传输 | Server / Client |
| 04 | [json_config.py](mcp/04_json_config.py) | JSON 配置加载 | 配置驱动 |

### 10. 模型提供商 (`model_providers/`)

支持 20+ LLM 提供商，统一 async 接口。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [openai.py](model_providers/01_openai.py) | OpenAI（原生 async） | `OpenAIChat`, `response()`, `response_stream()` |
| 02 | [deepseek.py](model_providers/02_deepseek.py) | DeepSeek（含推理模型） | `DeepSeek`, `reasoning_content` |
| 03 | [zhipuai.py](model_providers/03_zhipuai.py) | 智谱 AI | `ZhipuAI` |
| 04 | [custom_endpoint.py](model_providers/04_custom_endpoint.py) | 自定义 OpenAI 兼容端点 | `OpenAILike`, `base_url` |
| 05 | [litellm.py](model_providers/05_litellm.py) | LiteLLM 统一接口 | `LiteLLM` |

### 11. 技能系统 (`skills/`)

基于 `SKILL.md` 的能力扩展。

| # | 示例 | 描述 |
|---|------|------|
| 01 | [skill_basics.py](skills/01_skill_basics.py) | 技能基础 |
| 02 | [web_research.py](skills/02_web_research.py) | 网络研究技能 |
| 03 | [custom_skill.py](skills/03_custom_skill.py) | 自定义技能 |
| 04 | [skills_with_agent.py](skills/04_skills_with_agent.py) | 技能与 Agent 集成 |

### 12. DeepAgent (`deep_agent/`)

内置工具的增强版 Agent，支持代码执行、文件操作、网页搜索、子任务委派。

| # | 示例 | 描述 | 关键概念 |
|---|------|------|----------|
| 01 | [basic.py](deep_agent/01_basic.py) | 基础用法 | `DeepAgent`, 内置工具 |
| 02 | [file_operations.py](deep_agent/02_file_operations.py) | 文件操作 | `BuiltinFileTool` |
| 03 | [code_execute_demo.py](deep_agent/03_code_execute_demo.py) | 代码执行 | `BuiltinExecuteTool` |
| 04 | [web_search_demo.py](deep_agent/04_web_search_demo.py) | 网页搜索 | `BuiltinWebSearchTool` |
| 05 | [subagent_demo.py](deep_agent/05_subagent_demo.py) | 子 Agent 委派 | `BuiltinTaskTool`, subagent |

### 13. 可观测性 (`observability/`)

监控、追踪和调试。

| # | 示例 | 描述 |
|---|------|------|
| 01 | [langfuse.py](observability/01_langfuse.py) | Langfuse 集成 |
| 02 | [token_tracking.py](observability/02_token_tracking.py) | Token 追踪 |

### 14. ACP 协议 (`acp/`)

Agent Communication Protocol 集成。

| # | 示例 | 描述 |
|---|------|------|
| 01 | [acp_demo.py](acp/acp_demo.py) | ACP 协议演示 |

### 15. 完整应用 (`applications/`)

端到端的应用示例。

| 示例 | 描述 |
|------|------|
| [llm_os/main.py](applications/llm_os/main.py) | LLM OS - 综合 AI 助手 (Streamlit) |
| [deep_research/main.py](applications/deep_research/main.py) | 深度研究助手 |
| [customer_service/main.py](applications/customer_service/main.py) | 客服系统 |

---

## Async-First 架构亮点

Agentica 采用与 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)、[PydanticAI](https://ai.pydantic.dev/) 一致的 async-first 设计。

### 1. 原生 Async API

```python
import asyncio
from agentica import Agent

async def main():
    agent = Agent()
    # 非流式
    response = await agent.run("Hello")
    # 流式
    async for chunk in agent.run_stream("Hello"):
        print(chunk.content, end="")

asyncio.run(main())
```

### 2. 并行工具执行

模型返回多个工具调用时，框架自动通过 `asyncio.gather()` 并行执行：

```python
# 3 个工具调用 → 并行执行，耗时 = max(t1, t2, t3) 而非 t1 + t2 + t3
agent = Agent(tools=[search_web, query_db, call_api])
await agent.run("搜索新闻、查询数据库、调用 API")
```

### 3. 并行 Agent 执行

```python
# 3 个 Agent 并行运行
res1, res2, res3 = await asyncio.gather(
    agent_en.run(text),
    agent_zh.run(text),
    agent_fr.run(text),
)
```

### 4. Sync/Async 工具混用

```python
def sync_tool(x: int) -> str:       # sync → 自动 run_in_executor
    return str(x * 2)

async def async_tool(x: int) -> str:  # async → 直接 await
    await asyncio.sleep(0.1)
    return str(x * 3)

agent = Agent(tools=[sync_tool, async_tool])  # 混用无感
```

### 5. 同步适配器（兼容非 async 场景）

```python
# 脚本 / CLI / 非 async 环境
response = agent.run_sync("Hello")
for chunk in agent.run_stream_sync("Hello"):
    print(chunk.content, end="")
```

---

## 学习路径

### 入门

1. `basic/01_hello_world.py` — 第一个 Agent（`await agent.run()`）
2. `basic/03_stream_output.py` — 流式输出（5 种方式对比）
3. `basic/06_vision.py` — 多模态图片理解
4. `tools/01_custom_tool.py` — 自定义工具

### 进阶

1. `agent_patterns/02_parallelization.py` — **并行执行**（asyncio.gather 性能对比）
2. `tools/02_async_tool.py` — Async 工具（原生无阻塞）
3. `agent_patterns/01_agent_as_tool.py` — Agent 组合
4. `guardrails/01_input_guardrail.py` — 安全护栏
5. `rag/01_naive_rag.py` — RAG 知识库

### 高级

1. `agent_patterns/03_team_collaboration.py` — 多 Agent 团队协作
2. `workflow/01_data_pipeline.py` — 工作流编排
3. `deep_agent/05_subagent_demo.py` — 子 Agent 委派
4. `mcp/01_stdio.py` — MCP 协议集成
5. `applications/llm_os/main.py` — 完整应用

---

## 架构速览

```
┌─────────────────────────────────────────────────────┐
│                    用户 API 层                        │
│  agent.run()          async 非流式                   │
│  agent.run_stream()   async 流式 (AsyncIterator)     │
│  agent.run_sync()     sync 适配器                    │
│  agent.run_stream_sync() sync 流式适配器             │
├─────────────────────────────────────────────────────┤
│                    执行引擎层                         │
│  _run_impl()          唯一 async 执行引擎            │
│  asyncio.gather()     并行工具执行                   │
├─────────────────────────────────────────────────────┤
│                    Model 层 (async-only)             │
│  model.invoke() / invoke_stream()                    │
│  model.response() / response_stream()                │
│  model.run_function_calls()  并行工具调度            │
├─────────────────────────────────────────────────────┤
│                    Tool 层                           │
│  FunctionCall.execute()      async-only              │
│    async func → await        直接调用                │
│    sync func  → run_in_executor  自动包装            │
└─────────────────────────────────────────────────────┘
```

---

## LLM OS

Let's build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

- LLMs are the kernel process of an emerging operating system.
- This process (LLM) can solve problems by coordinating other resources (memory, computation tools).

### Running the LLM OS

```bash
# 1. 安装依赖
pip install agentica streamlit text2vec sqlalchemy lancedb pyarrow

# 2. 配置 API Key
echo "OPENAI_API_KEY=sk-xxx" > .env

# 3. 启动
cd examples/applications/llm_os
streamlit run main.py
```

![llm_os](https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png)

- Open [localhost:8501](http://localhost:8501) to view your LLM OS.
- `Web search`: 北京今天天气?
- `Code execution`: 帮我计算下 [168, 151, 171, 105, 124, 159, 153, 132, 112.2] 的平均值
- `RAG`: 添加 PDF 到知识库，然后提问
