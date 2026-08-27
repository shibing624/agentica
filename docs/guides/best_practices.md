# Agentica 最佳实践

> Agent 开发的推荐模式、技巧和常见问题解决方案

## 目录

- [Agent 配方（Recipes）](#agent-配方recipes)
- [Agent 设计原则](#agent-设计原则)
- [提示词工程](#提示词工程)
- [工具使用](#工具使用)
- [内存管理](#内存管理)
- [历史消息定制](#历史消息定制)
- [RAG 最佳实践](#rag-最佳实践)
- [多轮对话](#多轮对话)
- [团队协作](#团队协作)
- [性能优化](#性能优化)
- [错误处理](#错误处理)
- [生产部署](#生产部署)
- [常见问题](#常见问题)

---

## Agent 配方（Recipes）

`Agent` 参数很多，但 90% 的场景用以下 5 种模板就够。直接 copy 改名字。

### 1. 一次性脚本（最简）

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
print(agent.run_sync("一句话介绍北京").content)
```

适用：CLI 工具、Jupyter 实验、单次调用。

### 2. 多轮对话

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=5,         # 滑窗大小，控制 context 预算
    instructions="You are a helpful assistant.",
)
```

适用：聊天机器人、客服 Bot、需要记住上下文的对话型 Agent。

### 3. 工具型 Agent（自定义工具组合）

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

WORK_DIR = "./workspace"
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        BuiltinWebSearchTool(),
        BuiltinFileTool(work_dir=WORK_DIR),
        BuiltinExecuteTool(work_dir=WORK_DIR),
    ],
    instructions="You can search the web, read/write files, and run shell commands.",
)
```

适用：定制化场景，只装需要的工具。**注意 `BuiltinFileTool` / `BuiltinExecuteTool` 需要传 `work_dir`，否则文件操作没有沙箱根目录**。

### 4. 多用户 + 长期记忆 + 会话归档

每个用户一个 Agent 实例。**不要**在 `run()` 里切 `user_id`——hooks 状态、auto_archive、记忆抽取都依赖 workspace 当前 user，运行中切会写错位置。

```python
from agentica import Agent, OpenAIChat, Workspace, WorkspaceMemoryConfig

def create_agent(user_id: str) -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=Workspace("~/.agentica/workspace", user_id=user_id),
        session_id=user_id,                       # 会话日志按 user 切到 ~/.agentica/projects/<slug>/{user_id}.jsonl
        enable_long_term_memory=True,             # ← 必须显式开启，否则下面的 config 全废
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,                    # 每 run 后归档对话（零 LLM 成本）
            auto_extract_memory=True,             # LLM 抽取记忆条目（每 run 一次额外 LLM 调用）
        ),
        add_history_to_context=True,
        num_history_turns=5,
    )
```

适用：客服系统、多租户 SaaS、需要跨会话长期记忆的产品。

### 5. 完全体（DeepAgent）

```python
from agentica import DeepAgent

agent = DeepAgent()  # 内置工具 + 压缩 + 长期记忆 + skills + MCP
print(agent.run_sync("Research RAG 最新进展并写到 report.md").content)
```

适用：CLI、Gateway、长任务。所有 batteries-included 默认值都打开。

### 各场景的 Agent 参数速查

| 参数 | 何时该开 | 默认 |
|---|---|---|
| `add_history_to_context` | 多轮对话场景 | `False` |
| `num_history_turns` | 上下文窗口预算紧时调小 | `3` |
| `enable_long_term_memory` | 需要跨会话记忆/归档时 | `False` |
| `workspace` | 需要长期记忆/skill/AGENTS.md 时 | `None` |
| `session_id` | 需要落盘 jsonl 会话日志时 | `None`（不落盘） |
| `tools` | 需要工具调用时 | `None` |
| `history_config` | 长会话省 token 时 | `HistoryConfig()` |
| `history_filter` | 自定义历史过滤 | `None` |
| `prompt_config` | 改 system prompt 行为 | `PromptConfig()` |
| `tool_config` | 改工具行为（压缩、MCP 自动加载等） | `ToolConfig()` |
| `enable_tracing` + `LANGFUSE_*` env | 生产可观测性 | `False` |
| `RunConfig(max_cost_usd=...)` | 单次调用控成本 | 不限 |

---

---

## Agent 设计原则

### 1. 单一职责

每个 Agent 应该专注于一个特定任务。

```python
# ✅ 好的设计：专注的 Agent
researcher = Agent(
    name="Researcher",
    instructions=["专注于信息搜索和整理"],
    tools=[DuckDuckGoTool(), ArxivTool()],
)

writer = Agent(
    name="Writer",
    instructions=["专注于内容写作"],
)

# ❌ 避免：功能过多的 Agent
do_everything_agent = Agent(
    name="DoEverything",
    instructions=["搜索、写作、编程、分析..."],
    tools=[...20个工具...],
)
```

### 2. 清晰的指令

提供明确、具体的指令。

```python
# ✅ 好的指令
agent = Agent(
    instructions=[
        "你是一个 Python 代码审查专家",
        "检查代码时关注：安全性、性能、可读性",
        "使用中文回复",
        "发现问题时提供修复建议",
    ],
)

# ❌ 模糊的指令
agent = Agent(
    instructions=["你是一个助手"],
)
```

### 3. 适当的工具数量

工具过多会降低准确性。

```python
# ✅ 推荐：3-7 个相关工具
agent = Agent(
    tools=[
        DuckDuckGoTool(),
        UrlCrawlerTool(),
        BuiltinFileTool(work_dir="./workspace"),
    ],
)

# ❌ 避免：过多工具
agent = Agent(
    tools=[...15个工具...],  # 模型可能混淆
)
```

---

## 提示词工程

### 1. 结构化指令

```python
agent = Agent(
    instructions=[
        # 角色定义
        "你是一个专业的数据分析师",
        
        # 能力描述
        "你可以：分析数据、生成图表、撰写报告",
        
        # 行为约束
        "分析时始终验证数据质量",
        "使用 Python 进行数据处理",
        
        # 输出格式
        "报告格式：摘要 -> 详细分析 -> 结论",
    ],
)
```

### 2. 使用系统提示词

```python
from agentica.agent.config import PromptConfig

# 静态系统提示词
agent = Agent(
    prompt_config=PromptConfig(
        system_prompt="你是一个友好的助手，使用简洁的语言回答问题。",
    ),
)

# 动态系统提示词
def get_system_prompt(agent):
    from datetime import datetime
    return f"当前时间: {datetime.now()}\n你是一个智能助手。"

agent = Agent(
    prompt_config=PromptConfig(system_prompt=get_system_prompt),
)
```

### 3. 动态指令

```python
def get_instructions(agent):
    base = ["你是一个助手"]
    
    # 根据上下文添加指令
    if agent.session_state.get("mode") == "expert":
        base.append("使用专业术语回答")
    else:
        base.append("使用简单易懂的语言")
    
    return base

agent = Agent(instructions=get_instructions)
```

### 4. Few-shot 示例

```python
agent = Agent(
    instructions=[
        "将用户输入转换为 SQL 查询",
        "",
        "示例：",
        "输入：查找所有年龄大于 30 的用户",
        "输出：SELECT * FROM users WHERE age > 30",
        "",
        "输入：统计每个城市的用户数",
        "输出：SELECT city, COUNT(*) FROM users GROUP BY city",
    ],
)
```

---

## 工具使用

### 1. 工具描述要清晰

```python
def search_products(
    query: str,
    category: str = None,
    max_price: float = None,
) -> str:
    """搜索产品目录
    
    在产品数据库中搜索匹配的商品。
    
    Args:
        query: 搜索关键词，如 "iPhone" 或 "笔记本电脑"
        category: 产品类别，可选值：electronics, clothing, books
        max_price: 最高价格限制（人民币）
    
    Returns:
        JSON 格式的产品列表，包含名称、价格、描述
    
    Example:
        search_products("手机", category="electronics", max_price=5000)
    """
    ...
```

### 2. 工具返回格式

```python
# ✅ 结构化返回
def get_weather(city: str) -> str:
    data = fetch_weather(city)
    return json.dumps({
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "humidity": data["humidity"],
    }, ensure_ascii=False)

# ❌ 避免：非结构化返回
def get_weather(city: str) -> str:
    return f"天气很好，温度25度"  # 难以解析
```

### 3. 错误处理

```python
def safe_api_call(url: str) -> str:
    """安全的 API 调用"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "错误：请求超时，请稍后重试"
    except requests.HTTPError as e:
        return f"错误：HTTP {e.response.status_code}"
    except Exception as e:
        return f"错误：{str(e)}"
```

### 4. 工具限制

```python
from agentica.agent.config import ToolConfig

agent = Agent(
    tools=[...],
    tool_config=ToolConfig(tool_call_limit=10),  # 限制工具调用次数
)
```

---

## 内存管理

### 1. 会话持久化

```python
from agentica import Agent, SqliteDb, AgentMemory

# 使用数据库持久化会话
db = SqliteDb(table_name="sessions", db_file="agent.db")

agent = Agent(
    session_id="user-123-session",
    memory=AgentMemory.with_db(db=db),
)

# 会话会自动保存和恢复
```

### 2. 长期记忆

```python
agent = Agent(
    user_id="user-123",
    memory=AgentMemory.with_db(
        db=db,
        create_user_memories=True,  # 启用长期记忆
    ),
)

# Agent 会自动记住用户偏好
# "记住我喜欢 Python" -> 保存到长期记忆
```

### 3. 历史消息管理

```python
agent = Agent(
    add_history_to_context=True,  # 添加历史到上下文
    num_history_turns=5,           # 最近 5 轮对话
)
```

### 4. 会话摘要

```python
agent = Agent(
    memory=AgentMemory(
        create_session_summary=True,  # 生成会话摘要
    ),
)

# 长对话会自动生成摘要，减少 token 使用
```

---

## 历史消息定制

`Agent` 默认把最近 N 轮历史原样塞进 prompt（`num_history_turns`）。长会话场景下这会很快撑爆 context，常见痛点：

- 搜索 / fetch_url 等工具的结果几 KB 起步，后续轮次根本用不上但还在烧 token
- AI 上一轮回复啰嗦了 2000 字，下一轮其实只需要结论
- 用户 query 总带一个固定前缀（例如"用纯文本回复 ..."），不该污染历史

`Agent` 提供两层 API，从声明式快捷到完全自定义：

### 层 1：声明式（覆盖 80%）

```python
from agentica import Agent, OpenAIChat, HistoryConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=10,
    history_config=HistoryConfig(
        excluded_tools=["search_*", "web_search"],   # glob 匹配工具名，整条结果丢掉
        assistant_max_chars=200,                      # AI 回复在历史里截断到 200 字
    ),
)
```

`excluded_tools` 在丢掉 tool 消息时**会自动同步从前一条 assistant 消息里剥掉对应的 `tool_calls`**——OpenAI API 强制要求"每个 tool_call 必须有匹配的 tool 结果"，不剥就 400。

### 层 2：自定义 callable（任意 Python 表达力）

```python
def my_filter(history):
    out = []
    for m in history:
        # 剥用户 prompt 前缀
        if m.role == "user" and isinstance(m.content, str):
            m = m.model_copy(update={"content": m.content.removeprefix("用纯文本回复 ")})
        # AI 回复保留 reasoning 摘要、删掉正文
        if m.role == "assistant" and m.tool_call_error:
            continue   # 整条丢
        out.append(m)
    return out

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    history_filter=my_filter,
)
```

### 执行管线

```
working_memory.get_messages_from_last_n_runs(...)   # 已有的 tool 结果截断
        ↓
HistoryConfig.excluded_tools                         # 丢工具结果 + 同步 tool_calls
        ↓
HistoryConfig.assistant_max_chars                    # 截 AI 回复
        ↓
Agent.history_filter(history)                        # 用户 callable
        ↓
consistency 修复                                      # 自动剥孤立的 tool_calls（callable 没清干净时兜底）
        ↓
messages_for_model
```

每一层都接收**前面层处理后**的 history。`history_filter` 是最后一道，能看到所有内置规则都跑完之后的结果再做精修。

### 边界 & 注意

1. **不修改原始 messages**。pipeline 在 history 副本上跑，`agent.working_memory.runs` 永远是完整原始数据，下次 run 重新过滤。改 `history_config` / `history_filter` 立刻生效，不会污染历史。
2. **callable 删 tool 消息时，可以不清 `tool_calls`**——consistency 修复会自动剥孤立的，不会让你的 callable 出 OpenAI API 400。但能清就清，省一道兜底。
3. **`excluded_tools` 用 `fnmatch` 风格**：`search_*`、`web_*` 直接能用，不是正则。
4. **不会动 system / 当前 user message**。pipeline 只过滤 history 部分。

完整 4 个场景的可运行示例：[`examples/memory/03_history_filter.py`](https://github.com/shibing624/agentica/blob/main/examples/memory/03_history_filter.py)。

---

## RAG 最佳实践

### 1. 知识库配置

```python
from agentica import Knowledge
from agentica.vectordb import LanceDb
from agentica.emb import OpenAIEmbedder

knowledge = Knowledge(
    data_path="./documents",
    vector_db=LanceDb(
        table_name="docs",
        uri="./lancedb",
        embedder=OpenAIEmbedder(),
    ),
    chunk_size=1000,      # 适当的分块大小
    num_documents=5,      # 检索文档数
)

# 加载知识库
knowledge.load(recreate=False, upsert=True)
```

### 2. 检索增强

```python
from agentica.agent.config import ToolConfig

agent = Agent(
    knowledge=knowledge,
    tool_config=ToolConfig(add_references=True),  # 添加引用到响应
    instructions=[
        "基于知识库回答问题",
        "如果知识库中没有相关信息，明确告知用户",
        "引用来源时注明文档名称",
    ],
)
```

### 3. Agentic RAG

让 Agent 主动搜索知识库。

```python
from agentica.agent.config import ToolConfig

agent = Agent(
    knowledge=knowledge,
    tool_config=ToolConfig(search_knowledge=True),  # Agent 可以主动搜索
    instructions=[
        "遇到专业问题时，先搜索知识库",
        "综合多个来源的信息回答",
    ],
)
```

### 4. 混合检索

```python
from agentica.vectordb import LanceDb

db = LanceDb(
    search_type="hybrid",  # 混合检索：向量 + 关键词
    reranker=CohereReranker(),  # 重排序
)
```

---

## 多轮对话

### 1. 启用多轮策略

```python
agent = Agent(
    enable_multi_round=True,
    max_rounds=50,        # 最大轮数
    max_tokens=100000,    # token 限制
)
```

### 2. 监控工具调用进度

```python
async for response in agent.run_stream("complex task"):
    if response.event == "ToolCallStarted":
        print(f"Tool started: {response.content}")
    elif response.event == "ToolCallCompleted":
        print(f"Tool completed: {response.content}")
    elif response.event == "RunResponse":
        print(response.content, end="")
```

### 3. 上下文压缩

```python
from agentica import CompressionManager
from agentica.agent.config import ToolConfig

agent = Agent(
    tool_config=ToolConfig(
        enable_evict=True,           # Layer 1, default on
        enable_auto_compact=True,    # Layer 2 auto path, default on
        compression_manager=CompressionManager(
            compress_token_limit=50000,
        ),
    ),
)
```

---

## 多 Agent 协作

### 1. Agent 作为工具（编排器模式）

```python
researcher = Agent(
    name="Researcher",
    role="研究员",
    instructions=["负责信息搜索"],
    tools=[DuckDuckGoTool()],
)

writer = Agent(
    name="Writer",
    role="写手",
    instructions=["负责内容创作"],
)

# 主 Agent 把同伴 Agent 当工具调用
leader = Agent(
    name="Leader",
    instructions=[
        "协调任务完成",
        "需要研究时调用 research 工具",
        "需要写作时调用 write 工具",
    ],
    tools=[
        researcher.as_tool(tool_name="research", tool_description="进行深度研究"),
        writer.as_tool(tool_name="write", tool_description="撰写文章"),
    ],
)

leader.print_response("写一篇关于 AI 的文章")
```

### 2. Agent 作为工具

```python
# 将 Agent 转换为工具
research_tool = researcher.as_tool(
    tool_name="research",
    tool_description="进行深度研究",
)

main_agent = Agent(
    tools=[research_tool, other_tools...],
)
```

### 3. Workflow 编排

```python
from agentica import Workflow, RunResponse

class ArticleWorkflow(Workflow):
    researcher: Agent
    writer: Agent
    reviewer: Agent
    
    def run(self, topic: str) -> RunResponse:
        # 1. 研究
        research = self.researcher.run(f"研究: {topic}")
        
        # 2. 写作
        draft = self.writer.run(f"基于研究写文章:\n{research.content}")
        
        # 3. 审核
        final = self.reviewer.run(f"审核并改进:\n{draft.content}")
        
        return RunResponse(content=final.content)
```

---

## 性能优化

### 1. 流式输出

```python
# 同步流式输出提升用户体验
for chunk in agent.run_stream_sync("问题"):
    print(chunk.content, end="", flush=True)

# 异步流式输出
async for chunk in agent.run_stream("问题"):
    print(chunk.content, end="", flush=True)
```

### 2. 异步执行

```python
import asyncio

async def process_queries(queries):
    tasks = [agent.run(q) for q in queries]
    return await asyncio.gather(*tasks)

results = asyncio.run(process_queries(["问题1", "问题2", "问题3"]))
```

### 3. 模型选择

```python
# 简单任务用小模型
simple_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)

# 复杂任务用大模型
complex_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
)
```

### 4. 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str) -> str:
    """缓存搜索结果"""
    return do_search(query)
```

### 5. Token 管理

```python
from agentica import count_tokens

# 检查 token 使用
tokens = count_tokens(messages, model_id="gpt-4o")
if tokens > 100000:
    # 压缩或截断
    ...
```

---

## 错误处理

### 1. 重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
)
def robust_agent_call(agent, message):
    return agent.run(message)
```

### 2. 超时处理

```python
agent = Agent(
    model=OpenAIChat(
        timeout=60,  # 请求超时
        max_retries=3,
    ),
)
```

### 3. 优雅降级

```python
try:
    response = agent.run("问题")
except Exception as e:
    logger.error(f"Agent 错误: {e}")
    response = RunResponse(content="抱歉，我遇到了一些问题，请稍后重试。")
```

### 4. 工具错误处理

```python
from agentica.tools import ToolCallException

def risky_tool(data: str) -> str:
    try:
        return process(data)
    except ValueError as e:
        raise ToolCallException(
            user_message=f"处理失败: {e}",
            stop_execution=False,  # 继续执行
        )
```

---

## 生产部署

### 1. 环境配置

```python
import os

# 使用环境变量
agent = Agent(
    model=OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    ),
)
```

### 2. 日志记录

```python
from agentica.utils.log import logger, set_log_level_to_debug

# 开发环境
set_log_level_to_debug()

# 生产环境
import logging
logger.setLevel(logging.WARNING)
```

### 3. 监控指标

```python
# 响应中包含指标
response = agent.run("问题")
print(response.metrics)
# {
#     "input_tokens": 100,
#     "output_tokens": 200,
#     "time_to_first_token": 0.5,
#     "total_time": 2.3,
# }
```

### 4. API 服务

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
agent = Agent(...)

class Query(BaseModel):
    message: str
    session_id: str = None

@app.post("/chat")
async def chat(query: Query):
    agent.session_id = query.session_id
    response = await agent.run(query.message)
    return {"content": response.content}
```

### 5. 安全考虑

```python
from agentica import Agent, BuiltinExecuteTool, BuiltinFileTool
from agentica.agent.config import SandboxConfig

# 限制工具权限
agent = Agent(
    tools=[
        BuiltinExecuteTool(work_dir="./safe_dir"),
        BuiltinFileTool(
            work_dir="./safe_dir",
            sandbox_config=SandboxConfig(enabled=True, writable_dirs=["./safe_dir"]),
        ),
    ],
)

# 输入验证
def validate_input(message: str) -> str:
    if len(message) > 10000:
        raise ValueError("输入过长")
    # 其他验证...
    return message
```

---

## 常见问题

### Q: Agent 不调用工具？

**A:** 检查以下几点：
1. 工具描述是否清晰
2. 指令中是否提到使用工具
3. 模型是否支持工具调用

```python
agent = Agent(
    tools=[my_tool],
    instructions=["使用 my_tool 工具完成任务"],
)
```

### Q: 响应太慢？

**A:** 优化建议：
1. 使用流式输出
2. 减少工具数量
3. 使用更快的模型
4. 启用压缩

### Q: Token 超限？

**A:** 解决方案：
1. 启用压缩
2. 减少历史消息
3. 使用会话摘要
4. 分块处理长文本

### Q: 结果不准确？

**A:** 改进方法：
1. 优化提示词
2. 添加 Few-shot 示例
3. 使用更强的模型
4. 添加知识库

### Q: 会话日志没有落盘到 `~/.agentica/projects/...`？

**A:** `Agent` 默认 `session_id=None`，此时不会创建 `SessionLog`，也就不会写 jsonl。要持久化对话日志必须显式传 `session_id`：

```python
agent = Agent(
    model=...,
    session_id=user_id,  # 多用户场景常用 user_id 作为 session_id
)
log = agent.session_log
log.format_trace()
log.export(f"{user_id}.jsonl")
```

落盘路径：`~/.agentica/projects/<user>/<project-slug>/<session_id>.jsonl`。CLI 总会生成 `session_id`；Web `/traces` 读的是同一份文件。`enable_session_log=False` 仍然关闭落盘（服务自有会话库时用）。

源码位置：`agent/base.py::_init_execution`，`session_id is None` 时 `_session_log` 直接为 `None`。`SessionLog.analyze()` 与 Gateway `GET /api/sessions/{id}/trace/analysis` 同一 payload。

### Q: 配了 `long_term_memory_config` / `Workspace`，但长期记忆和对话归档没生效？

**A:** 从 v1.3.7 起 `Agent.__init__` 会主动检测这种错配并打 warning：

```
WARNING agentica.agent.base - long_term_memory_config has auto_archive / auto_extract_memory enabled,
but enable_long_term_memory=False. These settings will be IGNORED. Pass enable_long_term_memory=True to activate.
```

根因：自动归档（`ConversationArchiveHooks`）和自动抽取记忆（`MemoryExtractHooks`）的注入条件是 **`enable_long_term_memory=True` 且 `workspace is not None`** 双闸门，仅配 `long_term_memory_config` 不会打开开关：

```python
agent = Agent(
    model=...,
    workspace=Workspace("~/.agentica/workspace", user_id=user_id),
    enable_long_term_memory=True,             # 关键：必须显式开启
    long_term_memory_config=WorkspaceMemoryConfig(
        auto_archive=True,
        auto_extract_memory=True,
    ),
)
```

`DeepAgent` 已默认 `enable_long_term_memory=True`，开箱即用。普通 `Agent` 需自己打开。

源码位置：`agent/base.py::_post_init` 中以 `enable_long_term_memory and workspace is not None` 作为 hooks 注入门控。

### Q: 多用户场景怎么隔离会话和记忆？

**A:** 一个用户一个 Agent 实例，构造时把 `user_id` 同时传给 `Agent` 和 `session_id`：

```python
def create_agent(user_id: str) -> Agent:
    return Agent(
        model=...,
        workspace="~/.agentica/workspace",   # 字符串走便捷路径
        user_id=user_id,                      # workspace 落到 users/<user_id>/
        session_id=user_id,                   # 会话日志按 user 切文件
        enable_long_term_memory=True,
        long_term_memory_config=WorkspaceMemoryConfig(auto_archive=True),
    )
```

不要试图在 `agent.run()` 里切 `user_id`——`auto_archive` / 记忆 hooks 会把数据写错位置。

---

*文档最后更新: 2026-04-27*
