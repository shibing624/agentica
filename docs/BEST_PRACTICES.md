# Agentica 最佳实践

> Agent 开发的推荐模式、技巧和常见问题解决方案

## 目录

- [Agent 设计原则](#agent-设计原则)
- [提示词工程](#提示词工程)
- [工具使用](#工具使用)
- [内存管理](#内存管理)
- [RAG 最佳实践](#rag-最佳实践)
- [多轮对话](#多轮对话)
- [团队协作](#团队协作)
- [性能优化](#性能优化)
- [错误处理](#错误处理)
- [生产部署](#生产部署)

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
        FileTool(),
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
# 静态系统提示词
agent = Agent(
    system_prompt="你是一个友好的助手，使用简洁的语言回答问题。",
)

# 动态系统提示词
def get_system_prompt(agent):
    return f"""
    当前时间: {datetime.now()}
    用户ID: {agent.user_id}
    会话ID: {agent.session_id}
    
    你是一个智能助手。
    """

agent = Agent(system_prompt=get_system_prompt)
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
agent = Agent(
    tools=[...],
    tool_call_limit=10,  # 限制工具调用次数
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
    add_history_to_messages=True,  # 添加历史到上下文
    num_history_responses=5,       # 最近 5 轮对话
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
agent = Agent(
    knowledge=knowledge,
    add_references=True,  # 添加引用到响应
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
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,  # Agent 可以主动搜索
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

### 2. 监控多轮进度

```python
for response in agent.run("复杂任务", stream=True):
    if response.event == "MultiRoundTurn":
        print(f"轮次 {response.extra_data.round}")
    elif response.event == "MultiRoundToolCall":
        print(f"调用工具: {response.content}")
    elif response.event == "MultiRoundCompleted":
        print("任务完成")
```

### 3. 上下文压缩

```python
from agentica import CompressionManager

agent = Agent(
    compression_manager=CompressionManager(
        compress_tool_results=True,
        compress_token_limit=50000,
    ),
    compress_tool_results=True,
)
```

---

## 团队协作

### 1. Agent 团队

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

# 主 Agent 协调团队
leader = Agent(
    name="Leader",
    team=[researcher, writer],
    instructions=[
        "协调团队完成任务",
        "研究任务交给 Researcher",
        "写作任务交给 Writer",
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
# 流式输出提升用户体验
for chunk in agent.run("问题", stream=True):
    print(chunk.content, end="", flush=True)
```

### 2. 异步执行

```python
import asyncio

async def process_queries(queries):
    tasks = [agent.arun(q) for q in queries]
    return await asyncio.gather(*tasks)

results = asyncio.run(process_queries(["问题1", "问题2", "问题3"]))
```

### 3. 模型选择

```python
# 简单任务用小模型
simple_agent = Agent(
    model=OpenAIChat(model="gpt-4o-mini"),
)

# 复杂任务用大模型
complex_agent = Agent(
    model=OpenAIChat(model="gpt-4o"),
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
    response = await agent.arun(query.message)
    return {"content": response.content}
```

### 5. 安全考虑

```python
# 限制工具权限
agent = Agent(
    tools=[
        ShellTool(allowed_commands=["ls", "cat"]),  # 白名单
        FileTool(base_dir="./safe_dir"),  # 限制目录
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
    support_tool_calls=True,
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

---

*文档最后更新: 2025-12-20*
