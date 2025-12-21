# Agentica 项目技术实现文档

> 本文档详述 Agentica 项目结构，方便开发者和 AI 编程助手快速了解代码架构。

## 项目概述

**Agentica** 是一个功能强大的 AI Agent 框架，支持多模型、多工具、多轮对话、RAG（检索增强生成）、工作流编排、MCP（Model Context Protocol）集成、Temporal 分布式工作流和 Skills 技能系统。

**项目路径**: `/agentica`

---

## 1. 核心模块结构

### 1.1 agentica/ 目录核心文件

| 文件 | 主要职责 |
|------|----------|
| `agent.py` | **核心 Agent 类**，实现 AI 代理的完整功能 |
| `deep_agent.py` | **DeepAgent 类**，内置工具的增强版 Agent |
| `deep_tools.py` | **内置工具集**，DeepAgent 使用的文件/搜索/执行工具 |
| `memory.py` | **内存管理系统**，包含会话记忆和长期记忆 |
| `guardrails.py` | **护栏系统**，Agent 输入/输出验证和过滤 |
| `workflow.py` | **工作流引擎**，支持多步骤任务编排 |
| `cli.py` | **命令行接口**，交互式 AI 助手 |
| `run_response.py` | Agent 运行响应模型 |
| `media.py` | 多媒体（图片/音频/视频）处理 |
| `document.py` | 文档数据模型 |
| `template.py` | Prompt 模板系统 |
| `config.py` | 全局配置管理 |
| `reasoning.py` | 推理步骤模型 |
| `agent_session.py` | Agent 会话管理 |
| `workflow_session.py` | Workflow 会话管理 |

### 1.2 子目录结构

```
agentica/
├── model/           # 模型层 - 多提供商 LLM 支持
├── tools/           # 工具系统 - 40+ 内置工具
├── memory.py        # 内存管理
├── guardrails.py    # Agent 输入/输出护栏
├── db/              # 数据库层 - 会话/记忆持久化
├── vectordb/        # 向量数据库 - RAG 支持
├── knowledge/       # 知识库系统
├── emb/             # 嵌入模型
├── mcp/             # MCP 协议支持
├── compression/     # 上下文压缩
├── temporal/        # Temporal 分布式工作流
├── skills/          # Agent Skills 技能系统
├── reranker/        # 重排序器
├── file/            # 文件处理
└── utils/           # 工具函数
```

---

## 2. Agent 系统

### 2.1 核心类: `Agent` (`agentica/agent.py`)

Agent 是一个基于 `@dataclass` 的核心类，提供完整的 AI 代理功能。

#### 主要属性

```python
@dataclass(init=False)
class Agent:
    # 核心设置
    model: Optional[Model] = None              # LLM 模型
    name: Optional[str] = None                 # Agent 名称
    agent_id: Optional[str] = None             # Agent UUID
    
    # 用户/会话
    user_id: Optional[str] = None              # 用户 ID
    session_id: Optional[str] = None           # 会话 ID
    session_state: Dict[str, Any]              # 会话状态
    
    # 内存系统
    memory: AgentMemory                        # Agent 内存
    add_history_to_messages: bool = False      # 是否添加历史到消息
    num_history_responses: int = 3             # 历史响应数量
    
    # 知识库 (RAG)
    knowledge: Optional[Knowledge] = None      # 知识库
    add_references: bool = False               # 是否添加引用
    
    # 工具系统
    tools: Optional[List[...]] = None          # 工具列表
    support_tool_calls: bool = True            # 是否支持工具调用
    tool_call_limit: Optional[int] = None      # 工具调用限制
    
    # 多轮对话策略
    enable_multi_round: bool = False           # 启用多轮策略
    max_rounds: int = 100                      # 最大轮数
    max_tokens: int = 128000                   # 最大 token 数
    
    # 压缩设置
    compress_tool_results: bool = False        # 压缩工具结果
    compression_manager: Optional[CompressionManager] = None
    
    # 提示词设置
    system_prompt: Optional[Union[str, Callable]] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    description: Optional[str] = None
    
    # 响应设置
    response_model: Optional[Type[Any]] = None # 结构化输出模型
    structured_outputs: bool = False           # 使用结构化输出
    
    # 团队协作
    team: Optional[List["Agent"]] = None       # Agent 团队
    role: Optional[str] = None                 # 团队角色
```

#### 核心方法

| 方法 | 描述 |
|------|------|
| `run(message, stream=False)` | 运行 Agent，支持流式和非流式 |
| `arun(message, stream=False)` | 异步运行 Agent |
| `update_model()` | 更新/初始化模型配置 |
| `get_tools()` | 获取所有可用工具 |
| `get_transfer_function()` | 获取团队任务转移函数 |
| `deep_copy()` | 深拷贝 Agent 实例 |
| `search_knowledge_base()` | 搜索知识库（Agentic RAG） |
| `get_chat_history()` | 获取聊天历史 |

### 2.2 多轮对话策略

Agent 支持多轮对话策略，通过以下参数控制：

```python
agent = Agent(
    enable_multi_round=True,    # 启用多轮策略
    max_rounds=100,             # 最大轮数
    max_tokens=128000,          # 上下文 token 限制
)
```

**运行事件** (`RunEvent` 枚举)：
- `multi_round_turn` - 多轮对话轮次
- `multi_round_tool_call` - 多轮工具调用
- `multi_round_tool_result` - 多轮工具结果
- `multi_round_completed` - 多轮完成

### 2.3 DeepAgent (`agentica/deep_agent.py`)

DeepAgent 是 Agent 的增强版本，自动包含内置工具：

```python
@dataclass(init=False)
class DeepAgent(Agent):
    """
    增强版 Agent，自动包含内置工具：
    - 文件工具: ls, read_file, write_file, edit_file, glob, grep
    - 执行工具: execute
    - 网络工具: web_search, fetch_url
    - 任务管理: write_todos, read_todos
    - 子代理: task
    """
    
    # 配置选项
    work_dir: Optional[str] = None
    include_file_tools: bool = True
    include_execute: bool = True
    include_web_search: bool = True
    include_fetch_url: bool = True
    include_todos: bool = True
    include_task: bool = True
```

**使用示例**：
```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(
    model=OpenAIChat(id="gpt-4o"),
    description="A powerful coding assistant",
)

response = agent.run("List all Python files in the current directory")
```

### 2.4 内置工具 (`agentica/deep_tools.py`)

DeepAgent 的内置工具类：

| 工具类 | 功能 | 暴露函数 |
|--------|------|----------|
| `BuiltinFileTool` | 文件系统操作 | ls, read_file, write_file, edit_file, glob, grep |
| `BuiltinExecuteTool` | 命令执行 | execute |
| `BuiltinWebSearchTool` | 网页搜索 | web_search |
| `BuiltinFetchUrlTool` | URL 抓取 | fetch_url |
| `BuiltinTodoTool` | 任务管理 | write_todos, read_todos |
| `BuiltinTaskTool` | 子代理任务 | task |

```python
# 获取内置工具
from agentica.deep_tools import get_builtin_tools

tools = get_builtin_tools(
    base_dir="/path/to/work",
    include_file_tools=True,
    include_execute=True,
    include_web_search=True,
    include_fetch_url=True,
    include_todos=True,
    include_task=True,
)
```

---

## 3. Model 层

### 3.1 目录结构 (`agentica/model/`)

```
model/
├── __init__.py
├── base.py              # Model 基类
├── message.py           # 消息模型
├── response.py          # 响应模型
├── content.py           # 内容模型 (Image, Video, Audio)
├── base_audio_model.py  # 音频模型基类
│
├── openai/              # OpenAI 模型
│   ├── chat.py          # OpenAIChat
│   ├── audio.py         # 音频模型
│   └── like.py          # OpenAI 兼容接口
│
├── anthropic/           # Anthropic Claude
│   └── claude.py
│
├── google/              # Google Gemini
│   ├── gemini.py
│   └── gemini_openai.py
│
├── azure/               # Azure OpenAI
│   └── openai_chat.py
│
├── deepseek/            # DeepSeek
├── moonshot/            # Moonshot (月之暗面)
├── zhipuai/             # 智谱 AI
├── qwen/                # 通义千问
├── doubao/              # 豆包
├── yi/                  # 零一万物
├── ollama/              # Ollama 本地模型
├── groq/                # Groq
├── together/            # Together AI
├── mistral/             # Mistral
├── cohere/              # Cohere
├── huggingface/         # HuggingFace
├── xai/                 # xAI Grok
├── aws/                 # AWS Bedrock
├── vertexai/            # Google Vertex AI
├── nvidia/              # NVIDIA
├── fireworks/           # Fireworks
├── sambanova/           # SambaNova
└── openrouter/          # OpenRouter
```

### 3.2 Model 基类 (`agentica/model/base.py`)

```python
class Model(BaseModel):
    id: str = Field(..., alias="model")        # 模型 ID
    name: Optional[str] = None                 # 模型名称
    provider: Optional[str] = None             # 提供商
    
    # 工具支持
    tools: Optional[List[Union[ModelTool, Dict]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_tools: bool = True
    tool_call_limit: Optional[int] = None
    
    # 函数管理
    functions: Optional[Dict[str, Function]] = None
    function_call_stack: Optional[List[FunctionCall]] = None
    
    # 核心方法
    def response(self, messages: List[Message]) -> ModelResponse
    async def aresponse(self, messages: List[Message]) -> ModelResponse
    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]
    def add_tool(self, tool, strict=False, agent=None) -> None
    def run_function_calls(...) -> Iterator[ModelResponse]
```

### 3.3 Message 模型 (`agentica/model/message.py`)

```python
class Message(BaseModel):
    role: str                                  # system/user/assistant/tool
    content: Optional[Union[List[Any], str]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # 多模态支持
    audio: Optional[Any] = None
    images: Optional[Sequence[Any]] = None
    videos: Optional[Sequence[Any]] = None
    
    # 推理内容
    thinking: Optional[str] = None
    reasoning_content: Optional[str] = None
    
    # 压缩内容
    compressed_content: Optional[str] = None
    
    # 引用 (RAG)
    references: Optional[MessageReferences] = None

# 便捷类
class SystemMessage(Message): role = "system"
class UserMessage(Message): role = "user"
class AssistantMessage(Message): role = "assistant"
class ToolMessage(Message): role = "tool"
```

---

## 4. Tools 系统

### 4.1 目录结构 (`agentica/tools/`)

共 **43 个工具文件**：

| 工具文件 | 功能 |
|----------|------|
| `base.py` | **工具基类和接口** |
| `browser_tool.py` | 浏览器自动化工具 |
| `code_tool.py` | 代码工具 |
| `mcp_tool.py` | MCP 工具集成 |
| `workspace_tool.py` | 工作区管理工具 |
| `video_analysis_tool.py` | 视频分析 |
| `memori_tool.py` | 记忆工具 |
| `baidu_search_tool.py` | 百度搜索 |
| `run_nb_code_tool.py` | Notebook 代码执行 |
| `jina_tool.py` | Jina AI 工具 |
| `edit_tool.py` | 文件编辑工具 |
| `yfinance_tool.py` | 金融数据 |
| `volc_tts_tool.py` | 火山 TTS |
| `video_download_tool.py` | 视频下载 |
| `run_python_code_tool.py` | Python 代码执行 |
| `calculator_tool.py` | 计算器 |
| `search_serper_tool.py` | Serper 搜索 |
| `search_exa_tool.py` | Exa 搜索 |
| `cogvideo_tool.py` | 智谱视频生成 |
| `arxiv_tool.py` | arXiv 论文搜索 |
| `search_bocha_tool.py` | Bocha 搜索 |
| `url_crawler_tool.py` | URL 爬虫 |
| `file_tool.py` | 文件操作 |
| `sql_tool.py` | SQL 查询 |
| `image_analysis_tool.py` | 图像分析 |
| `duckduckgo_tool.py` | DuckDuckGo 搜索 |
| `cogview_tool.py` | 智谱图像生成 |
| `dalle_tool.py` | DALL-E 图像生成 |
| `apify_tool.py` | Apify 爬虫 |
| `dblp_tool.py` | DBLP 论文搜索 |
| `web_search_pro_tool.py` | 专业网页搜索 |
| `newspaper_tool.py` | 新闻提取 |
| `hackernews_tool.py` | Hacker News |
| `airflow_tool.py` | Airflow 集成 |
| `resend_tools.py` | 邮件发送 |
| `wikipedia_tool.py` | Wikipedia |
| `shell_tool.py` | Shell 命令 |
| `string_tool.py` | 字符串处理 |
| `ocr_tool.py` | OCR 识别 |
| `weather_tool.py` | 天气查询 |
| `text_analysis_tool.py` | 文本分析 |

### 4.2 工具基类 (`agentica/tools/base.py`)

```python
class Function(BaseModel):
    """函数定义模型"""
    name: str                                  # 函数名
    description: Optional[str] = None          # 描述
    parameters: Dict[str, Any]                 # JSON Schema 参数
    entrypoint: Optional[Callable] = None      # 实际函数
    strict: Optional[bool] = None              # 严格模式
    show_result: bool = False                  # 显示结果
    stop_after_tool_call: bool = False         # 调用后停止
    pre_hook: Optional[Callable] = None        # 前置钩子
    post_hook: Optional[Callable] = None       # 后置钩子
    
    @classmethod
    def from_callable(cls, c: Callable, strict=False) -> "Function"

class FunctionCall(BaseModel):
    """函数调用模型"""
    function: Function
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    call_id: Optional[str] = None
    error: Optional[str] = None
    
    def execute(self) -> bool

class Tool:
    """工具管理类"""
    def __init__(self, name: str = "tool")
    def register(self, function: Callable, sanitize_arguments=True)
    functions: Dict[str, Function]

class ToolCallException(Exception):
    """工具调用异常"""
    user_message: Optional[Union[str, Message]]
    agent_message: Optional[Union[str, Message]]
    stop_execution: bool = False

class StopAgentRun(ToolCallException):
    """停止 Agent 执行的异常"""
```

---

## 5. Memory 系统 (`agentica/memory.py`)

### 5.1 核心类

```python
class AgentMemory(BaseModel):
    """Agent 内存管理"""
    runs: List[AgentRun] = []                  # 运行历史
    messages: List[Message] = []               # 消息列表
    
    # 会话摘要
    create_session_summary: bool = False
    update_session_summary_after_run: bool = True
    summary: Optional[SessionSummary] = None
    summarizer: Optional[MemorySummarizer] = None
    
    # 用户记忆（长期记忆）
    create_user_memories: bool = False
    update_user_memories_after_run: bool = True
    db: Optional[BaseDb] = None                # 持久化数据库
    user_id: Optional[str] = None
    retrieval: MemoryRetrieval = MemoryRetrieval.last_n
    memories: Optional[List[Memory]] = None
    num_memories: Optional[int] = None
    classifier: Optional[MemoryClassifier] = None
    manager: Optional[MemoryManager] = None
    
    # 工厂方法
    @classmethod
    def with_db(cls, db: BaseDb, user_id=None, ...) -> "AgentMemory"
    
    # 核心方法
    def add_run(self, agent_run: AgentRun) -> None
    def add_message(self, message: Message) -> None
    def get_messages_from_last_n_runs(self, last_n=None) -> List[Message]
    def load_user_memories(self) -> None
    def update_memory(self, input: str, force=False) -> Optional[str]
    def update_summary(self) -> Optional[SessionSummary]
```

### 5.2 MemoryManager 类

```python
class MemoryManager(BaseModel):
    """记忆管理器 - CRUD 操作和智能搜索"""
    mode: Literal["model", "rule"] = "rule"    # 模式
    model: Optional[Model] = None
    db: Optional[BaseDb] = None
    user_id: Optional[str] = None
    
    # CRUD 操作
    def add_memory(self, memory: str) -> str
    def update_memory(self, id: str, memory: str) -> str
    def delete_memory(self, id: str) -> str
    def clear_memory(self) -> str
    
    # 用户记忆操作
    def get_user_memories(self, user_id: Optional[str] = None) -> List[MemoryRow]
    def add_user_memory(self, memory: Memory, user_id: Optional[str] = None) -> str
    def delete_user_memory(self, memory_id: str, user_id: Optional[str] = None) -> str
    def replace_user_memory(self, memory_id: str, memory: Memory, user_id: Optional[str] = None) -> str
    def clear_user_memories(self, user_id: Optional[str] = None) -> str
    
    # 智能搜索
    def search_user_memories(
        self,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        retrieval_method: Literal["last_n", "first_n", "keyword", "agentic"] = "last_n",
        user_id: Optional[str] = None,
    ) -> List[MemoryRow]
    
    async def asearch_user_memories(...) -> List[MemoryRow]  # 异步版本
```

**搜索方法**：
- `last_n`: 返回最近的记忆
- `first_n`: 返回最早的记忆
- `keyword`: 关键词匹配搜索
- `agentic`: 使用 LLM 进行语义相似度搜索

### 5.3 其他 Memory 类

```python
class MemoryClassifier(BaseModel):
    """记忆分类器 - 判断是否需要记忆"""
    def run(self, message: str) -> str  # 返回 "yes" 或 "no"

class MemorySummarizer(BaseModel):
    """会话摘要生成器"""
    def run(self, message_pairs: List[Tuple[Message, Message]]) -> SessionSummary

class WorkflowMemory(BaseModel):
    """Workflow 内存"""
    runs: List[WorkflowRun] = []

class MemoryRetrieval(str, Enum):
    """记忆检索模式"""
    last_n = "last_n"
    first_n = "first_n"
    only_user = "only_user"
```

---

## 6. VectorDB 系统 (`agentica/vectordb/`)

```
vectordb/
├── base.py              # VectorDb 基类
├── memory_vectordb.py   # 内存向量数据库
├── chromadb_vectordb.py # ChromaDB
├── lancedb_vectordb.py  # LanceDB
├── pgvectordb.py        # PostgreSQL pgvector
├── pineconedb.py        # Pinecone
└── qdrantdb.py          # Qdrant
```

### VectorDb 基类

```python
class Distance(str, Enum):
    cosine = "cosine"
    l2 = "l2"
    max_inner_product = "max_inner_product"

class SearchType(str, Enum):
    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"

class VectorDb(ABC):
    """向量数据库基类"""
    def create(self) -> None
    def doc_exists(self, document: Document) -> bool
    def insert(self, documents: List[Document], filters=None) -> None
    def upsert(self, documents: List[Document], filters=None) -> None
    def search(self, query: str, limit=5, filters=None) -> List[Document]
    def vector_search(self, query: str, limit=5) -> List[Document]
    def keyword_search(self, query: str, limit=5) -> List[Document]
    def hybrid_search(self, query: str, limit=5) -> List[Document]
    def drop(self) -> None
    def exists(self) -> bool
    def delete(self) -> bool
```

---

## 7. Compression 系统 (`agentica/compression/`)

### CompressionManager (`agentica/compression/manager.py`)

```python
@dataclass
class CompressionManager:
    """
    工具结果压缩管理器。
    
    压缩触发条件：
    1. Token 阈值: 上下文 token 超过 compress_token_limit
    2. 工具数量阈值: 未压缩工具调用超过 compress_tool_results_limit
    """
    model: Optional[Any] = None                # 压缩用模型
    compress_tool_results: bool = True
    compress_tool_results_limit: Optional[int] = None  # 工具数量阈值（默认 3）
    compress_token_limit: Optional[int] = None         # Token 阈值
    compress_tool_call_instructions: Optional[str] = None  # 自定义压缩提示词
    
    # 核心方法
    def should_compress(self, messages, tools=None, model=None, response_format=None) -> bool
    def compress(self, messages: List[Message]) -> None
    async def acompress(self, messages: List[Message]) -> None  # 异步并行压缩
    def get_compression_ratio(self) -> float
    def get_stats(self) -> Dict[str, Any]
```

**使用示例**：
```python
from agentica.compression import CompressionManager
from agentica.model.openai import OpenAIChat

# 基于工具数量压缩
compression_manager = CompressionManager(
    model=OpenAIChat(id="gpt-4o-mini"),
    compress_tool_results_limit=5,
)

# 基于 Token 数量压缩
compression_manager = CompressionManager(
    model=OpenAIChat(id="gpt-4o-mini"),
    compress_token_limit=10000,
)
```

**压缩策略**：
- 保留关键信息：数字、日期、实体、标识符
- 移除冗余：介绍语、过渡词、格式化内容

---

## 8. Database 层 (`agentica/db/`)

### 8.1 目录结构

```
db/
├── base.py      # BaseDb 抽象基类
├── sqlite.py    # SQLite 实现
├── postgres.py  # PostgreSQL 实现
├── memory.py    # 内存数据库
└── json.py      # JSON 文件存储
```

### 8.2 数据模型

```python
class SessionRow(BaseModel):
    """会话记录"""
    session_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    memory: Optional[Dict[str, Any]] = None
    agent_data: Optional[Dict[str, Any]] = None
    user_data: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

class MemoryRow(BaseModel):
    """记忆记录"""
    id: Optional[str] = None  # 自动生成 MD5 哈希
    user_id: Optional[str] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class MetricsRow(BaseModel):
    """指标记录"""
    id: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

class KnowledgeRow(BaseModel):
    """知识文档记录 (RAG)"""
    id: str
    name: str
    description: str = ""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    doc_type: Optional[str] = None  # pdf, txt, url 等
    size: Optional[int] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
```

### 8.3 BaseDb 抽象基类

```python
class BaseDb(ABC):
    """
    统一数据库抽象，管理多个表：
    - sessions: Agent/Workflow 会话历史
    - memories: 用户记忆（长期记忆）
    - metrics: 使用指标和统计
    - knowledge: RAG 知识文档
    """
    
    def __init__(
        self,
        session_table: Optional[str] = None,    # 默认 "agentica_sessions"
        memory_table: Optional[str] = None,     # 默认 "agentica_memories"
        metrics_table: Optional[str] = None,    # 默认 "agentica_metrics"
        knowledge_table: Optional[str] = None,  # 可选
    ):
    
    # Session 操作
    @abstractmethod
    def create_session_table(self) -> None
    @abstractmethod
    def read_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionRow]
    @abstractmethod
    def upsert_session(self, session: SessionRow) -> Optional[SessionRow]
    @abstractmethod
    def delete_session(self, session_id: str) -> None
    @abstractmethod
    def get_all_session_ids(self, user_id=None, agent_id=None) -> List[str]
    @abstractmethod
    def get_all_sessions(self, user_id=None, agent_id=None) -> List[SessionRow]
    
    # Memory 操作
    @abstractmethod
    def create_memory_table(self) -> None
    @abstractmethod
    def read_memories(self, user_id=None, limit=None, sort=None) -> List[MemoryRow]
    @abstractmethod
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]
    @abstractmethod
    def delete_memory(self, memory_id: str) -> None
    @abstractmethod
    def memory_exists(self, memory: MemoryRow) -> bool
    @abstractmethod
    def clear_memories(self, user_id: Optional[str] = None) -> bool
    
    # Metrics 操作
    @abstractmethod
    def create_metrics_table(self) -> None
    @abstractmethod
    def insert_metrics(self, metrics: MetricsRow) -> None
    @abstractmethod
    def get_metrics(self, agent_id=None, session_id=None, limit=None) -> List[MetricsRow]
    
    # Knowledge 操作
    @abstractmethod
    def create_knowledge_table(self) -> None
    @abstractmethod
    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]
    @abstractmethod
    def read_knowledge(self, knowledge_id: str) -> Optional[KnowledgeRow]
    @abstractmethod
    def get_all_knowledge(self, doc_type=None, status=None, limit=None) -> List[KnowledgeRow]
    @abstractmethod
    def delete_knowledge(self, knowledge_id: str) -> None
    @abstractmethod
    def clear_knowledge(self) -> bool
    
    # 生命周期
    def create(self) -> None  # 创建所有表
    @abstractmethod
    def drop(self) -> None
    @abstractmethod
    def upgrade_schema(self) -> None
```

**使用示例**：
```python
from agentica.db.sqlite import SqliteDb
from agentica import Agent

db = SqliteDb(db_file="agent.db")
agent = Agent(db=db)

# 用于 RAG 知识库
contents_db = SqliteDb(db_file="data.db", knowledge_table="knowledge_contents")
```

---

## 9. Temporal 分布式工作流 (`agentica/temporal/`)

Temporal 提供持久化执行能力，支持长时间运行的工作流、故障恢复和并行执行。

### 9.1 目录结构

```
temporal/
├── __init__.py      # 统一导出
├── activities.py    # Activity 定义（LLM 调用）
├── workflows.py     # Workflow 定义（编排逻辑）
└── client.py        # Temporal 客户端封装
```

### 9.2 Activities (`activities.py`)

```python
@dataclass
class AgentActivityInput:
    """Activity 输入"""
    message: str
    agent_name: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None  # 图片 URL 或路径

@dataclass
class AgentActivityOutput:
    """Activity 输出"""
    content: str
    agent_name: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@activity.defn
async def run_agent_activity(input: AgentActivityInput) -> AgentActivityOutput:
    """执行 Agent 作为 Temporal Activity"""
```

### 9.3 Workflows (`workflows.py`)

```python
@dataclass
class WorkflowInput:
    """Workflow 输入"""
    message: str
    agent_configs: Optional[List[Dict[str, Any]]] = None

@dataclass
class TranslationInput:
    """翻译 Workflow 输入"""
    text: str
    target_language: str = "Chinese"
    num_translations: int = 3

@dataclass
class WorkflowOutput:
    """Workflow 输出"""
    content: str
    steps: List[AgentActivityOutput] = field(default_factory=list)

# 预定义 Workflows
@workflow.defn
class AgentWorkflow:
    """单 Agent 工作流"""
    
@workflow.defn
class SequentialAgentWorkflow:
    """顺序多 Agent 工作流（流水线）"""
    
@workflow.defn
class ParallelAgentWorkflow:
    """并行多 Agent 工作流"""
    
@workflow.defn
class ParallelTranslationWorkflow:
    """并行翻译工作流（多翻译 + 最佳选择）"""
```

### 9.4 TemporalClient (`client.py`)

```python
@dataclass
class WorkflowResult:
    """Workflow 执行结果"""
    workflow_id: str
    content: str
    steps: List[Any] = field(default_factory=list)

class TemporalClient:
    """Temporal 客户端封装"""
    
    def __init__(
        self,
        host: str = "localhost:7233",
        namespace: str = "default",
        task_queue: str = "agentica-task-queue",
    ):
    
    async def connect(self) -> "TemporalClient"
    async def start_workflow(self, workflow_class, input, workflow_id=None, task_queue=None) -> str
    async def get_result(self, workflow_id: str) -> WorkflowResult
    async def get_status(self, workflow_id: str) -> str
    def get_handle(self, workflow_id: str) -> WorkflowHandle
    async def cancel_workflow(self, workflow_id: str) -> None
    async def terminate_workflow(self, workflow_id: str, reason: str = "Terminated by user") -> None
```

**使用示例**：
```python
from agentica.temporal import (
    TemporalClient, AgentWorkflow, WorkflowInput,
    ParallelTranslationWorkflow, TranslationInput,
)

# 启动 Workflow
client = TemporalClient()
await client.connect()

workflow_id = await client.start_workflow(
    AgentWorkflow,
    WorkflowInput(message="What is AI?"),
)

result = await client.get_result(workflow_id)
print(result.content)
```

**Worker 配置**（需要禁用沙箱）：
```python
from temporalio.worker import Worker, UnsandboxedWorkflowRunner
from agentica.temporal.workflows import AgentWorkflow
from agentica.temporal.activities import run_agent_activity

worker = Worker(
    client,
    task_queue="agentica-task-queue",
    workflows=[AgentWorkflow],
    activities=[run_agent_activity],
    workflow_runner=UnsandboxedWorkflowRunner(),  # 关键：禁用沙箱
)
```

---

## 10. Skills 技能系统 (`agentica/skills/`)

Skills 是 Anthropic 提出的提升 Agent 能力的方法，通过注入文本指令到系统提示词中。

### 10.1 目录结构

```
skills/
├── __init__.py        # 统一导出
├── skill.py           # Skill 数据类
├── skill_registry.py  # 技能注册表
└── skill_loader.py    # 技能加载器
```

### 10.2 Skill 类 (`skill.py`)

```python
@dataclass
class Skill:
    """
    技能定义，从 SKILL.md 文件加载。
    
    SKILL.md 格式：
    ```markdown
    ---
    name: My Skill
    description: A skill for doing something useful.
    license: MIT
    allowed-tools:
      - shell
      - python
    ---
    
    # My Skill
    ## Overview
    This skill helps you do something useful...
    ```
    """
    name: str
    description: str
    content: str  # Markdown 正文（指令）
    path: Path    # 技能目录路径
    
    # 可选元数据
    license: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: str = "project"  # project, user, managed
    
    @classmethod
    def from_skill_md(cls, skill_md_path: Path, location: str = "project") -> Optional["Skill"]
    
    def get_prompt(self) -> str  # 获取完整提示词
    def to_xml(self) -> str      # XML 格式输出
    def to_dict(self) -> Dict[str, Any]
```

### 10.3 SkillRegistry (`skill_registry.py`)

```python
class SkillRegistry:
    """
    技能注册表，管理已加载的技能。
    优先级：project > user > managed
    """
    
    def register(self, skill: Skill) -> bool
    def get(self, name: str) -> Optional[Skill]
    def exists(self, name: str) -> bool
    def list_all(self) -> List[Skill]
    def list_by_location(self, location: str) -> List[Skill]
    def remove(self, name: str) -> bool
    def clear(self)
    def generate_skills_prompt(self, char_budget: int = 10000) -> str
    def get_skill_instruction(self) -> str

# 全局注册表
def get_skill_registry() -> SkillRegistry
def reset_skill_registry()
```

### 10.4 SkillLoader (`skill_loader.py`)

```python
class SkillLoader:
    """
    技能加载器，从标准目录发现和加载技能。
    
    搜索路径（优先级顺序）：
    1. .claude/skills (项目级)
    2. .agentica/skills (项目级)
    3. ~/.claude/skills (用户级)
    4. ~/.agentica/skills (用户级)
    """
    
    def __init__(self, project_root: Optional[Path] = None)
    def get_search_paths(self) -> List[tuple]
    def discover_skills(self, skills_dir: Path) -> List[Path]
    def load_skill(self, skill_md_path: Path, location: str) -> Optional[Skill]
    def load_skill_from_dir(self, skill_dir: str, location: str = "project") -> Optional[Skill]
    def load_all(self, registry: Optional[SkillRegistry] = None) -> SkillRegistry
    def reload(self, registry: Optional[SkillRegistry] = None) -> SkillRegistry
    
    @staticmethod
    def list_skill_files(directory: str) -> str
    @staticmethod
    def read_skill_file(file_path: str) -> str

# 便捷函数
def load_skills(project_root: Optional[Path] = None) -> SkillRegistry
def get_available_skills() -> List[Skill]
def register_skill(skill_dir: str, location: str = "project") -> Optional[Skill]
def register_skills(skill_dirs: List[str], location: str = "project") -> List[Skill]
def list_skill_files(directory: str) -> str
def read_skill_file(file_path: str) -> str
```

**使用示例**：
```python
from agentica.skills import load_skills, get_available_skills, register_skill

# 加载所有技能
registry = load_skills()

# 获取可用技能
skills = get_available_skills()

# 注册单个技能
skill = register_skill("./my-skills/web-research")
```

---

## 11. Guardrails 护栏系统

Guardrails 是对 Agent 输入/输出进行验证、过滤或阻止的检查机制。

### 11.1 Agent Guardrails (`agentica/guardrails.py`)

用于验证 Agent 的输入和输出。

```
执行流程：
┌─────────────────────────────────────────────────────────────┐
│  1. User Input Received                                      │
│         ↓                                                    │
│  2. InputGuardrail.run() ──→ Check if input is valid         │
│         ↓                                                    │
│     ┌─ tripwire_triggered=False → Continue execution         │
│     └─ tripwire_triggered=True  → Raise exception, halt      │
│         ↓                                                    │
│  3. Agent Processing (LLM call, tool calls, etc.)            │
│         ↓                                                    │
│  4. OutputGuardrail.run() ──→ Check if output is valid       │
│         ↓                                                    │
│     ┌─ tripwire_triggered=False → Return output              │
│     └─ tripwire_triggered=True  → Raise exception, halt      │
│         ↓                                                    │
│  5. Return Result to User                                    │
└─────────────────────────────────────────────────────────────┘
```

#### 核心类

```python
@dataclass
class GuardrailFunctionOutput:
    """护栏函数输出"""
    output_info: Any = None           # 检查信息
    tripwire_triggered: bool = False  # 是否触发阻止
    
    @classmethod
    def allow(cls, output_info=None) -> "GuardrailFunctionOutput"
    @classmethod
    def block(cls, output_info=None) -> "GuardrailFunctionOutput"

@dataclass
class InputGuardrail(Generic[TContext]):
    """输入护栏 - 检查 Agent 输入"""
    guardrail_function: InputGuardrailFunc
    name: Optional[str] = None
    run_in_parallel: bool = True  # 是否与 Agent 并行运行
    
    async def run(self, agent, input_data, context=None) -> InputGuardrailResult

@dataclass
class OutputGuardrail(Generic[TContext]):
    """输出护栏 - 检查 Agent 输出"""
    guardrail_function: OutputGuardrailFunc
    name: Optional[str] = None
    
    async def run(self, agent, agent_output, context=None) -> OutputGuardrailResult
```

#### 装饰器

```python
@input_guardrail
async def check_topic(ctx, agent, input_data):
    if "off-topic" in str(input_data):
        return GuardrailFunctionOutput.block({"reason": "off-topic content"})
    return GuardrailFunctionOutput.allow()

@output_guardrail
async def check_sensitive_data(ctx, agent, output):
    if "password" in str(output).lower():
        return GuardrailFunctionOutput.block({"reason": "sensitive data"})
    return GuardrailFunctionOutput.allow()
```

#### 使用示例

```python
from agentica import Agent
from agentica.guardrails import input_guardrail, output_guardrail, GuardrailFunctionOutput

@input_guardrail
def check_input(ctx, agent, input_data):
    if len(str(input_data)) > 10000:
        return GuardrailFunctionOutput.block({"reason": "input too long"})
    return GuardrailFunctionOutput.allow()

@output_guardrail
def check_output(ctx, agent, output):
    if "error" in str(output).lower():
        return GuardrailFunctionOutput.block({"reason": "error in output"})
    return GuardrailFunctionOutput.allow()

agent = Agent(
    name="GuardedAgent",
    input_guardrails=[check_input],
    output_guardrails=[check_output],
)
```

### 11.2 Tool Guardrails (`agentica/tools/guardrails.py`)

用于验证工具的输入和输出。

```
执行流程：
┌─────────────────────────────────────────────────────────────┐
│  1. Tool Call Received                                       │
│         ↓                                                    │
│  2. ToolInputGuardrail.run() ──→ Check if input is valid     │
│         ↓                                                    │
│     ┌─ allow ──────────→ Continue execution                  │
│     ├─ reject_content ─→ Return message, skip tool execution │
│     └─ raise_exception → Raise exception, halt execution     │
│         ↓                                                    │
│  3. Tool Function Execution                                  │
│         ↓                                                    │
│  4. ToolOutputGuardrail.run() ──→ Check if output is valid   │
│         ↓                                                    │
│     ┌─ allow ──────────→ Return original result              │
│     ├─ reject_content ─→ Replace result with message         │
│     └─ raise_exception → Raise exception, halt execution     │
│         ↓                                                    │
│  5. Return Result to LLM                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 核心类

```python
@dataclass
class ToolGuardrailFunctionOutput:
    """工具护栏函数输出"""
    output_info: Any = None
    behavior: Union[AllowBehavior, RejectContentBehavior, RaiseExceptionBehavior]
    
    @classmethod
    def allow(cls, output_info=None) -> "ToolGuardrailFunctionOutput"
    @classmethod
    def reject_content(cls, message: str, output_info=None) -> "ToolGuardrailFunctionOutput"
    @classmethod
    def raise_exception(cls, output_info=None) -> "ToolGuardrailFunctionOutput"

@dataclass
class ToolContext:
    """工具上下文"""
    tool_name: str
    tool_arguments: Optional[str] = None
    tool_call_id: Optional[str] = None
    agent: Optional["Agent[Any]"] = None

@dataclass
class ToolInputGuardrail(Generic[TContext]):
    """工具输入护栏"""
    guardrail_function: ToolInputGuardrailFunc
    name: Optional[str] = None

@dataclass
class ToolOutputGuardrail(Generic[TContext]):
    """工具输出护栏"""
    guardrail_function: ToolOutputGuardrailFunc
    name: Optional[str] = None
```

#### 使用示例

```python
from agentica.tools.guardrails import (
    tool_input_guardrail, tool_output_guardrail,
    ToolGuardrailFunctionOutput, ToolInputGuardrailData, ToolOutputGuardrailData
)
import json

@tool_input_guardrail
def check_file_path(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    args = json.loads(data.context.tool_arguments or "{}")
    if "/etc/" in str(args) or "/root/" in str(args):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Access to system directories is forbidden"
        )
    return ToolGuardrailFunctionOutput.allow()

@tool_output_guardrail
def sanitize_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    if "password" in str(data.output).lower():
        return ToolGuardrailFunctionOutput.reject_content(
            message="[REDACTED - sensitive data removed]"
        )
    return ToolGuardrailFunctionOutput.allow()
```

---

## 12. CLI 命令行接口 (`agentica/cli.py`)

### 12.1 功能特性

```
交互功能：
  Enter           提交消息
  Alt+Enter       插入换行（多行输入）
  Ctrl+J          插入换行（替代方式）
  @filename       文件引用（自动补全并注入内容）
  /command        命令（自动补全）

命令：
  /help           显示帮助
  /clear          清屏并重置对话
  /tools          列出可用工具
  /exit, /quit    退出
```

### 12.2 工具注册表

```python
# 可通过 --tools 参数启用的额外工具
TOOL_REGISTRY = {
    # AI/ML 工具
    'cogvideo': ('cogvideo', 'CogVideoTool'),
    'cogview': ('cogview', 'CogViewTool'),
    'dalle': ('dalle', 'DalleTool'),
    'image_analysis': ('image_analysis', 'ImageAnalysisTool'),
    'ocr': ('ocr', 'OcrTool'),
    'video_analysis': ('video_analysis', 'VideoAnalysisTool'),
    'volc_tts': ('volc_tts', 'VolcTtsTool'),
    
    # 搜索工具
    'arxiv': ('arxiv', 'ArxivTool'),
    'baidu_search': ('baidu_search', 'BaiduSearchTool'),
    'dblp': ('dblp', 'DblpTool'),
    'duckduckgo': ('duckduckgo', 'DuckDuckGoTool'),
    'search_bocha': ('search_bocha', 'SearchBochaTool'),
    'search_exa': ('search_exa', 'SearchExaTool'),
    'search_serper': ('search_serper', 'SearchSerperTool'),
    'web_search_pro': ('web_search_pro', 'WebSearchProTool'),
    'wikipedia': ('wikipedia', 'WikipediaTool'),
    
    # 网络工具
    'browser': ('browser', 'BrowserTool'),
    'jina': ('jina', 'JinaTool'),
    'newspaper': ('newspaper', 'NewspaperTool'),
    'url_crawler': ('url_crawler', 'UrlCrawlerTool'),
    
    # 文件/代码工具
    'calculator': ('calculator', 'CalculatorTool'),
    'code': ('code', 'CodeTool'),
    'edit': ('edit', 'EditTool'),
    'file': ('file', 'FileTool'),
    'run_nb_code': ('run_nb_code', 'RunNbCodeTool'),
    'run_python_code': ('run_python_code', 'RunPythonCodeTool'),
    'shell': ('shell', 'ShellTool'),
    'workspace': ('workspace', 'WorkspaceTool'),
    
    # 数据工具
    'hackernews': ('hackernews', 'HackerNewsTool'),
    'sql': ('sql', 'SqlTool'),
    'weather': ('weather', 'WeatherTool'),
    'yfinance': ('yfinance', 'YFinanceTool'),
    
    # 集成工具
    'airflow': ('airflow', 'AirflowTool'),
    'apify': ('apify', 'ApifyTool'),
    'mcp': ('mcp', 'MCPTool'),
    'memori': ('memori', 'MemoriTool'),
    'skill': ('skill', 'SkillTool'),
    'video_download': ('video_download', 'VideoDownloadTool'),
}
```

### 12.3 使用方式

```bash
# 交互模式
agentica

# 带查询的非交互模式
agentica --query "List Python files"

# 指定模型
agentica --model_provider openai --model_name gpt-4o

# 启用额外工具
agentica --tools calculator shell wikipedia

# 指定工作目录
agentica --work_dir /path/to/project
```

---

## 13. Token 计数 (`agentica/utils/tokens.py`)

```python
# 文本 Token 计数
def count_text_tokens(text: str, model_id: str = "gpt-4o") -> int

# 图像 Token 计数 (基于 OpenAI 公式)
def count_image_tokens(image: Image) -> int

# 音频 Token 估算
def count_audio_tokens(audio: Audio, duration=None) -> int

# 视频 Token 估算
def count_video_tokens(video: Video, duration=None, fps=1.0) -> int

# 工具定义 Token 计数
def count_tool_tokens(tools: Sequence, model_id="gpt-4o") -> int

# 消息 Token 计数
def count_message_tokens(message: Message, model_id="gpt-4o") -> int

# 总 Token 计数
def count_tokens(
    messages: List[Message],
    tools: Optional[List] = None,
    model_id: str = "gpt-4o",
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
) -> int
```

**支持的分词器**：
- **tiktoken**: OpenAI 模型
- **HuggingFace tokenizers**: Llama-3, Llama-2, Cohere Command-R

---

## 14. Workflow 系统 (`agentica/workflow.py`)

```python
class Workflow(BaseModel):
    """工作流引擎"""
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # 用户/会话
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_state: Dict[str, Any] = Field(default_factory=dict)
    
    # 内存和数据库
    memory: WorkflowMemory = WorkflowMemory()
    db: Optional[BaseDb] = None
    
    # 子类需实现的方法
    def run(self, *args, **kwargs):
        """子类实现具体工作流逻辑"""
        pass
    
    # 内部方法
    def run_workflow(self, *args, **kwargs)  # 包装 run 方法
    def load_session(self, force=False) -> Optional[str]
    def read_from_storage(self) -> Optional[WorkflowSession]
    def write_to_storage(self) -> Optional[WorkflowSession]
    def deep_copy(self, update=None) -> "Workflow"
```

**使用示例**：
```python
class MyWorkflow(Workflow):
    agent1: Agent = Field(...)
    agent2: Agent = Field(...)
    
    def run(self, input_data: str) -> RunResponse:
        result1 = self.agent1.run(input_data)
        result2 = self.agent2.run(result1.content)
        return RunResponse(content=result2.content)
```

---

## 15. 其他重要模块

### 15.1 Media 模块 (`agentica/media.py`)

```python
class Image(BaseModel):
    url: Optional[str] = None                  # 远程 URL
    filepath: Optional[Union[Path, str]] = None # 本地路径
    content: Optional[Any] = None              # 字节内容
    format: Optional[str] = None               # png, jpeg, webp, gif
    detail: Optional[str] = None               # low, medium, high, auto

class Audio(BaseModel):
    content: Optional[Any] = None
    filepath: Optional[Union[Path, str]] = None
    url: Optional[str] = None
    format: Optional[str] = None

class Video(BaseModel):
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None
    format: Optional[str] = None

class AudioResponse(BaseModel):
    """模型音频响应"""
    id: Optional[str] = None
    content: Optional[str] = None              # Base64 编码
    transcript: Optional[str] = None
    sample_rate: Optional[int] = 24000
```

### 15.2 Knowledge 系统 (`agentica/knowledge/`)

```
knowledge/
├── base.py                    # Knowledge 基类
├── langchain_knowledge.py     # LangChain 集成
└── llamaindex_knowledge.py    # LlamaIndex 集成
```

```python
class Knowledge(BaseModel):
    """知识库管理"""
    data_path: Optional[Union[str, List[str]]] = None
    vector_db: Optional[VectorDb] = None
    num_documents: int = 3
    chunk_size: int = 2000
    chunk: bool = True
    
    # 核心方法
    def search(self, query: str, num_documents=None) -> List[Document]
    def load(self, recreate=False, upsert=False) -> None
    def load_documents(self, documents: List[Document]) -> None
    def read_file(self, path: Path) -> List[Document]
    def read_url(self, url: str) -> List[Document]
    def chunk_document(self, document: Document) -> List[Document]
```

### 15.3 MCP 模块 (`agentica/mcp/`)

```
mcp/
├── client.py    # MCP 客户端
├── server.py    # MCP 服务器实现
├── config.py    # MCP 配置
└── README.md    # MCP 文档
```

```python
class MCPServer(abc.ABC):
    """MCP 服务器基类"""
    async def connect(self)
    async def cleanup(self)
    async def list_tools(self) -> List[MCPTool]
    async def call_tool(self, tool_name: str, arguments: Dict) -> CallToolResult

class MCPServerStdio(MCPServer):
    """Stdio 传输的 MCP 服务器"""
    
class MCPServerSse(MCPServer):
    """SSE 传输的 MCP 服务器"""
    
class MCPServerStreamableHttp(MCPServer):
    """Streamable HTTP 传输的 MCP 服务器"""

class MCPClient:
    """MCP 客户端"""
    async def connect(self) -> None
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any
    async def list_tools(self) -> List[MCPTool]
    async def cleanup(self) -> None
```

---

## 16. 公开 API (`__init__.py`)

主要导出的类和函数：

```python
# Agent
from agentica import Agent, DeepAgent, Workflow

# Model
from agentica import (
    Model, Message, OpenAIChat, AzureOpenAIChat,
    Moonshot, DeepSeek, Qwen, ZhipuAI, Doubao, Yi, Grok, Together
)

# Memory
from agentica import AgentMemory, Memory, MemoryManager, WorkflowMemory

# Database
from agentica import BaseDb, SqliteDb, PostgresDb, InMemoryDb, JsonDb

# Knowledge & RAG
from agentica import Knowledge, VectorDb, InMemoryVectorDb, Document

# Tools
from agentica import Tool, Function, FunctionCall

# Guardrails
from agentica.guardrails import (
    InputGuardrail, OutputGuardrail,
    input_guardrail, output_guardrail,
    GuardrailFunctionOutput,
)
from agentica.tools.guardrails import (
    ToolInputGuardrail, ToolOutputGuardrail,
    tool_input_guardrail, tool_output_guardrail,
    ToolGuardrailFunctionOutput,
)

# Compression & Tokens
from agentica import CompressionManager, count_tokens, count_text_tokens

# MCP
from agentica import MCPConfig

# Temporal
from agentica.temporal import (
    TemporalClient, AgentWorkflow, SequentialAgentWorkflow,
    ParallelAgentWorkflow, ParallelTranslationWorkflow,
    WorkflowInput, WorkflowOutput, TranslationInput,
)

# Skills
from agentica.skills import (
    Skill, SkillRegistry, SkillLoader,
    load_skills, get_available_skills, register_skill,
)

# Response
from agentica import RunResponse, RunEvent, pprint_run_response
```

---

## 17. 示例目录 (`examples/`)

包含 50+ 示例文件，涵盖：
- 基础 LLM 调用
- 工具使用
- RAG 实现
- 多 Agent 协作
- Workflow 编排
- MCP 集成
- 长期记忆
- Token 压缩
- Temporal 分布式工作流
- Skills 技能系统
- Guardrails 护栏系统

---

## 18. 开发指南

### 18.1 添加新模型

1. 在 `agentica/model/` 下创建新目录
2. 继承 `Model` 基类
3. 实现 `response()` 和 `aresponse()` 方法
4. 在 `__init__.py` 中导出

### 18.2 添加新工具

1. 在 `agentica/tools/` 下创建新文件
2. 继承 `Tool` 类或使用 `@tool` 装饰器
3. 定义函数签名和描述
4. 在 `__init__.py` 中导出（可选）

### 18.3 添加新向量数据库

1. 在 `agentica/vectordb/` 下创建新文件
2. 继承 `VectorDb` 基类
3. 实现 `insert()`, `search()`, `upsert()` 等方法

### 18.4 添加新 Skill

1. 创建技能目录（如 `.agentica/skills/my-skill/`）
2. 创建 `SKILL.md` 文件，包含 YAML frontmatter 和指令
3. 可选：添加 `scripts/`, `references/`, `assets/` 子目录

