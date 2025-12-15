# Agentica 项目技术实现文档

> 本文档详述 Agentica 项目结构，方便开发者和 AI 编程助手快速了解代码架构。

## 项目概述

**Agentica** 是一个功能强大的 AI Agent 框架，支持多模型、多工具、多轮对话、RAG（检索增强生成）、工作流编排和 MCP（Model Context Protocol）集成。

**项目路径**: `/agentica`

---

## 1. 核心模块结构

### 1.1 agentica/ 目录核心文件

| 文件 | 主要职责 |
|------|----------|
| `agent.py` | **核心 Agent 类**，实现 AI 代理的完整功能 |
| `memory.py` | **内存管理系统**，包含会话记忆和长期记忆 |
| `workflow.py` | **工作流引擎**，支持多步骤任务编排 |
| `cli.py` | 命令行接口工具 |
| `react_agent.py` | ReACT 框架实现的推理 Agent |
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
├── db/              # 数据库层 - 会话/记忆持久化
├── vectordb/        # 向量数据库 - RAG 支持
├── knowledge/       # 知识库系统
├── emb/             # 嵌入模型
├── mcp/             # MCP 协议支持
├── compression/     # 上下文压缩
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

### 2.3 特殊 Agent 类型

#### Python 代码执行

使用 `Agent` + `RunPythonCodeTool` 实现 Python 代码执行：

```python
from agentica import Agent, OpenAIChat, RunPythonCodeTool

agent = Agent(
    name="Python Agent",
    model=OpenAIChat(),
    tools=[RunPythonCodeTool(save_and_run=True, pip_install=True)],
    instructions=["You are an expert Python programmer."],
    markdown=True,
)
```

#### ReactAgent (`agentica/react_agent.py`)

基于 ReACT 框架的推理 Agent：

```python
class ReactAgent:
    """使用 Thought -> Action -> Result 循环的推理 Agent"""
    def run(self, user_query, max_iterations=5) -> str
    def execute_action(self, action_str) -> str
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

## 5. Memory 和 VectorDB

### 5.1 Memory 系统 (`agentica/memory.py`)

#### 核心类

```python
class AgentMemory(BaseModel):
    """Agent 内存管理"""
    runs: List[AgentRun] = []                  # 运行历史
    messages: List[Message] = []               # 消息列表
    
    # 会话摘要
    create_session_summary: bool = False
    summary: Optional[SessionSummary] = None
    summarizer: Optional[MemorySummarizer] = None
    
    # 用户记忆（长期记忆）
    create_user_memories: bool = False
    db: Optional[BaseDb] = None                # 持久化数据库
    user_id: Optional[str] = None
    memories: Optional[List[Memory]] = None
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

class MemoryManager(BaseModel):
    """记忆管理器 - 添加/更新/删除记忆"""
    mode: Literal["model", "rule"] = "rule"    # 模式
    model: Optional[Model] = None
    db: Optional[BaseDb] = None
    
    def add_memory(self, memory: str) -> str
    def update_memory(self, id: str, memory: str) -> str
    def delete_memory(self, id: str) -> str
    def clear_memory(self) -> str

class MemoryClassifier(BaseModel):
    """记忆分类器 - 判断是否需要记忆"""
    def run(self, message: str) -> str  # 返回 "yes" 或 "no"

class MemorySummarizer(BaseModel):
    """会话摘要生成器"""
    def run(self, message_pairs: List[Tuple[Message, Message]]) -> SessionSummary

class WorkflowMemory(BaseModel):
    """Workflow 内存"""
    runs: List[WorkflowRun] = []
```

### 5.2 VectorDB 系统 (`agentica/vectordb/`)

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

#### VectorDb 基类

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

## 6. Compression 和 Token 计数

### 6.1 CompressionManager (`agentica/compression/manager.py`)

```python
@dataclass
class CompressionManager:
    """工具结果压缩管理器"""
    model: Optional[Any] = None                # 压缩用模型
    compress_tool_results: bool = True
    compress_tool_results_limit: Optional[int] = None  # 工具数量阈值
    compress_token_limit: Optional[int] = None         # Token 阈值
    compress_tool_call_instructions: Optional[str] = None
    
    # 核心方法
    def should_compress(self, messages, tools=None, model=None) -> bool
    def compress(self, messages: List[Message]) -> None
    async def acompress(self, messages: List[Message]) -> None
    def get_compression_ratio(self) -> float
    def get_stats(self) -> Dict[str, Any]
```

**压缩策略**：
- 保留关键信息：数字、日期、实体、标识符
- 移除冗余：介绍语、过渡词、格式化内容

### 6.2 Token 计数 (`agentica/utils/tokens.py`)

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

## 7. Workflow 系统

### 7.1 Workflow 类 (`agentica/workflow.py`)

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

## 8. 其他重要模块

### 8.1 Media 模块 (`agentica/media.py`)

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

### 8.2 Knowledge 系统 (`agentica/knowledge/`)

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

### 8.3 MCP 模块 (`agentica/mcp/`)

```
mcp/
├── client.py    # MCP 客户端
├── server.py    # MCP 服务器实现
├── config.py    # MCP 配置
└── README.md    # MCP 文档
```

#### MCPServer 类

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
```

#### MCPClient 类

```python
class MCPClient:
    """MCP 客户端"""
    async def connect(self) -> None
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any
    async def list_tools(self) -> List[MCPTool]
    async def cleanup(self) -> None
```

### 8.4 Database 层 (`agentica/db/`)

```
db/
├── base.py      # BaseDb 抽象基类
├── sqlite.py    # SQLite 实现
├── postgres.py  # PostgreSQL 实现
├── memory.py    # 内存数据库
└── json.py      # JSON 文件存储
```

```python
class BaseDb(ABC):
    """统一数据库抽象"""
    # Session 操作
    def create_session_table(self) -> None
    def read_session(self, session_id: str) -> Optional[SessionRow]
    def upsert_session(self, session: SessionRow) -> Optional[SessionRow]
    def delete_session(self, session_id: str) -> None
    
    # Memory 操作
    def create_memory_table(self) -> None
    def read_memories(self, user_id=None, limit=None) -> List[MemoryRow]
    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]
    def delete_memory(self, memory_id: str) -> None
    
    # Metrics 操作
    def create_metrics_table(self) -> None
    def insert_metrics(self, metrics: MetricsRow) -> None
    
    # Knowledge 操作 (RAG)
    def create_knowledge_table(self) -> None
    def upsert_knowledge(self, knowledge: KnowledgeRow) -> Optional[KnowledgeRow]
```

---

## 9. 公开 API (`__init__.py`)

主要导出的类和函数：

```python
# Agent
from agentica import Agent, Workflow

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

# Compression & Tokens
from agentica import CompressionManager, count_tokens, count_text_tokens

# MCP
from agentica import MCPConfig

# Response
from agentica import RunResponse, RunEvent, pprint_run_response
```

---

## 10. 示例目录 (`examples/`)

包含 50+ 示例文件，涵盖：
- 基础 LLM 调用
- 工具使用
- RAG 实现
- 多 Agent 协作
- Workflow 编排
- MCP 集成
- 长期记忆
- Token 压缩

---

## 11. 开发指南

### 11.1 添加新模型

1. 在 `agentica/model/` 下创建新目录
2. 继承 `Model` 基类
3. 实现 `response()` 和 `aresponse()` 方法
4. 在 `__init__.py` 中导出

### 11.2 添加新工具

1. 在 `agentica/tools/` 下创建新文件
2. 继承 `Tool` 类或使用 `@tool` 装饰器
3. 定义函数签名和描述
4. 在 `__init__.py` 中导出（可选）

### 11.3 添加新向量数据库

1. 在 `agentica/vectordb/` 下创建新文件
2. 继承 `VectorDb` 基类
3. 实现 `insert()`, `search()`, `upsert()` 等方法

---

*文档最后更新: 2025-12-15*
