# Agentica 长期记忆功能

## 概述

Agentica 支持长期记忆功能，允许 Agent 在多轮对话中记住用户的个人信息、偏好等。

## 支持的存储后端

| 类名 | 描述 | 检索方式 |
|------|------|---------|
| `CsvMemoryDb` | CSV 文件存储 | last_n, first_n |
| `SqliteMemoryDb` | SQLite 数据库存储 | last_n, first_n |
| `QdrantMemoryDb` | Qdrant 向量数据库 | semantic (语义搜索), keyword (关键词回退) |

## 检索模式 (MemoryRetrieval)

| 模式 | 描述 |
|------|------|
| `last_n` | 返回最近 N 条记忆 |
| `first_n` | 返回最早 N 条记忆 |
| `semantic` | 语义搜索（需要 QdrantMemoryDb + embedder） |

---

## QdrantMemoryDb 向量数据库

### 特性

- **语义搜索**：使用 embedding 模型进行向量相似度搜索
- **关键词回退**：当 embedder 不可用时，自动降级到关键词搜索
- **本地磁盘存储**：默认使用 `~/.agentica/qdrant_memory` 存储数据
- **兼容新旧 API**：自动适配 qdrant-client 新版 `query_points` 和旧版 `search` API

### 初始化参数

```python
from agentica import QdrantMemoryDb, OpenAIEmb

db = QdrantMemoryDb(
    collection="agent_memory",      # 集合名称
    embedder=OpenAIEmb(),           # Embedding 模型（可选，None 则使用关键词搜索）
    path="~/.agentica/qdrant_memory",  # 本地存储路径
    on_disk=True,                   # 是否使用磁盘存储
    url=None,                       # 远程 Qdrant 服务器 URL
    port=6333,                      # Qdrant 服务器端口
    api_key=None,                   # Qdrant Cloud API Key
)
```

### 支持的 Embedding 模型

| 类名 | API 提供商 |
|------|-----------|
| `OpenAIEmb` | OpenAI |
| `ZhipuAIEmb` | 智谱 AI |

---

## 使用示例

### 1. QdrantMemoryDb + 语义搜索

```python
from agentica import (
    Agent, OpenAIChat, AgentMemory, QdrantMemoryDb, OpenAIEmb,
    MemoryClassifier, MemoryRetrieval
)

# 创建带 embedder 的 QdrantMemoryDb
db = QdrantMemoryDb(
    collection="my_memory",
    embedder=OpenAIEmb(),  # 使用 OpenAI embedding
    on_disk=True,
)

# 创建 AgentMemory
memory = AgentMemory(
    db=db,
    user_id="user_123",
    num_memories=5,
    retrieval=MemoryRetrieval.semantic,  # 语义检索
    create_user_memories=True,
    update_user_memories_after_run=True,
    classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
    semantic_score_threshold=0.5,  # 相似度阈值
)

# 创建 Agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    memory=memory,
)

# 对话
agent.print_response("我叫张三，是一名软件工程师")

# 搜索记忆
relevant = memory.search_memories("张三的职业是什么？")
```

### 2. QdrantMemoryDb + 关键词搜索（无 embedder）

```python
from agentica import QdrantMemoryDb, AgentMemory, MemoryRetrieval

# 不传 embedder，自动使用关键词搜索
db = QdrantMemoryDb(
    collection="keyword_memory",
    embedder=None,  # 无 embedder
    on_disk=True,
    path="~/.agentica/qdrant_keyword",  # 使用不同路径避免冲突
)

memory = AgentMemory(
    db=db,
    user_id="user_456",
    num_memories=10,
    retrieval=MemoryRetrieval.semantic,  # 会自动回退到关键词搜索
    create_user_memories=True,
)

# 搜索（使用关键词匹配）
results = memory.search_memories("Python 编程")
```

### 3. CsvMemoryDb（传统方式，兼容旧代码）

```python
from agentica import CsvMemoryDb, AgentMemory, MemoryRetrieval, MemoryClassifier, OpenAIChat

# 使用 CsvMemoryDb（与之前完全兼容）
db = CsvMemoryDb(csv_file_path="outputs/memory.csv")

memory = AgentMemory(
    db=db,
    user_id="user_789",
    num_memories=10,
    retrieval=MemoryRetrieval.last_n,  # 返回最近 N 条
    create_user_memories=True,
    update_user_memories_after_run=True,
    classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini')),
)
```

---

## API 参考

### AgentMemory 参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `db` | `MemoryDb` | `QdrantMemoryDb()` | 存储后端 |
| `user_id` | `str` | `None` | 用户 ID |
| `num_memories` | `int` | `None` | 检索记忆数量 |
| `retrieval` | `MemoryRetrieval` | `last_n` | 检索模式 |
| `create_user_memories` | `bool` | `False` | 是否创建用户记忆 |
| `update_user_memories_after_run` | `bool` | `True` | 运行后是否更新记忆 |
| `classifier` | `MemoryClassifier` | `None` | 记忆分类器 |
| `semantic_score_threshold` | `float` | `0.5` | 语义搜索相似度阈值 |

### AgentMemory 方法

```python
# 搜索记忆（语义或关键词）
memories = memory.search_memories(query="搜索内容", limit=5)

# 获取格式化的相关记忆字符串
formatted = memory.get_relevant_memories_str(query="搜索内容", limit=5)

# 加载用户记忆
memory.load_user_memories(query="可选的搜索查询")
```

### QdrantMemoryDb 方法

```python
# 搜索记忆
results = db.search_memories(
    query="搜索内容",
    user_id="user_id",
    limit=5,
    score_threshold=0.5
)

# 获取格式化的相关记忆
formatted = db.get_relevant_memories(
    query="搜索内容",
    user_id="user_id",
    limit=5
)

# 插入/更新记忆
db.upsert_memory(memory_row)

# 读取记忆
memories = db.read_memories(user_id="user_id", limit=10, sort="desc")

# 删除记忆
db.delete_memory(id="memory_id")

# 清空记忆
db.clear()
```

---

## 注意事项

1. **并发访问**：同一路径的 Qdrant 本地存储不支持并发访问，多个实例需使用不同路径
2. **Embedding 模型**：语义搜索需要 embedding 模型，若不可用会自动降级到关键词搜索
3. **依赖安装**：使用 QdrantMemoryDb 需安装 `pip install qdrant-client`
4. **路径展开**：支持 `~` 路径，会自动展开为用户目录
