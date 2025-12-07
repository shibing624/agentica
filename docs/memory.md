我来分析 `ReActAgent` 的 `long_term_memory_mode` 以及 `long_term_memory` 的实现逻辑。首先让我查看相关代码。


让我读取关键的实现文件：


现在让我查看 embedding 模型的实现，了解默认使用什么 API：


现在我已经收集了足够的信息，让我给你详细分析：

---

## `long_term_memory_mode="both"` 详细分析

### 1. `long_term_memory_mode` 的三种模式

在 `ReActAgent` 中，`long_term_memory_mode` 有三个选项：

| 模式 | `_static_control` | `_agent_control` | 含义 |
|------|------------------|------------------|------|
| `"static_control"` | ✅ | ❌ | 开发者控制：每次 reply 开始时自动检索，结束时自动记录 |
| `"agent_control"` | ❌ | ✅ | Agent 控制：注册工具函数让 Agent 自主决定何时检索/记录 |
| `"both"` (默认) | ✅ | ✅ | 两者兼有 |

相关代码：
```python:148:155:src/agentscope/agent/_react_agent.py
self._static_control = long_term_memory and long_term_memory_mode in [
    "static_control",
    "both",
]
self._agent_control = long_term_memory and long_term_memory_mode in [
    "agent_control",
    "both",
]
```

### 2. 两种控制模式的具体行为

**Static Control（静态控制）**：
- **检索**：在 `reply()` 开始时调用 `_retrieve_from_long_term_memory()` 自动检索相关信息，并添加到 memory 中
- **记录**：在 `reply()` 结束时（第 381-388 行）自动将对话记录到长期记忆

**Agent Control（Agent 控制）**：
- 注册两个工具函数给 Agent 使用：
  - `retrieve_from_memory(keywords, limit)` - 按关键词检索
  - `record_to_memory(thinking, content)` - 记录内容

```python:160:167:src/agentscope/agent/_react_agent.py
if self._agent_control:
    self.toolkit.register_tool_function(
        long_term_memory.record_to_memory,
    )
    self.toolkit.register_tool_function(
        long_term_memory.retrieve_from_memory,
    )
```

---

## 3. Long-Term Memory 实现：基于 mem0

是的，**直接调用 mem0 库**。核心实现在 `Mem0LongTermMemory` 类中：

```python:253:253:src/agentscope/memory/_mem0_long_term_memory.py
self.long_term_working_memory = mem0.AsyncMemory(mem0_config)
```

### 存储和检索流程

| 操作 | 方法 | 底层调用 |
|------|------|---------|
| 记录 | `record()` / `record_to_memory()` | `mem0.AsyncMemory.add()` |
| 检索 | `retrieve()` / `retrieve_from_memory()` | `mem0.AsyncMemory.search()` |

### 向量存储

默认使用 **Qdrant** 本地存储：

```python:246:252:src/agentscope/memory/_mem0_long_term_memory.py
on_disk = kwargs.get("on_disk", True)
mem0_config.vector_store = (
    mem0.vector_stores.configs.VectorStoreConfig(
        config={"on_disk": on_disk},
    )
)
```

---

## 4. Embedding API 分析

**AgentScope 没有指定默认的 embedding 模型**，需要用户显式传入。

在 `Mem0LongTermMemory.__init__()` 中：
- 如果不传 `mem0_config`，则 `embedding_model` 是**必需参数**
- AgentScope 通过 `AgentScopeEmbedding` 包装类将自己的 embedding 模型适配到 mem0

```python:220:225:src/agentscope/memory/_mem0_long_term_memory.py
if model is None or embedding_model is None:
    raise ValueError(
        "model and embedding_model are required if mem0_config "
        "is not provided",
    )
```

### 支持的 Embedding 模型

AgentScope 提供以下 embedding 实现供用户选择：

| 类名 | API 提供商 |
|------|-----------|
| `OpenAITextEmbedding` | OpenAI |
| `DashScopeTextEmbedding` | 阿里云 DashScope |
| `DashScopeMultiModalEmbedding` | 阿里云 DashScope (多模态) |
| `GeminiTextEmbedding` | Google Gemini |
| `OllamaTextEmbedding` | Ollama (本地) |

### 使用示例

```python
from agentscope.embedding import OpenAITextEmbedding
from agentscope.memory import Mem0LongTermMemory

embedding_model = OpenAITextEmbedding(
    model_name="text-embedding-3-small",
    dimensions=1536,
)

long_term_memory = Mem0LongTermMemory(
    agent_name="my_agent",
    model=chat_model,
    embedding_model=embedding_model,  # 必须显式指定
)
```

---

## 总结

| 问题 | 答案 |
|------|------|
| `both` 模式含义 | 同时启用静态控制（自动检索/记录）和 Agent 控制（工具函数） |
| 是否使用 mem0 | ✅ 是，直接使用 `mem0.AsyncMemory` |
| 默认向量存储 | Qdrant（本地磁盘存储，`on_disk=True`） |
| 默认 embedding API | **无默认**，必须用户显式传入 `embedding_model` 参数 |