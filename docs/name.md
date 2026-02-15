# Agentica 统一命名重构方案

## 一、命名原则

1. **不缩写**：使用完整英文单词，如 `Embedding` 而非 `Emb`
2. **类名 = 厂商 + 功能**：如 `OpenAIEmbedding`、`JinaRerank`、`DeepSeekChat`
3. **Model 模块统一用 `Chat` 后缀**：表示"对话模型"，不用 `LLM`（LLM 太泛）
4. **VectorDB 参数名与类型名对齐**：`embedder` → `embedding`，`reranker` → `reranker`（保留，因为是实例）

---

## 二、Embedding 模块 (`agentica/emb/`)

### 2.1 目录重命名

```
agentica/emb/ → agentica/embedding/
```

### 2.2 基类重命名

| 旧文件名 | 新文件名 | 旧类名 | 新类名 |
|---|---|---|---|
| `base.py` | `base.py` | `Emb` | `Embedding` |

### 2.3 实现类重命名

| 旧文件名 | 新文件名 | 旧类名 | 新类名 |
|---|---|---|---|
| `openai_emb.py` | `openai.py` | `OpenAIEmb` | `OpenAIEmbedding` |
| `azure_openai_emb.py` | `azure_openai.py` | `AzureOpenAIEmb` | `AzureOpenAIEmbedding` |
| `ollama_emb.py` | `ollama.py` | `OllamaEmb` | `OllamaEmbedding` |
| `huggingface_emb.py` | `huggingface.py` | `HuggingfaceEmb` | `HuggingfaceEmbedding` |
| `gemini_emb.py` | `gemini.py` | `GeminiEmb` | `GeminiEmbedding` |
| `jina_emb.py` | `jina.py` | `JinaEmb` | `JinaEmbedding` |
| `zhipuai_emb.py` | `zhipuai.py` | `ZhipuAIEmb` | `ZhipuAIEmbedding` |
| `together_emb.py` | `together.py` | `TogetherEmb` | `TogetherEmbedding` |
| `fireworks_emb.py` | `fireworks.py` | `FireworksEmb` | `FireworksEmbedding` |
| `hash_emb.py` | `hash.py` | `HashEmb` | `HashEmbedding` |
| `http_emb.py` | `http.py` | `HttpEmb` | `HttpEmbedding` |
| `mulanai_emb.py` | `mulanai.py` | `MulanAIEmb` | `MulanAIEmbedding` |

### 2.4 `__init__.py` 导出

```python
from agentica.embedding.base import Embedding
```

---

## 三、Rerank 模块 (`agentica/rerank/`)

### 3.1 目录保持不变

```
agentica/rerank/  （不变）
```

### 3.2 基类重命名

| 旧文件名 | 新文件名 | 旧类名 | 新类名 |
|---|---|---|---|
| `base.py` | `base.py` | `Reranker` | `Rerank` |

### 3.3 实现类重命名

| 旧文件名 | 新文件名 | 旧类名 | 新类名 |
|---|---|---|---|
| `cohere.py` | `cohere.py` | `CohereReranker` | `CohereRerank` |
| `jina.py` | `jina.py` | `JinaReranker` | `JinaRerank` |
| `zhipuai.py` | `zhipuai.py` | `ZhipuAIReranker` | `ZhipuAIRerank` |

### 3.4 `__init__.py` 导出

```python
from agentica.rerank.base import Rerank
```

---

## 四、Model 模块 (`agentica/model/`)

### 4.1 统一后缀规则

- **通用模型**：`厂商名 + Chat`（如 `OpenAIChat`）
- **已经是简洁名的保持不变**：如 `Gemini`、`Claude`、`Ollama`、`Groq` 等（这些本身就是产品名，不需要后缀）
- **删除所有 `LLM` 别名**

### 4.2 类名修改

| 旧类名 | 新类名 | 说明 |
|---|---|---|
| `OpenAIChat` | `OpenAIChat` | 不变 |
| `OpenAILLM`（别名） | **删除** | |
| `OpenAILike` | `OpenAILike` | 不变，这是基类 |
| `AzureOpenAIChat` | `AzureOpenAIChat` | 不变 |
| `AzureOpenAILLM`（别名） | **删除** | |
| `Moonshot` | `MoonshotChat` | 统一加 Chat 后缀 |
| `MoonshotChat`（别名） | `MoonshotChat` | 变为主名 |
| `MoonshotLLM`（别名） | **删除** | |
| `DeepSeek` | `DeepSeekChat` | 统一加 Chat 后缀 |
| `DeepSeekChat`（别名） | `DeepSeekChat` | 变为主名 |
| `DeepSeekLLM`（别名） | **删除** | |
| `Doubao` | `DoubaoChat` | 统一加 Chat 后缀 |
| `DoubaoChat`（别名） | `DoubaoChat` | 变为主名 |
| `Together` | `TogetherChat` | 统一加 Chat 后缀 |
| `TogetherChat`（别名） | `TogetherChat` | 变为主名 |
| `TogetherLLM`（别名） | **删除** | |
| `Grok` | `GrokChat` | 统一加 Chat 后缀 |
| `GrokChat`（别名） | `GrokChat` | 变为主名 |
| `Yi` | `YiChat` | 统一加 Chat 后缀 |
| `YiChat`（别名） | `YiChat` | 变为主名 |
| `YiLLM`（别名） | **删除** | |
| `Qwen` | `QwenChat` | 统一加 Chat 后缀 |
| `ZhipuAI` | `ZhipuAIChat` | 统一加 Chat 后缀 |
| `ZhipuAIChat`（别名） | `ZhipuAIChat` | 变为主名 |
| `ZhipuAILLM`（别名） | **删除** | |
| `LiteLLM` | `LiteLLMChat` | 统一加 Chat 后缀 |

**保持不变的类**（产品名本身就是唯一标识）：

| 类名 | 说明 |
|---|---|
| `Claude` | Anthropic 的产品名 |
| `Gemini` | Google 的产品名 |
| `Ollama` | 本地模型框架名 |
| `Groq` | 推理加速平台名 |
| `Cohere` | Cohere 的产品名 |
| `Fireworks` | Fireworks AI 平台名 |
| `Nvidia` | Nvidia NIM 平台名 |
| `OpenRouter` | OpenRouter 平台名 |
| `Sambanova` | Sambanova 平台名 |

### 4.3 文件名修改

| 旧文件路径 | 新文件路径 |
|---|---|
| `model/moonshot/chat.py` | 不变（类名改为 `MoonshotChat`） |
| `model/deepseek/chat.py` | 不变（类名改为 `DeepSeekChat`） |
| `model/doubao/chat.py` | 不变（类名改为 `DoubaoChat`） |
| `model/together/together.py` | `model/together/chat.py`（类名改为 `TogetherChat`） |
| `model/xai/grok.py` | `model/xai/chat.py`（类名改为 `GrokChat`） |
| `model/yi/chat.py` | 不变（类名改为 `YiChat`） |
| `model/qwen/chat.py` | 不变（类名改为 `QwenChat`） |
| `model/zhipuai/chat.py` | 不变（类名改为 `ZhipuAIChat`） |
| `model/litellm/chat.py` | 不变（类名改为 `LiteLLMChat`） |

---

## 五、VectorDB 参数名修改

所有 VectorDB 实现类中的参数名统一修改：

| 旧参数名 | 新参数名 | 类型变化 |
|---|---|---|
| `embedder: Emb` | `embedding: Embedding` | `Emb` → `Embedding` |
| `reranker: Reranker` | `reranker: Rerank` | `Reranker` → `Rerank` |

### 5.1 受影响的文件

| 文件 | 修改内容 |
|---|---|
| `vectordb/lancedb_vectordb.py` | `embedder` → `embedding`，类型 `Emb` → `Embedding`，`Reranker` → `Rerank` |
| `vectordb/chromadb_vectordb.py` | 同上 |
| `vectordb/memory_vectordb.py` | 同上 |
| `vectordb/pgvectordb.py` | 同上 |
| `vectordb/pineconedb.py` | 同上 |
| `vectordb/qdrantdb.py` | 同上 |

### 5.2 Document 类修改

| 文件 | 修改内容 |
|---|---|
| `document.py` | `embedder: Optional[Emb]` → `embedding: Optional[Embedding]`，`embed(embedder)` → `embed(embedding)` |

### 5.3 其他受影响文件

| 文件 | 修改内容 |
|---|---|
| `memory/search.py` | `embedder` 参数 → `embedding` |
| `workspace.py` | `embedder` 参数 → `embedding` |

---

## 六、`__init__.py` 导出修改

### 6.1 删除所有 LLM 别名

```python
# 删除这些行
from agentica.model.openai.chat import OpenAIChat as OpenAILLM
from agentica.model.azure.openai_chat import AzureOpenAIChat as AzureOpenAILLM
from agentica.model.moonshot import Moonshot as MoonshotLLM
from agentica.model.deepseek.chat import DeepSeek as DeepSeekLLM
from agentica.model.together.together import Together as TogetherLLM
from agentica.model.yi.chat import Yi as YiLLM
from agentica.model.zhipuai.chat import ZhipuAI as ZhipuAILLM
```

### 6.2 更新 Embedding 导入

```python
# 旧
"Emb": ("agentica.emb.base", "Emb"),
"OpenAIEmb": ("agentica.emb.openai_emb", "OpenAIEmb"),
...

# 新
"Embedding": ("agentica.embedding.base", "Embedding"),
"OpenAIEmbedding": ("agentica.embedding.openai", "OpenAIEmbedding"),
...
```

### 6.3 更新 Rerank 导入

```python
# 旧
"Reranker": ("agentica.rerank.base", "Reranker"),
"CohereReranker": ("agentica.rerank.cohere", "CohereReranker"),
...

# 新
"Rerank": ("agentica.rerank.base", "Rerank"),
"CohereRerank": ("agentica.rerank.cohere", "CohereRerank"),
...
```

### 6.4 更新 Model 导入

```python
# 旧
from agentica.model.moonshot import Moonshot
from agentica.model.moonshot import Moonshot as MoonshotChat
from agentica.model.moonshot import Moonshot as MoonshotLLM

# 新
from agentica.model.moonshot.chat import MoonshotChat
```

---

## 七、Examples 和 Tests 修改

所有引用旧名称的地方统一替换：

| 旧名称 | 新名称 |
|---|---|
| `OpenAIEmb()` | `OpenAIEmbedding()` |
| `ZhipuAIEmb()` | `ZhipuAIEmbedding()` |
| `JinaEmb()` | `JinaEmbedding()` |
| `GeminiEmb()` | `GeminiEmbedding()` |
| `HuggingfaceEmb()` | `HuggingfaceEmbedding()` |
| `ZhipuAIReranker()` | `ZhipuAIRerank()` |
| `CohereReranker()` | `CohereRerank()` |
| `JinaReranker()` | `JinaRerank()` |
| `OpenAILLM` | `OpenAIChat` |
| `DeepSeekLLM` | `DeepSeekChat` |
| `ZhipuAILLM` | `ZhipuAIChat` |
| `embedder=...` | `embedding=...`（VectorDB 构造参数处） |
| `reranker=...` | `reranker=...`（不变） |

---

## 八、修改后的使用示例

```python
from agentica import Knowledge, LanceDb, SearchType
from agentica import OpenAIEmbedding, ZhipuAIRerank

knowledge = Knowledge(
    data_path=pdf_path,
    vector_db=LanceDb(
        table_name='paper_sample',
        uri='tmp/paper_lancedb',
        search_type=SearchType.vector,
        embedding=OpenAIEmbedding(),
        reranker=ZhipuAIRerank(),
    )
)
```

```python
from agentica import Agent, DeepSeekChat

agent = Agent(model=DeepSeekChat())
```

---

## 九、命名规范总结

| 模块 | 基类名 | 类名模式 | 示例 |
|---|---|---|---|
| Embedding | `Embedding` | `厂商Embedding` | `OpenAIEmbedding`, `JinaEmbedding` |
| Rerank | `Rerank` | `厂商Rerank` | `CohereRerank`, `ZhipuAIRerank` |
| Model | `Model` | `厂商Chat` 或 `产品名` | `OpenAIChat`, `DeepSeekChat`, `Claude`, `Gemini` |
| VectorDB | `VectorDb` | `产品Db` | `LanceDb`, `ChromaDb`, `QdrantDb` |
| Knowledge | `Knowledge` | `框架Knowledge` | `Knowledge`, `LlamaIndexKnowledge` |
