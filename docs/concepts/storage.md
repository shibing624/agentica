# Storage: Database & Media

Agentica 的存储抽象层：会话/记忆/指标用统一数据库接口，多模态生成内容用 Media Artifact 模型。

## Database Layer (`agentica/db/`)

统一存储抽象，覆盖 sessions、memories、metrics、knowledge 四类表。

**Base class** `BaseDb`（`agentica/db/base.py`）：CRUD 抽象方法。

**Implementations**：

| 实现 | 说明 |
|------|------|
| `SqliteDb` | 文件型，无外部依赖（默认） |
| `PostgresDb` | 生产级关系型 |
| `MysqlDb` | MySQL / MariaDB |
| `RedisDb` | 高速缓存 / 会话存储 |
| `JsonDb` | 文件型 JSON（开发用） |
| `InMemoryDb` | 仅运行时内存（测试用） |

**Features**：

- `filter_base64_media()` 防止大 base64 字符串落库
- `SessionRow`、`MemoryRow`、`MetricsRow` 类型化 schema（`agentica/db/base.py`）
- 支持长期记忆、会话历史、用量指标

```python
from agentica import SqliteDb
db = SqliteDb(db_file="~/.agentica/data/agent.db")
```

## Media Artifacts (`agentica/media.py`)

生成式多模态内容的 Pydantic 模型，均继承 `Media` 基类（`id`、`original_prompt`、`revised_prompt`）：

| 模型 | 字段 | 说明 |
|------|------|------|
| `VideoArtifact` | `url`, `eta`, `length` | 远程视频 + 元数据 |
| `ImageArtifact` | `url` 或 `content`, `mime_type`, `alt_text` | 远程 URL 或图片 bytes |
| `AudioArtifact` | `url` 或 `base64_audio`, `mime_type`, `length` | 音频（校验器强制 URL 与 inline 二选一） |

输入侧媒体模型 `Video` / `Audio` / `Image`（本地 filepath / bytes / URL，互斥校验）可从 `agentica` 顶层直接导入。

## 下一步

- [Agent](agent.md) -- Agent 如何使用存储
- [Memory & Workspace](memory.md) -- 长期记忆机制
- [API 参考](../api/agent.md) -- 存储 API
