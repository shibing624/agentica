# Agentica TODO List

> 项目待办事项和改进计划

## 高优先级

### 1. 数据库 Schema 迁移功能
- [ ] `agentica/db/postgres.py:646` - 实现 `upgrade_schema()` 方法
- [ ] `agentica/db/sqlite.py:654` - 实现 `upgrade_schema()` 方法
- [ ] `agentica/db/json.py:455` - 实现 `upgrade_schema()` 方法
- [ ] `agentica/db/memory.py:296` - 实现 `upgrade_schema()` 方法

### 2. 核心模块单元测试
- [ ] `tests/test_agent.py` - Agent 核心类测试
- [ ] `tests/test_deep_agent.py` - DeepAgent 测试
- [ ] `tests/test_workflow.py` - Workflow 工作流测试
- [ ] `tests/test_memory.py` - Memory 系统测试
- [ ] `tests/test_cli.py` - CLI 命令行测试
- [ ] `tests/test_knowledge.py` - Knowledge 知识库测试

## 中优先级

### 3. 代码中的 TODO 注释
- [ ] `agentica/utils/markdown_converter.py:357` - Fix json type
- [ ] `agentica/utils/markdown_converter.py:733` - Deal with kwargs
- [ ] `agentica/utils/markdown_converter.py:746` - Deal with kwargs  
- [ ] `agentica/utils/markdown_converter.py:779` - Fix kwargs type

### 4. Tools 单元测试
现有 40+ 工具，仅少数有测试：
- [x] `test_edit_tool.py`
- [x] `test_jina_tool.py`
- [x] `test_url_crawler.py`
- [x] `test_skill_tool.py`
- [ ] 其他 tools 需要添加测试

### 5. 文档完善
- [ ] API 参考文档
- [ ] 工具使用指南
- [ ] 最佳实践文档

## 低优先级

### 6. 类型注解完善
- [ ] `agentica/vectordb/lancedb_vectordb.py:290` - 添加 pandas 类型提示

### 7. 功能增强
- [ ] `agentica/tools/browser_tool.py:480` - 添加左右滚动功能
- [ ] `agentica/tools/browser_tool.py:581` - 检查 shuffle 的必要性

## 已完成

### CLI 优化 (2024-12)
- [x] 优化 CLI 日志显示，类似 Cursor 风格
- [x] 添加工具图标和颜色区分
- [x] 支持 stream_intermediate_steps 显示工具调用
- [x] write_todos 显示 todo 列表
- [x] read_file 显示文件名和行号范围
- [x] execute 显示完整命令

---

## 项目结构概览

```
agentica/
├── agent.py          # 核心 Agent 类 (~3900 行)
├── deep_agent.py     # DeepAgent 子类 (~324 行)
├── deep_tools.py     # 内置工具集 (~1121 行)
├── cli.py            # CLI 界面 (~911 行)
├── workflow.py       # 工作流引擎
├── memory.py         # 记忆系统
├── knowledge/        # 知识库模块
├── tools/            # 40+ 工具实现
├── model/            # LLM 模型适配
├── db/               # 数据库适配
└── vectordb/         # 向量数据库
```

## 测试覆盖情况

| 模块 | 测试状态 |
|------|----------|
| Agent | ❌ 缺失 |
| DeepAgent | ❌ 缺失 |
| Workflow | ❌ 缺失 |
| Memory | ❌ 缺失 |
| CLI | ❌ 缺失 |
| Knowledge | ❌ 缺失 |
| Guardrails | ✅ 已有 |
| Edit Tool | ✅ 已有 |
| Jina Tool | ✅ 已有 |
| URL Crawler | ✅ 已有 |
