# Agentica 工具重新设计方案

## 1. 设计原则

### 1.1 工具分层

| 层级 | 位置 | 职责 | 特点 |
|------|------|------|------|
| **基础层** | `deep_tools.py` | 提供 Agent 运行的基础能力 | 精简、稳定、立即可用 |
| **增益层** | `tools/*.py` | 提供特定领域的增强功能 | 专业、可选、有明确增益 |

### 1.2 核心原则

1. **不重复**: 增益层工具不重复实现基础层功能
2. **有增益**: 每个增益层工具必须提供基础层没有的价值
3. **正交性**: 工具之间职责清晰，不重叠

---

## 2. 基础层工具升级（deep_tools.py）

### 2.1 BuiltinFileTool 升级

#### edit_file 增强版

```python
def edit_file(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    allow_multiple_matches: bool = False,
) -> str:
    """
    精确替换文件中特定字符串，支持多行替换

    核心特性：
    1. 精确字符串匹配（非正则）
    2. 支持多行内容替换
    3. 顺序替换：按文件位置从左到右、从上到下依次替换
    4. 默认要求 old_string 在文件中唯一存在（防止误替换）
    5. 可通过 replace_all=True 替换所有匹配
    6. 可通过 allow_multiple_matches=True 跳过唯一性检查（只替换第一个）

    Args:
        file_path: 文件路径（支持绝对路径和相对于 base_dir 的路径）
        old_string: 要替换的原始字符串（精确匹配，包含空格和换行）
        new_string: 用于替换的新字符串
        replace_all: 是否替换所有匹配项，默认 False（只替换第一个）
        allow_multiple_matches: 当存在多个匹配时是否允许继续，默认 False
            - False: 发现多个匹配时抛出错误，提示用户提供更多上下文
            - True: 即使有多个匹配也继续，只替换第一个出现的

    Returns:
        操作结果描述

    Examples:
        # 单行替换
        edit_file("main.py", "def foo():", "def bar():")

        # 多行替换（保留缩进）
        edit_file(
            "main.py",
            old_string='''def old_func():
                pass''',
            new_string='''def new_func():
                return True'''
        )
    """
```

**实现关键点**:

```python
class StrReplaceEditor:
    """
    基于精确字符串匹配的文件编辑器

    算法：
    1. 读取整个文件内容
    2. 查找 old_string 的所有匹配位置
    3. 如果匹配数 > 1 且 replace_all=False 且 allow_multiple_matches=False，报错
    4. 按位置顺序（从左到右）执行替换
    5. 写回文件
    """

    def replace(
        self,
        content: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        allow_multiple_matches: bool = False
    ) -> tuple[str, int]:
        """
        执行替换

        Returns:
            (new_content, replacement_count)
        """
        # 查找所有匹配位置
        matches = []
        start = 0
        while True:
            idx = content.find(old_string, start)
            if idx == -1:
                break
            matches.append(idx)
            start = idx + 1

        if not matches:
            raise ValueError(f"String not found: {old_string[:50]}...")

        if len(matches) > 1 and not replace_all and not allow_multiple_matches:
            # 显示匹配位置的上下文，帮助用户定位
            contexts = []
            for idx in matches[:3]:  # 最多显示前3个
                line_num = content[:idx].count('\n') + 1
                context = content[idx:idx+len(old_string)+30].replace('\n', '\\n')
                contexts.append(f"  - Line {line_num}: ...{context}...")

            raise ValueError(
                f"Found {len(matches)} matches for the string. "
                f"Use replace_all=True to replace all, or provide more context to make it unique.\n"
                f"Matches found at:\n" + '\n'.join(contexts)
            )

        # 执行替换
        if replace_all:
            new_content = content.replace(old_string, new_string)
            return new_content, len(matches)
        else:
            # 只替换第一个（按位置顺序）
            idx = matches[0]
            new_content = content[:idx] + new_string + content[idx + len(old_string):]
            return new_content, 1
```

#### 新增 multi_edit（批量编辑）

```python
def multi_edit(
    self,
    file_path: str,
    edits: list[dict],
) -> str:
    """
    在同一个文件上顺序执行多个编辑操作

    用于一次修改文件的多个位置，保证原子性（全部成功或全部失败）

    Args:
        file_path: 文件路径
        edits: 编辑操作列表，每个操作包含 old_string 和 new_string
            [
                {"old_string": "...", "new_string": "..."},
                {"old_string": "...", "new_string": "..."},
            ]

    注意：
    - 编辑按数组顺序执行
    - 后续编辑看到的是前面编辑后的内容
    - 任意一步失败，所有修改都不会生效

    Example:
        multi_edit("main.py", [
            {"old_string": "def foo():", "new_string": "def bar():"},
            {"old_string": "x = 1", "new_string": "x = 2"},
        ])
    """
```

---

## 3. 增益层工具设计（tools/ 目录）

### 3.1 工具清单

| 工具 | 状态 | 说明 |
|------|------|------|
| FileTool | **删除** | 功能与基础层 BuiltinFileTool 重合 |
| EditTool | **删除** | 基础层 edit_file 已升级，无需单独存在 |
| CodeTool | **保留重构** | 专注于代码分析、格式化、lint 等代码特有功能 |
| LspTool | **新增** | 提供 LSP 支持（定义跳转、引用查找等） |

### 3.2 CodeTool - 代码分析工具

**定位**: 提供代码理解、分析、质量检查功能（不涉及文件编辑）

```python
class CodeTool(Tool):
    """
    代码分析工具 - 提供代码理解、分析和质量检查功能

    与基础层的区别：
    - BuiltinFileTool: 文件读写、字符串替换
    - CodeTool: 代码语义分析、格式化建议、质量检查

    功能：
    - 代码结构分析（类、函数、变量）
    - 代码大纲生成
    - 代码格式化（调用 black/autopep8 等）
    - 代码 Lint（调用 pylint/flake8 等）
    - 符号查找
    """

    def __init__(
        self,
        work_dir: Optional[str] = None,
        enable_analysis: bool = True,      # analyze_code
        enable_outline: bool = True,       # get_code_outline
        enable_format: bool = True,        # format_code
        enable_lint: bool = True,          # lint_code
        enable_symbols: bool = True,       # find_symbols
    ):
        ...

    def analyze_code(self, file_path: str) -> str:
        """
        分析 Python 代码结构

        Returns:
            JSON 格式的分析结果：
            - imports: 导入语句列表
            - functions: 函数列表（名称、参数、行号、文档字符串）
            - classes: 类列表（名称、方法、基类、行号）
            - global_variables: 全局变量
            - total_lines: 总行数
            - complexity_estimate: 复杂度估计
        """

    def get_code_outline(self, file_path: str) -> str:
        """
        生成代码大纲（Markdown 格式）

        Returns:
            Markdown 格式的代码结构，便于 LLM 快速理解代码组织
        """

    def format_code(self, file_path: str, formatter: str = "auto") -> str:
        """
        格式化代码文件

        Args:
            formatter: 格式化工具，auto 时根据文件类型自动选择
                - Python: black, autopep8, yapf
                - JS/TS: prettier

        Returns:
            格式化结果描述
        """

    def lint_code(self, file_path: str, linter: str = "auto") -> str:
        """
        检查代码质量问题

        Args:
            linter: lint 工具，auto 时根据文件类型自动选择
                - Python: pylint, flake8
                - JS/TS: eslint

        Returns:
            发现的问题列表
        """

    def find_symbols(self, file_path: str, symbol_type: str = "all", pattern: str = "") -> str:
        """
        在代码中查找符号（函数、类、变量）

        Args:
            symbol_type: all, function, class, variable
            pattern: 名称过滤（支持正则）
        """
```

### 3.3 LspTool - LSP 工具

**定位**: 提供基于 Language Server Protocol 的代码导航和分析

```python
class LspTool(Tool):
    """
    LSP 代码导航工具 - 提供精确的代码理解和导航

    与基础层的区别：
    - BuiltinFileTool + grep: 基于文本的搜索和替换
    - LspTool: 基于语义的代码导航（定义、引用、类型等）

    前置要求：需要安装对应的 LSP 服务器
        - Python: pip install pyright 或 python-lsp-server
        - TypeScript: npm install -g typescript-language-server
    """

    def __init__(
        self,
        work_dir: Optional[str] = None,
        servers: Optional[list[str]] = None,  # ["pyright", "typescript"]
        enable_definition: bool = True,
        enable_references: bool = True,
        enable_hover: bool = True,
        enable_diagnostics: bool = True,
        enable_formatting: bool = False,  # 与 CodeTool.format_code 区分，默认关闭
    ):
        ...

    def goto_definition(self, file_path: str, line: int, character: int) -> str:
        """
        跳转到符号定义位置

        Args:
            line: 行号（基于 0）
            character: 列号（基于 0）

        Returns:
            定义位置列表 [{"file": "...", "line": 10, "character": 5}]
        """

    def find_references(self, file_path: str, line: int, character: int) -> str:
        """
        查找符号的所有引用

        Returns:
            引用位置列表
        """

    def hover_info(self, file_path: str, line: int, character: int) -> str:
        """
        获取光标位置的类型/文档信息

        Returns:
            悬停提示内容（类型签名、文档字符串等）
        """

    def get_diagnostics(self, file_path: str) -> str:
        """
        获取文件的实时诊断信息（错误、警告）

        Returns:
            诊断列表 [{"severity": "error", "line": 10, "message": "..."}]
        """
```

---

## 4. 工具对比矩阵

### 4.1 文件操作

| 功能 | BuiltinFileTool (基础) | CodeTool (增益) | LspTool (增益) |
|------|------------------------|-----------------|----------------|
| 读取文件 | ✅ read_file | - | - |
| 写入文件 | ✅ write_file | - | - |
| 编辑文件（字符串替换） | ✅ edit_file / multi_edit | - | - |
| 列出目录 | ✅ ls | - | - |
| 文件搜索 | ✅ glob | - | - |
| 内容搜索 | ✅ grep | - | - |
| 代码分析 | - | ✅ analyze_code | - |
| 代码大纲 | - | ✅ get_code_outline | - |
| 代码格式化 | - | ✅ format_code | ⚠️ lsp_format |
| 代码 Lint | - | ✅ lint_code | - |
| 符号查找 | - | ✅ find_symbols (文本) | ✅ lsp_symbol |
| 跳转定义 | - | - | ✅ goto_definition |
| 查找引用 | - | - | ✅ find_references |
| 类型提示 | - | - | ✅ hover_info |

### 4.2 使用建议

| 场景 | 推荐工具 |
|------|----------|
| 普通文件读写、编辑 | `BuiltinFileTool` |
| 批量文件编辑 | `BuiltinFileTool.multi_edit` |
| 理解代码结构、生成大纲 | `CodeTool` |
| 代码格式化、Lint | `CodeTool` |
| 精确跳转定义、查找引用 | `LspTool` |
| 查看类型签名、文档 | `LspTool` |

---

## 5. 目录结构

```
agentica/
├── deep_tools.py              # 基础层：BuiltinFileTool, BuiltinExecuteTool 等
├── tools/
│   ├── __init__.py
│   ├── base.py               # Tool 基类
│   ├── code_tool.py          # 增益层：代码分析（重构后的 CodeTool）
│   └── lsp_tool.py           # 增益层：LSP 导航（新增）
│   # 以下文件删除
│   # - file_tool.py          # 删除：与 BuiltinFileTool 重合
│   # - edit_tool.py          # 删除：基础层 edit_file 已覆盖
```

---

## 6. 迁移计划

### Phase 1: 升级基础层
1. 升级 `deep_tools.py` 中的 `edit_file`
2. 新增 `multi_edit` 到 `BuiltinFileTool`

### Phase 2: 重构增益层
1. 重构 `CodeTool` - 移除编辑功能，专注代码分析
2. 新增 `LspTool`

### Phase 3: 清理旧代码
1. 删除 `tools/file_tool.py`
2. 删除 `tools/edit_tool.py`
3. 更新 `tools/__init__.py` 导出

### Phase 4: 文档更新
1. 更新工具使用文档
2. 添加工具选择指南

---

## 7. 命名讨论

### edit_file 的新名字（可选）

当前保留 `edit_file`，但可以考虑以下替代：

| 候选名 | 说明 |
|--------|------|
| `edit_file` | 保持现状，用户熟悉 |
| `str_replace` | 强调字符串替换特性 |
| `replace_in_file` | 明确是在文件中替换 |
| `apply_edit` | 类似其他编辑器术语 |

**建议**: 保持 `edit_file`，因为：
1. 用户已经熟悉这个名字
2. 功能仍是编辑文件，只是更精确
3. 与 `write_file` 形成区分

### multi_edit 的命名

保持 `multi_edit`，清晰表达"批量编辑"含义。
