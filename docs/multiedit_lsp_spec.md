# MultiEdit & LSP 工具实现方案

## 1. MultiEdit 批量编辑工具

### 1.1 核心设计思路

MultiEdit 是对 Edit 工具的批量包装，核心特性：

1. **顺序执行**: 按顺序应用多个编辑操作，保证原子性
2. **行级精确匹配**: 基于行号+内容哈希匹配，比纯字符串替换更稳定
3. **失败回滚**: 任意一步失败时，可选回滚所有修改
4. **冲突检测**: 检测多个编辑操作之间的行号冲突

### 1.2 数据结构

```python
from dataclasses import dataclass
from typing import List, Optional, Literal
from enum import Enum

class EditType(Enum):
    REPLACE = "replace"      # 字符串替换
    INSERT = "insert"        # 在指定行插入
    DELETE = "delete"        # 删除指定行
    PATCH = "patch"          # V4A/Unified diff

@dataclass
class EditOperation:
    """单个编辑操作"""
    edit_type: EditType
    file_path: str

    # For REPLACE
    old_string: Optional[str] = None
    new_string: Optional[str] = None

    # For INSERT/DELETE (基于1的行号)
    line_number: Optional[int] = None
    content: Optional[str] = None  # For INSERT

    # For PATCH
    patch_content: Optional[str] = None

    # 匹配验证（可选，增强稳定性）
    context_before: Optional[str] = None  # 修改前的上下文行
    context_after: Optional[str] = None   # 修改后的上下文行

@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    file_path: str
    edit_type: EditType
    message: str
    line_changes: int = 0  # 行数变化（用于后续编辑的偏移计算）
    error: Optional[str] = None

@dataclass
class MultiEditResult:
    """批量编辑结果"""
    success: bool
    results: List[EditResult]
    total_changes: int
    rolled_back: bool = False
    error_index: Optional[int] = None  # 失败的编辑索引
```

### 1.3 核心算法

#### 行号偏移追踪

当多个编辑操作作用于同一文件时，前面的修改会影响后续操作的行号。需要实时追踪行偏移。

```python
class LineOffsetTracker:
    """追踪文件编辑导致的行号偏移"""

    def __init__(self):
        # file_path -> {original_line: adjusted_line}
        self._offsets: dict[str, dict[int, int]] = {}

    def register_change(self, file_path: str, start_line: int, line_delta: int):
        """
        注册一次修改导致的行变化

        Args:
            start_line: 修改开始的原始行号
            line_delta: 行数变化（新增为正，删除为负）
        """
        if file_path not in self._offsets:
            self._offsets[file_path] = {}

        offsets = self._offsets[file_path]

        # 更新该文件所有后续行号的偏移
        for orig_line in list(offsets.keys()):
            if orig_line >= start_line:
                offsets[orig_line] += line_delta

        # 记录新插入点的偏移
        if start_line not in offsets:
            offsets[start_line] = line_delta

    def get_adjusted_line(self, file_path: str, original_line: int) -> int:
        """获取调整后的行号"""
        if file_path not in self._offsets:
            return original_line

        # 找到小于等于 original_line 的最大偏移点
        offset = 0
        for line, delta in self._offsets[file_path].items():
            if line <= original_line:
                offset += delta

        return original_line + offset
```

#### 冲突检测

```python
def detect_conflicts(operations: List[EditOperation]) -> List[tuple[int, int, str]]:
    """
    检测编辑操作之间的冲突

    Returns:
        冲突列表，每个元素为 (index1, index2, conflict_type)
    """
    conflicts = []
    file_ranges: dict[str, list[tuple[int, int, int]]] = {}  # file -> [(start, end, op_index)]

    for i, op in enumerate(operations):
        if op.file_path not in file_ranges:
            file_ranges[op.file_path] = []

        # 计算操作影响的行范围
        start, end = _get_operation_range(op)

        # 检查与之前操作的冲突
        for prev_start, prev_end, prev_idx in file_ranges[op.file_path]:
            # 范围重叠检测
            if start <= prev_end and end >= prev_start:
                conflicts.append((prev_idx, i, "range_overlap"))

        file_ranges[op.file_path].append((start, end, i))

    return conflicts
```

#### 原子性执行

```python
class MultiEditExecutor:
    """批量编辑执行器"""

    def __init__(self, fs: FileSystem, auto_rollback: bool = True):
        self.fs = fs
        self.auto_rollback = auto_rollback
        self._backup: dict[str, str] = {}  # 文件备份用于回滚

    def execute(self, operations: List[EditOperation]) -> MultiEditResult:
        """执行批量编辑"""
        results = []
        tracker = LineOffsetTracker()

        # 预检测冲突
        conflicts = detect_conflicts(operations)
        if conflicts:
            return MultiEditResult(
                success=False,
                results=[],
                total_changes=0,
                error_index=conflicts[0][0],
                error_message=f"Conflict detected between operations {conflicts[0][0]} and {conflicts[0][1]}"
            )

        # 备份所有将被修改的文件
        files_to_modify = set(op.file_path for op in operations)
        for file_path in files_to_modify:
            self._backup[file_path] = self.fs.read(file_path)

        # 顺序执行编辑
        for idx, op in enumerate(operations):
            try:
                # 调整行号（基于之前的编辑）
                op = self._adjust_operation(op, tracker)

                # 执行单个编辑
                result = self._execute_single(op)
                results.append(result)

                if not result.success:
                    if self.auto_rollback:
                        self._rollback()
                    return MultiEditResult(
                        success=False,
                        results=results,
                        total_changes=sum(r.line_changes for r in results),
                        rolled_back=self.auto_rollback,
                        error_index=idx
                    )

                # 更新行偏移
                if result.line_changes != 0:
                    tracker.register_change(
                        op.file_path,
                        op.line_number or 1,
                        result.line_changes
                    )

            except Exception as e:
                if self.auto_rollback:
                    self._rollback()
                return MultiEditResult(
                    success=False,
                    results=results,
                    total_changes=sum(r.line_changes for r in results),
                    rolled_back=self.auto_rollback,
                    error_index=idx,
                    error_message=str(e)
                )

        return MultiEditResult(
            success=True,
            results=results,
            total_changes=sum(r.line_changes for r in results)
        )

    def _rollback(self):
        """回滚所有修改"""
        for file_path, content in self._backup.items():
            self.fs.write(file_path, content)
        self._backup.clear()

    def _execute_single(self, op: EditOperation) -> EditResult:
        """执行单个编辑操作"""
        if op.edit_type == EditType.REPLACE:
            return self._do_replace(op)
        elif op.edit_type == EditType.INSERT:
            return self._do_insert(op)
        elif op.edit_type == EditType.DELETE:
            return self._do_delete(op)
        elif op.edit_type == EditType.PATCH:
            return self._do_patch(op)
        else:
            raise ValueError(f"Unknown edit type: {op.edit_type}")
```

### 1.4 Tool 接口

```python
from agentica.tools.base import Tool

class MultiEditTool(Tool):
    """
    批量文件编辑工具

    支持在一个函数调用中执行多个编辑操作，保证原子性。
    任意一个编辑失败时，所有修改可选回滚。
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        auto_rollback: bool = True,
        detect_conflicts: bool = True
    ):
        super().__init__(name="multi_edit")
        fs = FileSystem()
        resolver = PathResolver(Path(base_dir) if base_dir else Path.cwd())
        self.executor = MultiEditExecutor(fs, auto_rollback)
        self.detect_conflicts = detect_conflicts

        self.register(self.multi_edit)
        self.register(self.multi_edit_files)  # 简化的文件列表接口

    def multi_edit(
        self,
        operations: List[dict],
        rollback_on_error: bool = True
    ) -> str:
        """
        执行批量编辑操作

        Args:
            operations: 编辑操作列表，每个操作是一个字典，包含：
                - edit_type: "replace" | "insert" | "delete" | "patch"
                - file_path: 文件路径
                - 其他参数根据 edit_type 变化

                replace 类型:
                - old_string: 要替换的字符串（必须唯一存在）
                - new_string: 新字符串
                - context_before: 可选，用于验证的上下文

                insert 类型:
                - line_number: 插入位置的行号（基于1）
                - content: 要插入的内容

                delete 类型:
                - line_number: 要删除的行号
                - line_count: 删除的行数（默认1）

                patch 类型:
                - patch_content: V4A 或 Unified diff 格式的 patch

            rollback_on_error: 出错时是否回滚所有修改

        Returns:
            JSON 格式的执行结果

        Example:
            operations=[
                {
                    "edit_type": "replace",
                    "file_path": "src/main.py",
                    "old_string": "def old_func():",
                    "new_string": "def new_func():"
                },
                {
                    "edit_type": "insert",
                    "file_path": "src/main.py",
                    "line_number": 10,
                    "content": "import logging"
                }
            ]
        """
        # 解析操作
        ops = [self._parse_operation(op) for op in operations]

        # 执行
        result = self.executor.execute(ops)

        # 格式化返回
        return json.dumps({
            "success": result.success,
            "total_changes": result.total_changes,
            "rolled_back": result.rolled_back,
            "error_index": result.error_index,
            "details": [
                {
                    "file": r.file_path,
                    "type": r.edit_type.value,
                    "success": r.success,
                    "message": r.message,
                    "line_changes": r.line_changes
                }
                for r in result.results
            ]
        }, indent=2)

    def multi_edit_files(
        self,
        file_paths: List[str],
        edit_instructions: str
    ) -> str:
        """
        简化的批量编辑接口 - 让 LLM 生成编辑指令

        Args:
            file_paths: 要编辑的文件列表
            edit_instructions: 自然语言编辑指令，格式为：

                FILE: path/to/file1.py
                REPLACE:
                ```
                old content
                ```
                WITH:
                ```
                new content
                ```

                FILE: path/to/file2.py
                INSERT AT LINE 10:
                ```
                new line
                ```

        Returns:
            执行结果
        """
        # 解析 edit_instructions 为 EditOperation 列表
        operations = self._parse_instructions(edit_instructions, file_paths)
        return self.multi_edit([op.__dict__ for op in operations])
```

---

## 2. LSP 工具

### 2.1 架构设计

```
┌─────────────────────────────────────────────┐
│           LspTool (Tool 层)                 │
│  - goto_definition                          │
│  - find_references                          │
│  - get_diagnostics                          │
│  - hover_info                               │
│  - format_document                          │
├─────────────────────────────────────────────┤
│           LspAtom (原子层)                   │
│  - 封装 LspClient 的底层调用                │
├─────────────────────────────────────────────┤
│           LspClient (客户端层)               │
│  - JSON-RPC 通信                            │
│  - 服务器生命周期管理                       │
│  - 请求/响应映射                            │
├─────────────────────────────────────────────┤
│           LSP Server (外部进程)              │
│  - pyright, pylsp, typescript-language-server, etc. │
└─────────────────────────────────────────────┘
```

### 2.2 LspClient 实现

```python
import subprocess
import json
import threading
import queue
from typing import Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LspConfig:
    """LSP 服务器配置"""
    name: str
    command: list[str]  # 启动命令，如 ["pyright-langserver", "--stdio"]
    language_ids: dict[str, str]  # 扩展名 -> language_id 映射
    initialization_options: Optional[dict] = None

# 预设配置
DEFAULT_LSP_SERVERS = {
    "pyright": LspConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        language_ids={".py": "python"},
        initialization_options={
            "diagnostics": {"enable": True},
            "python": {"analysis": {"typeCheckingMode": "basic"}}
        }
    ),
    "pylsp": LspConfig(
        name="pylsp",
        command=["pylsp"],
        language_ids={".py": "python"}
    ),
    "typescript": LspConfig(
        name="typescript-language-server",
        command=["typescript-language-server", "--stdio"],
        language_ids={
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact"
        }
    )
}

class JsonRpcClient:
    """通用 JSON-RPC 客户端"""

    def __init__(self):
        self._id_counter = 0
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

    def send_request(self, method: str, params: dict) -> Any:
        """发送同步请求"""
        with self._lock:
            self._id_counter += 1
            req_id = self._id_counter
            self._pending[req_id] = queue.Queue()

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }

        self._send_message(message)

        # 等待响应
        response = self._pending[req_id].get(timeout=30)
        del self._pending[req_id]

        if "error" in response:
            raise LspError(response["error"])

        return response.get("result")

    def send_notification(self, method: str, params: dict) -> None:
        """发送通知（无需响应）"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self._send_message(message)

    def _send_message(self, message: dict) -> None:
        raise NotImplementedError()

    def _handle_message(self, message: dict) -> None:
        """处理接收到的消息"""
        if "id" in message:
            # 响应消息
            if message["id"] in self._pending:
                self._pending[message["id"]].put(message)

class LspClient(JsonRpcClient):
    """LSP 客户端"""

    def __init__(self, config: LspConfig, workspace_path: Path):
        super().__init__()
        self.config = config
        self.workspace_path = workspace_path
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._initialized = False

    def start(self) -> None:
        """启动 LSP 服务器进程"""
        self._process = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.workspace_path
        )
        self._running = True

        # 启动读取线程
        self._reader_thread = threading.Thread(target=self._read_loop)
        self._reader_thread.daemon = True
        self._reader_thread.start()

        # 发送初始化请求
        self._initialize()

    def stop(self) -> None:
        """停止 LSP 服务器"""
        if self._initialized:
            self.send_notification("shutdown", {})
            self.send_notification("exit", {})

        self._running = False
        if self._process:
            self._process.terminate()
            self._process.wait()

    def _initialize(self) -> None:
        """发送 LSP initialize 请求"""
        result = self.send_request("initialize", {
            "processId": None,
            "rootUri": self.workspace_path.as_uri(),
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "completion": {"dynamicRegistration": False},
                    "hover": {"dynamicRegistration": False},
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "documentFormatting": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True}
                }
            },
            "initializationOptions": self.config.initialization_options or {}
        })
        self._initialized = True
        self.send_notification("initialized", {})

    def _send_message(self, message: dict) -> None:
        """通过 stdin 发送消息"""
        content = json.dumps(message)
        header = f"Content-Length: {len(content.encode('utf-8'))}\r\n\r\n"
        data = header + content

        self._process.stdin.write(data.encode('utf-8'))
        self._process.stdin.flush()

    def _read_loop(self) -> None:
        """从 stdout 读取消息的循环"""
        while self._running:
            try:
                message = self._read_message()
                if message:
                    self._handle_message(message)
            except Exception as e:
                if self._running:
                    print(f"LSP read error: {e}")

    def _read_message(self) -> Optional[dict]:
        """读取单个 LSP 消息"""
        # 读取 header
        header = b""
        while True:
            byte = self._process.stdout.read(1)
            if not byte:
                return None
            header += byte
            if header.endswith(b"\r\n\r\n"):
                break

        # 解析 Content-Length
        content_length = 0
        for line in header.decode('utf-8').strip().split('\r\n'):
            if line.startswith('Content-Length:'):
                content_length = int(line.split(':')[1].strip())
                break

        if content_length == 0:
            return None

        # 读取 content
        content = self._process.stdout.read(content_length)
        return json.loads(content.decode('utf-8'))

    # LSP 便捷方法
    def text_document_did_open(self, file_path: Path, content: str) -> None:
        """通知服务器打开文档"""
        lang_id = self._get_language_id(file_path)
        self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_path.as_uri(),
                "languageId": lang_id,
                "version": 1,
                "text": content
            }
        })

    def text_document_did_change(self, file_path: Path, content: str) -> None:
        """通知服务器文档变更"""
        self.send_notification("textDocument/didChange", {
            "textDocument": {"uri": file_path.as_uri(), "version": 2},
            "contentChanges": [{"text": content}]
        })

    def goto_definition(self, file_path: Path, line: int, character: int) -> list:
        """跳转到定义"""
        return self.send_request("textDocument/definition", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character}
        }) or []

    def find_references(self, file_path: Path, line: int, character: int) -> list:
        """查找引用"""
        return self.send_request("textDocument/references", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": True}
        }) or []

    def hover(self, file_path: Path, line: int, character: int) -> Optional[dict]:
        """悬停提示"""
        return self.send_request("textDocument/hover", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character}
        })

    def get_diagnostics(self, file_path: Path) -> list:
        """获取诊断信息（需要服务器主动推送）"""
        # LSP 诊断通常是服务器推送的，需要缓存机制
        return self._diagnostics_cache.get(str(file_path), [])

    def format_document(self, file_path: Path) -> list:
        """格式化文档"""
        return self.send_request("textDocument/formatting", {
            "textDocument": {"uri": file_path.as_uri()},
            "options": {"tabSize": 4, "insertSpaces": True}
        }) or []

    def _get_language_id(self, file_path: Path) -> str:
        """根据文件扩展名获取 language id"""
        ext = file_path.suffix
        return self.config.language_ids.get(ext, "plaintext")


class LspServerManager:
    """管理多个 LSP 服务器实例"""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self._clients: dict[str, LspClient] = {}
        self._file_to_client: dict[str, LspClient] = {}  # 文件扩展名到客户端的映射

    def register_server(self, config: LspConfig) -> None:
        """注册 LSP 服务器配置"""
        client = LspClient(config, self.workspace_path)
        client.start()

        self._clients[config.name] = client

        # 映射文件扩展名到客户端
        for ext in config.language_ids.keys():
            self._file_to_client[ext] = client

    def get_client_for_file(self, file_path: str) -> Optional[LspClient]:
        """根据文件路径获取对应的 LSP 客户端"""
        ext = Path(file_path).suffix
        return self._file_to_client.get(ext)

    def shutdown_all(self) -> None:
        """关闭所有 LSP 服务器"""
        for client in self._clients.values():
            client.stop()
        self._clients.clear()
        self._file_to_client.clear()
```

### 2.3 LSP Tool 接口

```python
class LspTool(Tool):
    """
    LSP (Language Server Protocol) 工具

    提供基于 LSP 的代码分析功能，包括：
    - 跳转到定义
    - 查找引用
    - 悬停提示
    - 诊断信息
    - 代码格式化
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        servers: Optional[list[str]] = None
    ):
        super().__init__(name="lsp")
        self.workspace_path = Path(base_dir) if base_dir else Path.cwd()
        self.manager = LspServerManager(self.workspace_path)

        # 启动配置的 LSP 服务器
        servers = servers or ["pyright", "pylsp"]
        for server_name in servers:
            if server_name in DEFAULT_LSP_SERVERS:
                self.manager.register_server(DEFAULT_LSP_SERVERS[server_name])

        self.register(self.goto_definition)
        self.register(self.find_references)
        self.register(self.hover_info)
        self.register(self.get_diagnostics)
        self.register(self.format_document)

    def goto_definition(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> str:
        """
        跳转到符号定义位置

        Args:
            file_path: 文件路径（绝对路径）
            line: 行号（基于0）
            character: 列号（基于0）

        Returns:
            定义位置列表，JSON 格式
        """
        client = self.manager.get_client_for_file(file_path)
        if not client:
            return json.dumps({"error": f"No LSP server for file: {file_path}"})

        try:
            # 先通知服务器打开文档
            content = Path(file_path).read_text()
            client.text_document_did_open(Path(file_path), content)

            results = client.goto_definition(Path(file_path), line, character)

            return json.dumps({
                "definitions": [
                    {
                        "file": r["uri"].replace("file://", ""),
                        "line": r["range"]["start"]["line"],
                        "character": r["range"]["start"]["character"]
                    }
                    for r in results
                ]
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def find_references(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> str:
        """
        查找符号的所有引用

        Args:
            file_path: 文件路径
            line: 行号（基于0）
            character: 列号（基于0）

        Returns:
            引用位置列表，JSON 格式
        """
        client = self.manager.get_client_for_file(file_path)
        if not client:
            return json.dumps({"error": f"No LSP server for file: {file_path}"})

        try:
            content = Path(file_path).read_text()
            client.text_document_did_open(Path(file_path), content)

            results = client.find_references(Path(file_path), line, character)

            return json.dumps({
                "references": [
                    {
                        "file": r["uri"].replace("file://", ""),
                        "line": r["range"]["start"]["line"],
                        "character": r["range"]["start"]["character"]
                    }
                    for r in results
                ]
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def hover_info(
        self,
        file_path: str,
        line: int,
        character: int
    ) -> str:
        """
        获取光标位置的悬停提示信息

        Args:
            file_path: 文件路径
            line: 行号（基于0）
            character: 列号（基于0）

        Returns:
            悬停提示内容
        """
        client = self.manager.get_client_for_file(file_path)
        if not client:
            return f"No LSP server for file: {file_path}"

        try:
            content = Path(file_path).read_text()
            client.text_document_did_open(Path(file_path), content)

            result = client.hover(Path(file_path), line, character)
            if result and "contents" in result:
                contents = result["contents"]
                if isinstance(contents, dict):
                    return contents.get("value", str(contents))
                return str(contents)
            return "No hover information available"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_diagnostics(self, file_path: str) -> str:
        """
        获取文件的诊断信息（错误、警告等）

        Args:
            file_path: 文件路径

        Returns:
            诊断信息列表，JSON 格式
        """
        client = self.manager.get_client_for_file(file_path)
        if not client:
            return json.dumps({"error": f"No LSP server for file: {file_path}"})

        try:
            content = Path(file_path).read_text()
            client.text_document_did_open(Path(file_path), content)

            # 等待服务器推送诊断（简单实现：sleep 一下）
            import time
            time.sleep(0.5)

            diagnostics = client.get_diagnostics(Path(file_path))

            return json.dumps({
                "diagnostics": [
                    {
                        "severity": d.get("severity", 1),  # 1=Error, 2=Warning, 3=Info, 4=Hint
                        "line": d["range"]["start"]["line"],
                        "character": d["range"]["start"]["character"],
                        "message": d["message"],
                        "code": d.get("code")
                    }
                    for d in diagnostics
                ]
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_document(self, file_path: str) -> str:
        """
        格式化文档

        Args:
            file_path: 文件路径

        Returns:
            格式化后的内容或错误信息
        """
        client = self.manager.get_client_for_file(file_path)
        if not client:
            return f"No LSP server for file: {file_path}"

        try:
            content = Path(file_path).read_text()
            client.text_document_did_open(Path(file_path), content)

            edits = client.format_document(Path(file_path))

            if not edits:
                return "No formatting changes needed or server doesn't support formatting"

            # 应用编辑（简化版：假设只有一个 edit）
            new_content = content
            for edit in edits:
                start = edit["range"]["start"]
                end = edit["range"]["end"]
                lines = content.split('\n')

                # 计算字节位置（简化处理）
                start_pos = sum(len(lines[i]) + 1 for i in range(start["line"])) + start["character"]
                end_pos = sum(len(lines[i]) + 1 for i in range(end["line"])) + end["character"]

                new_content = new_content[:start_pos] + edit["newText"] + new_content[end_pos:]

            # 写回文件
            Path(file_path).write_text(new_content)

            return f"Document formatted: {file_path}"
        except Exception as e:
            return f"Error formatting document: {str(e)}"
```

### 2.4 与 CodingTool 的整合

```python
class CodingTool(Tool):
    """代码开发工具 - 整合编辑、搜索、LSP"""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        enable_lsp: bool = True,
        lsp_servers: Optional[list[str]] = None
    ):
        super().__init__(name="coding")

        # 初始化原子层
        fs = FileSystem()
        resolver = PathResolver(Path(base_dir) if base_dir else Path.cwd())

        self.file = FileAtom(fs, resolver)
        self.edit = EditAtom(fs, resolver)
        self.search = SearchAtom(fs, resolver)
        self.multi_edit = MultiEditExecutor(fs)

        # LSP
        if enable_lsp:
            self.lsp = LspTool(base_dir, lsp_servers)
            # 将 LSP 的函数注册到 CodingTool
            for name, func in self.lsp.functions.items():
                self.functions[f"lsp_{name}"] = func
        else:
            self.lsp = None

        # 注册核心函数
        self.register(self.edit_file)
        self.register(self.multi_edit_files)
        self.register(self.apply_patch)
        self.register(self.analyze_code)
        self.register(self.format_code)

        if self.lsp:
            self.register(self.goto_definition)
            self.register(self.find_references)

    def goto_definition(self, file_path: str, symbol: str) -> str:
        """
        跳转到符号定义（高级封装）

        先使用 grep 找到符号位置，然后使用 LSP 获取定义
        """
        # 1. 搜索符号位置
        matches = self.search.grep(symbol, file_path, output_mode="content")
        if not matches:
            return f"Symbol '{symbol}' not found in {file_path}"

        # 2. 使用第一个匹配位置调用 LSP
        first_match = matches[0]
        return self.lsp.goto_definition(
            first_match.file,
            first_match.line - 1,  # 转为0-based
            first_match.column
        )

    def find_references(self, file_path: str, symbol: str) -> str:
        """查找符号引用（高级封装）"""
        matches = self.search.grep(symbol, file_path, output_mode="content")
        if not matches:
            return f"Symbol '{symbol}' not found in {file_path}"

        first_match = matches[0]
        return self.lsp.find_references(
            first_match.file,
            first_match.line - 1,
            first_match.column
        )

    def multi_edit_files(
        self,
        operations: list[dict],
        rollback_on_error: bool = True
    ) -> str:
        """批量编辑文件"""
        ops = [self._parse_edit_op(op) for op in operations]
        result = self.multi_edit.execute(ops)

        return json.dumps({
            "success": result.success,
            "total_changes": result.total_changes,
            "rolled_back": result.rolled_back,
            "error_index": result.error_index
        }, indent=2)
```

---

## 3. 使用示例

### 3.1 MultiEdit 使用

```python
from agentica.tools.scenes import CodingTool

tool = CodingTool()

# 执行批量编辑
result = tool.multi_edit_files([
    {
        "edit_type": "replace",
        "file_path": "src/main.py",
        "old_string": "def old_func():",
        "new_string": "def new_func():"
    },
    {
        "edit_type": "insert",
        "file_path": "src/main.py",
        "line_number": 10,
        "content": "import logging\n"
    },
    {
        "edit_type": "replace",
        "file_path": "src/utils.py",
        "old_string": "class Helper:",
        "new_string": "class Utils:"
    }
])
```

### 3.2 LSP 使用

```python
from agentica.tools import LspTool

tool = LspTool(servers=["pyright"])

# 跳转到定义
result = tool.goto_definition("src/main.py", line=10, character=15)

# 查找引用
result = tool.find_references("src/main.py", line=10, character=15)

# 获取诊断
diags = tool.get_diagnostics("src/main.py")
```

---

## 4. 依赖项

```toml
# pyproject.toml 新增依赖
[project.optional-dependencies]
lsp = [
    "pyright",  # Python LSP
    "python-lsp-server",  # pylsp
]
```

---

## 5. 实现优先级

| 阶段 | 功能 | 优先级 |
|------|------|--------|
| Phase 1 | MultiEdit 核心实现（字符串替换） | P0 |
| Phase 1 | LineOffsetTracker | P0 |
| Phase 2 | MultiEdit 行级操作（insert/delete） | P1 |
| Phase 2 | LspClient JSON-RPC 基础 | P1 |
| Phase 3 | LSP goto_definition | P1 |
| Phase 3 | LSP find_references | P1 |
| Phase 4 | LSP diagnostics | P2 |
| Phase 4 | LSP format_document | P2 |
| Phase 5 | 多 LSP 服务器支持 | P3 |
