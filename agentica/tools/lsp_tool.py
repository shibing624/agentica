# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: LSP (Language Server Protocol) tool - provides semantic code navigation.

This tool provides code understanding capabilities based on LSP servers:
- Go to definition
- Find references
- Hover information (type signatures, docs)
- Diagnostics (errors, warnings)

Prerequisites:
    - Python: pip install pyright or python-lsp-server
    - TypeScript: npm install -g typescript-language-server

Difference from built-in tools:
    - BuiltinFileTool + grep: Text-based search and replace
    - CodeTool: AST-based code analysis, formatting
    - LspTool: Semantic code navigation (definition, references, types)
"""
import json
import os
import time
import subprocess
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from agentica.tools.base import Tool
from agentica.utils.log import logger


@dataclass
class LspConfig:
    """LSP server configuration."""
    name: str
    command: List[str]  # e.g., ["pyright-langserver", "--stdio"]
    language_ids: Dict[str, str]  # extension -> language_id mapping
    initialization_options: Optional[Dict] = None


# Default LSP server configurations
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
    """Generic JSON-RPC client for LSP communication."""

    def __init__(self, process: subprocess.Popen):
        self._process = process
        self._id_counter = 0
        self._pending: Dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Read messages from stdout in a background thread."""
        while self._running:
            try:
                message = self._read_message()
                if message:
                    self._handle_message(message)
            except Exception as e:
                if self._running:
                    logger.warning(f"LSP read error: {e}")

    def _read_message(self) -> Optional[Dict]:
        """Read a single LSP message from stdout."""
        # Read header
        header = b""
        while True:
            byte = self._process.stdout.read(1)
            if not byte:
                return None
            header += byte
            if header.endswith(b"\r\n\r\n"):
                break

        # Parse Content-Length
        content_length = 0
        for line in header.decode('utf-8').strip().split('\r\n'):
            if line.startswith('Content-Length:'):
                content_length = int(line.split(':')[1].strip())
                break

        if content_length == 0:
            return None

        # Read content
        content = self._process.stdout.read(content_length)
        return json.loads(content.decode('utf-8'))

    def _handle_message(self, message: Dict) -> None:
        """Handle received message (response to pending request)."""
        if "id" in message and message["id"] in self._pending:
            self._pending[message["id"]].put(message)

    def send_request(self, method: str, params: Dict) -> Any:
        """Send a JSON-RPC request and wait for response."""
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

        # Wait for response with timeout
        try:
            response = self._pending[req_id].get(timeout=30)
            del self._pending[req_id]
        except queue.Empty:
            del self._pending[req_id]
            raise TimeoutError(f"LSP request timeout: {method}")

        if "error" in response:
            raise LspError(response["error"])

        return response.get("result")

    def send_notification(self, method: str, params: Dict) -> None:
        """Send a JSON-RPC notification (no response needed)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self._send_message(message)

    def _send_message(self, message: Dict) -> None:
        """Send a message via stdin."""
        content = json.dumps(message)
        header = f"Content-Length: {len(content.encode('utf-8'))}\r\n\r\n"
        data = header + content

        self._process.stdin.write(data.encode('utf-8'))
        self._process.stdin.flush()

    def stop(self) -> None:
        """Stop the client."""
        self._running = False


class LspError(Exception):
    """LSP error."""
    pass


class LspClient:
    """LSP client for a specific language server."""

    def __init__(self, config: LspConfig, workspace_path: Path):
        self.config = config
        self.workspace_path = workspace_path
        self._process: Optional[subprocess.Popen] = None
        self._rpc: Optional[JsonRpcClient] = None
        self._initialized = False
        self._version_counter: Dict[str, int] = {}

    def start(self) -> None:
        """Start the LSP server process."""
        try:
            self._process = subprocess.Popen(
                self.config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.workspace_path
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"LSP server not found: {self.config.command[0]}. Please install it.") from e

        self._rpc = JsonRpcClient(self._process)
        self._initialize()

    def stop(self) -> None:
        """Stop the LSP server."""
        if self._initialized and self._rpc:
            try:
                self._rpc.send_notification("shutdown", {})
                self._rpc.send_notification("exit", {})
            except Exception:
                pass

        if self._rpc:
            self._rpc.stop()

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _initialize(self) -> None:
        """Send LSP initialize request."""
        result = self._rpc.send_request("initialize", {
            "processId": os.getpid(),
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
        self._rpc.send_notification("initialized", {})

    def _get_language_id(self, file_path: Path) -> str:
        """Get LSP language ID for a file."""
        return self.config.language_ids.get(file_path.suffix, "plaintext")

    def _get_version(self, file_path: Path) -> int:
        """Get document version counter."""
        path_str = str(file_path)
        self._version_counter[path_str] = self._version_counter.get(path_str, 0) + 1
        return self._version_counter[path_str]

    def text_document_did_open(self, file_path: Path, content: str) -> None:
        """Notify server that a document is open."""
        self._rpc.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_path.as_uri(),
                "languageId": self._get_language_id(file_path),
                "version": self._get_version(file_path),
                "text": content
            }
        })

    def text_document_did_change(self, file_path: Path, content: str) -> None:
        """Notify server that a document has changed."""
        self._rpc.send_notification("textDocument/didChange", {
            "textDocument": {"uri": file_path.as_uri(), "version": self._get_version(file_path)},
            "contentChanges": [{"text": content}]
        })

    def goto_definition(self, file_path: Path, line: int, character: int) -> List[Dict]:
        """Go to definition of symbol at position.

        Args:
            line: 0-based line number
            character: 0-based column number

        Returns:
            List of locations [{"uri": "...", "range": {...}}]
        """
        result = self._rpc.send_request("textDocument/definition", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character}
        })
        return result if result else []

    def find_references(self, file_path: Path, line: int, character: int) -> List[Dict]:
        """Find all references to symbol at position."""
        result = self._rpc.send_request("textDocument/references", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": True}
        })
        return result if result else []

    def hover(self, file_path: Path, line: int, character: int) -> Optional[Dict]:
        """Get hover information (type signature, docs) at position."""
        return self._rpc.send_request("textDocument/hover", {
            "textDocument": {"uri": file_path.as_uri()},
            "position": {"line": line, "character": character}
        })

    def format_document(self, file_path: Path) -> List[Dict]:
        """Format entire document."""
        return self._rpc.send_request("textDocument/formatting", {
            "textDocument": {"uri": file_path.as_uri()},
            "options": {"tabSize": 4, "insertSpaces": True}
        }) or []


class LspServerManager:
    """Manages multiple LSP server instances."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self._clients: Dict[str, LspClient] = {}
        self._ext_to_client: Dict[str, LspClient] = {}

    def register_server(self, config: LspConfig) -> None:
        """Register and start an LSP server."""
        client = LspClient(config, self.workspace_path)
        client.start()

        self._clients[config.name] = client

        # Map file extensions to this client
        for ext in config.language_ids.keys():
            self._ext_to_client[ext] = client

        logger.debug(f"Registered LSP server: {config.name}")

    def get_client(self, file_path: str) -> Optional[LspClient]:
        """Get LSP client for a file path."""
        ext = Path(file_path).suffix
        return self._ext_to_client.get(ext)

    def shutdown_all(self) -> None:
        """Shutdown all LSP servers."""
        for client in self._clients.values():
            client.stop()
        self._clients.clear()
        self._ext_to_client.clear()


class LspTool(Tool):
    """
    LSP (Language Server Protocol) tool - semantic code navigation.

    Provides code understanding capabilities:
    - goto_definition: Jump to symbol definition
    - find_references: Find all references to a symbol
    - hover_info: Get type signature and documentation
    - format_document: Format code using LSP formatter

    Prerequisites:
        For Python: pip install pyright
        For TypeScript: npm install -g typescript-language-server

    Example:
        lsp = LspTool(servers=["pyright"])

        # Go to definition
        result = lsp.goto_definition("src/main.py", line=10, character=15)

        # Find references
        result = lsp.find_references("src/main.py", line=10, character=15)
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            servers: Optional[List[str]] = None,
            enable_definition: bool = True,
            enable_references: bool = True,
            enable_hover: bool = True,
            enable_formatting: bool = False,
    ):
        """
        Initialize LspTool.

        Args:
            work_dir: Workspace root path. Defaults to current directory.
            servers: List of LSP servers to use. ["pyright"] for Python, ["typescript"] for TS/JS.
            enable_definition: Enable goto_definition function.
            enable_references: Enable find_references function.
            enable_hover: Enable hover_info function.
            enable_formatting: Enable format_document function.
        """
        super().__init__(name="lsp_tool")
        self.workspace_path = Path(work_dir) if work_dir else Path.cwd()
        self.manager = LspServerManager(self.workspace_path)

        # Start configured servers
        servers = servers or ["pyright"]
        for server_name in servers:
            if server_name in DEFAULT_LSP_SERVERS:
                try:
                    self.manager.register_server(DEFAULT_LSP_SERVERS[server_name])
                except RuntimeError as e:
                    logger.warning(f"Failed to start LSP server {server_name}: {e}")

        # Register functions
        if enable_definition:
            self.register(self.goto_definition)
        if enable_references:
            self.register(self.find_references)
        if enable_hover:
            self.register(self.hover_info)
        if enable_formatting:
            self.register(self.format_document)

    def _get_client(self, file_path: str) -> Optional[LspClient]:
        """Get LSP client for file."""
        client = self.manager.get_client(file_path)
        if client is None:
            ext = Path(file_path).suffix
            supported = []
            for config in DEFAULT_LSP_SERVERS.values():
                supported.extend(config.language_ids.keys())
            logger.warning(f"No LSP server for {ext} files. Supported: {supported}")
        return client

    def _open_document(self, client: LspClient, file_path: Path) -> None:
        """Open document in LSP server."""
        try:
            content = file_path.read_text(encoding='utf-8')
            client.text_document_did_open(file_path, content)
        except Exception as e:
            logger.warning(f"Failed to open document {file_path}: {e}")

    def goto_definition(self, file_path: str, line: int, character: int) -> str:
        """Jump to symbol definition location.

        Args:
            file_path: Absolute path to the file.
            line: Line number (0-based).
            character: Column number (0-based, character position in line).

        Returns:
            JSON-formatted list of definition locations.

        Example:
            # Cursor on "my_function" at line 10, column 5
            result = lsp.goto_definition("src/main.py", line=9, character=4)
            # Returns: [{"file": "/path/to/file.py", "line": 20, "character": 0}]
        """
        client = self._get_client(file_path)
        if not client:
            return json.dumps({"error": f"No LSP server for file: {file_path}"})

        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return json.dumps({"error": f"File not found: {file_path}"})

            self._open_document(client, path)

            # Small delay to allow server to analyze the file
            time.sleep(0.5)

            results = client.goto_definition(path, line, character)

            # Format result
            definitions = []
            for r in results:
                uri = r.get("uri", "")
                range_info = r.get("range", {})
                start = range_info.get("start", {})
                definitions.append({
                    "file": uri.replace("file://", ""),
                    "line": start.get("line", 0),
                    "character": start.get("character", 0)
                })

            return json.dumps({"definitions": definitions}, indent=2)

        except Exception as e:
            logger.error(f"LSP goto_definition error: {e}")
            return json.dumps({"error": str(e)})

    def find_references(self, file_path: str, line: int, character: int) -> str:
        """Find all references to a symbol.

        Args:
            file_path: Absolute path to the file.
            line: Line number (0-based).
            character: Column number (0-based).

        Returns:
            JSON-formatted list of reference locations.
        """
        client = self._get_client(file_path)
        if not client:
            return json.dumps({"error": f"No LSP server for file: {file_path}"})

        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return json.dumps({"error": f"File not found: {file_path}"})

            self._open_document(client, path)

            # Small delay to allow server to analyze the file
            time.sleep(0.5)

            results = client.find_references(path, line, character)

            references = []
            for r in results:
                uri = r.get("uri", "")
                range_info = r.get("range", {})
                start = range_info.get("start", {})
                references.append({
                    "file": uri.replace("file://", ""),
                    "line": start.get("line", 0),
                    "character": start.get("character", 0)
                })

            return json.dumps({"references": references}, indent=2)

        except Exception as e:
            logger.error(f"LSP find_references error: {e}")
            return json.dumps({"error": str(e)})

    def hover_info(self, file_path: str, line: int, character: int) -> str:
        """Get hover information (type signature, documentation) at cursor position.

        Args:
            file_path: Absolute path to the file.
            line: Line number (0-based).
            character: Column number (0-based).

        Returns:
            Hover information as string.
        """
        client = self._get_client(file_path)
        if not client:
            return f"No LSP server for file: {file_path}"

        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return f"File not found: {file_path}"

            self._open_document(client, path)

            # Small delay to allow server to analyze the file
            time.sleep(0.5)

            result = client.hover(path, line, character)
            if not result:
                return "No hover information available"

            contents = result.get("contents")
            if not contents:
                return "No hover information available"

            # Parse hover contents (can be string or MarkupContent)
            if isinstance(contents, dict):
                return contents.get("value", str(contents))
            elif isinstance(contents, list) and contents:
                # Array of MarkedString
                parts = []
                for item in contents:
                    if isinstance(item, dict):
                        parts.append(item.get("value", str(item)))
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            else:
                return str(contents)

        except Exception as e:
            logger.error(f"LSP hover error: {e}")
            return f"Error: {str(e)}"

    def format_document(self, file_path: str) -> str:
        """Format document using LSP formatter.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Result message.
        """
        client = self._get_client(file_path)
        if not client:
            return f"No LSP server for file: {file_path}"

        try:
            path = Path(file_path).resolve()
            if not path.exists():
                return f"File not found: {file_path}"

            self._open_document(client, path)

            edits = client.format_document(path)

            if not edits:
                return "No formatting changes needed"

            # Apply edits
            content = path.read_text(encoding='utf-8')
            # Sort edits in reverse order (to preserve positions)
            sorted_edits = sorted(
                edits,
                key=lambda e: (e["range"]["start"]["line"], e["range"]["start"]["character"]),
                reverse=True
            )

            lines = content.split("\n")
            for edit in sorted_edits:
                range_info = edit["range"]
                start = range_info["start"]
                end = range_info["end"]
                new_text = edit["newText"]

                # Convert to 0-based indices
                start_line = start["line"]
                start_char = start["character"]
                end_line = end["line"]
                end_char = end["character"]

                # Apply edit
                if start_line == end_line:
                    lines[start_line] = lines[start_line][:start_char] + new_text + lines[start_line][end_char:]
                else:
                    # Multi-line edit
                    before = lines[start_line][:start_char]
                    after = lines[end_line][end_char:]
                    new_lines = new_text.split("\n")
                    lines[start_line] = before + new_lines[0]
                    lines[start_line + 1:end_line + 1] = new_lines[1:] if len(new_lines) > 1 else []
                    if new_lines:
                        lines[start_line] += after

            # Write back
            path.write_text("\n".join(lines), encoding='utf-8')
            return f"Document formatted: {file_path}"

        except Exception as e:
            logger.error(f"LSP format error: {e}")
            return f"Error formatting document: {str(e)}"

    def __del__(self):
        """Cleanup LSP servers on deletion."""
        if hasattr(self, 'manager'):
            self.manager.shutdown_all()


if __name__ == '__main__':
    # Test example
    print("LspTool test - requires pylsp to be installed")
    print("Install with: pip install python-lsp-server")

    try:
        tool = LspTool(servers=["pylsp"])

        # Create test file with absolute path
        test_file = str(Path("test_lsp.py").resolve())
        with open(test_file, "w") as f:
            f.write("""def hello():
    return "world"

x = hello()
""")

        # Wait for server to initialize
        time.sleep(1)

        print(f"\nTest file: {test_file}")

        print("\nTest hover_info:")
        result = tool.hover_info(test_file, line=3, character=4)  # On 'hello' call
        print(result)

        print("\nTest goto_definition:")
        result = tool.goto_definition(test_file, line=3, character=4)
        print(result)

        print("\nTest find_references:")
        result = tool.find_references(test_file, line=3, character=4)
        print(result)

        # Cleanup
        import os
        if os.path.exists(test_file):
            os.remove(test_file)

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
