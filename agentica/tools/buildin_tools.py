# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in tools for Agent

Built-in tool set for Agent, including:
- ls: List directory contents
- read_file: Read file content
- write_file: Write file content
- edit_file: Edit file (string replacement)
- multi_edit_file: Apply multiple edits to a file atomically
- glob: File pattern matching
- grep: Search file content
- execute: Execute command
- web_search: Web search (implemented using BaiduSearch)
- fetch_url: Fetch URL content (implemented using UrlCrawler)
- write_todos: Create and manage task list
- task: Launch subagent to handle complex tasks
"""
import asyncio
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime
import time
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Optional, List, Dict, Any, Literal, Tuple, TYPE_CHECKING, Union

import aiofiles

from agentica.tools.builtin.task_state_tools import BuiltinMemoryTool, BuiltinTodoTool
from agentica.tools.builtin.web_tools import BuiltinFetchUrlTool, BuiltinWebSearchTool
from agentica.tools.base import Tool
from agentica.tools.builtin_task_tool import BuiltinTaskTool  # re-export after extraction
from agentica.tools.safety import check_command_safety, redact_sensitive_text
from agentica.security.redact import redact_tool_outputs_enabled
from agentica.utils.log import logger
from agentica.utils.string import truncate_if_too_long

# grep self-imposed timeout (seconds). Covers both the rg subprocess and the
# pure-Python fallback so a missing rg can't walk huge trees for the outer
# 120s executor timeout. grep is marked manages_own_timeout=True. 30s gives rg
# enough headroom on large monorepos (a whole-tree grep can take ~25s) while
# still bounding a genuine hang (no rg + pure-Python fallback over millions of
# files) well under the outer executor limit.
_GREP_TIMEOUT = 30


def _interpret_exit_code(command: str, exit_code: int) -> Optional[str]:
    """Return a human-readable note when a non-zero exit code is non-erroneous.

    Returns None when the exit code is 0 or genuinely signals an error.
    The note is appended to the tool result so the model doesn't waste
    turns investigating expected exit codes.
    """
    if exit_code == 0:
        return None

    # Extract the last command in a pipeline/chain — that determines the
    # exit code. Handles `cmd1 && cmd2`, `cmd1 | cmd2`, `cmd1; cmd2`.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: Dict[str, Dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep": {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg": {1: "No matches found (not an error)"},
        "ag": {1: "No matches found (not an error)"},
        "ack": {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff": {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find": {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test": {1: "Condition evaluated to false (expected, not an error)"},
        "[": {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl": {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal
        "git": {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
        # pytest
        "pytest": {1: "Tests failed", 5: "No tests collected"},
        "python": {1: "Script exited with error"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


# Patterns that signal the script source itself is broken (rather than the
# logic under test), so the LLM should re-read / rewrite the source instead
# of retrying the same command. The classic case: a model substitutes JSON
# `null` / `true` / `false` for Python `None` / `True` / `False` when
# generating an inline heredoc, producing NameError at runtime.
_PYTHON_ERROR_HINTS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"NameError: name '(null|true|false)' is not defined"),
        "Python source contains a JSON literal "
        "(`null`/`true`/`false`). Rewrite using `None`/`True`/`False`.",
    ),
    (
        re.compile(r"\bSyntaxError:\s*invalid syntax", re.IGNORECASE),
        "Python source has a SyntaxError. Re-read the file and fix the "
        "offending line — do not re-run the same command.",
    ),
    (
        re.compile(r"\bIndentationError\b"),
        "Python source has inconsistent indentation. Re-read the file and "
        "fix the indentation before retrying.",
    ),
    (
        re.compile(r"json\.decoder\.JSONDecodeError|json\.JSONDecodeError"),
        "Input is not valid JSON. Inspect the raw payload before retrying.",
    ),
    (
        re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]"),
        "Missing Python dependency. Install it (pip install <pkg>) or "
        "switch to a stdlib alternative — re-running won't help.",
    ),
]


def _detect_python_error_hint(output: str) -> Optional[str]:
    """Return a short remediation hint when the output shows a script-source
    error (NameError, SyntaxError, ...) rather than a logic failure.

    Empty when nothing matches — callers must treat ``None`` as no-op.
    """
    if not output:
        return None
    for pattern, hint in _PYTHON_ERROR_HINTS:
        if pattern.search(output):
            return hint
    return None

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model


# ─── File safety guards ──────────────────────────────────────────────────────
# Ported from hermes-agent tools/file_tools.py

# Paths that would hang the process (infinite output or blocking input)
_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

# Sensitive system paths that file tools should refuse to write to
_SENSITIVE_PATH_PREFIXES = ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/", "/private/var/run/")

# Maximum consecutive identical reads before hard-blocking
_MAX_CONSECUTIVE_READS = 4


def _is_blocked_device(filepath: str) -> bool:
    """Return True if the path would hang the process."""
    normalized = os.path.expanduser(filepath)
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio
    if normalized.startswith("/proc/") and normalized.endswith(
        ("/fd/0", "/fd/1", "/fd/2")
    ):
        return True
    return False


def _check_sensitive_write_path(filepath: str) -> Optional[str]:
    """Return an error message if the path targets a sensitive system location."""
    try:
        resolved = str(Path(filepath).expanduser().resolve())
    except (OSError, ValueError):
        resolved = filepath
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix):
            return (
                f"Refusing to write to sensitive system path: {filepath}\n"
                "Use the execute tool with sudo if you need to modify system files."
            )
    # Home-directory sensitive locations
    home = str(Path.home())
    for sensitive in ("/.ssh/", "/.gnupg/", "/.aws/credentials"):
        if resolved.startswith(home + sensitive):
            return (
                f"Refusing to write to sensitive path: {filepath}\n"
                "This could compromise system security."
            )
    return None


class BuiltinFileTool(Tool):
    """
    Built-in file system tool providing ls, read_file, write_file, edit_file, multi_edit_file, glob, grep functions.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            max_read_lines: int = 500,
            max_line_length: int = 2000,
            sandbox_config=None,
            diagnostics_checker=None,
    ):
        """
        Initialize BuiltinFileTool.

        Args:
            work_dir: Work directory for file operations, defaults to current working directory
            max_read_lines: Maximum number of lines to read by default
            max_line_length: Maximum length per line, longer lines will be truncated
            sandbox_config: SandboxConfig instance for path restriction enforcement
            diagnostics_checker: Optional LspDiagnosticsChecker. When set, file
                edits append newly-introduced LSP diagnostics to the tool result.
        """
        super().__init__(name="builtin_file_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.max_read_lines = max_read_lines
        self.max_line_length = max_line_length
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._sandbox_config = sandbox_config
        self.diagnostics_checker = diagnostics_checker

        # mtime cache: detect external modifications before edit/write.
        # Key: resolved absolute path (str), Value: {"mtime": float}
        self._file_read_state: Dict[str, Dict[str, Any]] = {}

        # Consecutive-read tracker: prevents LLM read-loop death spirals.
        # Dict mapping (path, offset, limit) -> consecutive count.
        # Per-key tracking is safe under concurrent reads (no cross-key interference).
        self._read_consecutive_counts: Dict[tuple, int] = {}
        self._read_last_key: Optional[tuple] = None

        # File snapshots for workspace rollback: {abs_path: [content_before_1, ...]}
        # Stores previous file content before each write/edit, supporting undo.
        self._file_snapshots: Dict[str, List[str]] = {}

        # Register all file operation functions.
        # Read-only tools are concurrency_safe (can run in parallel with each other).
        # Write tools (write_file, edit_file, multi_edit_file) stay serialised.
        self.register(self.ls, concurrency_safe=True, is_read_only=True)
        self.register(self.read_file, concurrency_safe=True, is_read_only=True)
        self.register(self.write_file, sanitize_arguments=False, is_destructive=True)
        self.register(self.edit_file, sanitize_arguments=False, is_destructive=True)
        self.register(self.multi_edit_file, sanitize_arguments=False, is_destructive=True)
        self.register(self.glob, concurrency_safe=True, is_read_only=True)
        self.register(self.grep, concurrency_safe=True, is_read_only=True)
        # grep enforces its own 30s timeout on both the rg and pure-Python
        # fallback paths, so skip the outer 120s executor wrapper — a missing
        # rg used to let the fallback walk huge trees for the full 120s.
        self.functions["grep"].manages_own_timeout = True
        self.register(self.undo_edit, is_destructive=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path, supporting absolute, relative, and ~ paths.

        - ~ paths are expanded to user home directory
        - Absolute paths are used directly
        - Relative paths are resolved relative to work_dir
        """
        # Expand ~ to user home directory
        if path.startswith("~"):
            return Path(path).expanduser()
        p = Path(path)
        if p.is_absolute():
            return p
        return self.work_dir / p

    def _get_file_lock(self, path: str) -> asyncio.Lock:
        """Get or create a per-file asyncio.Lock to serialize concurrent edits."""
        return self._file_locks.setdefault(path, asyncio.Lock())

    async def _diagnostics_snapshot(self, path: "Path") -> None:
        """Capture a pre-edit diagnostics baseline (off the event loop).

        No-op when no checker is attached. Cheap on repeat edits (the checker
        caches the baseline per file).
        """
        if self.diagnostics_checker is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.diagnostics_checker.snapshot_before, str(path))

    async def _diagnostics_after(self, path: "Path") -> str:
        """Return formatted newly-introduced diagnostics (off the event loop)."""
        if self.diagnostics_checker is None:
            return ""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self.diagnostics_checker.report_after, str(path)
            )
        except Exception as e:
            logger.warning(f"Diagnostics check failed for {path}: {e}")
            return ""

    def _preflight_edit_check(self, file_path: str) -> Optional[str]:
        """Run shared preflight checks for edit_file / multi_edit_file.

        - Sensitive-path guard: raises PermissionError immediately.
        - mtime staleness guard: returns a Warning string (not an error) —
          the edit is still rejected but the model sees advice, not a crash.
        Returns None if the edit is safe to proceed.
        """
        path = self._resolve_path(file_path)

        # Sensitive path guard (e.g. /etc/passwd, ~/.ssh)
        sensitive_err = _check_sensitive_write_path(str(path))
        if sensitive_err:
            raise PermissionError(sensitive_err)

        # mtime guard: detect external modifications since last read.
        abs_path = str(path.resolve())
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            current_mtime = None
        if current_mtime is not None and abs_path in self._file_read_state:
            prev_mtime = self._file_read_state[abs_path].get("mtime")
            if prev_mtime is not None and current_mtime != prev_mtime:
                logger.warning(
                    f"File '{file_path}' was modified externally since last read "
                    f"(mtime {prev_mtime} -> {current_mtime}). "
                    f"Please re-read the file before editing."
                )
                return (
                    f"Warning: File '{file_path}' was modified externally since your last read. "
                    f"Please re-read the file with read_file() before editing to avoid "
                    f"overwriting someone else's changes."
                )

        return None


    def _validate_path(self, path: str) -> str:
        """Validate path against sandbox restrictions and blocked device files.

        Always checks:
        - Path must not resolve to a known device file (/dev/zero, etc.)

        When sandbox is enabled, also checks:
        - Path components do not match any blocked_paths entries
        - Uses path component matching (not substring) to avoid false positives
        - For write operations, caller should use _validate_write_path instead

        Raises:
            PermissionError: If path is blocked by sandbox config or is a device file
        """
        resolved = self._resolve_path(path).resolve()

        # Device-file guard: always active regardless of sandbox setting.
        # Reading /dev/zero or /dev/random hangs indefinitely or exhausts memory.
        if str(resolved) in self.BLOCKED_DEVICE_FILES:
            raise PermissionError(
                f"Reading device file '{path}' is blocked for safety. "
                f"Resolved path: {resolved}"
            )

        if self._sandbox_config is None or not self._sandbox_config.enabled:
            return path
        resolved_parts = set(resolved.parts)
        for blocked in self._sandbox_config.blocked_paths:
            if blocked in resolved_parts:
                raise PermissionError(f"Sandbox: access to path containing '{blocked}' is blocked")
        return path

    def _validate_write_path(self, path: str) -> str:
        """Validate that a write operation is allowed under sandbox restrictions.

        Checks blocked_paths and writable_dirs whitelist.

        Raises:
            PermissionError: If write is not allowed
        """
        self._validate_path(path)
        if self._sandbox_config is None or not self._sandbox_config.enabled:
            return path
        resolved = str(self._resolve_path(path).resolve())
        # If writable_dirs is configured, enforce whitelist
        if self._sandbox_config.writable_dirs:
            allowed = False
            for wd in self._sandbox_config.writable_dirs:
                wd_resolved = str(Path(wd).expanduser().resolve())
                if resolved.startswith(wd_resolved):
                    allowed = True
                    break
            if not allowed:
                # Also allow work_dir
                work_dir_str = str(self.work_dir.resolve())
                if not resolved.startswith(work_dir_str):
                    raise PermissionError(
                        f"Sandbox: write to '{path}' is not allowed. "
                        f"Writable dirs: {self._sandbox_config.writable_dirs}"
                    )
        return path

    async def ls(self, directory: str = ".") -> str:
        """List the immediate entries (files AND subdirectories) of a directory, NON-recursive.

        Usage:
        - The directory parameter can be absolute, relative, or `~`-prefixed
        - Returns one entry per immediate child — both files and subdirectories.
          Each entry has a ``type`` field: ``"file"`` or ``"dir"``. Hidden entries
          (names starting with ``.``) are included.
        - This tool does NOT recurse. To search a directory tree, use ``glob``
          (e.g. ``glob("**/*.py", path=...)``) or ``grep`` instead — calling ``ls``
          repeatedly on every subdirectory wastes turns.
        - Useful for discovering what's at a known path before ``read_file`` / ``edit_file``.

        Args:
            directory: Directory path to list (default: current working directory).

        Returns:
            JSON-formatted list, e.g. ``[{"name": "src", "path": "...", "type": "dir"},
            {"name": "main.py", "path": "...", "type": "file"}]``.
        """
        self._validate_path(directory)
        dir_path = self._resolve_path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        def _ls_sync():
            items = []
            for item in sorted(dir_path.iterdir()):
                item_type = "dir" if item.is_dir() else "file"
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": item_type,
                })
            return items

        items = await asyncio.get_event_loop().run_in_executor(None, _ls_sync)

        logger.debug(f"Listed {len(items)} items in {dir_path}")
        result = json.dumps(items, ensure_ascii=False, indent=2)
        result = truncate_if_too_long(result)
        return str(result)

    # Maximum file size (bytes) for read_file.  Larger files must use offset+limit.
    # Mirrors CC's FileReadTool maxSizeBytes (256KB).
    MAX_FILE_SIZE_BYTES = 256_000

    # Device files that must never be read: reading /dev/zero or /dev/random
    # hangs indefinitely or exhausts memory.  Absolute paths only — checked
    # after resolving the input path so symlinks cannot bypass the guard.
    BLOCKED_DEVICE_FILES: frozenset = frozenset({
        "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
        "/dev/tty", "/dev/stdin", "/dev/stdout", "/dev/stderr",
        "/dev/mem", "/dev/kmem", "/dev/port",
    })

    async def read_file(
            self,
            file_path: str,
            offset: int = 0,
            limit: Optional[int] = 500,
    ) -> str:
        """Reads a file from the filesystem. You can access any file directly by using this tool.
        Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

        Usage:
        - The file_path parameter may be absolute, relative to the working directory, or `~`-prefixed
        - Relative paths are resolved relative to the base working directory
        - By default, it reads up to 500 lines starting from the beginning of the file
        - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
        - First scan: read_file(path, limit=100) to see file structure
        - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
        - Only omit limit (read full file) when necessary for editing
        - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
        - Any lines longer than 2000 characters will be truncated
        - Results are returned in numbered-line format (line_number + content), starting at line 1
        - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
        - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
        - You should ALWAYS make sure a file has been read before editing it.

        Args:
            file_path: File path for md/txt/py/etc. Supports absolute paths, relative paths, and `~`
            offset: Starting line number (0-based)
            limit: Maximum number of lines to read, defaults to 500

        Returns:
            File content with line numbers
        """
        self._validate_path(file_path)
        path = self._resolve_path(file_path)

        # ── Device path guard ─────────────────────────────────────
        if _is_blocked_device(str(path)):
            raise PermissionError(
                f"Cannot read '{file_path}': this is a device file "
                "that would block or produce infinite output."
            )

        if not path.exists():
            # Reset consecutive tracker for this path
            for k in [k for k in self._read_consecutive_counts if k[0] == file_path]:
                del self._read_consecutive_counts[k]
            self._read_last_key = None
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        abs_path = str(path.resolve())

        # --- Large-file guard (mirrors CC's maxSizeBytes) ---
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = None
        if file_size is not None and file_size > self.MAX_FILE_SIZE_BYTES:
            loop = asyncio.get_running_loop()
            total_lines = await loop.run_in_executor(
                None, lambda: sum(1 for _ in open(path, errors='ignore'))
            )
            raise ValueError(
                f"File too large ({file_size:,} bytes, {total_lines:,} lines). "
                f"Use offset and limit to read specific sections. "
                f"Example: read_file('{file_path}', offset=0, limit=100)"
            )

        limit = limit if limit is not None else self.max_read_lines
        max_line_len = self.max_line_length

        # Async streaming read — only read the lines we need
        output_lines = []
        total_lines = 0
        end_line = offset + limit
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            async for line in f:
                total_lines += 1
                if total_lines > offset and total_lines <= end_line:
                    line = line.rstrip('\n\r')
                    if len(line) > max_line_len:
                        line = line[:max_line_len] + "..."
                    output_lines.append(f"{total_lines:6d}\t{line}")

        result = "\n".join(output_lines)

        # Add file info if truncated
        actual_end = min(offset + len(output_lines), total_lines)
        if actual_end < total_lines:
            result += f"\n\n[Showing lines {offset + 1}-{actual_end} of {total_lines} total lines]"

        # Record mtime so edit_file can detect external modifications
        try:
            self._file_read_state[abs_path] = {"mtime": path.stat().st_mtime}
        except OSError:
            pass

        # ── Consecutive-read loop detection ───────────────────────
        read_key = (file_path, offset, limit)
        self._read_consecutive_counts[read_key] = self._read_consecutive_counts.get(read_key, 0) + 1
        count = self._read_consecutive_counts[read_key]
        self._read_last_key = read_key

        if count >= _MAX_CONSECUTIVE_READS:
            # Soft signal (not an error): tell the agent to stop re-reading.
            # Not a FileNotFoundError or the like -- the file exists and the
            # read succeeded; we simply refuse to return the content again.
            return (
                f"BLOCKED: You have read this exact file region {count} times in a row. "
                "The content has NOT changed. You already have this information. "
                "STOP re-reading and proceed with your task."
            )
        if count >= 3:
            result += (
                f"\n\n[Warning: You have read this exact file region {count} times "
                "consecutively. The content has not changed. Use the information "
                "you already have.]"
            )

        logger.debug(f"Read file {file_path}: lines {offset + 1}-{actual_end}, total {total_lines} lines")
        return result

    async def write_file(self, file_path: str, content: str) -> str:
        """Writes content to a file in the filesystem.

        Usage:
        - If this is an existing file, you MUST use read_file first to read the file's contents.
          This tool will create a new file or OVERWRITE the existing file entirely.
        - Prefer edit_file for modifying existing files — it only sends the diff.
          Only use write_file to create NEW files or for complete rewrites.
        - The file_path can be relative (e.g., "tmp/script.py", "./outputs/data.txt") or absolute path.
          Relative paths are resolved relative to the base working directory.
        - The tool returns the actual absolute path of the created file — ALWAYS use this returned
          path for subsequent operations (read_file, execute, etc.). Do NOT guess or construct paths.
        - Parent directories will be created automatically if they don't exist.

        Args:
            file_path: File path (relative or absolute). Examples: "tmp/script.py", "outputs/result.txt", "./tmp/main.py", use './tmp/' prefix file path for temporary files
            content: File content to write

        Returns:
            Operation result message containing the actual absolute path of the file
        """
        self._validate_write_path(file_path)
        path = self._resolve_path(file_path)

        # ── Sensitive path guard ──────────────────────────────────
        sensitive_err = _check_sensitive_write_path(str(path))
        if sensitive_err:
            raise PermissionError(sensitive_err)

        # ── File staleness check ──────────────────────────────────
        stale_warning = ""
        abs_path = str(path.resolve()) if path.exists() else None
        if abs_path and abs_path in self._file_read_state:
            read_mtime = self._file_read_state[abs_path].get("mtime")
            if read_mtime is not None:
                try:
                    current_mtime = path.stat().st_mtime
                    if current_mtime != read_mtime:
                        stale_warning = (
                            f"\nWarning: {file_path} was modified since you last read it "
                            "(external edit or concurrent agent). Consider re-reading."
                        )
                except OSError:
                    pass

        # Reset consecutive-read tracker (write breaks the loop)
        self._read_last_key = None
        self._read_consecutive_counts.clear()

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        action = "Created" if not path.exists() else "Updated"

        # ── Snapshot for rollback ─────────────────────────────────
        await self._diagnostics_snapshot(path)
        if path.exists() and path.is_file():
            try:
                old_content = path.read_text(encoding='utf-8', errors='ignore')
                abs_snap = str(path.resolve())
                self._file_snapshots.setdefault(abs_snap, []).append(old_content)
            except OSError:
                pass

        # Atomic write: write to temp file then rename to avoid partial writes
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.close(tmp_fd)
            async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            # Atomic rename
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Return absolute path to help LLM use correct path in subsequent operations
        absolute_path = str(path.resolve())
        try:
            self._file_read_state[absolute_path] = {"mtime": path.stat().st_mtime}
        except OSError:
            self._file_read_state.pop(absolute_path, None)
        logger.debug(f"{action} file: {absolute_path}, file content length: {len(content)} characters")
        diag_text = await self._diagnostics_after(path)
        suffix = f"\n\n{diag_text}" if diag_text else ""
        return f"{action} file, absolute path: {absolute_path}{stale_warning}{suffix}"

    async def edit_file(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Replace a specific string in a file.

        You MUST use read_file at least once before editing a file.
        This tool will error if the file has been modified externally since your last read.

        Uses literal string matching (NOT regex). Multi-line strings are supported.
        Prefer this tool over write_file or shell `sed` for targeted changes.

        When editing text from read_file output, ensure you preserve the exact indentation
        (tabs/spaces) as it appears in the file. The line number prefix in read_file output
        is metadata only — never include it in old_string or new_string.

        The edit will FAIL if old_string is not unique in the file. Either provide a
        larger string with more surrounding context to make it unique, or use
        replace_all=True to change every instance.

        For multiple edits to the SAME file, prefer `multi_edit_file` to apply them
        atomically in one call. If you call `edit_file` multiple times on the same file
        in parallel, they will be serialized automatically to avoid race conditions.
        File paths may be absolute, relative to the working directory, or `~`-prefixed.

        Args:
            file_path: The path to the file to edit. Supports absolute paths, relative
                      paths, and `~`. Relative paths resolve from the working directory.
            old_string: The existing text to find and replace. Must match exactly.
            new_string: The replacement text.
            replace_all: Whether to replace all occurrences. Default: False (replace first
                        match only; errors if multiple matches found).

        Returns:
            Operation result message

        Examples:
            edit_file("app.py", "def foo():", "def bar():")
            edit_file("config.py", "DEBUG = True", "DEBUG = False")
            edit_file("test.py", "old_name", "new_name", replace_all=True)
        """
        self._validate_write_path(file_path)
        path = self._resolve_path(file_path)
        path_key = str(path)

        # Shared preflight: sensitive path (raises) + mtime staleness (warning)
        preflight_warning = self._preflight_edit_check(file_path)
        if preflight_warning:
            # mtime staleness is a soft-reject: return warning string so the
            # model can re-read and retry rather than seeing a crash.
            return preflight_warning

        abs_path = str(path.resolve())

        # Reset consecutive-read tracker (edit breaks the loop)
        self._read_last_key = None
        self._read_consecutive_counts.clear()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        await self._diagnostics_snapshot(path)

        # Per-file lock to serialize concurrent edits on the same file
        lock = self._get_file_lock(path_key)
        async with lock:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # ── Snapshot for rollback before edit ─────────────────
            self._file_snapshots.setdefault(abs_path, []).append(content)

            result = self._str_replace(content, old_string, new_string, replace_all)

            if not result["success"]:
                raise ValueError(result["error"])

            # Atomic write back
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                os.close(tmp_fd)
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(result["new_content"])
                os.replace(tmp_path, str(path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        logger.debug(f"Replaced {result['count']} occurrence(s) in {file_path}")
        try:
            self._file_read_state[abs_path] = {"mtime": path.stat().st_mtime}
        except OSError:
            self._file_read_state.pop(abs_path, None)
        diag_text = await self._diagnostics_after(path)
        suffix = f"\n\n{diag_text}" if diag_text else ""
        return f"Successfully replaced {result['count']} occurrence(s) in '{file_path}'{suffix}"

    async def multi_edit_file(
            self,
            file_path: str,
            edits: List[Dict[str, Any]],
            continue_on_error: bool = True,
    ) -> str:
        """Apply multiple edits to a single file.

        Edits are applied sequentially on the same in-memory content, then
        the result is written back atomically once.

        This is preferred over multiple parallel `edit_file` calls when you need
        to make several changes to the same file — it is faster, uses fewer tokens,
        and the write-back step is atomic.

        Failure semantics — controlled by ``continue_on_error``:

        - ``True`` (default, recommended): each edit is independent. Successful
          edits are written, failed edits are reported back per-item so the
          model can retry just the failing ones. This avoids the "1-failure
          poisons all 7 edits" footgun where a slightly wrong ``old_string``
          forces the model to redo every edit from scratch.
        - ``False``: classic atomic mode. If ANY edit fails, NO edits are
          applied. Use only when the edits genuinely depend on each other
          (e.g. an earlier edit introduces text the next edit references).

        Args:
            file_path: Path to the file to edit.
            edits: List of edit operations. Each dict must contain:
                - old_string (str): The existing text to find
                - new_string (str): The replacement text
                - replace_all (bool, optional): Replace all occurrences. Default: False
            continue_on_error: Apply successful edits even when some fail.
                Default: True.

        Returns:
            Summary listing which edits applied and which failed.
        """
        self._validate_write_path(file_path)
        path = self._resolve_path(file_path)
        path_key = str(path)

        # Shared preflight: sensitive path (raises) + mtime staleness (warning)
        preflight_warning = self._preflight_edit_check(file_path)
        if preflight_warning:
            return preflight_warning

        abs_path = str(path.resolve())

        # Reset consecutive-read tracker (multi-edit breaks the loop)
        self._read_last_key = None
        self._read_consecutive_counts.clear()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")
        if not edits:
            raise ValueError("'edits' list cannot be empty.")

        await self._diagnostics_snapshot(path)

        lock = self._get_file_lock(path_key)
        async with lock:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # ── Snapshot for rollback before edits ────────────────
            self._file_snapshots.setdefault(abs_path, []).append(content)

            # Apply edits sequentially on in-memory content
            results: List[str] = []
            failures: List[str] = []
            applied_count = 0
            for i, edit in enumerate(edits):
                old_string = edit.get("old_string", "")
                new_string = edit.get("new_string", "")
                replace_all = edit.get("replace_all", False)

                if not old_string:
                    err = f"Edit {i + 1}/{len(edits)}: empty old_string"
                    if continue_on_error:
                        failures.append(err)
                        results.append(f"Edit {i + 1}: SKIPPED ({err})")
                        continue
                    raise ValueError(f"{err}. No changes were made.")

                result = self._str_replace(content, old_string, new_string, replace_all)
                if not result["success"]:
                    err = f"Edit {i + 1}/{len(edits)}: {result['error']}"
                    if continue_on_error:
                        failures.append(err)
                        results.append(f"Edit {i + 1}: FAILED ({result['error']})")
                        continue
                    raise ValueError(f"{err}. No changes were made.")

                content = result["new_content"]
                applied_count += 1
                results.append(f"Edit {i + 1}: replaced {result['count']} occurrence(s)")

            # If nothing applied (all failed in best-effort mode), don't touch the file —
            # surface the failures so the LLM can retry without producing an empty diff.
            if applied_count == 0:
                summary_lines = [
                    f"No edits applied to '{file_path}' ({len(failures)}/{len(edits)} failed):",
                    *results,
                ]
                return "\n".join(summary_lines)

            # Atomic write (once)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                os.close(tmp_fd)
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
                os.replace(tmp_path, str(path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        if failures:
            header = (
                f"Partially applied edits to '{file_path}' "
                f"({applied_count}/{len(edits)} succeeded, {len(failures)} failed). "
                "Successful edits have been written. "
                "Re-issue ONLY the failing edits (likely with adjusted old_string) — "
                "do not resend the successful ones:"
            )
        else:
            header = f"Successfully applied {len(edits)} edits to '{file_path}':"
        summary = header + "\n" + "\n".join(results)
        logger.debug(summary)
        # Update mtime state so subsequent edits don't trigger stale-read warning
        try:
            self._file_read_state[abs_path] = {"mtime": path.stat().st_mtime}
        except OSError:
            self._file_read_state.pop(abs_path, None)
        diag_text = await self._diagnostics_after(path)
        if diag_text:
            summary += f"\n\n{diag_text}"
        return summary

    @staticmethod
    def _normalize_quotes(s: str) -> str:
        """Replace curly/typographic quotes with their ASCII equivalents.

        LLMs sometimes emit curly quotes (\u201c\u201d\u2018\u2019) when the
        source file uses straight ASCII quotes, causing exact-match failures.
        """
        return (
            s.replace('\u201c', '"').replace('\u201d', '"')   # left/right double
             .replace('\u2018', "'").replace('\u2019', "'")   # left/right single
             .replace('\u2032', "'").replace('\u2033', '"')   # prime / double prime
        )

    def _str_replace(
            self,
            content: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> dict:
        """Internal string replacement logic.

        Tries exact match first.  If that fails, retries after normalizing
        curly/typographic quotes in old_string to ASCII equivalents — LLMs
        sometimes emit curly quotes when the file uses straight ASCII quotes.

        Returns:
            {"success": bool, "new_content": str, "count": int, "error": str}
        """
        # Find all match positions
        matches = []
        start = 0
        while True:
            idx = content.find(old_string, start)
            if idx == -1:
                break
            matches.append(idx)
            start = idx + len(old_string)

        # Quote-normalization fallback: if exact match failed, retry with
        # normalized quotes.  We search the normalized content for the
        # normalized needle, then map positions back to the original content
        # so the actual replacement preserves the file's quote style.
        if not matches:
            norm_content = self._normalize_quotes(content)
            norm_old = self._normalize_quotes(old_string)
            if norm_old != old_string and norm_old in norm_content:
                # Find positions in normalized content, then use them on original
                # (lengths are identical because _normalize_quotes is 1-to-1).
                start = 0
                while True:
                    idx = norm_content.find(norm_old, start)
                    if idx == -1:
                        break
                    matches.append(idx)
                    start = idx + len(norm_old)
                if matches:
                    # Replace using the normalized needle on the original content
                    # (positions are identical because character counts don't change).
                    old_string = norm_old
                    content_for_replace = norm_content
                else:
                    content_for_replace = content
            else:
                content_for_replace = content
        else:
            content_for_replace = content

        if not matches:
            display_old = old_string[:100] + "..." if len(old_string) > 100 else old_string
            return {
                "success": False,
                "error": f"String not found: '{display_old}'",
                "new_content": content,
                "count": 0,
            }

        # If not replace_all and multiple matches, show context for each match
        if not replace_all and len(matches) > 1:
            contexts = []
            for idx in matches[:3]:  # Show first 3 matches
                line_num = content_for_replace[:idx].count('\n') + 1
                # Get surrounding context (up to 50 chars around the match)
                context_start = max(0, idx - 20)
                context_end = min(len(content_for_replace), idx + len(old_string) + 30)
                context = content_for_replace[context_start:context_end].replace('\n', '\\n')
                contexts.append(f"  Line {line_num}: ...{context}...")

            error_msg = (
                f"Found {len(matches)} occurrences of the string.\n"
                f"Use replace_all=True to replace all, or provide more context to make it unique.\n"
                f"Matches found at:\n" + '\n'.join(contexts)
            )
            if len(matches) > 3:
                error_msg += f"\n  ... and {len(matches) - 3} more"

            return {
                "success": False,
                "error": error_msg,
                "new_content": content,
                "count": len(matches),
            }

        # Perform replacement
        if replace_all:
            new_content = content_for_replace.replace(old_string, new_string)
            count = len(matches)
        else:
            # Replace only the first match (leftmost)
            idx = matches[0]
            new_content = content_for_replace[:idx] + new_string + content_for_replace[idx + len(old_string):]
            count = 1

        return {
            "success": True,
            "new_content": new_content,
            "count": count,
            "error": None,
        }

    async def glob(self, pattern: str, path: str = ".") -> str:
        """Find files matching a glob pattern (supports recursive search with `**`).

        Usage:
        - This tool searches for files by matching standard glob wildcards, returns JSON formatted absolute file paths
        - Core glob wildcards (key differences):
        1. `*`: Matches any files in the **current specified single directory** (non-recursive, no deep subdirectories)
        2. `**`: Matches any directories recursively (penetrates all deep subdirectories for cross-level search)
        3. `?`: Matches any single character (e.g., "file?.txt" matches "file1.txt", "filea.txt")
        - Patterns can be absolute (starting with `/`, e.g., "/home/user/*.py") or relative (e.g., "docs/*.md")
        - Automatically excludes common useless directories (.git, __pycache__, etc.) to filter valid files
        - Returns empty JSON list if no matching files are found

        Examples (clear parameter correspondence and function explanation):
        - pattern: `*.py`, path: "." - Find all Python files in the current working directory (non-recursive)
        - pattern: `*.txt`, path: "." - Find all text files in the current working directory (non-recursive)
        - pattern: `**/*.md`, path: "/path/to/subdir/" - Find all markdown files in all levels under /path/to/subdir/ (recursive)
        - pattern: `subdir/*.md`, path: "." - Find all markdown files directly in the "subdir" folder (non-recursive, no deep subdirs)

        Args:
            pattern: Valid glob search pattern, e.g., "*.py", "**/*.md", "src/?*.js"
            path: Starting search directory (relative or absolute), defaults to current working directory (".").

        Returns:
            JSON formatted string of sorted absolute file paths (filtered to exclude ignored directories).
        """
        self._validate_path(path)
        base_path = self._resolve_path(path)

        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Run glob in executor to avoid blocking on large directory trees
        def _glob_sync():
            matches = list(base_path.glob(pattern))
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            return sorted(
                str(m) for m in matches
                if not set(m.parts).intersection(ignore_dirs)
            )

        filtered = await asyncio.get_event_loop().run_in_executor(None, _glob_sync)

        logger.debug(f"Glob found {len(filtered)} files matching pattern '{pattern}' in directory '{path}'")
        # Convert to formatted JSON string
        result = json.dumps(filtered, ensure_ascii=False, indent=2)
        # Truncate if content exceeds the limit to avoid excessive output
        result = truncate_if_too_long(result)
        return str(result)

    async def grep(
            self,
            pattern: str,
            path: str = ".",
            *,
            include: Optional[str] = None,
            output_mode: Literal["content", "files_with_matches", "count"] = "content",
            case_insensitive: bool = False,
            multiline: bool = False,
            context_lines: int = 0,
            before_context: int = 0,
            after_context: int = 0,
            limit: int = 100,
            fixed_strings: bool = False,
    ) -> str:
        """Search file contents for a pattern using ripgrep (rg).

        Default output is matching lines with `file:line_number:content`.
        Switch to "files_with_matches" only when you just need a path list,
        or "count" when you only need totals — both modes drop the actual code,
        which usually forces a follow-up read_file.

        Usage:
        - Powered by ripgrep for speed (falls back to pure Python if rg missing)
        - The pattern parameter supports regex by default (e.g., 'class \\w+', 'def \\w+')
        - Use fixed_strings=True to treat pattern as literal text (no regex)
        - The path parameter specifies the search directory (default: current working directory)
        - The include parameter filters files by glob (e.g., "*.py", "*.{ts,tsx}")
        - output_mode (plain string):
          - "content" (default): matching lines with file path + line numbers
          - "files_with_matches": list of matching file paths only
          - "count": match count per file
        - Add context lines in "content" mode via context_lines / before_context / after_context.

        Args:
            pattern: Text/regex to search for
            path: Starting directory for search (default: ".")
            include: File glob filter, e.g., "*.py", "*.{js,ts}" (maps to rg --glob)
            output_mode: "content" (default), "files_with_matches", or "count". Do NOT pass a dict.
            case_insensitive: Ignore case when matching (default: False)
            multiline: Enable multiline matching where . matches newlines (default: False)
            context_lines: Show N lines before and after each match (default: 0, content mode only)
            before_context: Show N lines before each match (default: 0, content mode only)
            after_context: Show N lines after each match (default: 0, content mode only)
            limit: Maximum results to return (default: 100)
            fixed_strings: Treat pattern as literal text, not regex (default: False)

        Returns:
            Search results as formatted string

        Examples:
            grep(pattern="def _close_box")                     # show matches with line numbers
            grep(pattern="TODO", include="*.py", context_lines=2)
            grep(pattern="class \\w+", include="*.py")
            grep(pattern="enable_agentic_prompt", context_lines=3)
            grep(pattern="exact phrase", fixed_strings=True)
            grep(pattern="import", include="*.py", output_mode="count")
            grep(pattern="Foo", output_mode="files_with_matches")  # only when paths suffice
        """
        # Resolve and validate path
        self._validate_path(path)
        base_path = self._resolve_path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Check if rg is available
        rg_path = shutil.which("rg")
        if rg_path is None:
            return await self._run_grep_fallback(
                pattern, path, include, output_mode, limit, fixed_strings,
                case_insensitive,
            )

        # Build rg command arguments
        cmd: List[str] = [rg_path]

        # Output mode flags
        if output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count")
        else:  # content
            cmd.append("--line-number")

        # Matching options
        if fixed_strings:
            cmd.append("--fixed-strings")
        if case_insensitive:
            cmd.append("--ignore-case")
        if multiline:
            cmd.extend(["--multiline", "--multiline-dotall"])

        # Context lines (content mode only)
        if output_mode == "content":
            if context_lines > 0:
                cmd.extend(["--context", str(context_lines)])
            else:
                if before_context > 0:
                    cmd.extend(["--before-context", str(before_context)])
                if after_context > 0:
                    cmd.extend(["--after-context", str(after_context)])

        # File filter
        if include:
            cmd.extend(["--glob", include])

        # Result limit: for content mode, limit matches per file
        if output_mode == "content":
            cmd.extend(["--max-count", str(limit)])

        # Exclude common irrelevant directories (rg already ignores .git via .gitignore)
        for d in ["__pycache__", "node_modules", ".venv", "venv", ".idea", ".pytest_cache"]:
            cmd.extend(["--glob", f"!{d}/"])

        # Pattern and path
        cmd.append("--")
        cmd.append(pattern)
        cmd.append(str(base_path))

        # rg is normally millisecond-fast; a hard _GREP_TIMEOUT catches hangs
        # (pathological regex, huge binary files, or zombie processes).
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_GREP_TIMEOUT)
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            raise TimeoutError(f"grep timed out after {_GREP_TIMEOUT} seconds")
        except FileNotFoundError:
            return await self._run_grep_fallback(
                pattern, path, include, output_mode, limit, fixed_strings,
                case_insensitive,
            )

        # rg exit codes: 0=matches found, 1=no matches, 2=error
        if proc.returncode == 2:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"grep(rg) failed: {err}")

        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return f"No matches found for '{pattern}'"

        # Truncate result lines for files_with_matches / count
        if output_mode in ("files_with_matches", "count"):
            lines = output.split("\n")
            if len(lines) > limit:
                output = "\n".join(lines[:limit])
                output += f"\n... ({len(lines) - limit} more results truncated)"

        result = truncate_if_too_long(output)
        logger.debug(f"Grep(rg) for '{pattern}': result length {len(result)} chars")
        return result

    async def _run_grep_fallback(
            self,
            pattern: str,
            path: str,
            include: Optional[str],
            output_mode: str,
            limit: int,
            fixed_strings: bool,
            case_insensitive: bool = False,
    ) -> str:
        """Run the pure-Python fallback in an executor with a hard timeout.

        The fallback walks the tree in a thread, so on timeout we can only
        drop the result — the thread keeps running — but the tool returns a
        clear timeout error at ``_GREP_TIMEOUT`` instead of hanging to the
        outer 120s executor limit.
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._grep_fallback, pattern, path, include,
                    output_mode, limit, fixed_strings, case_insensitive,
                ),
                timeout=_GREP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"grep timed out after {_GREP_TIMEOUT} seconds")

    def _grep_fallback(
            self,
            pattern: str,
            path: str,
            include: Optional[str],
            output_mode: str,
            limit: int,
            fixed_strings: bool,
            case_insensitive: bool = False,
    ) -> str:
        """Fallback grep using pure Python when ripgrep is not available."""
        base_path = self._resolve_path(path)

        # Compile regex
        regex_pattern = None
        if not fixed_strings:
            try:
                flags = re.IGNORECASE if case_insensitive else 0
                regex_pattern = re.compile(pattern, flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        # Determine files to search
        if include:
            files = list(base_path.glob(f"**/{include}"))
        else:
            files = list(base_path.glob("**/*"))

        # Exclude directories and ignored paths
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
        files = [f for f in files if f.is_file() and not set(f.parts).intersection(ignore_dirs)]

        results = []
        file_counts = {}

        match_pattern = pattern.lower() if (case_insensitive and fixed_strings) else pattern

        for fp in files:
            if len(results) >= limit:
                break

            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except OSError:
                # Per-file read failure shouldn't abort the whole grep —
                # skip the unreadable file and continue.
                continue

            file_matches = []
            for line_num, line in enumerate(lines, 1):
                if fixed_strings:
                    check_line = line.lower() if case_insensitive else line
                    matched = match_pattern in check_line
                else:
                    matched = regex_pattern.search(line)
                if matched:
                    file_matches.append({
                        "line_num": line_num,
                        "content": line.strip()[:200],
                    })

            if file_matches:
                file_counts[str(fp)] = len(file_matches)
                if output_mode == "content":
                    for match in file_matches[:limit - len(results)]:
                        results.append(f"{fp}:{match['line_num']}: {match['content']}")
                elif output_mode == "files_with_matches":
                    results.append(str(fp))

        # Format output
        if output_mode == "count":
            output_lines = [f"{p}:{c}" for p, c in file_counts.items()]
            result = "\n".join(output_lines) if output_lines else f"No matches found for '{pattern}'"
        elif output_mode == "files_with_matches":
            result = "\n".join(sorted(set(results))) if results else f"No matches found for '{pattern}'"
        else:  # content
            result = "\n".join(results) if results else f"No matches found for '{pattern}'"

        result = truncate_if_too_long(result)
        logger.debug(f"Grep(fallback) for '{pattern}': found {len(file_counts)} files, result length: {len(result)} chars")
        return result

    async def undo_edit(self, file_path: str) -> str:
        """Undo the last edit or write to a file, restoring the previous version.

        Each write_file() and edit_file() call automatically snapshots the file's
        content before modification. This tool restores the most recent snapshot,
        effectively undoing the last change. Can be called multiple times to step
        back through multiple edits.

        Args:
            file_path: Path to the file to restore

        Returns:
            Confirmation message or error if no previous version exists
        """
        self._validate_write_path(file_path)
        path = self._resolve_path(file_path)

        # ── Reuse safety guards from write_file ───────────────────
        sensitive_err = _check_sensitive_write_path(str(path))
        if sensitive_err:
            raise PermissionError(sensitive_err)

        abs_path = str(path.resolve())
        snapshots = self._file_snapshots.get(abs_path)
        if not snapshots:
            raise FileNotFoundError(
                f"No previous version available for '{file_path}'. "
                "Only files modified in this session can be undone."
            )
        previous = snapshots.pop()

        # ── Atomic restore with per-file lock ─────────────────────
        path_key = str(path)
        lock = self._get_file_lock(path_key)
        async with lock:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                os.close(tmp_fd)
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(previous)
                os.replace(tmp_path, str(path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        # Update mtime state
        try:
            self._file_read_state[abs_path] = {"mtime": path.stat().st_mtime}
        except OSError:
            pass
        remaining = len(snapshots)
        return (
            f"Restored '{file_path}' to previous version ({len(previous)} chars). "
            f"{remaining} more undo(s) available."
        )


class BuiltinExecuteTool(Tool):
    """
    Built-in command execution tool using async subprocess.
    Exposed as execute function for consistent naming in Agent.
    """

    def __init__(self, work_dir: Optional[str] = None, timeout: int = 120, max_timeout: int = 600,
                 max_output_length: int = 20000, sandbox_config=None):
        """
        Initialize BuiltinExecuteTool.

        Args:
            work_dir: Work directory for command execution
            timeout: Default command execution timeout in seconds
            max_timeout: Maximum allowed timeout in seconds
            max_output_length: Maximum length of output to return
            sandbox_config: SandboxConfig instance for command restriction enforcement
        """
        super().__init__(name="builtin_execute_tool")
        self._work_dir: Optional[Path] = Path(work_dir) if work_dir else None
        self._timeout = timeout
        self._max_timeout = max_timeout
        self._max_output_length = max_output_length
        self._sandbox_config = sandbox_config
        # Override timeout from sandbox config if set
        if sandbox_config and sandbox_config.enabled and sandbox_config.max_execution_time:
            self._timeout = sandbox_config.max_execution_time
        # Import ShellTool for its syntax-fix helpers
        from agentica.tools.shell_tool import ShellTool
        self._shell = ShellTool(work_dir=work_dir, timeout=timeout)
        self.register(self.execute, is_destructive=True)
        # Large bash outputs are persisted to disk (context gets preview only).
        # read_file keeps max_result_size_chars=None (never persist — avoids
        # reading its own persisted output file in a loop).
        self.functions["execute"].max_result_size_chars = 50_000
        # Execute tool manages its own timeout internally via asyncio.wait_for
        # on the subprocess. Skip the outer timeout wrapper in Model.run_function_calls.
        self.functions["execute"].manages_own_timeout = True

    async def execute(self, command: str, timeout: Optional[int] = None) -> str:
        """Executes a shell command, capturing both stdout and stderr.

        IMPORTANT — Use dedicated tools instead of bash equivalents:
        - File search:    Use glob tool    (NOT find, ls -R, or locate)
        - Content search: Use grep tool    (NOT grep, rg, or ag)
        - Read files:     Use read_file    (NOT cat, head, tail, less, or more)
        - Edit files:     Use edit_file    (NOT sed, awk, or perl -i)
        - Write files:    Use write_file   (NOT echo >, tee, or cat <<EOF)
        - List files:     Use ls tool      (NOT ls command in bash)

        The execute tool is for commands that have NO dedicated tool equivalent:
        git, python, pytest, pip, npm, make, docker, curl (POST), etc.

        Before executing:
        1. Verify target directory exists (use ls tool first if unsure)
        2. Always quote file paths with spaces: cd "/path with spaces/"
        3. Use absolute paths; avoid cd when possible

        Usage notes:
        - Commands timeout after 120 seconds by default
        - You may specify a custom timeout up to 600 seconds (10 min) for long-running commands
        - Use '&&' to chain dependent commands; use ';' for independent commands
        - DO NOT use newlines in commands (newlines ok inside quoted strings)
        - For Python code, the tool auto-converts `python3 -c "..."` to heredoc format
        - When issuing multiple independent commands, make multiple execute calls in parallel

        Git safety:
        - Prefer creating new commits over amending existing ones
        - Before destructive operations (git reset --hard, git push --force),
          consider safer alternatives and check with the user first
        - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign)
          unless the user explicitly requests it

        Good examples:
            - execute(command="python3 /path/to/script.py")
            - execute(command="pytest /path/to/tests/ -v --tb=short")
            - execute(command="git status")
            - execute(command="npm install && npm test", timeout=300)

        Bad examples (use dedicated tools instead):
            - execute(command="find . -name '*.py'")   → use glob(pattern="**/*.py")
            - execute(command="grep -r 'TODO' .")      → use grep(pattern="TODO")
            - execute(command="cat file.txt")           → use read_file(file_path="file.txt")
            - execute(command="sed -i 's/old/new/' f")  → use edit_file(...)

        Args:
            command: shell command to execute
            timeout: optional timeout in seconds (default 120, max 600)

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        # Apply timeout: use provided value, clamped to max
        effective_timeout = self._timeout
        if timeout is not None:
            effective_timeout = min(max(1, timeout), self._max_timeout)

        # Use ShellTool's syntax fixers (python -c → heredoc conversion, null/true/false fix)
        command = self._shell._convert_python_c_to_heredoc(command)

        # Sandbox: check blocked commands (best-effort, not a true security sandbox)
        if self._sandbox_config and self._sandbox_config.enabled:
            cmd_lower = command.lower().strip()
            for blocked in self._sandbox_config.blocked_commands:
                # Use regex word boundary to reduce false positives (e.g. "rm" in "format")
                # while still catching the actual dangerous patterns
                pattern = re.escape(blocked.lower())
                if re.search(r'(?:^|[\s;|&])' + pattern, cmd_lower):
                    logger.warning(f"Sandbox: blocked command: {command[:100]}")
                    raise PermissionError(
                        "Sandbox blocked this command for security reasons."
                    )

            # Sandbox: check allowed_commands whitelist (prefix match on first token)
            # Only enforced when allowed_commands is explicitly set (non-None).
            allowed = self._sandbox_config.allowed_commands
            if allowed is not None:
                # Extract the first token (bare executable name, strip path prefix)
                first_token = cmd_lower.split()[0] if cmd_lower.split() else ""
                # Normalize: strip leading path (e.g. "/usr/bin/python3" → "python3")
                first_token_base = os.path.basename(first_token)
                if not any(
                    first_token_base == a.lower() or first_token_base.startswith(a.lower())
                    for a in allowed
                ):
                    logger.warning(
                        f"Sandbox: command '{first_token_base}' not in allowed_commands "
                        f"{allowed}: {command[:100]}"
                    )
                    raise PermissionError(
                        f"Sandbox blocked this command — '{first_token_base}' is not "
                        f"in the allowed_commands list: {allowed}"
                    )

        # Safety: check dangerous command patterns (always active, independent of sandbox)
        safety = check_command_safety(command)
        if safety["action"] == "block":
            logger.warning(f"Safety blocked command: {safety['reason']} — {command[:100]}")
            raise PermissionError(f"{safety['reason']}. Use a safer alternative.")
        if safety["action"] == "warn":
            logger.info(f"Safety warning: {safety['reason']} — {command[:100]}")

        logger.debug(f"Executing command: {command}")
        cwd = str(self._work_dir) if self._work_dir else None
        proc = None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            # Graceful termination: SIGTERM first, then SIGKILL
            if proc is not None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                except ProcessLookupError:
                    pass
            logger.warning(f"Command timed out after {effective_timeout}s: {command}")
            raise TimeoutError(
                f"Command timed out after {effective_timeout} seconds"
            )

        # Combine stdout and stderr
        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")

        output = "\n".join(output_parts).strip()

        # Truncate if too long — 40% head + 60% tail strategy
        # (head preserves early context/errors, tail preserves final results)
        if len(output) > self._max_output_length:
            head_chars = int(self._max_output_length * 0.4)
            tail_chars = self._max_output_length - head_chars
            omitted = len(output) - head_chars - tail_chars
            output = (
                output[:head_chars]
                + f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted"
                  f" out of {len(output)} total] ...\n\n"
                + output[-tail_chars:]
            )

        # Add exit code info with semantic interpretation
        if proc.returncode and proc.returncode != 0:
            hint = _interpret_exit_code(command, proc.returncode)
            exit_line = f"\n\n[Exit code: {proc.returncode}]"
            if hint:
                exit_line += f"\n(Note: {hint})"
            output = f"{output}{exit_line}"

        logger.debug(f"Command exit code: {proc.returncode}")
        if not output:
            output = f"Command executed successfully (exit code: {proc.returncode})"

        # Detect language-level errors in stderr (NameError, SyntaxError, ...)
        # so the LLM doesn't mistake a malformed Python literal (e.g. `null`
        # leaking from a JSON template into Python source) for a logic bug
        # in the script under test. Append a short structured hint that
        # nudges it to inspect the source rather than re-running blindly.
        py_hint = _detect_python_error_hint(output)
        if py_hint:
            output = f"{output}\n\n[Heuristic: {py_hint}]"

        # Redact sensitive text only when the operator opts in. Default is
        # off because rewriting tool output corrupts byte-exact round-trips
        # for downstream edit_file calls.
        if redact_tool_outputs_enabled():
            output = redact_sensitive_text(output)

        # A non-zero exit code that is NOT covered by _interpret_exit_code
        # (i.e. no benign-hint returned) is a real failure — raise so the
        # runtime records it via function_call.error, keeping a single source
        # of truth for error state.
        if (
            proc.returncode
            and proc.returncode != 0
            and _interpret_exit_code(command, proc.returncode) is None
        ):
            raise RuntimeError(
                f"Command exited with code {proc.returncode}.\n{output}"
            )

        return output


# BuiltinWebSearchTool / BuiltinFetchUrlTool now live in
# ``agentica.tools.builtin.web_tools`` and are re-exported here for backwards
# compatibility. Keep this module as the stable legacy import path while the
# canonical implementation migrates into focused builtin modules.


# BuiltinTodoTool / BuiltinMemoryTool now live in
# ``agentica.tools.builtin.task_state_tools`` and are re-exported here for
# backwards compatibility. Keep ``buildin_tools.py`` as the stable legacy import
# path while the canonical implementations migrate into focused builtin modules.


def get_builtin_tools(
        work_dir: Optional[str] = None,
        include_file_tools: bool = True,
        include_execute: bool = True,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        include_todos: bool = True,
        include_task: bool = True,
        include_skills: bool = False,
        include_user_input: bool = False,
        task_model: Optional["Model"] = None,
        custom_skill_dirs: Optional[List[str]] = None,
        user_input_callback=None,
        sandbox_config=None,
        enable_diagnostics: bool = False,
        diagnostics_servers: Optional[List[str]] = None,
        diagnostics_errors_only: bool = True,
    ) -> List[Tool]:
    """
    Get the list of built-in tools for Agent.

    Args:
        work_dir: Work directory for file operations
        include_file_tools: Whether to include file tools (ls, read_file, write_file, edit_file, glob, grep)
        include_execute: Whether to include code execution tool
        include_web_search: Whether to include web search tool
        include_fetch_url: Whether to include URL fetching tool
        include_todos: Whether to include task management tools
        include_task: Whether to include subagent task tool
        include_skills: Whether to include skill tool for executing skills (default: False)
        include_user_input: Whether to include user input tool for human-in-the-loop (default: False)
        task_model: Optional model override for subagents spawned by the
            ``task`` tool. When ``None`` the parent agent's model is cloned.
        custom_skill_dirs: Custom skill directories to load (optional)
        user_input_callback: Custom callback for user input tool (optional)
        sandbox_config: SandboxConfig instance for security isolation (optional)
        enable_diagnostics: When True, start an LSP diagnostics checker and attach
            it to the file tool so write/edit results report newly-introduced
            type/import/syntax errors. Requires a language server (e.g. pyright)
            on PATH; degrades to a no-op if none is available. Default False.
        diagnostics_servers: LSP server names to use (default ["pyright"]).
        diagnostics_errors_only: When True (default), only severity "error"
            diagnostics are surfaced to the model.

    Returns:
        List of tools
    """
    tools = []

    if include_file_tools:
        diagnostics_checker = None
        if enable_diagnostics:
            from agentica.lsp_diagnostics import LspDiagnosticsChecker
            checker = LspDiagnosticsChecker(
                work_dir=work_dir,
                servers=diagnostics_servers,
                errors_only=diagnostics_errors_only,
            )
            # Only attach if a server actually started; otherwise stay a no-op.
            diagnostics_checker = checker if checker.available() else None
        tools.append(BuiltinFileTool(
            work_dir=work_dir,
            sandbox_config=sandbox_config,
            diagnostics_checker=diagnostics_checker,
        ))

    if include_execute:
        tools.append(BuiltinExecuteTool(work_dir=work_dir, sandbox_config=sandbox_config))

    if include_web_search:
        tools.append(BuiltinWebSearchTool())

    if include_fetch_url:
        tools.append(BuiltinFetchUrlTool())

    if include_todos:
        tools.append(BuiltinTodoTool())

    if include_task:
        tools.append(BuiltinTaskTool(model_override=task_model))

    if include_skills:
        from agentica.tools.skill_tool import SkillTool
        tools.append(SkillTool(custom_skill_dirs=custom_skill_dirs, auto_load=True))

    if include_user_input:
        from agentica.tools.user_input_tool import AskUserQuestionTool
        tools.append(AskUserQuestionTool(input_callback=user_input_callback))

    return tools


if __name__ == '__main__':
    # Test file tool
    file_tool = BuiltinFileTool()
    print("=== ls test ===")
    print(file_tool.ls("."))

    print("\n=== glob test ===")
    print(file_tool.glob("*.py", "."))

    # Test search tool
    search_tool = BuiltinWebSearchTool()
    print("\n=== web_search test ===")
    print(search_tool.web_search("Python programming", max_results=2))

    # Test todo tool
    todo_tool = BuiltinTodoTool()
    print("\n=== write_todos test ===")
    print(todo_tool.write_todos([
        {"content": "Task 1", "status": "in_progress"},
        {"content": "Task 2", "status": "pending"},
    ]))
