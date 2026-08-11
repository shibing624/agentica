# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in file tools (ls/read_file/write_file/edit_file/apply_patch/glob/grep/undo)
"""
import asyncio
import difflib
import json
import os
import re
import shutil
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, List, Dict, Literal, Tuple

import aiofiles

from agentica.tools.base import Tool
from agentica.tools.helpers import ToolDisplayOutput, file_change_meta, file_display_meta
from agentica.tools.patch_tool import apply_diff, parse_patch_envelope
from agentica.utils.async_utils import terminate_subprocess
from agentica.utils.log import logger
from agentica.utils.string import truncate_if_too_long

# grep self-imposed timeout (seconds). Covers both the rg subprocess and the
# pure-Python fallback so a missing rg can't walk huge trees for the outer
# 120s executor timeout. grep is marked manages_own_timeout=True. The default
# 3s fails fast on NFS / huge trees; the LLM may override via the `timeout` arg.
#
# Default timeout for the built-in ``glob`` tool, in seconds. Same rationale
# as ``_GREP_TIMEOUT``: caller can override without upper cap, but we must
# always be bounded — a bare ``**/pattern`` walk over ``$HOME`` or a stuck
# network mount will otherwise hang for the full outer harness limit (120s).
_GLOB_TIMEOUT = 3

# Default timeout (in seconds) for the built-in ``grep`` tool. Callers may
# override this per-invocation by passing ``timeout``, and the override is
# used as-is (no upper cap — the caller decides). A timeout must always be
# set: on a bad disk / network mount, or with a backtracking regex (e.g.
# nested .*.*) over a large file, grep can hang or go exponential — keep the
# default short so the model scopes the path instead of waiting.
_GREP_TIMEOUT = 3

_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

# Sensitive system paths that file tools should refuse to write to
_SENSITIVE_PATH_PREFIXES = ("/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/", "/private/var/run/")


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
    Built-in file system tool providing ls, read_file, write_file, edit_file,
    apply_patch, glob, and grep functions.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            max_read_lines: int = 500,
            max_line_length: int = 2000,
            sandbox_config=None,
            diagnostics_checker=None,
            consent_callback=None,
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
            consent_callback: Optional ``(prompt, options) -> str`` callback used by
                ``request_path_access`` to ask the user for permission to read or
                write a path that is otherwise blocked (outside the sandboxed
                ``writable_dirs``, or a sensitive system/credentials path). Same
                callback the CLI wires into ``ask_user_question``. When None,
                ``request_path_access`` fails closed (no human to ask).
        """
        super().__init__(name="builtin_file_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.max_read_lines = max_read_lines
        self.max_line_length = max_line_length
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._sandbox_config = sandbox_config
        self.diagnostics_checker = diagnostics_checker
        self._consent_callback = consent_callback
        # Paths the user has explicitly approved via request_path_access,
        # overriding sandbox writable_dirs / blocked_paths / sensitive-path
        # guards for the rest of this session. Not persisted across restarts.
        self._escalated_paths: List[str] = []

        # File snapshots for workspace rollback: {abs_path: [content_before_1, ...]}
        # Stores previous file content before each write/edit, supporting undo.
        self._file_snapshots: Dict[str, List[str]] = {}

        # Register all file operation functions.
        # Read-only tools are concurrency_safe (can run in parallel with each other).
        # Write tools stay serialised.
        self.register(self.ls, concurrency_safe=True, is_read_only=True)
        self.register(self.read_file, concurrency_safe=True, is_read_only=True)
        self.register(self.write_file, is_destructive=True)
        self.register(self.edit_file, is_destructive=True)
        self.register(self.apply_patch, is_destructive=True)
        self.register(self.request_path_access, is_destructive=False)
        self.register(self.glob, concurrency_safe=True, is_read_only=True)
        self.register(self.grep, concurrency_safe=True, is_read_only=True)
        # glob and grep enforce their own timeouts on both fast (rg / native
        # pathlib.glob) and fallback paths (default 3s, LLM-tunable with no
        # upper cap), so skip the outer 120s executor wrapper — otherwise a
        # bare ``**/*.log`` walk from ``$HOME`` or a stuck network mount used
        # to hang for the full 120s.
        self.functions["glob"].manages_own_timeout = True
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

    def _lexical_abs_path(self, path: Path) -> Path:
        """Return an absolute path without requiring the target to exist."""
        return Path(os.path.abspath(path.expanduser()))

    def _nearest_existing_parent(self, path: Path) -> Optional[Path]:
        """Find the nearest existing parent for a possibly missing path."""
        current = self._lexical_abs_path(path)
        if current.exists():
            return current if current.is_dir() else current.parent
        for parent in current.parents:
            if parent.exists():
                return parent
        return None

    def _missing_path_error(self, kind: str, raw_path: str, path: Path) -> str:
        """Build a missing-path error that exposes real path state."""
        resolved = self._lexical_abs_path(path)
        nearest_parent = self._nearest_existing_parent(path)

        lines = [
            f"{kind} not found: {raw_path}",
            f"Resolved path: {resolved}",
        ]
        if nearest_parent is not None:
            lines.append(f"Nearest existing parent: {nearest_parent}")
        else:
            lines.append("Nearest existing parent: <none>")

        lines.append(
            "Next step: use ls/glob/grep from the nearest existing parent; "
            "do not retry speculative absolute paths."
        )
        return "\n".join(lines)

    def _result_path(self, raw_path: str) -> str:
        """Return the lexical tool path relative to the configured work directory."""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.work_dir / path
        lexical_path = Path(os.path.abspath(path))
        work_roots = (
            Path(os.path.abspath(self.work_dir.expanduser())),
            self.work_dir.expanduser().resolve(),
        )
        for work_root in work_roots:
            try:
                return lexical_path.relative_to(work_root).as_posix()
            except ValueError:
                continue
        return lexical_path.as_posix()

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

    def _is_escalated(self, resolved: str) -> bool:
        """Whether `resolved` was previously approved via request_path_access."""
        return any(resolved.startswith(d) for d in self._escalated_paths)

    def _component_blocked(self, resolved: str) -> bool:
        """Whether `resolved` has a path component matching sandbox_config.blocked_paths
        (e.g. `.ssh`, `.env`) and hasn't already been approved via request_path_access."""
        if self._sandbox_config is None or not self._sandbox_config.enabled:
            return False
        if self._is_escalated(resolved):
            return False
        resolved_parts = set(Path(resolved).parts)
        return any(blocked in resolved_parts for blocked in self._sandbox_config.blocked_paths)

    def _validate_path(self, path: str) -> str:
        """Validate path against sandbox restrictions and blocked device files.

        Always checks:
        - Path must not resolve to a known device file (/dev/zero, etc.) — this
          is a crash/hang guard, not a permission decision, so it is NOT
          escalatable via request_path_access.

        When sandbox is enabled, also checks:
        - Path components do not match any blocked_paths entries, unless the
          path was already approved via request_path_access.
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
        if self._component_blocked(str(resolved)):
            matched = next(
                b for b in self._sandbox_config.blocked_paths if b in set(resolved.parts)
            )
            raise PermissionError(
                f"Sandbox: access to path containing '{matched}' is blocked. "
                f"Call request_path_access(path=<the path>, reason=<why>) to ask the "
                f"user for permission, then retry."
            )
        return path

    def _is_write_allowed(self, resolved: str) -> bool:
        """Whether `resolved` (an absolute path string) is inside a writable_dir,
        work_dir, or a previously-escalated path."""
        if self._is_escalated(resolved):
            return True
        for wd in self._sandbox_config.writable_dirs:
            wd_resolved = str(Path(wd).expanduser().resolve())
            if resolved.startswith(wd_resolved):
                return True
        return resolved.startswith(str(self.work_dir.resolve()))

    def _sensitive_write_guard(self, filepath: str) -> Optional[str]:
        """Return an error message if `filepath` targets a sensitive system/credentials
        location and hasn't already been approved via request_path_access.

        Wraps the stateless `_check_sensitive_write_path()` with per-instance
        escalation so a user-approved path bypasses the guard.
        """
        try:
            resolved = str(Path(filepath).expanduser().resolve())
        except (OSError, ValueError):
            resolved = filepath
        if self._is_escalated(resolved):
            return None
        err = _check_sensitive_write_path(filepath)
        if err is None:
            return None
        return err + (
            f"\nCall request_path_access(path={filepath!r}, reason=<why>) to ask "
            f"the user for permission, then retry."
        )

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
        if self._sandbox_config.writable_dirs and not self._is_write_allowed(resolved):
            raise PermissionError(
                f"Sandbox: write to '{path}' is not allowed in the current permission mode. "
                f"Writable dirs: {self._sandbox_config.writable_dirs}. "
                f"Call request_path_access(path=<the directory>, reason=<why>) to ask the "
                f"user for permission, then retry the write."
            )
        return path

    def request_path_access(self, path: str, reason: str) -> str:
        """Ask the user for permission to read or write a path that is otherwise blocked.

        Two independent restrictions can block a path, and both are escalatable
        with this tool:
        1. Sandbox scoping (the "auto"/"ask" permission tiers): writes are
           restricted to work_dir, reads to `.ssh`/`.env`/etc. within any path
           are blocked.
        2. Sensitive system/credentials paths (`/etc/`, `~/.ssh/`, `~/.aws/credentials`,
           etc.): writes there are refused by default in ANY permission mode,
           including "allow-all", since the model should not silently touch
           credentials or system files.

        Use this tool whenever a task legitimately requires touching such a
        path. It asks the user for a one-time yes/no confirmation; on approval,
        the path is whitelisted for the rest of this session so the original
        read/write can be retried immediately afterward.

        Args:
            path: The file or directory path you need access to.
            reason: A short, clear explanation of why this access is needed.

        Returns:
            JSON string with "granted": true/false and a "message" explaining
            the outcome. When granted, retry the original read/write immediately.
        """
        resolved = str(self._resolve_path(path).resolve())
        is_blocked = (
            self._component_blocked(resolved)
            or self._sensitive_write_guard(path) is not None
            or (
                self._sandbox_config is not None
                and self._sandbox_config.enabled
                and not self._is_write_allowed(resolved)
            )
        )
        if not is_blocked:
            return json.dumps({"granted": True, "message": f"'{path}' is already accessible."})
        if self._consent_callback is None:
            return json.dumps({
                "granted": False,
                "message": (
                    "No interactive user is available to grant this request in the current "
                    "session. Access to this path stays blocked."
                ),
            })
        prompt = (
            f"The agent wants access to a restricted path:\n"
            f"  Path: {resolved}\n"
            f"  Reason: {reason}\n\n"
            f"Allow access to this path for the rest of the session?"
        )
        try:
            answer = self._consent_callback(prompt, ["yes", "no"])
        except Exception as e:
            return json.dumps({"granted": False, "message": f"Failed to get user consent: {e}"})
        granted = str(answer).strip().lower() in ("yes", "y")
        if granted:
            grant_dir = resolved if Path(resolved).is_dir() else str(Path(resolved).parent)
            self._escalated_paths.append(grant_dir)
            return json.dumps({"granted": True, "message": f"User approved. '{grant_dir}' is now accessible for this session."})
        return json.dumps({"granted": False, "message": "User denied access to this path."})

    async def ls(self, directory: str = ".") -> str:
        """List the immediate files and subdirectories of a directory.

        Non-recursive. Use ``glob`` or ``grep`` for tree-wide search.
        Returns JSON objects with ``name``, ``path``, and ``type``.
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
    # read_file content cache: each entry is inherently bounded by
    # MAX_FILE_SIZE_BYTES; this caps the entry count (LRU eviction).
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
        """Reads a file from the filesystem. Reading a file that does not exist returns an error.

        Usage:
        - file_path may be absolute, relative to the working directory, or `~`-prefixed
        - Before calling, ground file_path in the user's exact input or a prior
          ls/glob/grep/write_file/apply_patch result. If you only know a module,
          class, or function name, locate the file first; do not use speculative
          absolute paths from memory or stale summaries.
        - Reads up to `limit` lines (default 500) starting from `offset` (0-based); use offset+limit to page through large files
        - Any line longer than 2000 characters is truncated
        - Results are returned with line-number prefixes (metadata only)
        - An empty file returns a system reminder in place of contents
        - Prefer one larger read over many small slices

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
            raise FileNotFoundError(self._missing_path_error("File", file_path, path))
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

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
        if total_lines == 0:
            result = (
                "<system-reminder>\n"
                f"File exists but is empty: {path.resolve()} (0 bytes, 0 lines).\n"
                "Use write_file to add content.\n"
                "</system-reminder>"
            )
        elif actual_end < total_lines:
            result += f"\n\n[Showing lines {offset + 1}-{actual_end} of {total_lines} total lines]"

        logger.debug(f"Read file {file_path}: lines {offset + 1}-{actual_end}, total {total_lines} lines")
        return result

    async def write_file(self, file_path: str, content: str) -> str:
        """Writes content to a file in the filesystem.

        Usage:
        - If this is an existing file, you MUST use read_file first to read the file's contents.
          This tool will create a new file or OVERWRITE the existing file entirely.
        - Prefer apply_patch for modifying existing files with context.
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
        sensitive_err = self._sensitive_write_guard(str(path))
        if sensitive_err:
            raise PermissionError(sensitive_err)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        action = "Created" if not path.exists() else "Updated"
        old_content = None

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
        logger.debug(f"{action} file: {absolute_path}, file content length: {len(content)} characters")
        diag_text = await self._diagnostics_after(path)
        suffix = f"\n\n{diag_text}" if diag_text else ""
        return ToolDisplayOutput(
            f"{action} file, absolute path: {absolute_path}{suffix}",
            file_display_meta([
                file_change_meta(
                    absolute_path,
                    "add" if action == "Created" else "update",
                    old_content,
                    content,
                )
            ]),
        )

    async def apply_patch(self, patch: str) -> str:
        """Apply one context patch across one or more text files.

        You MUST call read_file before every Update or Delete in the patch.
        Read the relevant current regions immediately before constructing the
        patch; never guess context from memory or stale output. Add File
        operations do not require a prior read.

        Every Update/Delete path must be grounded in the user's exact input or a
        prior tool result, preferably the read_file call you just used for that
        file. Do not construct long absolute paths from module names or stale
        summaries or speculative absolute paths.

        Use this when a change spans multiple files or needs multiple contextual
        hunks. Keep using edit_file only for one simple literal replacement.

        The patch must use exactly one envelope with one or more Add, Update, or
        Delete sections. All paths and hunks are validated against current file
        contents before any file is changed. If validation fails, no files are
        written. This is a JSON function tool, so pass the entire patch as the
        ``patch`` string.

        Keep each ``@@`` hunk's unchanged context short, stable, and unique.
        Read or re-read the relevant regions immediately before building the
        patch. A failed preflight reports every file and every context hunk that
        could be checked, showing the expected context and the actual current
        content, so regenerate those hunks from the exact current text
        instead of retrying the same patch.

        Example:
            *** Begin Patch
            *** Update File: app.py
            @@
             DEBUG = False
            -TIMEOUT = 10
            +TIMEOUT = 30
            *** Add File: tests/test_timeout.py
            +def test_timeout():
            +    assert True
            *** End Patch

        Args:
            patch: Complete multi-file patch envelope.

        Returns:
            Summary of files and line counts actually changed.
        """
        operations = parse_patch_envelope(patch)

        resolved = []
        seen_paths = set()
        for operation in operations:
            self._validate_write_path(operation.path)
            path = self._resolve_path(operation.path).resolve()
            path_key = str(path)
            if path_key in seen_paths:
                raise ValueError(
                    f"Patch paths resolve to the same file more than once: {operation.path!r}."
                )
            seen_paths.add(path_key)

            sensitive_err = self._sensitive_write_guard(path_key)
            if sensitive_err:
                raise PermissionError(sensitive_err)
            resolved.append((operation, path, path_key))

        # Acquire every target lock in stable order so no same-tool edit can
        # change a file between preflight and commit.
        lock_by_path = {
            path_key: self._get_file_lock(path_key)
            for _, _, path_key in resolved
        }
        async with AsyncExitStack() as stack:
            for path_key in sorted(lock_by_path):
                await stack.enter_async_context(lock_by_path[path_key])

            prepared = []
            preflight_errors = []
            for operation, path, path_key in resolved:
                result_path = self._result_path(operation.path)
                try:
                    if operation.action == "add":
                        if path.exists():
                            raise FileExistsError("Cannot add a file that already exists.")
                        old_content = None
                        new_content = apply_diff("", operation.diff, mode="create")
                    else:
                        if not path.exists():
                            raise FileNotFoundError("File not found.")
                        if not path.is_file():
                            raise IsADirectoryError("Path is not a file.")
                        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
                            old_content = await handle.read()
                        if operation.action == "update":
                            new_content = apply_diff(old_content, operation.diff, mode="default")
                            if new_content == old_content:
                                raise ValueError("Update does not change the file.")
                        else:
                            new_content = None
                except (FileExistsError, FileNotFoundError, IsADirectoryError, ValueError) as exc:
                    preflight_errors.append((result_path, exc))
                    continue

                added, removed = self._content_change_counts(old_content, new_content)
                prepared.append(
                    (operation, path, path_key, old_content, new_content, result_path, added, removed)
                )

            if preflight_errors:
                file_noun = "file" if len(preflight_errors) == 1 else "files"
                error_lines = [
                    f"Patch preflight failed for {len(preflight_errors)} {file_noun}; "
                    "no files were changed."
                ]
                for result_path, error in preflight_errors:
                    error_lines.append(f"- {result_path}:")
                    error_lines.extend(f"  {line}" for line in str(error).splitlines())
                error_lines.extend(("", self._patch_read_hint()))
                error_message = "\n".join(error_lines)
                if len(preflight_errors) == 1:
                    original_error = preflight_errors[0][1]
                    if isinstance(
                        original_error,
                        (FileExistsError, FileNotFoundError, IsADirectoryError),
                    ):
                        raise type(original_error)(error_message) from original_error
                raise ValueError(error_message) from preflight_errors[0][1]

            # Diagnostics and undo snapshots are captured only after every file
            # and hunk has passed preflight.
            for operation, path, path_key, old_content, *_ in prepared:
                await self._diagnostics_snapshot(path)
                if old_content is not None:
                    self._file_snapshots.setdefault(path_key, []).append(old_content)

            applied = []
            try:
                for operation, path, _, _, new_content, result_path, _, _ in prepared:
                    new_text = new_content or ""
                    if operation.action == "delete":
                        path.unlink()
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
                        try:
                            os.close(tmp_fd)
                            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as handle:
                                await handle.write(new_text)
                            os.replace(tmp_path, str(path))
                        except Exception:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                            raise
                    applied.append(result_path)
            except Exception as exc:
                applied_text = ", ".join(applied) if applied else "none"
                raise OSError(
                    f"Patch commit failed after changing {len(applied)} file(s) "
                    f"({applied_text}). The workspace may be partially modified: {exc}"
                ) from exc

        action_codes = {"add": "A", "update": "M", "delete": "D"}
        added_lines = sum(item[-2] for item in prepared)
        removed_lines = sum(item[-1] for item in prepared)
        file_noun = "file" if len(prepared) == 1 else "files"
        result_lines = [
            f"Successfully applied patch to {len(prepared)} {file_noun} "
            f"(+{added_lines} -{removed_lines}):",
            *(
                f"{action_codes[operation.action]} {result_path} (+{added} -{removed})"
                for operation, _, _, _, _, result_path, added, removed in prepared
            ),
        ]
        diagnostics = []
        for operation, path, _, _, _, result_path, _, _ in prepared:
            if operation.action != "delete":
                diag_text = await self._diagnostics_after(path)
                if diag_text:
                    diagnostics.append(f"{result_path}:\n{diag_text}")
        if diagnostics:
            result_lines.extend(("", "\n\n".join(diagnostics)))
        return ToolDisplayOutput(
            "\n".join(result_lines),
            file_display_meta([
                file_change_meta(result_path, operation.action, old_content, new_content)
                for operation, _, _, old_content, new_content, result_path, _, _ in prepared
            ]),
        )

    async def edit_file(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Replace a specific string in a file.

        Read the relevant region with read_file before editing so old_string
        matches the current content exactly. A String not found failure means
        the region must be re-read before retrying.

        file_path and old_string must both be grounded in recent context: use a
        path returned by ls/glob/grep/read_file/write_file/apply_patch, and copy
        old_string from the latest read_file output for that exact file. Do not
        use speculative absolute paths.

        Uses literal string matching (NOT regex). Prefer apply_patch for code
        edits, multi-hunk edits, and anything that needs surrounding context;
        use this tool only for one short, unique literal replacement.

        When editing text from read_file output, ensure you preserve the exact indentation
        (tabs/spaces) as it appears in the file. The line number prefix in read_file output
        is metadata only — never include it in old_string or new_string.

        The edit will FAIL if old_string is not unique in the file. Either provide a
        larger string with more surrounding context to make it unique, or use
        replace_all=True to change every instance.

        If you call `edit_file` multiple times on the same file in parallel,
        they will be serialized automatically to avoid race conditions.
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

        if not path.exists():
            raise FileNotFoundError(self._missing_path_error("File", file_path, path))
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        sensitive_err = self._sensitive_write_guard(str(path))
        if sensitive_err:
            raise PermissionError(sensitive_err)

        abs_path = str(path.resolve())

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
                raise ValueError(
                    self._build_edit_not_found_error(result["error"], content, old_string)
                )

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
        diag_text = await self._diagnostics_after(path)
        parts = [f"Successfully replaced {result['count']} occurrence(s) in '{file_path}'"]
        if diag_text:
            parts.append(diag_text)
        return ToolDisplayOutput(
            "\n\n".join(parts),
            file_display_meta([
                file_change_meta(abs_path, "update", content, result["new_content"])
            ]),
        )

    @staticmethod
    def _edit_read_hint() -> str:
        """Return the recovery action for an exact-string edit failure."""
        return (
            "Read or re-read the relevant region with read_file, copy the exact "
            "current text into old_string, then retry the edit."
        )

    @classmethod
    def _build_edit_not_found_error(
            cls, error: str, _content: str, _old_string: str
    ) -> str:
        """Assemble a stateless exact-string edit failure."""
        return error + "\n\n" + cls._edit_read_hint()

    @staticmethod
    def _patch_read_hint() -> str:
        """Return the recovery action for patch preflight failures."""
        return (
            "Read or re-read each failed region with read_file, rebuild those hunks "
            "from the exact current text with short unique context, then retry the patch."
        )

    @staticmethod
    def _content_change_counts(
            old_content: Optional[str], new_content: Optional[str]
    ) -> Tuple[int, int]:
        """Count added and removed content lines for one prepared file change."""
        old_text = old_content or ""
        new_text = new_content or ""
        added = 0
        removed = 0
        for line in difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(), lineterm=""
        ):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return added, removed

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

        Tries exact match first. If that fails, retries after normalizing
        curly/typographic quotes in both strings to ASCII equivalents. The
        normalized text is used only for locating matches; replacements are
        always sliced from the original content.

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

        # Quote-normalization fallback: normalization is length-preserving, so
        # positions found here map directly back to the original content.
        if not matches:
            norm_content = self._normalize_quotes(content)
            norm_old = self._normalize_quotes(old_string)
            if norm_content != content or norm_old != old_string:
                start = 0
                while True:
                    idx = norm_content.find(norm_old, start)
                    if idx == -1:
                        break
                    matches.append(idx)
                    start = idx + len(norm_old)

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
                line_num = content[:idx].count('\n') + 1
                # Get surrounding context (up to 50 chars around the match)
                context_start = max(0, idx - 20)
                context_end = min(len(content), idx + len(old_string) + 30)
                context = content[context_start:context_end].replace('\n', '\\n')
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

        # Build the result from original content so normalization cannot alter
        # unrelated punctuation elsewhere in the file.
        selected_matches = matches if replace_all else matches[:1]
        parts = []
        cursor = 0
        for idx in selected_matches:
            parts.extend((content[cursor:idx], new_string))
            cursor = idx + len(old_string)
        parts.append(content[cursor:])
        new_content = ''.join(parts)
        count = len(selected_matches)

        return {
            "success": True,
            "new_content": new_content,
            "count": count,
            "error": None,
        }

    async def glob(self, pattern: str, path: str = ".", timeout: Optional[int] = None) -> str:
        """Find files by name pattern, anywhere in a tree.

        `*` matches within one directory only; `**` recurses through every level,
        so "*.py" and "**/*.py" give very different results. Noise directories
        (.git, __pycache__, node_modules, .venv, ...) are always excluded.

        Use this to ground later read_file/grep/apply_patch calls. If you are
        unsure where a file lives, search from "." or another known existing
        directory; do not pass a speculative absolute path as `path`.

        Args:
            pattern: Glob pattern, e.g. "*.py", "**/*.md", "src/?*.js". May be
                absolute ("/home/user/*.py") or relative to `path`.
            path: Directory to search from (default: ".")
            timeout: Search timeout in seconds (default 3, no upper cap). Raise it
                only for a legitimately huge tree; a bare `**/...` walk from a home
                directory or a stuck network mount can otherwise run for minutes.

        Returns:
            JSON list of sorted absolute file paths, empty when nothing matches.
        """
        self._validate_path(path)
        base_path = self._resolve_path(path)

        if not base_path.exists():
            raise FileNotFoundError(self._missing_path_error("Directory", path, base_path))

        effective_timeout = timeout if timeout is not None else _GLOB_TIMEOUT

        # Run glob in executor to avoid blocking on large directory trees
        def _glob_sync():
            matches = list(base_path.glob(pattern))
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            return sorted(
                str(m) for m in matches
                if not set(m.parts).intersection(ignore_dirs)
            )

        loop = asyncio.get_event_loop()
        try:
            filtered = await asyncio.wait_for(
                loop.run_in_executor(None, _glob_sync),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"glob timed out after {effective_timeout} seconds "
                f"(pattern={pattern!r}, path={path!r})"
            )

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
            timeout: Optional[int] = None,
    ) -> str:
        """Search file contents for a regex pattern across a whole tree.

        Default output is matching lines as `file:line_number:content`. Switch to
        "files_with_matches" only when a path list is enough, or "count" when only
        totals matter — both drop the code itself and usually force a follow-up
        read_file.

        Ground the `path` argument before calling. If you only know a module,
        class, function, or filename fragment, search from "." or a known existing
        directory and let grep find the path; do not pass a speculative absolute
        path from memory or stale summaries.

        Args:
            pattern: Text/regex to search for
            path: File or directory to search (default: ".")
            include: File glob filter, e.g. "*.py", "*.{js,ts}"
            output_mode: "content" (default), "files_with_matches", or "count".
                Pass a plain string, never a dict.
            case_insensitive: Ignore case when matching (default: False)
            multiline: Let `.` match newlines so a pattern can span lines (default: False)
            context_lines: Lines to show before and after each match ("content" mode only)
            before_context: Lines to show before each match ("content" mode only)
            after_context: Lines to show after each match ("content" mode only)
            limit: Maximum results to return (default: 100)
            fixed_strings: Treat pattern as literal text, not regex (default: False)
            timeout: Search timeout in seconds (default 3, no upper cap). Raise it
                only for a legitimately huge tree; a backtracking pattern such as
                nested `.*.*` can otherwise run for an unbounded time.

        Returns:
            Search results as formatted string
        """
        # Resolve and validate path
        self._validate_path(path)
        base_path = self._resolve_path(path)
        if not base_path.exists():
            raise FileNotFoundError(self._missing_path_error("Path", path, base_path))

        # Effective timeout: the LLM-provided value is used as-is (no upper
        # cap — the caller decides); default _GREP_TIMEOUT when not provided.
        # A timeout must always be set (see _GREP_TIMEOUT comment).
        effective_timeout = timeout if timeout is not None else _GREP_TIMEOUT

        # Check if rg is available
        rg_path = shutil.which("rg")
        if rg_path is None:
            return await self._run_grep_fallback(
                pattern, path, include, output_mode, limit, fixed_strings,
                case_insensitive, effective_timeout,
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

        # rg is normally millisecond-fast; a hard effective_timeout catches hangs
        # (pathological regex, huge binary files, or a stuck network mount).
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            if proc is not None:
                await terminate_subprocess(proc)
            raise TimeoutError(
                f"grep timed out after {effective_timeout} seconds"
            ) from None
        except asyncio.CancelledError:
            if proc is not None:
                await terminate_subprocess(proc)
            raise
        except FileNotFoundError:
            return await self._run_grep_fallback(
                pattern, path, include, output_mode, limit, fixed_strings,
                case_insensitive, effective_timeout,
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
            timeout: int = _GREP_TIMEOUT,
    ) -> str:
        """Run the pure-Python fallback in an executor with a hard timeout.

        The fallback walks the tree in a thread, so on timeout we can only
        drop the result — the thread keeps running — but the tool returns a
        clear timeout error at ``timeout`` instead of hanging to the outer
        120s executor limit.
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._grep_fallback, pattern, path, include,
                    output_mode, limit, fixed_strings, case_insensitive,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"grep timed out after {timeout} seconds")

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
        if base_path.is_file():
            files = [base_path] if not include or base_path.match(include) else []
        elif include:
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

        Each write_file(), edit_file(), and apply_patch()
        update/delete automatically snapshots the file's content before modification.
        This tool restores the most recent snapshot,
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
        sensitive_err = self._sensitive_write_guard(str(path))
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

        remaining = len(snapshots)
        return (
            f"Restored '{file_path}' to previous version ({len(previous)} chars). "
            f"{remaining} more undo(s) available."
        )


# A trailing `&` forks the real work off the shell this tool holds. Under
# background=True that is broken rather than merely untracked: the registry
# would watch a shell that exits at once and announce a completion while the
# work runs on, so it is refused there. In the foreground it still runs — it is
# a standard idiom and callers may genuinely want a raw orphan — but the result
# says what was lost, because the forked child stays in the shell's process
# group and a cancelled or timed-out turn kills it along with the group.
#
# A long leading `sleep` is not work at all, it is a poll that re-blocks the
# very turn backgrounding just freed. The threshold matches the default
# foreground timeout: a shorter sleep is a plausible wait-for-boot retry
# (`sleep 2 && curl ...`), while reaching this one means the caller also had to
# raise `timeout`, which only a deliberate poll does. Matching only a leading
# `sleep` is deliberate: waiting on an external condition belongs in a loop that
# exits on success (`until curl -sf ...; do sleep 5; done`), which is both the
# better form and not a match.
_SELF_DETACHING_COMMAND = re.compile(r"(?<!&)&\s*$")
_LEADING_SLEEP = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)\b")
_MAX_FOREGROUND_SLEEP_SECONDS = 120

# Upper bound on a single `wait` call. The wait returns the instant the command
# exits, so this only caps how long one tool call may hold the turn: the caller
# comes back through the model loop periodically, which is what lets the user
