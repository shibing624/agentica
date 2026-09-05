# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in file tools (read_file/write_file/apply_patch/glob/grep)
"""
import asyncio
import difflib
import json
import os
import re
import shutil
import tempfile
from collections import deque
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import aiofiles

from agentica.tools.base import Tool
from agentica.tools.helpers import ToolDisplayOutput, file_change_meta, file_display_meta
from agentica.tools.patch import PatchContextError, PatchNoChangeError, apply_diff, parse_patch_envelope
from agentica.utils.async_utils import close_subprocess_transport, terminate_subprocess
from agentica.utils.log import logger
from agentica.utils.string import truncate_if_too_long

# grep/glob self-imposed timeout (seconds). Covers rg, pathlib.glob, and the
# pure-Python grep fallback so a missing rg or a stuck network mount cannot
# walk huge trees for the outer 120s executor timeout. Both tools are marked
# manages_own_timeout=True. Not exposed to the model: an unbounded override
# would defeat fail-fast, and a silent clamp is the same class of bug wait()
# already deleted. On timeout the error tells the model to narrow path.
# 20s matches kimi-cli / Claude Code. glob keeps a timeout too: pathlib.glob
# walks the whole tree before returning, so a match cap cannot bound NFS hangs.
_GLOB_TIMEOUT = 20
_GREP_TIMEOUT = 20
# Tail / negative-offset reads scan the file from the start (deque of N lines).
# A multi-GB log must not pin the turn; 20s matches grep/glob. Not exposed.
_READ_TIMEOUT = 20


def _effective_tail(tail) -> Optional[int]:
    """Last-N count from the ``tail`` argument, or None to page from the start.

    ``tail=0`` / omit means "not a tail read" — models send 0 when they want
    the default top-of-file page. A negative value is last ``|N|`` lines
    (same window as ``offset=-N``).
    """
    if tail is None or tail == "":
        return None
    try:
        n = int(tail)
    except (TypeError, ValueError):
        return None
    if n == 0:
        return None
    return abs(n)
_FURTHER_TRUNCATED = "... (further results truncated)"

# Directories every search skips. ``.agentica`` earns its place for a specific
# reason: with ``settings.worktree.root`` pointing inside the repository
# (``.agentica/worktrees``), a full second checkout lives under it, and a bare
# ``glob("**/*.py")`` would then return every file twice — once really, once in
# each worktree. Verified before fixing: the duplicate was returned, and the
# dangerous half of that is not the noise but an edit landing in the copy.
# ``grep`` shells out to ripgrep, which already skips it via .gitignore; the
# pure-Python fallback and ``glob`` walk the tree themselves and do not.
_NOISE_DIRS = frozenset({
    '.git', '.agentica', '__pycache__', 'node_modules', '.venv', 'venv',
    '.idea', '.pytest_cache',
})


def _nested_checkouts(base: "Path") -> tuple:
    """Worktrees of this repository that live under ``base``, from git itself.

    A worktree inside the checkout is a second copy of every file, so searches
    skip it — see ``agentica.worktrees.nested_worktrees`` for why this is asked
    of git instead of matched by name. Never the base itself: a session bound to
    that worktree searches from inside it and must see its own files.
    """
    try:
        from agentica.worktrees import nested_worktrees

        here = os.path.realpath(str(base))
        return tuple(
            path for path in nested_worktrees(str(base))
            if path != here and not here.startswith(path + os.sep)
        )
    except Exception:
        return ()


def _in_noise_dir(
    path: "Path",
    base: "Path",
    noise: Optional[frozenset] = None,
    nested: tuple = (),
) -> bool:
    """Whether ``path`` sits inside a skipped directory *below* ``base``.

    Only the part below the search root counts. A session whose work_dir is
    itself inside one of these names — which is exactly what a worktree under
    ``.agentica/worktrees`` is — must still see its own files; matching against
    the whole absolute path made that session's ``glob`` return nothing at all.
    """
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = path
    if set(relative.parts).intersection(noise if noise is not None else _NOISE_DIRS):
        return True
    if nested:
        resolved = os.path.realpath(str(path))
        return any(resolved.startswith(root + os.sep) for root in nested)
    return False


def _cap_output_lines(output: str, limit: int) -> str:
    """Keep the first ``limit`` lines of a grep payload (global, not per-file)."""
    if not output or limit < 0:
        return output
    lines = output.split("\n")
    if len(lines) <= limit:
        return output
    omitted = len(lines) - limit
    return "\n".join(lines[:limit]) + f"\n... ({omitted} more results truncated)"


async def _collect_rg_output(
    proc: asyncio.subprocess.Process,
    max_lines: Optional[int],
) -> Tuple[bytes, bytes, bool]:
    """Read rg stdout until EOF or ``max_lines``, then reap the process.

    A global cap that only ran after ``communicate()`` still let rg walk the
    whole tree and buffer every match. Stopping the read and killing rg is
    what makes ``limit`` bound I/O, not just the string we return. Reads one
    line past ``max_lines`` so a result set that ends exactly at the cap is
    not reported as truncated; the lookahead line is dropped.
    Returns ``(stdout, stderr, hit_cap)``.
    """
    chunks: List[bytes] = []
    hit_cap = False
    max_chunks = max_lines + 1 if max_lines is not None else None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        chunks.append(line)
        if max_chunks is not None and len(chunks) >= max_chunks:
            hit_cap = True
            break
    stdout = b"".join(chunks)
    if hit_cap:
        stdout = stdout[: sum(len(c) for c in chunks[:max_lines])]
        await terminate_subprocess(proc)
        return stdout, b"", True
    stderr = await proc.stderr.read()
    await proc.wait()
    return stdout, stderr, False

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


def is_sensitive_write_path(filepath: str) -> bool:
    """True if a write to ``filepath`` hits a sensitive system/credentials location."""
    return _check_sensitive_write_path(filepath) is not None


class BuiltinFileTool(Tool):
    """
    Built-in file system tool providing read_file, write_file, apply_patch,
    glob, and grep. Session path grants go through ``grant_path_access``.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            max_read_lines: int = 500,
            max_line_length: int = 2000,
            sandbox_config=None,
            diagnostics_checker=None,
            permission_mode: str = "allow-all",
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
            permission_mode: Current permission tier. An Agent holding this
                tool overwrites it with its own tier at wiring time; this
                argument only matters when the tool is used standalone.
        """
        super().__init__(name="builtin_file_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.max_read_lines = max_read_lines
        self.max_line_length = max_line_length
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._sandbox_config = sandbox_config
        self.diagnostics_checker = diagnostics_checker
        # Paths the user has explicitly approved via grant_path_access,
        # overriding sandbox writable_dirs / blocked_paths / sensitive-path
        # guards for the rest of this session. Not persisted across restarts.
        self._escalated_paths: List[str] = []
        self._permission_mode = permission_mode

        # Register all file operation functions.
        # Read-only tools are concurrency_safe (can run in parallel with each other).
        # Write tools stay serialised.
        self.register(self.read_file, concurrency_safe=True, is_read_only=True)
        self.register(self.write_file, is_destructive=True)
        self.register(self.apply_patch, is_destructive=True)
        self.register(self.glob, concurrency_safe=True, is_read_only=True)
        self.register(self.grep, concurrency_safe=True, is_read_only=True)
        # glob and grep enforce their own hardcoded timeouts on both fast
        # (rg / native pathlib.glob) and fallback paths, so skip the outer
        # 120s executor wrapper — otherwise a bare ``**/*.log`` walk from
        # ``$HOME`` or a stuck network mount used to hang for the full 120s.
        self.functions["glob"].manages_own_timeout = True
        self.functions["grep"].manages_own_timeout = True

    def set_work_dir(self, work_dir: str) -> None:
        """Point relative paths at another directory, mid-session.

        A long-running session that binds itself to a git worktree
        (``agentica/worktrees.py``) has to take its tools with it — otherwise
        ``read_file("agentica/peers.py")`` still reads the directory the session
        started in, and the isolation the worktree was for is a fiction.
        ``Agent.rebind_work_dir`` calls this; the sandbox's writable_dirs are
        updated there, on the shared SandboxConfig every tool holds.

        Not a tool function: it is registered nowhere, so the model cannot move
        its own file tools behind the agent's back — only ``rebind_work_dir``
        can, which moves everything at once.
        """
        self.work_dir = Path(work_dir)

    def set_permission_mode(self, mode: str) -> None:
        """Record the agent's permission tier (sandbox still reads SandboxConfig)."""
        self._permission_mode = mode

    def grant_path_access(self, path: str, *, prefix: bool) -> None:
        """Whitelist ``path`` for the rest of this session.

        Sensitive paths (``_check_sensitive_write_path``) always grant the
        exact file, never the parent directory — ``_is_escalated`` also
        bypasses the sensitive-path guard, so prefixing ``~/.ssh`` would
        unlock the whole tree.
        """
        resolved = str(self._resolve_path(path).resolve())
        sensitive = _check_sensitive_write_path(path) is not None
        if sensitive or not prefix:
            grant = resolved
        else:
            grant = resolved if Path(resolved).is_dir() else str(Path(resolved).parent)
        if grant not in self._escalated_paths:
            self._escalated_paths.append(grant)

    def _escalation_hint(self, path_expr: str) -> str:
        """Say what to do about a sandbox refusal. Parking happens before execute."""
        return " This path is blocked until the user grants access."

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
        """Whether `resolved` was previously approved via grant_path_access."""
        for granted in self._escalated_paths:
            if resolved == granted:
                return True
            prefix = granted.rstrip("/\\")
            if resolved.startswith(prefix + "/") or resolved.startswith(prefix + os.sep):
                return True
        return False

    def _component_blocked(self, resolved: str) -> bool:
        """Whether `resolved` has a path component matching sandbox_config.blocked_paths
        (e.g. `.ssh`, `.env`) and hasn't already been approved via grant_path_access."""
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
          escalatable via grant_path_access.

        When sandbox is enabled, also checks:
        - Path components do not match any blocked_paths entries, unless the
          path was already approved via grant_path_access.
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
                f"Sandbox: access to path containing '{matched}' is blocked."
                + self._escalation_hint("<the path>")
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
        location and hasn't already been approved via grant_path_access.

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
        return err + self._escalation_hint(repr(filepath))

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
                f"Writable dirs: {self._sandbox_config.writable_dirs}."
                + self._escalation_hint("<the directory>")
            )
        return path

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
            *,
            tail: Optional[int] = None,
    ) -> str:
        """Reads a file from the filesystem. A missing file returns an error.

        Two ways to page — pick one:

        - From the start (default): omit ``tail`` (or pass ``tail=0``).
          ``offset`` is 0-based (0 = first line). ``limit`` is how many
          lines (default 500). Next page: offset=500, limit=500. To read
          700 lines from the top, use ``limit=700``, not ``tail=700``.
        - From the end: ``tail=N`` with N>=1 is the last N lines. A file
          shorter than N returns the whole file. ``offset=-N`` is the same
          window; ``limit`` then keeps the oldest lines of it
          (offset=-50, limit=10 → lines N-49..N-40).

        A tail scan of a huge file times out after 20 seconds.
        file_path may be absolute, relative to the working directory, or
        `~`-prefixed. Lines longer than 2000 characters are truncated.
        Results have line-number prefixes. Prefer one larger read over
        many small slices.

        Args:
            file_path: File path for md/txt/py/etc. Absolute, relative, or `~`
            offset: 0-based start line when not using tail. Negative = window
                of that many lines at the end.
            limit: Lines to return from offset (default 500). From-the-start
                paging uses this, not tail.
            tail: Last N lines (N>=1). Omit or 0 = read from the start with
                offset/limit. Negative N is last |N| lines.

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

        n_tail = _effective_tail(tail)
        if n_tail is not None:
            offset, limit = -n_tail, n_tail

        limit = limit if limit is not None else self.max_read_lines
        max_line_len = self.max_line_length

        # --- Large-file guard (mirrors CC's maxSizeBytes) ---
        # Front-paged reads only: the guard keeps unbounded content out of the
        # context. A tail read holds at most `keep` lines on any file size, so
        # it is exempt — the guard used to fire first and reject the exact
        # tail call its own error message recommends for large files.
        if offset >= 0:
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
                    f"Use tail to read the end, "
                    f"e.g. read_file('{file_path}', tail=50)."
                )

        if offset < 0:
            try:
                result, total_lines, start_shown, actual_end = await asyncio.wait_for(
                    self._read_from_end(
                        path, keep=abs(offset), take=limit, max_line_len=max_line_len,
                    ),
                    timeout=_READ_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"read_file timed out after {_READ_TIMEOUT} seconds while "
                    f"reading from the end of {file_path}. Use execute with "
                    f"tail/rg on a narrower path."
                ) from None
        else:
            result, total_lines, start_shown, actual_end = await self._read_from_start(
                path, offset=offset, limit=limit, max_line_len=max_line_len,
            )

        if total_lines == 0:
            result = f"File is empty: {path}"
        elif start_shown > 1 or actual_end < total_lines:
            result += (
                f"\n\n[Showing lines {start_shown}-{actual_end} of {total_lines} total lines]"
            )

        logger.debug(
            f"Read file {file_path}: lines {start_shown}-{actual_end}, "
            f"total {total_lines} lines"
        )
        return result

    async def _read_from_start(
            self,
            path: Path,
            *,
            offset: int,
            limit: int,
            max_line_len: int,
    ) -> Tuple[str, int, int, int]:
        output_lines: List[str] = []
        total_lines = 0
        end_line = offset + limit
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            async for line in f:
                total_lines += 1
                if total_lines > offset and total_lines <= end_line:
                    text = line.rstrip('\n\r')
                    if len(text) > max_line_len:
                        text = text[:max_line_len] + "..."
                    output_lines.append(f"{total_lines:6d}\t{text}")
        if not output_lines:
            return "", total_lines, offset + 1, offset
        start_shown = offset + 1
        actual_end = offset + len(output_lines)
        return "\n".join(output_lines), total_lines, start_shown, actual_end

    async def _read_from_end(
            self,
            path: Path,
            *,
            keep: int,
            take: int,
            max_line_len: int,
    ) -> Tuple[str, int, int, int]:
        buf: deque = deque(maxlen=keep)
        total_lines = 0
        async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            async for line in f:
                total_lines += 1
                text = line.rstrip('\n\r')
                if len(text) > max_line_len:
                    text = text[:max_line_len] + "..."
                buf.append((total_lines, text))
        window = list(buf)[:take]
        if not window:
            return "", total_lines, 1, 0
        output_lines = [f"{n:6d}\t{text}" for n, text in window]
        return "\n".join(output_lines), total_lines, window[0][0], window[-1][0]

    async def write_file(self, file_path: str, content: str) -> str:
        """Writes content to a file in the filesystem.

        Creates a new file or overwrites an existing one entirely. Prefer
        apply_patch for modifying existing files. Use write_file for new files
        or whole-file rewrites, including HTML reports the user can open in a
        browser (inline CSS is fine). Parent directories are created if needed.
        Relative paths resolve against the working directory.

        Args:
            file_path: File path (relative or absolute). Examples: "tmp/script.py", "outputs/result.txt", "./tmp/main.py", use './tmp/' prefix file path for temporary files
            content: File content to write

        Returns:
            Operation result message containing the actual absolute path of the file
        """
        self._validate_write_path(file_path)
        path = self._resolve_path(file_path)

        # ── Sensitive path guard ──────────────────────────────────
        if self._sandbox_config is not None and self._sandbox_config.enabled:
            sensitive_err = self._sensitive_write_guard(str(path))
            if sensitive_err:
                raise PermissionError(sensitive_err)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        action = "Created" if not path.exists() else "Updated"
        old_content = None

        # ── Snapshot for diagnostics ──────────────────────────────
        await self._diagnostics_snapshot(path)
        if path.exists() and path.is_file():
            try:
                old_content = path.read_text(encoding='utf-8', errors='ignore')
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
        suffix = ("\n\n" + diag_text) if diag_text else ""
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

        Use this for code edits, multi-hunk edits, and changes that span
        multiple files. The same substitution in several files is one
        patch with several Update File hunks, not a shell rewriter.
        Related edits across files: parallel read_file, then one patch —
        not read-then-patch one file at a time.
        Use write_file for new files or whole-file rewrites.

        After ``@@``, the first character of each line is a space (keep an
        existing line), ``-`` (delete), or ``+`` (insert). To add a comment,
        emit a ``+`` line; a spaced copy of the file is a no-op.

        *** Begin Patch
        *** Update File: app.py
        @@
        +# matching via BFS
         def run():
        -    timeout = 10
        +    timeout = 30
        *** Update File: tests/test_app.py
        @@
        -assert run() == 10
        +assert run() == 30
        *** End Patch

        Context must match each file exactly — no whitespace or quote
        rewriting. Parallel ``read_file`` the files you will edit, then
        one patch covering all of them; do not reconstruct context from
        memory.
        All paths are validated before any file is changed. A markdown
        fence, heredoc, or omitted Begin/End around a File header is
        accepted. A keep line that forgot its leading space is recovered
        when it uniquely matches the current file; other hunk lines still
        need ' ', '-', or '+'.

        Args:
            patch: Begin/End Patch envelope. After @@: space keeps, '-' deletes, '+' inserts.

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

            if self._sandbox_config is not None and self._sandbox_config.enabled:
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
                                raise PatchNoChangeError(
                                    "Update does not change the file. "
                                    "A leading space keeps the line; "
                                    "to delete a line start it with '-'; "
                                    "to add a line start it with '+'."
                                )
                        else:
                            new_content = None
                except (FileExistsError, FileNotFoundError, IsADirectoryError, PatchContextError, ValueError) as exc:
                    preflight_errors.append((result_path, exc))
                    continue

                added, removed = self._content_change_counts(old_content, new_content)
                prepared.append(
                    (operation, path, path_key, old_content, new_content, result_path, added, removed)
                )

            if preflight_errors:
                file_noun = "file" if len(preflight_errors) == 1 else "files"
                kinds = set()
                for _, error in preflight_errors:
                    if isinstance(error, PatchContextError):
                        kinds.add("context")
                    elif isinstance(error, PatchNoChangeError):
                        kinds.add("noop")
                    elif isinstance(error, (FileExistsError, FileNotFoundError, IsADirectoryError)):
                        kinds.add("fs")
                    else:
                        kinds.add("grammar")
                if kinds == {"grammar"}:
                    headline = (
                        f"Malformed patch for {len(preflight_errors)} {file_noun}; "
                        "no files were changed."
                    )
                elif kinds == {"noop"}:
                    headline = (
                        f"Patch does not change {len(preflight_errors)} {file_noun}; "
                        "no files were changed."
                    )
                elif kinds == {"context"}:
                    headline = (
                        f"Patch context not found for {len(preflight_errors)} {file_noun}; "
                        "no files were changed."
                    )
                else:
                    headline = (
                        f"Patch not applied for {len(preflight_errors)} {file_noun}; "
                        "no files were changed."
                    )
                error_lines = [headline]
                for result_path, error in preflight_errors:
                    error_lines.append(f"- {result_path}:")
                    error_lines.extend(f"  {line}" for line in str(error).splitlines())
                error_message = "\n".join(error_lines)
                if len(preflight_errors) == 1:
                    original_error = preflight_errors[0][1]
                    if isinstance(
                        original_error,
                        (FileExistsError, FileNotFoundError, IsADirectoryError),
                    ):
                        raise type(original_error)(error_message) from original_error
                raise ValueError(error_message) from preflight_errors[0][1]

            # Diagnostics snapshots are captured only after every file
            # and hunk has passed preflight.
            for operation, path, path_key, old_content, *_ in prepared:
                await self._diagnostics_snapshot(path)

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

    async def glob(self, pattern: str, path: str = ".") -> str:
        """Find files by name pattern, anywhere in a tree.

        `*` matches within one directory only; `**` recurses through every level,
        so "*.py" and "**/*.py" give very different results. Noise directories
        (.git, __pycache__, node_modules, .venv, ...) are always excluded.
        Entries include subdirectories, so pattern "*" is how you list what a
        directory holds.

        Args:
            pattern: Glob pattern, e.g. "*.py", "**/*.md", "src/?*.js". May be
                absolute ("/home/user/*.py") or relative to `path`.
            path: Directory to search from (default: ".")

        Returns:
            JSON list of sorted absolute file paths, empty when nothing matches.
        """
        self._validate_path(path)
        base_path = self._resolve_path(path)

        if not base_path.exists():
            raise FileNotFoundError(self._missing_path_error("Directory", path, base_path))

        # Run glob in executor to avoid blocking on large directory trees
        def _glob_sync():
            matches = list(base_path.glob(pattern))
            nested = _nested_checkouts(base_path)
            return sorted(
                str(m) for m in matches
                if not _in_noise_dir(m, base_path, _NOISE_DIRS, nested)
            )

        loop = asyncio.get_event_loop()
        try:
            filtered = await asyncio.wait_for(
                loop.run_in_executor(None, _glob_sync),
                timeout=_GLOB_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"glob timed out after {_GLOB_TIMEOUT} seconds "
                f"(pattern={pattern!r}, path={path!r}). "
                f"Narrow `path` or use a more specific pattern; "
                f"do not retry the same walk."
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
            limit: int = 100,
    ) -> str:
        """Search file contents for a regex pattern.

        Uses ripgrep (``rg``) when it is on PATH, otherwise a Python fallback.
        Output is matching lines as `file:line_number:content`.

        Args:
            pattern: Regex to search for
            path: File or directory to search (default: ".")
            include: File glob filter, e.g. "*.ts" or "*.py"
            limit: Maximum matching lines (default: 100)

        Returns:
            Search results as formatted string
        """
        self._validate_path(path)
        base_path = self._resolve_path(path)
        if not base_path.exists():
            raise FileNotFoundError(self._missing_path_error("Path", path, base_path))

        rg_path = shutil.which("rg")
        if rg_path is None:
            return await self._run_grep_fallback(pattern, path, include, limit)

        cmd: List[str] = [rg_path, "--line-number"]
        if include:
            cmd.extend(["--glob", include])
        for d in sorted(_NOISE_DIRS - {'.git'}):
            cmd.extend(["--glob", f"!{d}/"])
        for root in _nested_checkouts(base_path):
            cmd.extend(["--glob", f"!{os.path.relpath(root, str(base_path))}/"])
        cmd.extend(["--", pattern, str(base_path)])

        proc = None
        drained = False
        hit_cap = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            max_lines = limit if limit >= 0 else None
            stdout, stderr, hit_cap = await asyncio.wait_for(
                _collect_rg_output(proc, max_lines),
                timeout=_GREP_TIMEOUT,
            )
            drained = True
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"grep timed out after {_GREP_TIMEOUT} seconds. "
                f"Narrow `path` or `include`."
            ) from None
        finally:
            if proc is not None and not drained:
                await terminate_subprocess(proc)
            if proc is not None:
                close_subprocess_transport(proc)

        output = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
        if hit_cap:
            output = _cap_output_lines(output, limit)
        if not output.strip():
            result = f"No matches found for '{pattern}'"
            if err:
                result = f"{result}\n{err}"
        else:
            result = output.rstrip("\n")
            if hit_cap:
                result = _cap_output_lines(result, limit)
        result = truncate_if_too_long(result)
        logger.debug(f"Grep(rg) for '{pattern}': result length {len(result)} chars")
        return result

    async def _run_grep_fallback(
            self,
            pattern: str,
            path: str,
            include: Optional[str],
            limit: int,
    ) -> str:
        loop = asyncio.get_event_loop()
        timeout = _GREP_TIMEOUT
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None, self._grep_fallback, pattern, path, include, limit,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"grep timed out after {timeout} seconds. "
                f"Narrow `path` or `include`."
            )

    def _grep_fallback(
            self,
            pattern: str,
            path: str,
            include: Optional[str],
            limit: int,
    ) -> str:
        base_path = self._resolve_path(path)
        try:
            regex_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        if base_path.is_file():
            files = [base_path] if not include or base_path.match(include) else []
        elif include:
            files = list(base_path.glob(f"**/{include}"))
        else:
            files = list(base_path.glob("**/*"))

        nested = _nested_checkouts(base_path)
        files = [
            f for f in files
            if f.is_file() and not _in_noise_dir(f, base_path, _NOISE_DIRS, nested)
        ]

        results = []
        n_emitted = 0
        for fp in files:
            if n_emitted >= limit:
                break
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as handle:
                    lines_in = handle.readlines()
            except OSError:
                continue
            for line_num, line in enumerate(lines_in, 1):
                if regex_pattern.search(line):
                    body = line.rstrip("\n")[:200]
                    results.append(f"{fp}:{line_num}: {body}")
                    n_emitted += 1
                    if n_emitted >= limit:
                        break

        result = "\n".join(results) if results else f"No matches found for '{pattern}'"
        result = truncate_if_too_long(result)
        logger.debug(f"Grep(fallback) for '{pattern}': result length {len(result)} chars")
        return result

