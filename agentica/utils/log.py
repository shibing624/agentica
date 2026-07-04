# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import os
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Iterator, Optional

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL


# ----------------------------------------------------------------------
# Per-run log context (concurrency-safe via contextvars)
# ----------------------------------------------------------------------
#
# ``agentica`` is an SDK: multiple agents / workflows / tools can run
# concurrently in a single process (asyncio tasks, thread-pool executors,
# subagents). When they all share a single log file, records interleave and
# it becomes impossible to reconstruct the full trace for a single query.
#
# We attach the active ``run_id`` (and optional ``parent_run_id`` for
# subagent/workflow children) to the logging record via ``ContextVar``s.
# ``ContextVar`` is the right primitive here: asyncio propagates the context
# per-Task automatically, and ``contextvars.copy_context()`` covers thread
# handoffs. The formatters below read these vars and emit a compact
# ``run=<8-hex>`` prefix so a user can do::
#
#     grep 'run=a1b2c3d4' ~/.agentica/logs/*.log
#
# to reconstruct a single query's full timeline even when concurrent runs
# are interleaved in the same file.
_run_id_var: ContextVar[Optional[str]] = ContextVar("agentica_run_id", default=None)
_parent_run_id_var: ContextVar[Optional[str]] = ContextVar("agentica_parent_run_id", default=None)


def _short(run_id: Optional[str]) -> Optional[str]:
    """Truncate a UUID-shaped run id to its first 8 hex chars for log display.

    Full UUIDs (36 chars) make every log line noisy; 8 hex chars are still
    unique enough within a single process's log window (~4B combinations)
    and stay readable. Non-UUID ids are returned as-is up to 8 chars.
    """
    if not run_id:
        return None
    return str(run_id).replace("-", "")[:8]


@contextmanager
def bind_run_context(
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind a run's identifiers to the log context for the duration of the block.

    Meant to wrap a single ``Agent.run`` / ``Workflow.run`` invocation so every
    log record emitted from that call (including from tools and subagents
    running on child asyncio Tasks) carries the same ``run=<id>`` prefix.

    ``ContextVar`` tokens are reset on exit to avoid leaking the id into
    unrelated code that happens to run afterwards on the same task.
    """
    tokens = []
    if run_id:
        tokens.append((_run_id_var, _run_id_var.set(_short(run_id))))
    if parent_run_id:
        tokens.append((_parent_run_id_var, _parent_run_id_var.set(_short(parent_run_id))))
    try:
        yield
    finally:
        # Reset in reverse order so nested bind_run_context calls unwind cleanly.
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except (ValueError, LookupError):
                # ContextVar.reset raises if the token was created in a
                # different Context (rare, but can happen across thread
                # boundaries). Fall back to clearing the value.
                var.set(None)


def _run_prefix() -> str:
    """Render the current run context as a formatter fragment.

    Returns an empty string when no run is bound (so SDK usage outside a
    ``Runner.run`` — e.g. module import time — doesn't gain a noisy
    ``run=none`` column). When a parent id is present we render it as
    ``run=<child>/parent=<parent>`` to make subagent hierarchies greppable.
    """
    rid = _run_id_var.get()
    if not rid:
        return ""
    pid = _parent_run_id_var.get()
    if pid:
        return f"run={rid}/parent={pid}"
    return f"run={rid}"


# CHAT_LEVEL is a custom log level dedicated to inter-agent conversation
# (semantic dialog flow), distinct from system-level INFO/DEBUG. Sits between
# INFO (20) and WARNING (30) so it surfaces above routine info logs but below
# real warnings. Used by Runner turn boundaries, Swarm assignments, and
# Subagent spawn/return so multi-agent debugging shows the conversation flow
# without being drowned in framework chatter.
CHAT_LEVEL = 25
logging.addLevelName(CHAT_LEVEL, "CHAT")


def _logger_chat(self, message, *args, **kwargs):
    """Log an inter-agent conversation event at the CHAT level.

    ``stacklevel=2`` makes Python's logging framework attribute the record
    to the caller of ``logger.chat(...)`` rather than to this helper, so
    log lines display the true source location.
    """
    if self.isEnabledFor(CHAT_LEVEL):
        kwargs.setdefault("stacklevel", 2)
        self._log(CHAT_LEVEL, message, args, **kwargs)


logging.Logger.chat = _logger_chat  # type: ignore[attr-defined]


@lru_cache(maxsize=4096)
def _dotted_module_from_path(pathname: str) -> str:
    """Convert a source file path to a loguru-style dotted module name.

    Walks up from the file looking for the first directory without an
    ``__init__.py`` — that's the import root. The path below it becomes the
    dotted module (``agentica/runner.py`` → ``agentica.runner``,
    ``tests/test_cli.py`` → ``tests.test_cli``).

    Falls back to the bare filename stem when the file is not inside any
    importable package (e.g. a one-off script). Cached because every log
    record from the same source file resolves to the same string.
    """
    if not pathname:
        return "?"
    try:
        abs_path = os.path.abspath(pathname)
    except (OSError, ValueError):
        return os.path.splitext(os.path.basename(pathname))[0] or "?"

    directory, filename = os.path.split(abs_path)
    stem = os.path.splitext(filename)[0]
    if not directory:
        return stem or "?"

    parts = [stem]
    current = directory
    while True:
        if not os.path.isfile(os.path.join(current, "__init__.py")):
            break
        parent, name = os.path.split(current)
        if not name or parent == current:
            break
        parts.append(name)
        current = parent

    return ".".join(reversed(parts)) if parts else stem


class LoguruStyleFormatter(logging.Formatter):
    """A formatter that mimics loguru's beautiful output style."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[34m',  # Blue
        'INFO': '\033[32m',  # Green
        'CHAT': '\033[36m',  # Cyan — distinct from INFO so dialog stands out
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'

    def format(self, record):
        # Get color for log level
        level_color = self.COLORS.get(record.levelname, '')

        # Format timestamp with green color (same format as file handler)
        timestamp = f"\033[32m{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}.{record.msecs:03.0f}\033[0m"

        # Format level with color and bold
        level = f"{level_color}{self.BOLD}{record.levelname:<8}{self.RESET}"

        # Format module info with cyan — full dotted path so the line is
        # actually clickable / greppable to the real file (loguru-style
        # ``module.path:function:line`` rather than the literal "agentica").
        module_path = _dotted_module_from_path(record.pathname)
        module_info = (
            f"{self.CYAN}{module_path}{self.RESET}:"
            f"{self.CYAN}{record.funcName}{self.RESET}:"
            f"{self.CYAN}{record.lineno}{self.RESET}"
        )

        # Format message with level color
        message = f"{level_color}{record.getMessage()}{self.RESET}"

        # Include the active run context (if any) between level and module so
        # concurrent runs stay visually separable. Empty string when unbound
        # (import-time logs, direct SDK helpers outside a Runner) keeps the
        # non-run case as clean as before.
        run_ctx = _run_prefix()
        if run_ctx:
            run_ctx_col = f"{self.CYAN}{run_ctx}{self.RESET}"
            return f"{timestamp} | {level} | {run_ctx_col} | {module_info} - {message}"
        return f"{timestamp} | {level} | {module_info} - {message}"


class _PlainLoguruStyleFormatter(logging.Formatter):
    """No-ANSI variant of :class:`LoguruStyleFormatter` for file handlers.

    File logs need the same dotted module locator (so grep / IDE click-through
    still works), but ANSI escape codes pollute the file.
    """

    def format(self, record):
        timestamp = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}"
            f".{record.msecs:03.0f}"
        )
        module_path = _dotted_module_from_path(record.pathname)
        run_ctx = _run_prefix()
        if run_ctx:
            return (
                f"{timestamp} | {record.levelname:<8} | {run_ctx} | "
                f"{module_path}:{record.funcName}:{record.lineno} - {record.getMessage()}"
            )
        return (
            f"{timestamp} | {record.levelname:<8} | "
            f"{module_path}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        )


def get_agentica_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Get agentica-specific logger that doesn't interfere with user's logging."""
    # Create a logger specifically for agentica
    logger = logging.getLogger("agentica")

    # Check if we need to add console handler
    has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers)
    # Check if we need to add file handler
    has_file_handler = log_file and any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers)

    # Set the logger level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Add console handler if not exists
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(LoguruStyleFormatter())
        logger.addHandler(console_handler)

    # Add file handler if specified and not exists
    if log_file and not has_file_handler:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # delay=True: defer opening the file until the first log record is
        # actually emitted. Without this, merely `import agentica` creates a
        # 0-byte YYYYMMDD.log every day (import → get_agentica_logger() at
        # module load → FileHandler.__init__ → open()), even when no INFO
        # event ever gets logged (e.g. pytest collection, IDE indexer,
        # a script that imports and exits). See ~/.agentica/logs/ history
        # for the 0-byte file pattern this eliminates.
        file_handler = logging.FileHandler(log_file, delay=True)
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        # Plain (no-ANSI) variant of LoguruStyleFormatter so the file copy
        # carries the same module.path:function:line locator as the console.
        file_handler.setFormatter(_PlainLoguruStyleFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid interfering with user's logging
    logger.propagate = False

    return logger


# Create the agentica logger instance
logger = get_agentica_logger(log_level=AGENTICA_LOG_LEVEL, log_file=AGENTICA_LOG_FILE)


def set_log_level_to_debug():
    """Set agentica logger to DEBUG level."""
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)


def set_log_level_to_info():
    """Set agentica logger to INFO level."""
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)


# ----------------------------------------------------------------------
# Log file retention (called from CLI startup)
# ----------------------------------------------------------------------
#
# We match filenames like ``20260705.log`` and ``20260705-12345.log``
# (date + optional ``-<pid>`` suffix added by CLI). Files that don't
# match this shape are never touched, so unrelated files in the same
# directory are safe.
_LOG_FILENAME_RE = re.compile(r"^(\d{8})(?:-\d+)?\.log$")


def cleanup_old_logs(log_dir: str, keep_days: int = 14) -> int:
    """Delete agentica log files older than ``keep_days`` in ``log_dir``.

    Only removes files whose name matches ``YYYYMMDD.log`` or
    ``YYYYMMDD-<pid>.log`` — the two shapes this module produces. Unrelated
    files are left alone so this is safe to call even if the log directory
    contains other content.

    Called lazily from the CLI startup path (not at module import) so pure
    SDK usage never touches the filesystem. Errors are swallowed with a
    debug log — a stale file failing to delete must never break user code.

    Returns the number of files removed (mostly useful for tests).
    """
    if keep_days <= 0 or not log_dir or not os.path.isdir(log_dir):
        return 0

    cutoff = datetime.now().date() - timedelta(days=keep_days)
    removed = 0
    try:
        entries = os.listdir(log_dir)
    except OSError:
        return 0

    for name in entries:
        m = _LOG_FILENAME_RE.match(name)
        if not m:
            continue
        try:
            file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if file_date >= cutoff:
            continue
        path = os.path.join(log_dir, name)
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:  # pragma: no cover — best-effort cleanup
            logger.debug("cleanup_old_logs: failed to remove %s: %s", path, exc)
    return removed


def print_llm_stream(msg):
    print(msg, end="", flush=True)


def suppress_console_logging():
    """Suppress console output for agentica logger (keep file logging).
    
    Use this in CLI mode to prevent logger output from interfering with
    interactive UI elements like spinners and status displays.
    
    Remove all non-file stream handlers, not just ``sys.stdout``. In TUI/IDE
    environments stdout/stderr may be wrapped, so identity checks against the
    current ``sys.stdout`` are not reliable enough.
    """
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)


def restore_console_logging(log_level: str = "INFO"):
    """Restore console logging for agentica logger.
    
    Args:
        log_level: Log level for the console handler
    """
    # Check if a non-file console handler already exists
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(LoguruStyleFormatter())
        logger.addHandler(console_handler)
