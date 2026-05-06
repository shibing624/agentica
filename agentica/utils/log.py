# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import os
import logging
from functools import lru_cache
from typing import Optional

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL


# CHAT_LEVEL is a custom log level dedicated to inter-agent conversation
# (semantic dialog flow), distinct from system-level INFO/DEBUG. Sits between
# INFO (20) and WARNING (30) so it surfaces above routine info logs but below
# real warnings. Used by Runner turn boundaries, Swarm assignments, and
# Subagent spawn/return so multi-agent debugging shows the conversation flow
# without being drowned in framework chatter.
CHAT_LEVEL = 25
logging.addLevelName(CHAT_LEVEL, "CHAT")


def _logger_chat(self, message, *args, **kwargs):
    """Log an inter-agent conversation event at the CHAT level."""
    if self.isEnabledFor(CHAT_LEVEL):
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

        file_handler = logging.FileHandler(log_file)
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
