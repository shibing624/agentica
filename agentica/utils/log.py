# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import os
import logging
from typing import Optional

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL


class LoguruStyleFormatter(logging.Formatter):
    """A formatter that mimics loguru's beautiful output style."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[34m',  # Blue
        'INFO': '\033[32m',  # Green
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

        # Format module info with cyan
        module_info = f"{self.CYAN}agentica{self.RESET}:{self.CYAN}{record.funcName}{self.RESET}:{self.CYAN}{record.lineno}{self.RESET}"

        # Format message with level color
        message = f"{level_color}{record.getMessage()}{self.RESET}"

        return f"{timestamp} | {level} | {module_info} - {message}"


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
        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | agentica:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
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
    """
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            logger.removeHandler(handler)


def restore_console_logging(log_level: str = "INFO"):
    """Restore console logging for agentica logger.
    
    Args:
        log_level: Log level for the console handler
    """
    # Check if console handler already exists
    has_console = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout 
        for h in logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(LoguruStyleFormatter())
        logger.addHandler(console_handler)
