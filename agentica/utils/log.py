# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import os
import loguru

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL

logger = loguru.logger

_logger_initialized = False


def _ensure_log_dir(log_file: str):
    """Ensure log directory exists."""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)


def configure_logger(log_level: str = "INFO", log_file: str = None):
    """Configure loguru logger."""
    global _logger_initialized

    logger.remove()
    logger.add(
        sink=sys.stdout,
        level=log_level,
        enqueue=True,  # 异步输出日志
        backtrace=False,  # 设置为 True 以打印回溯
        diagnose=False,  # 设置为 True 以自动显示变量
    )

    if log_file:
        _ensure_log_dir(log_file)
        logger.add(
            sink=log_file,
            level=log_level,
            rotation="1 week",
            retention="1 month",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

    _logger_initialized = True
    return logger


def _lazy_init_logger():
    """Lazy initialization of logger if not already initialized."""
    global _logger_initialized
    if not _logger_initialized:
        configure_logger(log_level=AGENTICA_LOG_LEVEL, log_file=AGENTICA_LOG_FILE)


def set_log_level_to_debug():
    """Set log level to DEBUG for all handlers."""
    global _logger_initialized

    logger.remove()  # 移除之前的日志处理器
    logger.add(
        sink=sys.stdout,
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    if AGENTICA_LOG_FILE:
        _ensure_log_dir(AGENTICA_LOG_FILE)
        logger.add(
            sink=AGENTICA_LOG_FILE,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    _logger_initialized = True


def set_log_level_to_info():
    """Set log level to INFO for all handlers."""
    global _logger_initialized

    logger.remove()  # 移除之前的日志处理器
    logger.add(
        sink=sys.stdout,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    if AGENTICA_LOG_FILE:
        _ensure_log_dir(AGENTICA_LOG_FILE)
        logger.add(
            sink=AGENTICA_LOG_FILE,
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

    _logger_initialized = True


_lazy_init_logger()


def print_llm_stream(msg):
    """Print message without newline."""
    print(msg, end="", flush=True)
