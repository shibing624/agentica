# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import loguru

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL

logger = loguru.logger


def configure_logger(log_level: str = "INFO", log_file: str = None):
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sink=sys.stdout,
        level=log_level,
        enqueue=True,  # 异步输出日志
        backtrace=False,  # 设置为 True 以打印回溯
        diagnose=False,  # 设置为 True 以自动显示变量
    )

    if log_file:
        logger.add(
            sink=log_file,
            level=log_level,
            rotation="1 week",
            retention="1 month",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

    return logger


# 初始化配置全局logger
configure_logger(log_level=AGENTICA_LOG_LEVEL, log_file=AGENTICA_LOG_FILE)


def set_log_level_to_debug():
    """Set log level to DEBUG for all handlers."""
    logger.remove()  # 移除之前的日志处理器
    logger.add(
        sink=sys.stdout,
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    if AGENTICA_LOG_FILE:
        logger.add(
            sink=AGENTICA_LOG_FILE,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )


def set_log_level_to_info():
    """Set log level to INFO for all handlers."""
    logger.remove()  # 移除之前的日志处理器
    logger.add(
        sink=sys.stdout,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    if AGENTICA_LOG_FILE:
        logger.add(
            sink=AGENTICA_LOG_FILE,
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )


def print_llm_stream(msg):
    """Print message without newline."""
    print(msg, end="", flush=True)
