# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

from loguru import logger as _logger

from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL


def get_logger(log_level: str = "INFO", log_file: str = None):
    """Get logger instance."""
    _logger.remove()
    _logger.add(
        sink=sys.stdout,
        level=log_level,
        enqueue=True,  # 异步输出日志
        backtrace=False,  # 设置为 True 以打印回溯
        diagnose=False,  # 设置为 True 以自动显示变量
    )
    if log_file:
        _logger.add(
            sink=log_file,
            level=log_level,
            rotation="1 week",
            retention="1 month",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    return _logger


logger = get_logger(log_level=AGENTICA_LOG_LEVEL, log_file=AGENTICA_LOG_FILE)


def set_log_level_to_debug():
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


def print_llm_stream(msg):
    print(msg, end="", flush=True)
