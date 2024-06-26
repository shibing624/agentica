# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

from loguru import logger


def get_logger(log_level: str = "INFO", log_file: str = None) -> logger:
    """Get logger instance."""
    logger.remove()
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
    else:
        logger.add(
            sink=sys.stdout,
            level=log_level,
            enqueue=True,  # 异步输出日志
            backtrace=False,  # 设置为 True 以打印回溯
            diagnose=False,  # 设置为 True 以自动显示变量
        )
    return logger


log_level = os.environ.get("LOG_LEVEL") or "INFO"
logger = get_logger(
    log_level=log_level,
    log_file=os.environ.get("LOG_FILE")
)


def set_log_level_to_debug():
    logger.remove()  # 移除之前的日志处理器
    logger.add(
        sink=sys.stdout,
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
