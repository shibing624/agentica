# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from datetime import datetime

AGENTICA_HOME = os.environ.get("AGENTICA_HOME", os.path.expanduser("~/.agentica"))

# Load environment variables from .env file
AGENTICA_DOTENV_PATH = os.environ.get("AGENTICA_DOTENV_PATH", f"{AGENTICA_HOME}/.env")
try:
    from dotenv import load_dotenv  # noqa
    from loguru import logger  # noqa, need to import logger here to avoid circular import

    if load_dotenv(AGENTICA_DOTENV_PATH, override=True):
        logger.info(f"Loaded AGENTICA_DOTENV_PATH: {AGENTICA_DOTENV_PATH}")
except ImportError:
    logger.debug("dotenv not installed, skipping...")

AGENTICA_DATA_DIR = os.environ.get("AGENTICA_DATA_DIR", f"{AGENTICA_HOME}/data")
AGENTICA_LOG_LEVEL = os.environ.get("AGENTICA_LOG_LEVEL", "INFO")
AGENTICA_LOG_FILE = os.environ.get("AGENTICA_LOG_FILE")
if AGENTICA_LOG_LEVEL.upper() == "DEBUG":
    formatted_date = datetime.now().strftime("%Y%m%d")
    default_log_file = f"{AGENTICA_HOME}/logs/{formatted_date}.log"
    AGENTICA_LOG_FILE = os.environ.get("AGENTICA_LOG_FILE", default_log_file)
    logger.debug(f"AGENTICA_LOG_LEVEL: DEBUG, AGENTICA_LOG_FILE: {AGENTICA_LOG_FILE}")

SMART_LLM = os.environ.get("SMART_LLM")
FAST_LLM = os.environ.get("FAST_LLM")
