# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from datetime import datetime
from dotenv import load_dotenv  # noqa
from loguru import logger  # noqa, need to import logger here to avoid circular import

AGENTICA_HOME = os.getenv("AGENTICA_HOME", os.path.expanduser("~/.agentica"))
if AGENTICA_HOME:
    os.makedirs(AGENTICA_HOME, exist_ok=True)

# Load environment variables from .env file
AGENTICA_DOTENV_PATH = os.getenv("AGENTICA_DOTENV_PATH", f"{AGENTICA_HOME}/.env")
load_dotenv(AGENTICA_DOTENV_PATH, override=True)

AGENTICA_DATA_DIR = os.getenv("AGENTICA_DATA_DIR", f"{AGENTICA_HOME}/data")
AGENTICA_LOG_LEVEL = os.getenv("AGENTICA_LOG_LEVEL", "INFO").upper()
AGENTICA_LOG_FILE = os.getenv("AGENTICA_LOG_FILE")
if AGENTICA_LOG_LEVEL == "DEBUG":
    formatted_date = datetime.now().strftime("%Y%m%d")
    default_log_file = f"{AGENTICA_HOME}/logs/{formatted_date}.log"
    AGENTICA_LOG_FILE = os.getenv("AGENTICA_LOG_FILE", default_log_file)
    logger.debug(f"AGENTICA_LOG_LEVEL: DEBUG, AGENTICA_LOG_FILE: {AGENTICA_LOG_FILE}")
    if os.path.exists(AGENTICA_DOTENV_PATH):
        logger.debug(f"Loaded AGENTICA_DOTENV_PATH: {AGENTICA_DOTENV_PATH}")
