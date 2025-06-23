# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from datetime import datetime
from dotenv import load_dotenv  # noqa

AGENTICA_HOME = os.getenv("AGENTICA_HOME", os.path.expanduser("~/.agentica"))
if AGENTICA_HOME:
    os.makedirs(AGENTICA_HOME, exist_ok=True)

# Load environment variables from .env file
AGENTICA_DOTENV_PATH = os.getenv("AGENTICA_DOTENV_PATH", f"{AGENTICA_HOME}/.env")
load_dotenv(AGENTICA_DOTENV_PATH, override=True)

AGENTICA_DATA_DIR = os.getenv("AGENTICA_DATA_DIR", f"{AGENTICA_HOME}/data")
AGENTICA_LOG_LEVEL = os.getenv("AGENTICA_LOG_LEVEL", "INFO").upper()

user_log_file = os.getenv("AGENTICA_LOG_FILE")

if user_log_file:
    # User specified a log file
    AGENTICA_LOG_FILE = user_log_file
elif AGENTICA_LOG_LEVEL == "DEBUG":
    # Use default log file if log level is DEBUG
    formatted_date = datetime.now().strftime("%Y%m%d")
    default_log_file = f"{AGENTICA_HOME}/logs/{formatted_date}.log"
    os.makedirs(os.path.dirname(default_log_file), exist_ok=True)
    AGENTICA_LOG_FILE = default_log_file
else:
    AGENTICA_LOG_FILE = ""
