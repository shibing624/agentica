# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from datetime import datetime
from dotenv import load_dotenv  # noqa

AGENTICA_HOME = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
if AGENTICA_HOME:
    os.makedirs(AGENTICA_HOME, exist_ok=True)

# Load environment variables from .env file
AGENTICA_DOTENV_PATH = os.path.expanduser(os.getenv("AGENTICA_DOTENV_PATH", f"{AGENTICA_HOME}/.env"))
load_dotenv(AGENTICA_DOTENV_PATH, override=True)
load_dotenv()

AGENTICA_DATA_DIR = os.getenv("AGENTICA_DATA_DIR", f"{AGENTICA_HOME}/data")
AGENTICA_SKILL_DIR = os.getenv("AGENTICA_DATA_DIR", f"{AGENTICA_HOME}/skill")
AGENTICA_LOG_LEVEL = os.getenv("AGENTICA_LOG_LEVEL", "INFO").upper()
user_log_file = os.getenv("AGENTICA_LOG_FILE")

if user_log_file:
    # User specified a log file
    AGENTICA_LOG_FILE = os.path.expanduser(user_log_file)
elif AGENTICA_LOG_LEVEL == "DEBUG":
    # Use default log file if log level is DEBUG
    formatted_date = datetime.now().strftime("%Y%m%d")
    default_log_file = f"{AGENTICA_HOME}/logs/{formatted_date}.log"
    os.makedirs(os.path.dirname(default_log_file), exist_ok=True)
    AGENTICA_LOG_FILE = default_log_file
else:
    AGENTICA_LOG_FILE = ""

# Langfuse configuration
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
LANGFUSE_TIMEOUT = int(os.getenv("LANGFUSE_TIMEOUT", "300"))
