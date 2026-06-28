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

# Load environment variables from .env file (project .env takes priority over global)
load_dotenv()
AGENTICA_DOTENV_PATH = os.path.expanduser(os.getenv("AGENTICA_DOTENV_PATH", f"{AGENTICA_HOME}/.env"))
load_dotenv(AGENTICA_DOTENV_PATH)

# Project the unified, hand-editable config (~/.agentica/config.yaml) into the
# process environment. This is the single source of truth shared by the SDK and
# the CLI. It runs AFTER .env so precedence (highest first) is:
#     shell env  >  .env  >  config.yaml
# Injection uses setdefault semantics, so an already-set variable is never
# overwritten. The SDK keeps reading plain env vars; nothing else has to change.
try:
    from agentica.global_config import apply_global_config

    apply_global_config()
except Exception:
    # Never let a malformed config.yaml break import of the SDK.
    pass

AGENTICA_SKILL_DIR = os.getenv("AGENTICA_SKILL_DIR", f"{AGENTICA_HOME}/skills")
AGENTICA_EXTRA_SKILL_PATHS = [
    os.path.expanduser(path) for path in os.getenv("AGENTICA_EXTRA_SKILL_PATH", "").split(os.pathsep) if path.strip()
]
AGENTICA_CRON_DIR = os.getenv("AGENTICA_CRON_DIR", f"{AGENTICA_HOME}/cron")
AGENTICA_WORKSPACE_DIR = os.getenv("AGENTICA_WORKSPACE_DIR", f"{AGENTICA_HOME}/workspace")
AGENTICA_PROJECTS_DIR = os.getenv("AGENTICA_PROJECTS_DIR", f"{AGENTICA_HOME}/projects")
AGENTICA_LOG_LEVEL = os.getenv("AGENTICA_LOG_LEVEL", "INFO").upper()
AGENTICA_MAX_MEMORY_CHARACTER_COUNT = int(os.getenv("AGENTICA_MAX_MEMORY_CHARACTER_COUNT", "40000"))
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
