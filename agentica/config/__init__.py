# -*- coding: utf-8 -*-
"""Process configuration: env paths, config.yaml profiles, per-project files.

``env`` is imported first so ``AGENTICA_*`` exists on this package. Profiles
and the project store are submodules; import them from there.

``importlib.reload`` of this package must re-read the environment. The
constants live in ``env``; reloading only this module would re-bind the
already-computed values.
"""

import importlib

from agentica.config import env as _env

if globals().get("_RELOAD_ENV"):
    _env = importlib.reload(_env)
else:
    _RELOAD_ENV = True

from agentica.config.env import (
    AGENTICA_CACHE_DIR,
    AGENTICA_CRON_DIR,
    AGENTICA_DOTENV_PATH,
    AGENTICA_EXTRA_SKILL_PATHS,
    AGENTICA_HOME,
    AGENTICA_LOG_FILE,
    AGENTICA_LOG_LEVEL,
    AGENTICA_MAX_MEMORY_CHARACTER_COUNT,
    AGENTICA_NUM_HISTORY_TURNS,
    AGENTICA_PROJECTS_DIR,
    AGENTICA_SKILL_DIR,
    AGENTICA_WORKSPACE_DIR,
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TIMEOUT,
    apply_global_config,
    get_setting,
)

__all__ = [
    "AGENTICA_CACHE_DIR",
    "AGENTICA_CRON_DIR",
    "AGENTICA_DOTENV_PATH",
    "AGENTICA_EXTRA_SKILL_PATHS",
    "AGENTICA_HOME",
    "AGENTICA_LOG_FILE",
    "AGENTICA_LOG_LEVEL",
    "AGENTICA_MAX_MEMORY_CHARACTER_COUNT",
    "AGENTICA_NUM_HISTORY_TURNS",
    "AGENTICA_PROJECTS_DIR",
    "AGENTICA_SKILL_DIR",
    "AGENTICA_WORKSPACE_DIR",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_TIMEOUT",
    "apply_global_config",
    "get_setting",
]
