# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified database module for agentica

Database implementations are lazy-loaded to improve import speed.
Only base types are imported eagerly.
"""
import importlib
from typing import TYPE_CHECKING

# Base types (fast import, no heavy dependencies)
from agentica.db.base import (
    BaseDb,
    SessionRow,
    MemoryRow,
    MetricsRow,
    KnowledgeRow,
    filter_base64_images,
    filter_base64_media,
    clean_media_placeholders,
    BASE64_PLACEHOLDER,
)

# Lazy imports mapping
_LAZY_DB_IMPORTS = {
    "SqliteDb": "agentica.db.sqlite",
    "PostgresDb": "agentica.db.postgres",
    "InMemoryDb": "agentica.db.memory",
    "JsonDb": "agentica.db.json",
    "MysqlDb": "agentica.db.mysql",
    "RedisDb": "agentica.db.redis",
}

_DB_CACHE = {}


def __getattr__(name: str):
    """Lazy import handler for database implementations."""
    if name in _LAZY_DB_IMPORTS:
        if name not in _DB_CACHE:
            module = importlib.import_module(_LAZY_DB_IMPORTS[name])
            _DB_CACHE[name] = getattr(module, name)
        return _DB_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all available names including lazy imports."""
    eager_names = [name for name in globals() if not name.startswith('_')]
    return sorted(set(eager_names) | set(_LAZY_DB_IMPORTS.keys()))


# Type hints for IDE support
if TYPE_CHECKING:
    from agentica.db.sqlite import SqliteDb
    from agentica.db.postgres import PostgresDb
    from agentica.db.memory import InMemoryDb
    from agentica.db.json import JsonDb
    from agentica.db.mysql import MysqlDb
    from agentica.db.redis import RedisDb


__all__ = [
    # Base types
    "BaseDb",
    "SessionRow",
    "MemoryRow",
    "MetricsRow",
    "KnowledgeRow",
    "filter_base64_images",
    "filter_base64_media",
    "clean_media_placeholders",
    "BASE64_PLACEHOLDER",
    # Lazy loaded implementations
    "SqliteDb",
    "PostgresDb",
    "InMemoryDb",
    "JsonDb",
    "MysqlDb",
    "RedisDb",
]
