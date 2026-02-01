# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Utils module with lazy loading for heavy dependencies.
"""
import importlib
from typing import TYPE_CHECKING

# Lazy imports mapping
_LAZY_UTILS_IMPORTS = {
    "count_tokens": "agentica.utils.tokens",
    "count_text_tokens": "agentica.utils.tokens",
    "count_image_tokens": "agentica.utils.tokens",
    "count_message_tokens": "agentica.utils.tokens",
    "count_tool_tokens": "agentica.utils.tokens",
}

_UTILS_CACHE = {}


def __getattr__(name: str):
    """Lazy import handler for heavy utils."""
    if name in _LAZY_UTILS_IMPORTS:
        if name not in _UTILS_CACHE:
            module = importlib.import_module(_LAZY_UTILS_IMPORTS[name])
            _UTILS_CACHE[name] = getattr(module, name)
        return _UTILS_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Type hints for IDE support
if TYPE_CHECKING:
    from agentica.utils.tokens import (
        count_tokens,
        count_text_tokens,
        count_image_tokens,
        count_message_tokens,
        count_tool_tokens,
    )
