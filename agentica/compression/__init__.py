# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Compression module for managing context compression.
"""
from agentica.compression.manager import CompressionManager
from agentica.compression.tool_result_storage import (
    maybe_persist_result,
    enforce_tool_result_budget,
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
)
from agentica.compression.tool_result_classification import (
    ToolResultClass,
    classify_tool_result,
    describe_media,
)

__all__ = [
    "CompressionManager",
    "maybe_persist_result",
    "enforce_tool_result_budget",
    "DEFAULT_MAX_RESULT_SIZE_CHARS",
    "MAX_TOOL_RESULTS_PER_MESSAGE_CHARS",
    "ToolResultClass",
    "classify_tool_result",
    "describe_media",
]
