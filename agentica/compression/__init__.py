# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Compression module for managing context compression.
"""
from agentica.compression.evict import (
    evict_context,
    evict_tool_results,
    shrink_tool_call_arguments,
    carries_tool_results,
    tool_result_blocks,
    is_irreducible_prompt_too_long,
    EvictionResult,
    EVICT_THRESHOLD_RATIO,
    EVICT_TARGET_RATIO,
)
from agentica.compression.manager import CompressionManager
from agentica.compression.tool_pairs import sanitize_tool_pairs
from agentica.compression.tool_result_storage import (
    maybe_persist_result,
    enforce_tool_batch_budget,
    can_recover_spill,
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    TOOL_BATCH_BUDGET_RATIO,
)
from agentica.compression.tool_result_classification import (
    ToolResultClass,
    classify_tool_result,
    describe_media,
)

__all__ = [
    "CompressionManager",
    "evict_context",
    "evict_tool_results",
    "shrink_tool_call_arguments",
    "carries_tool_results",
    "tool_result_blocks",
    "is_irreducible_prompt_too_long",
    "EvictionResult",
    "sanitize_tool_pairs",
    "EVICT_THRESHOLD_RATIO",
    "EVICT_TARGET_RATIO",
    "maybe_persist_result",
    "enforce_tool_batch_budget",
    "can_recover_spill",
    "DEFAULT_MAX_RESULT_SIZE_CHARS",
    "TOOL_BATCH_BUDGET_RATIO",
    "ToolResultClass",
    "classify_tool_result",
    "describe_media",
]
