# -*- coding: utf-8 -*-
"""Helpers for shrinking tool-call argument JSON without breaking validity."""

import json
from typing import Any


def omitted_tool_arg(n: int) -> str:
    """Placeholder for a string leaf dropped from context.

    Must not be a prefix of the original payload. ``head + "...[truncated]"``
    looks like real ``write_file`` / ``apply_patch`` content, and the model
    copies it into the next write.
    """
    return f"<evicted-tool-arg chars={n}>"


def shrink_tool_arg_leaves(value: Any, max_string_chars: int) -> Any:
    """Replace oversize string leaves with ``omitted_tool_arg``, keeping JSON shape."""
    if isinstance(value, str):
        if len(value) > max_string_chars:
            return omitted_tool_arg(len(value))
        return value
    if isinstance(value, dict):
        return {key: shrink_tool_arg_leaves(item, max_string_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [shrink_tool_arg_leaves(item, max_string_chars) for item in value]
    return value


def shrink_tool_call_arguments_json(arguments: str, max_string_chars: int = 200) -> str:
    """Shrink long string leaves inside tool-call arguments while preserving JSON."""
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments

    shrunken = shrink_tool_arg_leaves(parsed, max_string_chars)
    return json.dumps(shrunken, ensure_ascii=False)
