# -*- coding: utf-8 -*-
"""The eviction marker for tool-call arguments.

Layer 1 no longer writes it (arguments are never rewritten), but session logs
from before still carry it, and a model that saw it may emit it. These helpers
recognise both forms so ``get_function_call`` can refuse such a call.
"""

import json
from typing import Any, Optional

# Structured stand-in for one dropped string leaf. A JSON object, not a
# string: a string in this slot is an ``execute`` /
# ``write_file`` payload, and the model copies it into the next call.
OMITTED_ARG_KEY = "$evicted"

# String we used to write. Still in live JSONL; rewrite it so the model
# cannot copy it as content.
_LEGACY_OMITTED_PREFIX = "<evicted-tool-arg chars="


def omitted_tool_arg(n: int) -> dict:
    """Placeholder for a string leaf dropped from context."""
    return {OMITTED_ARG_KEY: int(n)}


def _legacy_omitted_chars(value: str) -> Optional[int]:
    """Char count if ``value`` is exactly the string marker we used to write.

    The model sometimes re-emits it without the closing ``>``. A command
    that only mentions the prefix has other text and returns None.
    """
    if not value.startswith(_LEGACY_OMITTED_PREFIX):
        return None
    rest = value[len(_LEGACY_OMITTED_PREFIX):]
    if rest.endswith(">"):
        rest = rest[:-1]
    if rest.isdecimal():
        return int(rest)
    return None


def _is_omitted_object(value: Any) -> bool:
    if not isinstance(value, dict) or len(value) != 1 or OMITTED_ARG_KEY not in value:
        return False
    n = value[OMITTED_ARG_KEY]
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def is_omitted_tool_arg(value: Any) -> bool:
    """True when ``value`` is the eviction placeholder for one argument."""
    if _is_omitted_object(value):
        return True
    if not isinstance(value, str):
        return False
    if _legacy_omitted_chars(value) is not None:
        return True
    if not value.startswith("{") or OMITTED_ARG_KEY not in value:
        return False
    try:
        parsed = json.loads(value)
    except ValueError:
        return False
    return _is_omitted_object(parsed)


def omitted_tool_args(value: Any) -> list:
    """Key paths under ``value`` whose leaf is an omitted argument."""
    found: list = []

    def walk(node: Any, path: str) -> None:
        if is_omitted_tool_arg(node):
            found.append(path)
            return
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(value, "")
    return found


def rewrite_omitted_arg_leaves(value: Any) -> Any:
    """Upgrade the old string marker to the structured object. Idempotent."""
    if isinstance(value, str):
        n = _legacy_omitted_chars(value)
        return omitted_tool_arg(n) if n is not None else value
    if isinstance(value, dict):
        if _is_omitted_object(value):
            return value
        return {key: rewrite_omitted_arg_leaves(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_omitted_arg_leaves(item) for item in value]
    return value
