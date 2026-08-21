# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Helper functions for tool response serialization.

Eliminates boilerplate json.dumps across all tool implementations.
"""
import json
from typing import Any, Dict, List, Optional


class ToolDisplayOutput(str):
    """A tool result string that also carries display-only metadata.

    Write tools know their own before/after content at execution time. A
    renderer that instead re-reads the file after the fact cannot reconstruct
    per-call diffs, because a batch of writes to one file has already reached
    its final state by the time the first result is rendered.

    Subclassing ``str`` keeps this transparent to every consumer that treats a
    tool result as text (LLM payload, logging, persistence, ``startswith``
    soft-error checks); only the presentation layer looks at ``display_meta``.
    """

    display_meta: Dict[str, Any]

    def __new__(cls, value: str, display_meta: Dict[str, Any]) -> "ToolDisplayOutput":
        obj = super().__new__(cls, value)
        obj.display_meta = display_meta
        return obj


def file_change_meta(
        path: str,
        action: str,
        before: Optional[str],
        after: Optional[str],
) -> Dict[str, Any]:
    """Describe one file change for display-only rendering."""
    return {"path": path, "action": action, "before": before, "after": after}


def file_display_meta(changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap one or more file changes as a write tool's display metadata.

    A single envelope shape covers both write tools: ``write_file`` reports
    one file, ``apply_patch`` reports every file it mutated atomically.
    """
    return {"files": changes}


def tool_error(message: str, **extra) -> str:
    """Return a JSON error string for tool handlers.

    Examples:
        >>> tool_error("file not found")
        '{"error": "file not found"}'
        >>> tool_error("bad input", success=False)
        '{"error": "bad input", "success": false}'
    """
    result: dict[str, Any] = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data: dict | None = None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg OR keyword arguments (not both):

    Examples:
        >>> tool_result(success=True, count=42)
        '{"success": true, "count": 42}'
        >>> tool_result({"key": "value"})
        '{"key": "value"}'
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(kwargs, ensure_ascii=False, indent=2)
