# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Response formatting utilities for gateway API responses.

Extracted from AgentService — these are pure functions with no state,
making them easy to test and reuse across different response paths
(SSE streaming, WebSocket, non-streaming chat).
"""
import difflib
import json
from typing import Any, Dict, List, Optional

from agentica.run_response import ToolCallInfo


def extract_metrics(agent: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics from the agent's last run_response.

    Args:
        agent: A DeepAgent instance (or None).

    Returns:
        The metrics dict from the agent's last run, or None.
    """
    if not agent:
        return None
    if agent.run_response and agent.run_response.metrics:
        return agent.run_response.metrics
    return None


# Local file-read tools: the call line is enough. Writes, search, memory,
# execute, task, and everything else send the body. Errors still pass through.
_HIDE_RESULT_TOOLS = frozenset({"read_file", "glob", "grep"})
# Only runaway payloads are clipped; execute / wait / peers stay intact below this.
_MAX_TOOL_RESULT_CHARS = 100_000


def _clip_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TOOL_RESULT_CHARS:
        extra = len(value) - _MAX_TOOL_RESULT_CHARS
        return value[:_MAX_TOOL_RESULT_CHARS] + f"\n... ({extra} more chars)"
    if isinstance(value, dict):
        return {k: _clip_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_value(v) for v in value]
    return value


def format_tool_call_args(tool_name: str, tool_args: dict) -> dict:
    """Format tool call arguments for frontend display.

    ``read_file`` / ``glob`` / ``grep`` keep a short one-liner (100 chars).
    Every other tool sends the full arguments so the web row can show
    input and result together.
    """
    raw = dict(tool_args or {})
    if tool_name in _HIDE_RESULT_TOOLS:
        display_args: dict = {}
        for k, v in raw.items():
            if isinstance(v, str) and len(v) > 100:
                display_args[k] = v[:100] + "..."
            else:
                display_args[k] = v
        return display_args
    return _clip_value(raw)


def _format_task_result(raw: str) -> str:
    """Turn the task tool's JSON payload into a full readable recap.

    The old web path kept only ``tool_count`` / ``tool_calls_summary`` and
    dropped ``result`` (the subagent's actual answer). Show every inner
    call plus the full answer.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        return raw
    name = data.get("subagent_name") or data.get("subagent_type") or "task"
    kind = data.get("subagent_type") or ""
    success = data.get("success")
    status = data.get("status") or ("ok" if success else "failed")
    tools = data.get("tool_calls_summary") or []
    n = data.get("tool_count") or len(tools)
    elapsed = data.get("execution_time")
    if elapsed is None:
        elapsed = data.get("elapsed_seconds")
    head = f"{name} ({kind})" if kind and kind != name else str(name)
    bits = [head, str(status)]
    if n:
        bits.append(f"{n} tools")
    if elapsed is not None:
        bits.append(f"{float(elapsed):.1f}s")
    lines = [" · ".join(bits)]
    if data.get("error"):
        lines.append(f"error: {data['error']}")
    if tools:
        lines.append("")
        for tc in tools:
            if not isinstance(tc, dict):
                lines.append(str(tc))
                continue
            tname = tc.get("name") or "tool"
            info = tc.get("info") or tc.get("input") or ""
            mark = " [error]" if tc.get("is_error") else ""
            lines.append(f"{tname} {info}{mark}".rstrip())
    body = data.get("result") or data.get("content") or ""
    if body:
        lines.append("")
        lines.append(str(body).rstrip())
    if data.get("next_action"):
        lines.append("")
        lines.append(str(data["next_action"]).rstrip())
    return "\n".join(lines).rstrip() + "\n"


def files_unified_diff(files: List[Dict[str, Any]]) -> str:
    """CLI-style unified diffs from write_file / apply_patch display_meta.

    Same shape as ``cli/display/stream.py`` ``_build_file_diff``: one
    ``diff -- path`` block per file, context of 2. Missing before on an
    add (or after on a delete) is treated as empty so a new file still
    renders as a full ``+`` dump.
    """
    parts: List[str] = []
    for change in files:
        path = str(change.get("path") or "")
        action = change.get("action") or "update"
        old_content = change.get("before")
        new_content = change.get("after")
        if action == "add" and old_content is None:
            old_content = ""
        if action == "delete" and new_content is None:
            new_content = ""
        if old_content is None or new_content is None:
            continue
        unified_lines = list(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=path,
            tofile=path,
            n=2,
            lineterm="",
        ))
        if not unified_lines:
            continue
        hunks = "\n".join(unified_lines[2:]).rstrip("\n")
        parts.append(f"diff -- {path}\n{hunks}")
    return "\n\n".join(parts)


def format_tool_result(tool_call: ToolCallInfo) -> tuple[str, str, dict]:
    """Format a tool result for frontend display.

    ``read_file`` / ``glob`` / ``grep`` send an empty body (the call line
    is enough). ``task`` is expanded into the inner tool list plus the
    subagent's full answer. ``write_file`` / ``apply_patch`` also send a
    unified ``diff`` in the extra dict when ``tool_display_meta.files``
    is present. Everything else is sent in full; only payloads past
    ``_MAX_TOOL_RESULT_CHARS`` are clipped.

    Returns:
        ``(tool_name, result_string, extra)``. ``extra`` may contain
        ``tool_call_id`` (so the web row can match a parallel batch) and
        ``diff`` (unified text for the web row).
    """
    t_name = tool_call.tool_name or "unknown"
    t_content = tool_call.content or ""
    is_error = tool_call.is_error
    extra: Dict[str, Any] = {}
    if tool_call.tool_call_id:
        extra["tool_call_id"] = tool_call.tool_call_id

    meta = tool_call.tool_display_meta or {}
    files = meta.get("files") if isinstance(meta, dict) else None
    if files:
        diff = files_unified_diff(files)
        if diff:
            if len(diff) > _MAX_TOOL_RESULT_CHARS:
                extra["diff"] = (
                    diff[:_MAX_TOOL_RESULT_CHARS]
                    + f"\n... ({len(diff) - _MAX_TOOL_RESULT_CHARS} more chars)"
                )
            else:
                extra["diff"] = diff

    if t_name == "task" and t_content:
        try:
            t_content = _format_task_result(str(t_content))
        except (ValueError, TypeError):
            pass

    if t_name in _HIDE_RESULT_TOOLS and not is_error:
        return t_name, "", extra

    if t_content:
        raw = str(t_content)
        if len(raw) > _MAX_TOOL_RESULT_CHARS:
            result_str = (
                raw[:_MAX_TOOL_RESULT_CHARS]
                + f"\n... ({len(raw) - _MAX_TOOL_RESULT_CHARS} more chars)"
            )
        else:
            result_str = raw
    else:
        result_str = "(no output)"
    if is_error:
        result_str = "Error: " + result_str

    return t_name, result_str, extra
