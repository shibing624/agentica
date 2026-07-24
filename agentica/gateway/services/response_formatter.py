# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Response formatting utilities for gateway API responses.

Extracted from AgentService — these are pure functions with no state,
making them easy to test and reuse across different response paths
(SSE streaming, WebSocket, non-streaming chat).
"""
import json
from typing import Any, Dict, Optional

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


def format_tool_call_args(tool_name: str, tool_args: dict) -> dict:
    """Format tool call arguments for frontend display.

    For file-editing tools (edit_file, multi_edit_file, write_file),
    computes diff metadata (lines added/deleted) instead of sending
    the full content to the frontend. For other tools, truncates long
    string arguments to 100 characters.

    Args:
        tool_name: The name of the tool being called.
        tool_args: The raw arguments dict from the tool call.

    Returns:
        A display-friendly arguments dict with metadata fields
        prefixed with ``_`` (e.g. ``_diff_add``, ``_diff_del``).
    """
    display_args: dict = {}

    if tool_name == "edit_file":
        old_s = tool_args.get("old_string", "")
        new_s = tool_args.get("new_string", "")
        display_args["_diff_add"] = new_s.count("\n") + (1 if new_s else 0)
        display_args["_diff_del"] = old_s.count("\n") + (1 if old_s else 0)
        fp = tool_args.get("file_path") or tool_args.get("file") or tool_args.get("path", "")
        if fp:
            display_args["file_path"] = fp

    elif tool_name == "multi_edit_file":
        edits = tool_args.get("edits", [])
        total_add = total_del = 0
        for ed in (edits if isinstance(edits, list) else []):
            old_s = ed.get("old_string", "")
            new_s = ed.get("new_string", "")
            total_del += old_s.count("\n") + (1 if old_s else 0)
            total_add += new_s.count("\n") + (1 if new_s else 0)
        display_args["_diff_add"] = total_add
        display_args["_diff_del"] = total_del
        display_args["_edit_count"] = len(edits) if isinstance(edits, list) else 0
        fp = tool_args.get("file_path") or tool_args.get("file") or tool_args.get("path", "")
        if fp:
            display_args["file_path"] = fp

    elif tool_name == "write_file":
        content = tool_args.get("content", "")
        display_args["_lines"] = content.count("\n") + (1 if content else 0)
        fp = tool_args.get("file_path") or tool_args.get("file") or tool_args.get("path", "")
        if fp:
            display_args["file_path"] = fp

    else:
        for k, v in tool_args.items():
            if isinstance(v, str) and len(v) > 100:
                display_args[k] = v[:100] + "..."
            else:
                display_args[k] = v

    return display_args


def format_tool_result(tool_call: ToolCallInfo) -> tuple[str, str, bool]:
    """Format a tool result for frontend display.

    For the ``task`` tool (subagent execution), parses the JSON result
    and produces structured metadata. For all other tools, truncates
    the result to 500 characters.

    Args:
        tool_call: The completed tool call, as carried by
                   ``RunResponse.tool_call`` on a ToolCallCompleted event.

    Returns:
        A tuple of (tool_name, result_string, is_task_meta).
        ``is_task_meta`` is True when the result is structured JSON
        from a subagent task execution.
    """
    t_name = tool_call.tool_name or "unknown"
    t_content = tool_call.content or ""
    is_error = tool_call.is_error

    # task tool: parse subagent JSON and produce structured metadata
    if t_name == "task" and t_content:
        try:
            task_data = json.loads(str(t_content))
            task_meta = {
                "_task_meta": True,
                "success": task_data.get("success", False),
                "tool_calls_summary": task_data.get("tool_calls_summary", []),
                "execution_time": task_data.get("execution_time"),
                "tool_count": task_data.get("tool_count", 0),
            }
            if not task_data.get("success"):
                task_meta["error"] = task_data.get("error", "Unknown error")
            return t_name, json.dumps(task_meta, ensure_ascii=False), True
        except (ValueError, TypeError):
            pass

    if t_content:
        result_str = str(t_content)[:500] + ("..." if len(str(t_content)) > 500 else "")
    else:
        result_str = "(no output)"
    if is_error:
        result_str = "Error: " + result_str

    return t_name, result_str, False
