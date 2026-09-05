# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tool-call line formatting for the CLI transcript
"""

import os
import re
import textwrap
from pathlib import Path
from typing import List, Optional

from rich.markup import escape
from rich.text import Text

from agentica.cli.runtime import TOOL_ICONS

from .console import remember_truncated

def _format_line_range(offset: int, limit: int) -> str:
    """Format line range as L{start}-{end}."""
    if offset < 0:
        keep = abs(offset)
        take = limit or keep
        return f"last {keep}" if take >= keep else f"oldest {take} of last {keep}"
    start = offset + 1 if offset else 1
    end = start + (limit or 500) - 1
    return f"L{start}-{end}"


def _shorten_path(file_path: str, work_dir: Optional[Path] = None) -> str:
    """Work-dir relative path, or the path as written when outside it.

    Normalisation is lexical (``normpath``, not ``resolve``): a ``..`` segment
    should collapse, but following symlinks would rewrite the path the caller
    typed (on macOS ``/tmp`` becomes ``/private/tmp``). ``~`` is kept as ``~``
    for the same reason — expanding it is longer than what was written and
    puts the user's name on screen.
    """
    if not file_path or file_path == ".":
        return "."
    written = str(file_path)
    p = Path(written).expanduser()
    root = Path(work_dir).expanduser() if work_dir is not None else Path.cwd()
    try:
        candidate = p if p.is_absolute() else root / p
        lexical = Path(os.path.normpath(candidate))
        return lexical.relative_to(Path(os.path.normpath(root))).as_posix()
    except ValueError:
        if written.startswith("~"):
            return written
        return Path(os.path.normpath(p)).as_posix() if p.is_absolute() else written


_PATCH_FILE_RE = re.compile(
    r"^\*\*\*\s*(?:Add|Update|Delete)\s+File:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def patch_file_paths(patch: str) -> List[str]:
    """File paths named in an apply_patch envelope, in order."""
    return [m.group(1).strip() for m in _PATCH_FILE_RE.finditer(patch or "") if m.group(1).strip()]


def _shorten_paths_in_command(command: str) -> str:
    """Shorten absolute paths embedded in a shell command."""
    cwd = str(Path.cwd())
    if cwd in command:
        command = command.replace(cwd + "/", "").replace(cwd, ".")
    return command


def _wrap_command_lines(command: str, width: int) -> List[str]:
    """Wrap a shell command for display while preserving explicit newlines.

    Long indivisible tokens (URLs, IDs, paths) stay intact instead of being
    split into misleading fragments. The full command is retained separately
    for Ctrl+O whenever the inline preview is folded.
    """
    wrapped: List[str] = []
    logical_lines = command.splitlines() or [""]
    for line in logical_lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(
            line,
            width=max(1, width),
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""])
    return wrapped


def format_execute_expand(command: str, output: str = "") -> str:
    """Plain-text pager view: the launch command, then the full result.

    Matches the transcript's shape closely enough to copy from — ``$`` for
    the command, ``⎿`` for the output — without Rich markup (the pager is
    ``less``).
    """
    parts = ["execute"]
    cmd_lines = str(command or "").splitlines() or ([""] if command else [])
    if cmd_lines:
        parts.append(f"$ {cmd_lines[0]}")
        parts.extend(f"  {line}" for line in cmd_lines[1:])
    if output:
        out_lines = str(output).splitlines() or [""]
        parts.append("")
        parts.append(f"    ⎿ {out_lines[0]}")
        parts.extend(f"      {line}" for line in out_lines[1:])
    return "\n".join(parts)


def _display_execute_command(
    console_instance, command: str, *, full: bool = False,
    tool_call_id: Optional[str] = None,
) -> None:
    """Render an execute command.

    Foreground calls keep a three-line preview (full text via Ctrl+O).
    Background calls show every wrapped line — the command is the identity of
    the detached job and must not hide behind a fold.
    """
    raw_command = str(command or "")
    display_command = _shorten_paths_in_command(raw_command)
    icon = TOOL_ICONS.get("execute", TOOL_ICONS["default"])
    header = f" {icon} execute "
    continuation = "   │ "
    width = max(1, int(getattr(console_instance, "width", 80) or 80) - len(header))
    command_lines = _wrap_command_lines(display_command, width)
    if full:
        visible_lines = command_lines
        omitted = 0
    else:
        visible_lines = command_lines[:3]
        omitted = len(command_lines) - len(visible_lines)

    for index, line in enumerate(visible_lines):
        rendered = Text()
        if index == 0:
            rendered.append(f" {icon} ")
            rendered.append("execute", style="bold magenta")
            rendered.append(" ")
        else:
            rendered.append(continuation, style="dim")
        rendered.append(line, style="dim")
        console_instance.print(rendered)

    if omitted > 0:
        hint = Text()
        hint.append(continuation, style="dim")
        hint.append(f"… +{omitted} lines (Ctrl+O to expand)", style="dim italic")
        console_instance.print(hint)
        remember_truncated(
            "execute",
            format_execute_expand(raw_command),
            key=f"execute:{tool_call_id}" if tool_call_id else None,
        )


def _format_handoff_display(
    tool_args: dict,
    *,
    body_key: str,
    meta_keys: tuple,
) -> str:
    """Format a work-handoff tool (task / delegate) with no truncation.

    Meta args stay on the first line as ``key=value``; the instruction body
    follows on its own lines so newlines in the brief stay readable.
    """
    body = str(tool_args.get(body_key, "") or "")
    meta: List[str] = []
    for key in meta_keys:
        value = tool_args.get(key)
        if value is None or value == "":
            continue
        meta.append(f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}")
    if body:
        indented = "\n    ".join(body.splitlines() or [""])
        if meta:
            return ", ".join(meta) + "\n    " + indented
        return indented
    return ", ".join(meta)


def format_tool_display(tool_name: str, tool_args: dict, work_dir: Optional[Path] = None) -> str:
    """Format tool call for user-friendly display."""
    # File reading tools - show filename and line range
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        filename = _shorten_path(file_path, work_dir)
        raw_tail = tool_args.get("tail")
        try:
            n_tail = int(raw_tail) if raw_tail not in (None, "") else 0
        except (TypeError, ValueError):
            n_tail = 0
        if n_tail:
            return f"{filename} (tail {abs(n_tail)})"
        offset = tool_args.get("offset", 0)
        limit = tool_args.get("limit", 500)
        line_range = _format_line_range(offset, limit)
        return f"{filename} ({line_range})"
    
    # File writing tools — same relative path as read_file, not basename.
    if tool_name == "write_file":
        file_path = str(tool_args.get("file_path", "") or "")
        return _shorten_path(file_path, work_dir) if file_path else ""

    if tool_name == "apply_patch":
        paths = [_shorten_path(p, work_dir) for p in patch_file_paths(str(tool_args.get("patch", "") or ""))]
        return ", ".join(paths)
    
    # Execute command - shorten absolute paths in command
    if tool_name == "execute":
        command = tool_args.get("command", "")
        return _shorten_paths_in_command(command)
    
    # Todo tools - list the todo items (show ALL todos, no truncation)
    if tool_name == "write_todos":
        todos = tool_args.get("todos", [])
        if isinstance(todos, list) and todos:
            todo_lines = []
            for todo in todos:
                if isinstance(todo, dict):
                    content = todo.get("content", "")
                    status = todo.get("status", "pending")
                    status_icon = "✓" if status == "completed" else "○" if status == "pending" else "◐"
                    todo_lines.append(f"{status_icon} {content}")
                else:
                    todo_lines.append(f"○ {str(todo)}")
            return "\n    ".join(todo_lines)
        return f"{len(todos)} items"
    
    # Web search - show search queries
    if tool_name == "web_search":
        queries = tool_args.get("queries", "")
        if isinstance(queries, list):
            return ", ".join(str(q)[:40] for q in queries[:3])
        return str(queries)[:80]
    
    # Fetch URL - show the URL
    if tool_name == "fetch_url":
        url = tool_args.get("url", "")
        if len(url) > 60:
            return url[:57] + "..."
        return url
    
    # glob/grep - show shortened path/pattern
    if tool_name == "glob":
        pattern = tool_args.get("pattern", "*")
        path = tool_args.get("path", ".")
        return f"{pattern} in {_shorten_path(path, work_dir)}"

    if tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", ".")
        return f"'{pattern[:40]}' in {_shorten_path(path, work_dir)}"
    
    # task / delegate — these hand off work; truncating the brief hides what
    # the user needs to audit. Show every arg in full (multi-line body below).
    if tool_name == "task":
        return _format_handoff_display(
            tool_args,
            body_key="description",
            meta_keys=("subagent_type", "timeout", "max_turns", "resume_from_run_id"),
        )

    if tool_name == "delegate":
        return _format_handoff_display(
            tool_args,
            body_key="task",
            meta_keys=("label", "work_dir", "model"),
        )

    # Peer messaging — show the full body; truncating here hides the handoff.
    if tool_name == "send_message":
        target = str(tool_args.get("target", "") or "")
        message = str(tool_args.get("message", "") or "")
        if not message:
            return f"→ {target}" if target else ""
        return f"→ {target}\n    {message}" if target else message

    if tool_name == "list_agents":
        return ""

    # Human-in-the-loop: the question and its options are rendered in full by
    # the TUI's prompt widget the moment this call parks. Repeating a clipped
    # copy on the call line shows the same question twice, neither in full.
    # The lasting record is the result block, which replays both sides.
    if tool_name == "ask_user_question":
        return ""

    # Default format for other tools
    brief_args = []
    for key, value in tool_args.items():
        if isinstance(value, str):
            if len(value) > 40:
                value = value[:37] + "..."
            brief_args.append(f"{key}={value!r}")
        elif isinstance(value, (int, float, bool)):
            brief_args.append(f"{key}={value}")
        elif isinstance(value, list):
            brief_args.append(f"{key}=[{len(value)} items]")
        elif isinstance(value, dict):
            brief_args.append(f"{key}={{...}}")
    
    args_str = ", ".join(brief_args[:3])
    if len(brief_args) > 3:
        args_str += ", ..."
    
    return args_str if args_str else ""


def _display_tool_impl(console_instance, tool_name: str, tool_args: dict,
                       tool_count: int = 0, tool_call_id: Optional[str] = None,
                       work_dir: Optional[Path] = None) -> None:
    """Shared implementation for displaying a tool call."""
    icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
    display_str = format_tool_display(tool_name, tool_args, work_dir=work_dir)

    # Add blank line between tools for readability
    if tool_count > 1:
        console_instance.print()

    if tool_name == "execute":
        bg = tool_args.get("background")
        full = bg is True or bg == "true" or bg == 1 or bg == "1"
        _display_execute_command(
            console_instance, tool_args.get("command", ""), full=full,
            tool_call_id=tool_call_id,
        )
    # Special handling for write_todos - multi-line display.
    # Note: in this repo "task" is the dedicated subagent-spawn tool, so we
    # avoid using "tasks" as the label here to prevent confusion.
    elif tool_name == "write_todos" and "\n" in display_str:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta]:")
        console_instance.print(f"    {display_str}", style="dim", highlight=False, markup=False)
    elif tool_name in ("send_message", "task", "delegate") and "\n" in display_str:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta]")
        for line in display_str.splitlines():
            console_instance.print(f"    {line}", style="dim", highlight=False, markup=False)
    elif display_str:
        console_instance.print(
            f" {icon} [bold magenta]{tool_name}[/bold magenta] [dim]{escape(display_str)}[/dim]"
        )
    else:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta]")
