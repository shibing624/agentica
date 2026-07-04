# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI display utilities - colors, formatting, stream display manager
"""
import difflib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from agentica.cli.config import get_console, TOOL_ICONS, BUILTIN_TOOLS


# Rich console color scheme (unified - no separate ANSI codes)
COLORS = {
    "user": "bright_cyan",
    "agent": "bright_green",
    "thinking": "yellow",
    "tool": "cyan",
    "error": "red",
}


def print_header(model_provider: str, model_name: str, work_dir: Optional[str] = None,
                 extra_tools: Optional[List[str]] = None, shell_mode: bool = False):
    """Print the application header with version and model information"""
    box_width = min(get_console().width, 80)
    get_console().print("=" * box_width, style="bright_cyan")
    get_console().print("  Agentica CLI - Interactive AI Assistant")
    get_console().print(f"  Model: [bright_green]{model_provider}/{model_name}[/bright_green]")

    # Working directory
    cwd = work_dir or os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    if len(cwd) > 50:
        cwd = "..." + cwd[-47:]
    get_console().print(f"  Working Directory: {cwd}")

    # Built-in tools (always shown)
    get_console().print(f"  Built-in Tools: [white]{', '.join(BUILTIN_TOOLS)}[/white]")

    # Extra tools info
    if extra_tools:
        tools_str = ", ".join(extra_tools)
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        get_console().print(f"  Extra Tools: [bright_green]{tools_str}[/bright_green]")

    # Log file location (helps users find logs when debugging)
    from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL
    if AGENTICA_LOG_FILE:
        log_path = AGENTICA_LOG_FILE
        if log_path.startswith(home):
            log_path = "~" + log_path[len(home):]
        get_console().print(f"  Log File ({AGENTICA_LOG_LEVEL}): [white]{log_path}[/white]")

    get_console().print("=" * box_width, style="bright_cyan")
    get_console().print()
    # Keyboard shortcuts
    get_console().print("  [bright_green]Enter[/bright_green]       Submit your message")
    get_console().print("  [bright_green]Ctrl+X[/bright_green]      Toggle Agent/Shell mode")
    get_console().print("  [bright_green]Ctrl+J[/bright_green]      Insert newline (Alt+Enter also works)")
    get_console().print("  [bright_green]Ctrl+D[/bright_green]      Exit")
    get_console().print("  [bright_green]Ctrl+C[/bright_green]      Interrupt current operation")
    get_console().print("  [bright_green]Alt+V[/bright_green]       Paste image from clipboard")
    get_console().print("  [bright_green]Ctrl+o[/bright_green]      Expand all truncated inputs/tool outputs in pager (Ctrl+o or Esc to return)")
    get_console().print()
    # Input features
    get_console().print("  [bright_green]@filename[/bright_green]   Type @ to auto-complete files and inject content")
    get_console().print("  [bright_green]/paste[/bright_green]      Paste image from clipboard")
    get_console().print("  [bright_green]/image[/bright_green]      Attach local image: /image <path>")
    get_console().print("  [bright_green]/command[/bright_green]    Type / to see available commands (try /help)")
    get_console().print()


def parse_file_mentions(text: str) -> Tuple[str, List[Path]]:
    """Parse @file mentions and return text with mentioned files.
    
    Uses lookbehind to avoid matching email addresses.
    """
    pattern = r"(?:^|(?<=\s))@([\w./-]+)"
    mentioned_files = []
    
    for match in re.finditer(pattern, text):
        file_path_str = match.group(1)
        file_path = Path(file_path_str).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if file_path.exists() and file_path.is_file():
            mentioned_files.append(file_path)
    
    # Remove @ mentions from text for cleaner display
    processed_text = re.sub(pattern, r'\1', text)
    return processed_text, mentioned_files


def inject_file_contents(prompt_text: str, mentioned_files: List[Path]) -> str:
    """Inject file contents into the prompt."""
    if not mentioned_files:
        return prompt_text
    
    context_parts = [prompt_text, "\n\n## Referenced Files\n"]
    for file_path in mentioned_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            # Limit file content to reasonable size
            if len(content) > 20000:
                content = content[:20000] + "\n... (file truncated)"
            context_parts.append(
                f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
            )
        except Exception as e:
            context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")
    
    return "\n".join(context_parts)


_PASTE_PATH_RE = re.compile(r"@\S*[\\/]pastes[\\/]paste_\S+\.txt")


# All blocks (user input or tool output) truncated in the CLI display during
# the current run. Remembered so the user can expand them on demand: Ctrl+o
# opens EVERY folded block in one pager (CC-style "expand all"), instead of
# only the most recent one. Cleared at the start of each run.
_truncated_blocks: List[Dict[str, str]] = []


def remember_truncated(title: str, content: str) -> None:
    """Stash a truncated block for on-demand expansion (Ctrl+o opens all)."""
    if not content:
        return
    _truncated_blocks.append({"title": title, "content": content})


def get_last_truncated() -> Dict[str, str]:
    """Return a copy of the most recent truncated block (or empty)."""
    if not _truncated_blocks:
        return {"title": "", "content": ""}
    return dict(_truncated_blocks[-1])


def get_truncated_blocks() -> List[Dict[str, str]]:
    """Return all truncated blocks accumulated this run (newest last)."""
    return [dict(b) for b in _truncated_blocks]


def clear_truncated_blocks() -> None:
    """Drop all remembered truncated blocks (called at run start)."""
    _truncated_blocks.clear()


def display_user_message(text: str, *, pasted_blocks: int = 0, pasted_lines: int = 0) -> None:
    """Display user message with file mentions colored.

    For long pasted content, shows a trimmed preview with line count.
    """
    # A new user turn starts here: drop the previous turn's folded blocks so
    # Ctrl+o expands the CURRENT turn (this query + its tool results), not
    # stale history. The clear is done here rather than in
    # StreamDisplayManager.__init__ because display_user_message runs BEFORE
    # the manager is created — clearing in __init__ would wipe the
    # just-remembered user input, which is exactly why Ctrl+o used to show
    # only tool results and never the long query.
    clear_truncated_blocks()
    cleaned = _PASTE_PATH_RE.sub("", text).strip()
    if not cleaned and pasted_blocks:
        cleaned = f"[Pasted text: {pasted_lines} lines]"

    # Show the message nearly in full; only fold when it exceeds 12 lines so
    # typical multi-line questions and pastes stay visible without Ctrl+o.
    lines = cleaned.split('\n')
    if len(lines) > 12:
        # Show first 10 lines + summary
        preview_lines = lines[:10]
        preview = '\n'.join(preview_lines)
        remaining = len(lines) - 10
        rich_text = Text()
        rich_text.append(preview, style=COLORS["user"])
        rich_text.append(f"\n... (+{remaining} more lines · Ctrl+o 展开)", style="dim")
        remember_truncated("User input", cleaned)
    else:
        pattern = r"(@[\w./-]+)"
        parts = re.split(pattern, cleaned)
        rich_text = Text()
        for part in parts:
            if part.startswith("@"):
                rich_text.append(part, style="magenta")
            else:
                rich_text.append(part, style=COLORS["user"])

    if pasted_blocks:
        suffix = "s" if pasted_blocks > 1 else ""
        rich_text.append(
            f" ({pasted_blocks} pasted block{suffix}, {pasted_lines} lines total)",
            style="dim",
        )

    get_console().print(rich_text)


def get_file_completions(document_text: str) -> List[str]:
    """Get file completions for @ mentions."""
    import glob as glob_module

    # Find the @ mention being typed
    match = re.search(r"@([\w./-]*)$", document_text)
    if not match:
        return []
    
    partial = match.group(1)
    
    if partial:
        # Search for files matching the partial path (current dir only, not recursive)
        search_pattern = f"{partial}*"
        matches = glob_module.glob(search_pattern, recursive=False)
        # Also search one level of subdirectories (limited depth)
        if os.sep not in partial and "/" not in partial:
            for d in os.listdir("."):
                if os.path.isdir(d) and not d.startswith("."):
                    sub_matches = glob_module.glob(os.path.join(d, f"{partial}*"))
                    matches.extend(sub_matches[:5])
    else:
        # Show files in current directory
        matches = glob_module.glob("*")
    
    # Filter to only files (not directories) and limit results
    completions = []
    seen = set()
    for m in matches[:20]:
        if m in seen:
            continue
        seen.add(m)
        if os.path.isfile(m):
            completions.append(m)
        elif os.path.isdir(m):
            completions.append(m + "/")
    
    return completions


def show_help(skills_registry=None):
    """Display categorized help information."""
    categories = {
        "Session": {
            "/new":             "Start a new chat session",
            "/clear, /reset":   "Clear screen and reset conversation",
            "/resume [name]":   "Resume a previous session",
            "/history":         "Show conversation history",
            "/save, /export":   "Save conversation to JSON (no system prompts)",
            "/retry":           "Retry the last message (resend to agent)",
            "/undo":            "Remove the last user/assistant exchange",
            "/compact":         "Compact context (summarize history)",
            "/btw <question>":  "Ephemeral side question (no tools, not saved)",
            "/queue":           "Queue: <prompt> | list | clear | remove <n>",
            "/steer <text>":    "Guide the running agent mid-task (no interrupt)",
            "/checkpoint":      "Durable file snapshots: list | create | diff | restore",
            "/background":      "Run prompt in background (/bg alias)",
            "/stop":            "Kill all running background tasks",
        },
        "Configure": {
            "/model [p/m]":     "Show or switch model",
            "/config":          "Show current configuration",
            "/cost, /usage":    "Show detailed token usage and cost",
            "/debug":           "Show debug info (model, history count)",
            "/reasoning":       "Toggle reasoning display: on | off",
            "/statusbar, /sb":  "Toggle the status bar",
        },
        "Tools & Skills": {
            "/tools":           "Manage tools: add | remove | info | search",
            "/skills":          "Manage skills: search | browse | install | remove | inspect | tap",
        },
        "Permissions": {
            "/permissions":     "View or set mode (allow-all/auto/strict)",
            "/yolo":            "Toggle YOLO mode (auto-approve all)",
        },
        "Media": {
            "/paste":           "Paste image from clipboard",
            "/image <path>":    "Attach a local image file",
        },
        "Other": {
            "/help":            "Show this help message",
            "/exit, /quit":     "Exit the CLI",
        },
    }

    get_console().print()
    get_console().print("  [bold]Available Commands[/bold]")
    get_console().print()

    for category, commands in categories.items():
        get_console().print(f"  [bold]-- {category} --[/bold]")
        for cmd, desc in commands.items():
            get_console().print(f"    [bright_green]{cmd:<18}[/bright_green] [dim]{desc}[/dim]")
        get_console().print()

    # Skill auto-commands
    if skills_registry and len(skills_registry) > 0:
        skill_cmds = skills_registry.auto_commands()
        if skill_cmds:
            get_console().print("  [bold]-- Skill Commands --[/bold]")
            for slug, skill in skill_cmds.items():
                desc = skill.description[:50] if skill.description else ""
                get_console().print(f"    [bright_green]{slug:<18}[/bright_green] [dim]{desc}[/dim]")
            get_console().print()

    get_console().print("  [bold]Keyboard Shortcuts[/bold]")
    get_console().print()
    shortcuts = {
        "Enter":             "Submit your message",
        "Ctrl+X":            "Toggle Agent/Shell mode ($ = shell, > = agent)",
        "Ctrl+J, Alt+Enter": "Insert newline for multi-line input",
        "Ctrl+D":            "Exit",
        "Ctrl+C":            "Interrupt current operation",
        "Tab, Right Arrow":  "Accept completion / auto-suggestion",
        "Alt+V":             "Paste image from clipboard",
        "Ctrl+o":            "Expand all truncated inputs/tool outputs in pager",
    }
    for key, desc in shortcuts.items():
        get_console().print(f"    [bright_green]{key:<20}[/bright_green] [dim]{desc}[/dim]")
    get_console().print()

    get_console().print("  [bold]Input Features[/bold]")
    get_console().print()
    get_console().print("    [bright_green]@filename[/bright_green]           Reference a file - content injected into prompt")
    get_console().print("    [bright_green]/command[/bright_green]            Type / to see slash commands with auto-complete")
    get_console().print()
    get_console().print("  [dim]Tip: type your message and press Enter to chat![/dim]")
    get_console().print()


def _extract_filename(file_path: str) -> str:
    """Extract filename from a file path."""
    return Path(file_path).name


def _format_line_range(offset: int, limit: int) -> str:
    """Format line range as L{start}-{end}."""
    start = offset + 1 if offset else 1
    end = start + (limit or 500) - 1
    return f"L{start}-{end}"


def _shorten_path(file_path: str) -> str:
    """Shorten a file path for display: prefer relative path, fallback to filename."""
    if not file_path or file_path == ".":
        return "."
    p = Path(file_path)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return p.name


def _shorten_paths_in_command(command: str) -> str:
    """Shorten absolute paths embedded in a shell command."""
    cwd = str(Path.cwd())
    if cwd in command:
        command = command.replace(cwd + "/", "").replace(cwd, ".")
    return command


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool call for user-friendly display."""
    # File reading tools - show filename and line range
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        filename = _extract_filename(file_path)
        offset = tool_args.get("offset", 0)
        limit = tool_args.get("limit", 500)
        line_range = _format_line_range(offset, limit)
        return f"{filename} ({line_range})"
    
    # File writing tools - show filename only
    if tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)
    
    # File editing tools - show filename only
    if tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)

    # Multi-edit: filename only (edit count goes in the completion summary)
    if tool_name == "multi_edit_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)
    
    # Execute command - shorten absolute paths in command
    if tool_name == "execute":
        command = tool_args.get("command", "")
        command = _shorten_paths_in_command(command)
        if len(command) > 300:
            return command[:297] + "..."
        return command
    
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
    
    # ls/glob/grep - show shortened path/pattern
    if tool_name == "ls":
        directory = tool_args.get("directory", ".")
        return _shorten_path(directory)

    if tool_name == "glob":
        pattern = tool_args.get("pattern", "*")
        path = tool_args.get("path", ".")
        return f"{pattern} in {_shorten_path(path)}"

    if tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", ".")
        include = tool_args.get("include", "")
        display = f"'{pattern[:40]}' in {_shorten_path(path)}"
        if include:
            display += f" ({include})"
        return display
    
    # Task tool - show description
    if tool_name == "task":
        description = tool_args.get("description", "")
        if len(description) > 80:
            return description[:77] + "..."
        return description
    
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
                       tool_count: int = 0) -> None:
    """Shared implementation for displaying a tool call."""
    icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
    display_str = format_tool_display(tool_name, tool_args)

    # Add blank line between tools for readability
    if tool_count > 1:
        console_instance.print()

    # Special handling for write_todos - multi-line display.
    # Note: in this repo "task" is the dedicated subagent-spawn tool, so we
    # avoid using "tasks" as the label here to prevent confusion.
    if tool_name == "write_todos" and "\n" in display_str:
        console_instance.print(f"  {icon} [bold magenta]{tool_name}[/bold magenta]:")
        console_instance.print(f"    {display_str}", style="dim")
    elif display_str:
        console_instance.print(f"  {icon} [bold magenta]{tool_name}[/bold magenta] [dim]{display_str}[/dim]")
    else:
        console_instance.print(f"  {icon} [bold magenta]{tool_name}[/bold magenta]")


_BOX_COLOR = "bright_yellow"
_BOX_DIM_COLOR = "dim"


class StreamDisplayManager:
    """Manages CLI output display state for streaming responses.

    LLM response text is wrapped in a ``╭─ Response ─╮ … ╰───╯`` box.
    Reasoning/thinking gets a separate ``╭─ Thinking ─╮`` box.
    """

    # Subagent rendering verbosity. Three explicit modes (see PR notes):
    #   "all"     — default. Show ``tool_started`` only (one line per call,
    #               consecutive same-tool dedup, ``[N]`` prefix when multiple
    #               subagents are concurrently active). Final response shown.
    #   "verbose" — also show ``tool_completed`` with elapsed time.
    #   "off"     — silent during execution; only the final response summary
    #               at ``subagent.end`` is shown.
    SUBAGENT_VERBOSITIES = ("all", "verbose", "off")

    def __init__(self, console_instance, subagent_verbosity: str = "all"):
        self.console = console_instance
        self._term_width = min(console_instance.width or 80, 120)
        if subagent_verbosity not in self.SUBAGENT_VERBOSITIES:
            subagent_verbosity = "all"
        self._subagent_verbosity = subagent_verbosity
        self.reset()

    def reset(self):
        """Reset state for a new response."""
        self.in_thinking = False
        self.thinking_shown = False
        self.tool_count = 0
        self.in_tool_section = False
        self.response_started = False
        self.has_content_output = False
        self._response_buffer = []
        self._box_opened = False
        self._thinking_box_opened = False
        self._line_buffer = ""  # accumulates tokens until newline for line-buffered output
        # Set of "task" tool_call_ids (or just a counter) for which we have
        # already streamed live subagent steps; the after-completion summary
        # in display_tool_result() should be suppressed in that case.
        self._subagent_live_shown = 0
        # Per-run state for batch prefixes + consecutive-tool dedup.
        # ``_subagent_index`` maps run_id → 1-based slot. Slots are reclaimed
        # at ``subagent.end`` so a long-lived parent can run many batches
        # without leaking. ``_subagent_last_tool`` stores the previous
        # (tool_name, info) per run_id to suppress consecutive duplicates
        # (e.g. an agent retrying the same read_file call).
        self._subagent_index: "OrderedDict[str, int]" = OrderedDict()
        self._subagent_last_tool: Dict[str, tuple] = {}
        self._next_subagent_slot: int = 0
        # write_file: old file content captured at tool-call START time (before
        # the overwrite) so the result display can show a real old→new diff.
        # Keyed by the raw file_path arg; popped when the result is rendered.
        self._write_old: Dict[str, str] = {}
        # Truncated blocks are cleared at the start of each user turn in
        # display_user_message(), NOT here. Clearing here would wipe the
        # user's just-remembered long query (display_user_message runs before
        # the manager is created), which used to make Ctrl+o show only tool
        # results and never the folded query.

    def _open_box(self, label: str = "Response"):
        w = self._term_width
        fill = max(0, w - len(label) - 5)
        self.console.print(f"[{_BOX_COLOR}]╭─ {label} {'─' * fill}╮[/{_BOX_COLOR}]")

    def _close_box(self, right_label: Optional[str] = None):
        """Close a box. If ``right_label`` is given, embed it on the right
        side of the closing rule, e.g. ``╰─────────── (15:32:08) ─╯``.
        """
        w = self._term_width
        if not right_label:
            self.console.print(f"[{_BOX_COLOR}]╰{'─' * (w - 2)}╯[/{_BOX_COLOR}]")
            return
        # Layout: ╰─...─ <right_label> ─╯  (one space padding around label)
        # Total width is ``w``; left rule + 1 space + label + 1 space + ─╯
        label_len = len(right_label)
        fill = max(2, w - label_len - 5)
        self.console.print(
            f"[{_BOX_COLOR}]╰{'─' * fill} [/{_BOX_COLOR}]"
            f"[{_BOX_DIM_COLOR}]{right_label}[/{_BOX_DIM_COLOR}]"
            f"[{_BOX_COLOR}] ─╯[/{_BOX_COLOR}]"
        )

    def _flush_line_buffer(self):
        """Flush any accumulated partial line to output."""
        if self._line_buffer:
            self.console.print(self._line_buffer, highlight=False, markup=False)
            self._line_buffer = ""

    def _stream_text(self, content: str):
        """Buffer tokens and output complete lines.

        Accumulate tokens and flush on each newline through the same
        console path used by box drawing, ensuring correct ordering.
        """
        self._line_buffer += content
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self.console.print(line, highlight=False, markup=False)

    def start_thinking(self):
        """Start thinking section with a box."""
        if not self.thinking_shown:
            self.console.print()
            self._open_box("Thinking")
            self._thinking_box_opened = True
            self.thinking_shown = True
            self.in_thinking = True

    def stream_thinking(self, content: str):
        """Stream thinking content with line-buffered output."""
        self._stream_text(content)

    def end_thinking(self):
        """End thinking section and close its box."""
        if self.in_thinking:
            self._flush_line_buffer()
            if self._thinking_box_opened:
                self._close_box()
                self._thinking_box_opened = False
            self.in_thinking = False
            self.response_started = False

    def start_tool_section(self):
        """Start tool section."""
        if not self.in_tool_section:
            if self.in_thinking:
                self.end_thinking()
            if self.has_content_output and not self.response_started:
                self.console.print()
            self.console.print()
            self.in_tool_section = True

    def display_tool(self, tool_name: str, tool_args: dict):
        """Display a single tool call.

        Read-only tools (``_DEFERRED_TOOLS``) skip the start-time call line and
        collapse into a single completion line that folds in elapsed time, e.g.
        ``  🔎 grep 'pat' in path - 5 lines (13ms)``. The live spinner still
        announces the running tool, so deferring the print costs no feedback.
        """
        if tool_name in self._DEFERRED_TOOLS:
            return
        if tool_name in self._WRITE_DIFF_TOOLS:
            # write_file: capture the on-disk content BEFORE the overwrite so
            # the result display can render a real old→new unified diff. edit
            # tools don't need this (their old/new come from the args).
            if tool_name == "write_file":
                self._stash_write_file_old(tool_args)
            return
        self.start_tool_section()
        self.tool_count += 1
        _display_tool_impl(self.console, tool_name, tool_args, self.tool_count)

    def _stash_write_file_old(self, tool_args: dict) -> None:
        """Best-effort read of a write_file target's current content (pre-write).

        Resolves the path the same way the user would (cwd-relative; the tool
        itself resolves against its base dir, but cwd is close enough for a
        display-only diff). Failures (missing/binary/large file) → empty
        string, which yields an all-additions "new file" diff.
        """
        fp = tool_args.get("file_path")
        if not fp:
            return
        try:
            p = Path(fp).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            if p.exists() and p.is_file() and p.stat().st_size < 512_000:
                self._write_old[fp] = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    
    @staticmethod
    def _fmt_elapsed(elapsed: Optional[float]) -> str:
        """Format elapsed seconds with ms-precision under 1s.

        Every tool call has a non-zero cost (subprocess spawn, file I/O,
        even pure-python work), so we always surface a number when one is
        provided — fast tools just report ``<1ms`` instead of being hidden.

        - None or negative   → ''            (no measurement available)
        - < 1ms              → ' (<1ms)'
        - < 1s               → ' (Nms)'      e.g. ' (5ms)', ' (120ms)'
        - < 10s              → ' (N.NNs)'    e.g. ' (1.23s)'
        - >= 10s             → ' (N.Ns)'     e.g. ' (12.3s)'
        """
        if elapsed is None or elapsed < 0:
            return ""
        if elapsed < 0.001:
            return " (<1ms)"
        if elapsed < 1.0:
            return f" ({int(round(elapsed * 1000))}ms)"
        if elapsed < 10.0:
            return f" ({elapsed:.2f}s)"
        return f" ({elapsed:.1f}s)"

    @staticmethod
    def _result_count_summary(tool_name: str, result_content: str) -> str:
        """One-word count summary for a deferred read-only tool's result."""
        if not result_content:
            return "no matches" if tool_name == "grep" else ""
        n = len(str(result_content).splitlines())
        if n == 0:
            return "no matches" if tool_name == "grep" else ""
        if tool_name == "grep":
            return f"{n} lines"
        if tool_name == "ls":
            return f"{n} items"
        if tool_name == "glob":
            return f"{n} files"
        if tool_name == "web_search":
            return f"{n} results"
        return f"{n} lines"

    def _display_deferred_merged(self, tool_name: str, tool_args: dict,
                                 result_content: str, is_error: bool,
                                 elapsed_str: str) -> None:
        """Print the single merged line for a deferred read-only tool.

        Format: ``  {icon} {name} {params} - {count} {elapsed}``
        (errors surface a truncated message instead of the count).
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        params = format_tool_display(tool_name, tool_args)
        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if params:
            line += f" [dim]{params}[/dim]"
        if is_error:
            err = str(result_content).replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(f"Tool error · {tool_name}", str(result_content))
            line += f" [red]- error: {err}{elapsed_str}[/red]"
        else:
            summary = self._result_count_summary(tool_name, result_content)
            if summary:
                line += f" [dim]- {summary}{elapsed_str}[/dim]"
            else:
                line += f" [dim]{elapsed_str}[/dim]"
        self.console.print(line)

    # Max diff lines shown inline beneath an edit_file/multi_edit_file summary.
    _EDIT_DIFF_MAX_LINES = 8

    def _display_edit_merged(self, tool_name: str, tool_args: dict,
                             result_content: str, is_error: bool,
                             elapsed_str: str) -> None:
        """One summary line + a truncated unified diff for edit tools.

        ``  ✎ edit_file config.py - ✓ 1 edit (120ms)`` followed by a short
        CC/opencode-style diff so the user sees the actual code change. Errors
        surface a truncated message instead. The full diff is remembered for
        Ctrl+o expansion when truncated.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        filename = _extract_filename(tool_args.get("file_path", ""))
        params = filename

        # Count edits from args (multi_edit_file) or treat edit_file as 1.
        if tool_name == "multi_edit_file":
            edits = tool_args.get("edits") or []
            n_edits = len(edits) if isinstance(edits, list) else 0
        else:
            n_edits = 1

        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if params:
            line += f" [dim]{params}[/dim]"
        if is_error:
            err = str(result_content).replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(f"Tool error · {tool_name}", str(result_content))
            line += f" [red]- error: {err}{elapsed_str}[/red]"
        else:
            unit = "edit" if n_edits == 1 else "edits"
            line += f" [dim]- ✓ {n_edits} {unit}{elapsed_str}[/dim]"
        self.console.print(line)

        # Render a truncated unified diff from the edit args (skip on error).
        if is_error:
            return
        diff_text = self._build_edit_diff(tool_name, tool_args, filename)
        if not diff_text:
            return
        diff_lines = diff_text.splitlines()
        show = diff_lines[: self._EDIT_DIFF_MAX_LINES]
        self.console.print(Syntax("\n".join(show) + "\n", "diff", theme="monokai",
                                  line_numbers=False))
        remaining = len(diff_lines) - self._EDIT_DIFF_MAX_LINES
        if remaining > 0:
            self.console.print(
                f"      [dim italic]... ({remaining} more diff lines · Ctrl+o 展开)[/dim italic]"
            )
            remember_truncated(f"Edit diff · {filename}", diff_text)

    @staticmethod
    def _build_edit_diff(tool_name: str, tool_args: dict, filename: str) -> str:
        """Build a unified diff string from edit_file/multi_edit_file args."""
        if tool_name == "multi_edit_file":
            edits = tool_args.get("edits") or []
            if not isinstance(edits, list) or not edits:
                return ""
            parts = []
            for i, e in enumerate(edits):
                if not isinstance(e, dict):
                    continue
                old = str(e.get("old_string", ""))
                new = str(e.get("new_string", ""))
                # splitlines() WITHOUT keepends + lineterm="" so every diff
                # line (incl. ---/+++/@@ headers) is uniformly newline-free;
                # joining with "\n" then guarantees clean line boundaries.
                # Using keepends=True + "".join() would glue "-old" onto the
                # next "+new" whenever the edited text lacks a trailing newline.
                d = list(difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    fromfile=f"a/{filename}#{i + 1}",
                    tofile=f"b/{filename}#{i + 1}",
                    n=2,
                    lineterm="",
                ))
                if d:
                    parts.append("\n".join(d))
            return "\n".join(parts).rstrip("\n")
        # edit_file
        old = str(tool_args.get("old_string", ""))
        new = str(tool_args.get("new_string", ""))
        d = list(difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=2,
            lineterm="",
        ))
        return "\n".join(d).rstrip("\n")

    def _display_write_merged(self, tool_name: str, tool_args: dict,
                              result_content: str, is_error: bool,
                              elapsed_str: str) -> None:
        """One summary line + a truncated old→new diff for write_file.

        ``  ✎ write_file config.py - ✓ created 42 lines (120ms)`` followed by
        a head/tail unified diff (old content was stashed at call-start; for a
        brand-new file the "old" side is empty so the diff is all additions).
        Errors surface a truncated message instead. The full diff is remembered
        for Ctrl+o expansion when truncated.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        filename = _extract_filename(tool_args.get("file_path", ""))
        new_content = str(tool_args.get("content", ""))
        result_str = str(result_content)
        created = "Created" in result_str

        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if filename:
            line += f" [dim]{filename}[/dim]"
        if is_error:
            err = result_str.replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(f"Tool error · {tool_name}", result_str)
            line += f" [red]- error: {err}{elapsed_str}[/red]"
        else:
            n_lines = len(new_content.splitlines())
            verb = "created" if created else "updated"
            unit = "line" if n_lines == 1 else "lines"
            line += f" [dim]- ✓ {verb} {n_lines} {unit}{elapsed_str}[/dim]"
        self.console.print(line)

        if is_error:
            return

        # Old content stashed at call start; pop so the slot is reusable.
        old_content = self._write_old.pop(tool_args.get("file_path", ""), "")
        diff_text = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=2,
            lineterm="",
        )).rstrip("\n")
        if not diff_text:
            return
        diff_lines = diff_text.splitlines()
        # Render the diff itself as head/tail (the new tail of a file is often
        # the most relevant part), with the middle hidden.
        n = len(diff_lines)
        head, tail = self._WRITE_DIFF_HEAD_LINES, self._WRITE_DIFF_TAIL_LINES
        if n <= head + tail:
            show = diff_lines
            hidden = 0
        else:
            show = diff_lines[:head] + diff_lines[-tail:]
            hidden = n - head - tail
        self.console.print(Syntax("\n".join(show) + "\n", "diff", theme="monokai",
                                  line_numbers=False))
        if hidden > 0:
            self.console.print(
                f"      [dim italic]... ({hidden} hidden diff lines · Ctrl+o 展开)[/dim italic]"
            )
            remember_truncated(f"Write diff · {filename}", diff_text)

    # Read-only tools whose call line is deferred to completion so the call
    # line and elapsed time collapse into ONE line, e.g.
    # ``  🔎 grep 'pat' in path - 5 lines (13ms)``. No separate result footer.
    _DEFERRED_TOOLS = frozenset({"glob", "grep", "ls", "read_file", "web_search", "fetch_url"})

    # Write tools that edit/create files: call line is deferred to completion
    # and rendered as one summary line + a truncated unified diff
    # (CC/opencode-style) so the user sees the actual code change, not an
    # absolute path or noise. write_file diffs the pre-write content (stashed
    # at call start) against the new content arg.
    _WRITE_DIFF_TOOLS = frozenset({"edit_file", "multi_edit_file", "write_file"})

    # Tools whose success result is pure noise on success. The call line itself
    # already tells the user what happened; errors are still surfaced.
    _SUPPRESS_RESULT_TOOLS = frozenset({"write_todos"})

    # Max result lines shown inline before folding (per-tool overrides below).
    _DEFAULT_MAX_RESULT_LINES = 4
    # execute: head/tail window (middle hidden) — the tail carries the
    # command's final status/output which is usually what the user needs.
    _EXECUTE_HEAD_LINES = 6
    _EXECUTE_TAIL_LINES = 4
    # write_file diff window (head/tail, middle hidden).
    _WRITE_DIFF_HEAD_LINES = 6
    _WRITE_DIFF_TAIL_LINES = 4

    def display_tool_result(self, tool_name: str, result_content: str,
                            is_error: bool = False, elapsed: float = None,
                            tool_args: Optional[dict] = None):
        """Display tool execution result.

        For ``_DEFERRED_TOOLS`` the call line was suppressed at start time, so
        here we emit the merged single line ``icon name params - count (elapsed)``.
        """
        elapsed_str = self._fmt_elapsed(elapsed)

        if tool_name in self._DEFERRED_TOOLS:
            self.start_tool_section()
            self._display_deferred_merged(
                tool_name, tool_args or {}, result_content, is_error, elapsed_str
            )
            return

        if tool_name in self._WRITE_DIFF_TOOLS:
            self.start_tool_section()
            if tool_name == "write_file":
                self._display_write_merged(
                    tool_name, tool_args or {}, result_content, is_error, elapsed_str
                )
            else:
                self._display_edit_merged(
                    tool_name, tool_args or {}, result_content, is_error, elapsed_str
                )
            return

        # Suppress noisy success results; the call line is enough.
        if tool_name in self._SUPPRESS_RESULT_TOOLS and not is_error:
            return

        if not result_content:
            self.console.print(f"    [dim]⎿ done{elapsed_str}[/dim]")
            return

        if tool_name == "task":
            self._display_task_result(result_content, is_error)
            return

        result_str = str(result_content)

        if tool_name in ("grep", "glob", "execute", "ls", "read_file"):
            cwd = str(Path.cwd())
            if cwd in result_str:
                result_str = result_str.replace(cwd + "/", "").replace(cwd, ".")

        lines = result_str.splitlines()

        # execute: head/tail window with the middle hidden — the tail carries
        # the command's final status/output, which is usually what the user
        # needs to see at a glance.
        if tool_name == "execute":
            self._display_head_tail(
                lines, self._EXECUTE_HEAD_LINES, self._EXECUTE_TAIL_LINES,
                prefix="    ⎿ ", cont_prefix="      ",
                style="dim red" if is_error else "dim",
                error_prefix="    ⎿ ⚠ " if is_error else None,
                truncated_title=f"Tool output · {tool_name}",
                full_content=result_str,
                elapsed_str=elapsed_str,
            )
            return

        max_lines = self._DEFAULT_MAX_RESULT_LINES
        max_line_width = 120

        style = "dim red" if is_error else "dim"
        prefix = "    ⎿ " if not is_error else "    ⎿ ⚠ "
        cont_prefix = "      "

        display_lines = lines[:max_lines]
        for i, line in enumerate(display_lines):
            if len(line) > max_line_width:
                line = line[:max_line_width - 3] + "..."
            p = prefix if i == 0 else cont_prefix
            self.console.print(f"{p}{line}", style=style)

        remaining = len(lines) - max_lines
        if remaining > 0:
            self.console.print(
                f"{cont_prefix}... ({remaining} more lines · Ctrl+o 展开)", style="dim italic"
            )
            remember_truncated(f"Tool output · {tool_name}", result_str)
        if elapsed_str:
            self.console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")

    def _display_head_tail(self, lines: List[str], head: int, tail: int,
                           *, prefix: str, cont_prefix: str, style: str,
                           error_prefix: Optional[str] = None,
                           truncated_title: str, full_content: str,
                           elapsed_str: str = "",
                           max_line_width: int = 120) -> None:
        """Render a head/tail window with the middle hidden.

        Shows the first ``head`` lines and the last ``tail`` lines; anything in
        between is collapsed into a ``... (N hidden lines · Ctrl+o 展开)``
        separator. The full content is remembered for Ctrl+o expansion. Used
        for execute output and write_file diffs where the tail matters.
        """
        first_prefix = error_prefix or prefix
        n = len(lines)
        if n <= head + tail:
            show = lines
            hidden = 0
        else:
            show = lines[:head] + lines[-tail:]
            hidden = n - head - tail

        for i, line in enumerate(show):
            if len(line) > max_line_width:
                line = line[:max_line_width - 3] + "..."
            p = first_prefix if i == 0 else cont_prefix
            self.console.print(f"{p}{line}", style=style)

        if hidden > 0:
            self.console.print(
                f"{cont_prefix}... ({hidden} hidden lines · Ctrl+o 展开)",
                style="dim italic",
            )
            remember_truncated(truncated_title, full_content)
        if elapsed_str:
            self.console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")
    
    def _display_task_result(self, result_content: str, is_error: bool = False):
        """Display subagent task result.

        When live subagent events were already streamed via ``handle_event``,
        skip the per-tool summary (avoid duplication) and just print a brief
        execution-summary footer.
        """
        try:
            data = json.loads(result_content)
        except (ValueError, TypeError):
            self.console.print(f"    ⎿ {str(result_content)[:120]}", style="dim")
            return

        success = data.get("success", False)
        tool_summary = data.get("tool_calls_summary", [])
        exec_time = data.get("execution_time")
        tool_count = data.get("tool_count", len(tool_summary))

        if not success:
            error_msg = data.get("error", "Unknown error")
            self.console.print(f"    ⎿ ⚠ {error_msg[:120]}", style="dim red")
            if self._subagent_live_shown > 0:
                self._subagent_live_shown -= 1
            return

        # Live events already rendered the tool calls + final response — only
        # print a one-line summary footer to avoid duplicating output.
        if self._subagent_live_shown > 0:
            self._subagent_live_shown -= 1
            summary_parts = []
            if tool_count > 0:
                summary_parts.append(f"{tool_count} tool uses")
            if exec_time is not None:
                summary_parts.append(f"{exec_time:.1f}s")
            if summary_parts:
                self.console.print(
                    f"    [dim italic]⎿ task done ({', '.join(summary_parts)})[/dim italic]"
                )
            return

        # Fallback (no live callback registered): render the recap.
        max_shown = 8
        for i, tc in enumerate(tool_summary[:max_shown]):
            name = tc.get("name", "")
            info = tc.get("info", "")
            if len(info) > 90:
                info = info[:87] + "..."
            if i == 0:
                self.console.print(f"    ⎿ ", end="", style="dim")
            else:
                self.console.print(f"      ", end="")
            self.console.print(f"{name}", end="", style="dim bold")
            if info:
                self.console.print(f" {info}", style="dim")
            else:
                self.console.print(style="dim")

        if len(tool_summary) > max_shown:
            remaining = len(tool_summary) - max_shown
            self.console.print(f"      ... and {remaining} more tool calls", style="dim italic")

        summary_parts = []
        if tool_count > 0:
            summary_parts.append(f"{tool_count} tool uses")
        if exec_time is not None:
            summary_parts.append(f"cost: {exec_time:.1f}s")
        if summary_parts:
            summary_str = ", ".join(summary_parts)
            self.console.print(f"    [dim italic]Execution Summary: {summary_str}[/dim italic]")

    # ------------------------------------------------------------------
    # Live event rendering (subagent progress + compaction)
    # ------------------------------------------------------------------

    # Indent prefix for subagent inner steps; visually nests them under the
    # parent ``task`` tool call line.
    _SUB_INDENT = "    └─ "
    _SUB_CONT_INDENT = "       "

    def handle_event(self, event: dict) -> None:
        """Dispatch a runtime event from the agent (subagent / compression).

        Called synchronously by Runner / BuiltinTaskTool from the asyncio
        thread. While these events fire, the parent run is awaiting tool
        execution or starting a new turn, so the main thread is not mutating
        display state — direct console output is safe.
        """
        et = event.get("type", "")
        if et.startswith("subagent."):
            self._handle_subagent_event(et, event)
        elif et.startswith("compact."):
            self._handle_compact_event(et, event)

    def _subagent_prefix(self, run_id: Optional[str]) -> str:
        """Return ``[N] `` if multiple subagents are concurrently active.

        Single-subagent runs render with no numeric prefix to avoid noise.
        """
        if not run_id or len(self._subagent_index) < 2:
            return ""
        slot = self._subagent_index.get(run_id)
        return f"[dim]\\[{slot}][/dim] " if slot is not None else ""

    def _handle_subagent_event(self, et: str, event: dict) -> None:
        verbosity = self._subagent_verbosity
        run_id = event.get("run_id")

        if et == "subagent.start":
            self._subagent_live_shown += 1
            self._next_subagent_slot += 1
            if run_id:
                self._subagent_index[run_id] = self._next_subagent_slot
                self._subagent_last_tool.pop(run_id, None)

            if verbosity == "off":
                return

            agent_name = event.get("agent_name", "Subagent")
            task = event.get("task", "")
            preview = task.replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            self.console.print(
                f"{self._SUB_INDENT}{self._subagent_prefix(run_id)}"
                f"[dim cyan]⮕ {agent_name}[/dim cyan] "
                f"[dim italic]{preview}[/dim italic]"
            )

        elif et == "subagent.tool_started":
            if verbosity == "off":
                return
            tool_name = event.get("tool_name", "")
            info = event.get("info", "") or ""
            # Consecutive same-(tool, args) dedup: an agent that retries the
            # exact same call (or a stuck loop) shouldn't produce N identical
            # CLI lines. Only suppress when the previous tool from THIS run
            # had the same key — different runs / interleaved tools still
            # render normally.
            key = (tool_name, info)
            if run_id and self._subagent_last_tool.get(run_id) == key:
                return
            if run_id:
                self._subagent_last_tool[run_id] = key
            if len(info) > 100:
                info = info[:97] + "..."
            line = (
                f"{self._SUB_INDENT}{self._subagent_prefix(run_id)}"
                f"[dim magenta]{tool_name}[/dim magenta]"
            )
            if info:
                line += f" [dim]{info}[/dim]"
            self.console.print(line)

        elif et == "subagent.tool_completed":
            # Default ``all`` mode is tool-first: completion is hidden because
            # the started line already told the user "agent is doing X".
            # Verbose mode adds completion + elapsed for debugging.
            if verbosity != "verbose":
                # Always surface errors though — silent failures are worse
                # than slightly noisier output.
                if not event.get("is_error"):
                    return
            tool_name = event.get("tool_name", "")
            info = event.get("info", "") or ""
            if len(info) > 100:
                info = info[:97] + "..."
            is_error = event.get("is_error", False)
            elapsed_str = self._fmt_elapsed(event.get("elapsed"))
            prefix = self._subagent_prefix(run_id)
            if is_error:
                self.console.print(
                    f"{self._SUB_INDENT}{prefix}"
                    f"[dim red]⚠ {tool_name}[/dim red] [dim]{info}[/dim]"
                )
                return
            line = (
                f"{self._SUB_INDENT}{prefix}"
                f"[dim green]✓ {tool_name}[/dim green]"
            )
            if info:
                line += f" [dim]{info}[/dim]"
            if elapsed_str:
                line += f"[dim]{elapsed_str}[/dim]"
            self.console.print(line)

        elif et == "subagent.end":
            # Reclaim the slot before rendering so the prefix on the final
            # line reflects the active count after this subagent exits.
            if run_id:
                self._subagent_index.pop(run_id, None)
                self._subagent_last_tool.pop(run_id, None)

            response = event.get("response", "") or ""
            if response:
                preview = response.replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                # Final response is shown in every mode (including ``off``):
                # it's the actual answer the parent agent will consume.
                self.console.print(
                    f"{self._SUB_INDENT}[dim cyan]⤷[/dim cyan] "
                    f"[dim italic]{preview}[/dim italic]"
                )

    def _handle_compact_event(self, et: str, event: dict) -> None:
        # Micro-compact is an expected per-turn maintenance pass and fires too
        # frequently to be useful in the CLI. Keep it silent; surface only the
        # heavier compaction stages that change conversation structure.
        if et == "compact.micro":
            return

        agent_name = event.get("agent_name", "Agent")
        # Subagent compactions get extra indent so they visually nest under
        # the parent task tool line.
        prefix = "    " if self._subagent_live_shown > 0 else "  "
        if et == "compact.rule_based":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            self.console.print(
                f"{prefix}[dim yellow]🗜 compact [/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.2f}s)[/dim]"
            )
        elif et == "compact.auto":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            self.console.print(
                f"{prefix}[dim yellow]🗜 compact (auto / LLM-summarised)[/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.1f}s)[/dim]"
            )
        elif et == "compact.reactive":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            self.console.print(
                f"{prefix}[dim yellow]🗜 compact (reactive · prompt_too_long)[/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.1f}s)[/dim]"
            )
    
    def end_tool_section(self):
        """End tool section."""
        if self.in_tool_section:
            self.in_tool_section = False
            self.response_started = False
    
    def start_response(self):
        """Start response section with a box."""
        if not self.response_started:
            if self.in_thinking:
                self.end_thinking()
            if self.in_tool_section:
                self.end_tool_section()
            self.console.print()
            self._open_box("Response")
            self._box_opened = True
            self.response_started = True

    def stream_response(self, content: str):
        """Stream response content with line-buffered output."""
        self.start_response()
        self._response_buffer.append(content)
        self._stream_text(content)
        self.has_content_output = True

    def finalize(self):
        """Finalize output: flush buffered text and close open boxes.

        The Response box gets a small completion timestamp embedded in its
        bottom-right corner, e.g. ``╰─...─ (15:32:08) ─╯``. Useful when a
        long session is reviewed later — you can see when each answer landed.
        """
        if self.in_thinking:
            self.end_thinking()
        if self.in_tool_section:
            self.end_tool_section()
        ts = time.strftime("%H:%M:%S", time.localtime())
        right_label = f"({ts})"
        if self.has_content_output:
            self._flush_line_buffer()
            if self._box_opened:
                self._close_box(right_label=right_label)
                self._box_opened = False
        elif self._box_opened:
            self._close_box(right_label=right_label)
            self._box_opened = False


def display_tool_call(tool_name: str, tool_args: dict) -> None:
    """Display a tool call with icon and colored tool name."""
    _display_tool_impl(get_console(), tool_name, tool_args)


def _has_markdown(text: str) -> bool:
    """Detect if text contains Markdown formatting worth rendering."""
    markers = ["```", "## ", "### ", "* ", "- [ ]", "| ", "**", "1. "]
    return any(m in text for m in markers)


def render_markdown_response(console_instance, text: str) -> None:
    """Render a complete response as rich Markdown if it contains formatting."""
    if _has_markdown(text):
        console_instance.print(Markdown(text))
    else:
        console_instance.print(text, style=COLORS["agent"])


def display_diff(console_instance, file_path: str, old_content: str, new_content: str) -> None:
    """Display unified diff between old and new file content."""
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
        lineterm="",
    ))
    if diff_lines:
        diff_text = "\n".join(diff_lines)
        console_instance.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))


def _format_tokens_short(n: int) -> str:
    """Format token count with K/M suffix for compact display."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def context_pct_style(pct: float) -> str:
    """Return Rich style name based on context usage percentage."""
    if pct >= 95:
        return "bold red"
    if pct >= 80:
        return "red"
    if pct >= 50:
        return "yellow"
    return "green"


def build_context_bar(pct: float, width: int = 10) -> str:
    """Build a visual context usage bar like [████░░░░░░]."""
    safe = max(0.0, min(100.0, pct))
    filled = round((safe / 100) * width)
    return f"[{'█' * filled}{'░' * max(0, width - filled)}]"


def display_token_stats(
    console_instance,
    cost_tracker,
    *,
    context_window: int = 128000,
    session_total_tokens: int = 0,
    tool_use_count: int = 0,
    elapsed_seconds: float = 0.0,
) -> None:
    """Display compact per-response stats footer with color-graded context.

    Format example::

        ctx 50.0% (64K / 128K) [████░░░░░░] · 2 tools · 5.32s · $0.0034
    """
    if cost_tracker is None:
        return

    if session_total_tokens <= 0:
        session_total_tokens = (
            cost_tracker.total_input_tokens + cost_tracker.total_output_tokens
        )

    used_pct = (
        session_total_tokens / context_window * 100 if context_window > 0 else 0.0
    )
    pct_style = context_pct_style(used_pct)
    bar = build_context_bar(used_pct)

    parts = [
        f"[{pct_style}]ctx {used_pct:.1f}%[/{pct_style}] "
        f"({_format_tokens_short(session_total_tokens)} / "
        f"{_format_tokens_short(context_window)}) "
        f"[{pct_style}]{bar}[/{pct_style}]"
    ]

    if tool_use_count > 0:
        label = "tool" if tool_use_count == 1 else "tools"
        parts.append(f"[dim]{tool_use_count} {label}[/dim]")

    if elapsed_seconds > 0:
        parts.append(f"[dim]{elapsed_seconds:.2f}s[/dim]")

    # Prompt-cache hits / writes (Anthropic-style, e.g. Venus proxying Claude).
    cache_read = sum(s.cache_read_tokens for s in cost_tracker.model_usage.values())
    cache_write = sum(s.cache_write_tokens for s in cost_tracker.model_usage.values())
    if cache_read or cache_write:
        seg = []
        if cache_read:
            seg.append(f"{_format_tokens_short(cache_read)} cache_read")
        if cache_write:
            seg.append(f"{_format_tokens_short(cache_write)} cache_write")
        parts.append(f"[dim]{' · '.join(seg)}[/dim]")

    cost = cost_tracker.total_cost_usd
    cost_str = f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"
    parts.append(f"[dim]{cost_str}[/dim]")

    console_instance.print(f"{'  ·  '.join(parts)}")


# ---------------------------------------------------------------------------
# Persistent TUI status bar (prompt_toolkit fragments)
# ---------------------------------------------------------------------------

def _ctx_bar_ansi(pct: float, width: int = 10) -> str:
    """Build a plain-text context usage bar for the status bar."""
    safe = max(0.0, min(100.0, pct))
    filled = round((safe / 100) * width)
    return f"{'█' * filled}{'░' * max(0, width - filled)}"


def _ctx_fg_style(pct: float) -> str:
    """Return a prompt_toolkit style class for context usage percentage."""
    if pct >= 95:
        return "class:sb-critical"
    if pct >= 80:
        return "class:sb-bad"
    if pct >= 50:
        return "class:sb-warn"
    return "class:sb-good"


def format_duration_compact(seconds: float) -> str:
    """Format seconds into compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def build_status_bar_fragments(
    *,
    model_name: str = "",
    model_provider: str = "",
    profile_name: str = "",
    context_tokens: int = 0,
    context_window: int = 0,
    cost_usd: float = 0.0,
    active_seconds: float = 0.0,
    last_turn_seconds: float = 0.0,
    spinner_text: str = "",
    terminal_width: int = 80,
):
    """Build prompt_toolkit formatted-text fragments for the persistent status bar.

    Time display uses *agent active time* (sum of all LLM + tool
    execution durations) rather than session wall-clock, plus the
    most recent turn's latency.

    Adapts to terminal width:
      <52 cols:  ▸ model · ⏱12.3s
      <76 cols:  ▸ model · 45% · $0.02 · ⏱12.3s
      >=76 cols: ▸ model │ 64K/128K │ [████░░] 45% │ $0.02 │ ⏱12.3s Σ1m45s

    The model label is rendered as ``provider/model`` when a provider is
    supplied (e.g. ``openai/gpt-4o``); a non-default profile name is shown
    as a ``profile:`` prefix on wider terminals.
    """
    base = model_name.split("/")[-1] if "/" in model_name else model_name
    if model_provider:
        label = f"{model_provider}/{base}"
    else:
        label = base
    if len(label) > 26:
        label = label[:23] + "..."
    # Optional profile prefix (only when not the default profile).
    profile_prefix = ""
    if profile_name and profile_name != "default":
        profile_prefix = f"profile:{profile_name} "
    pct = (context_tokens / context_window * 100) if context_window > 0 else 0.0
    pct_label = f"{pct:.0f}%"
    fg = _ctx_fg_style(pct)
    cost_str = f"${cost_usd:.4f}" if cost_usd < 0.01 else f"${cost_usd:.2f}"

    turn_str = f"⏱ {last_turn_seconds:.1f}s" if last_turn_seconds > 0 else ""
    total_str = f"Σ {format_duration_compact(active_seconds)}" if active_seconds > 0 else ""

    sep = ("class:sb-dim", " · ")

    if terminal_width < 52:
        frags = [
            ("class:sb", " ▸ "),
            ("class:sb-strong", label),
        ]
        if turn_str:
            frags.append(sep)
            frags.append(("class:sb", turn_str))
    elif terminal_width < 76:
        frags = [
            ("class:sb", " ▸ "),
            ("class:sb-strong", label),
            sep,
            (fg, pct_label),
            sep,
            ("class:sb-dim", cost_str),
        ]
        if turn_str:
            frags.append(sep)
            frags.append(("class:sb", turn_str))
    else:
        ctx_used = _format_tokens_short(context_tokens) if context_tokens else "0"
        ctx_total = _format_tokens_short(context_window) if context_window else "?"
        frags = [
            ("class:sb", " ▸ "),
            ("class:sb-strong", label),
            ("class:sb-dim", " │ "),
            ("class:sb", f"{ctx_used}/{ctx_total}"),
            ("class:sb-dim", " │ "),
            (fg, _ctx_bar_ansi(pct)),
            ("class:sb-dim", " "),
            (fg, pct_label),
            ("class:sb-dim", " │ "),
            ("class:sb", cost_str),
        ]
        if turn_str:
            frags.append(("class:sb-dim", " │ "))
            frags.append(("class:sb", turn_str))
        if total_str:
            frags.append(("class:sb-dim", "  "))
            frags.append(("class:sb-dim", total_str))

    # Non-default profile prefix (dim), shown on medium/wide terminals only
    # right after the leading marker to keep the narrow layout intact.
    if profile_prefix and terminal_width >= 52:
        frags.insert(1, ("class:sb-dim", profile_prefix))

    frags.append(("class:sb", " "))
    return frags
