# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI display utilities - colors, formatting, stream display manager
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from rich.text import Text

from agentica.cli.config import console, TOOL_ICONS


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
    box_width = 70
    console.print("=" * box_width, style="bright_cyan")
    console.print("  Agentica CLI - Interactive AI Assistant with DeepAgent")
    console.print(f"  Model: [bright_green]{model_provider}/{model_name}[/bright_green]")

    # Working directory
    cwd = work_dir or os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    if len(cwd) > 50:
        cwd = "..." + cwd[-47:]
    console.print(f"  Working Directory: {cwd}")

    # Built-in tools (always shown)
    builtin_tools = ["ls", "read_file", "write_file", "edit_file", "glob", "grep",
                     "execute", "web_search", "fetch_url", "write_todos", "read_todos", "task", "save_memory"]
    console.print(f"  Built-in Tools: [white]{', '.join(builtin_tools)}[/white]")

    # Extra tools info
    if extra_tools:
        tools_str = ", ".join(extra_tools)
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        console.print(f"  Extra Tools: [bright_green]{tools_str}[/bright_green]")

    console.print("=" * box_width, style="bright_cyan")
    console.print()
    # Keyboard shortcuts
    console.print("  [bright_green]Enter[/bright_green]       Submit your message")
    console.print("  [bright_green]Ctrl+X[/bright_green]      Toggle Agent/Shell mode")
    console.print("  [bright_green]Ctrl+J[/bright_green]      Insert newline (Alt+Enter also works)")
    console.print("  [bright_green]Ctrl+D[/bright_green]      Exit")
    console.print("  [bright_green]Ctrl+C[/bright_green]      Interrupt current operation")
    console.print()
    # Input features
    console.print("  [bright_green]@filename[/bright_green]   Type @ to auto-complete files and inject content")
    console.print("  [bright_green]/command[/bright_green]    Type / to see available commands (try /help)")
    console.print()


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
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


def inject_file_contents(prompt_text: str, mentioned_files: list[Path]) -> str:
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


def display_user_message(text: str) -> None:
    """Display user message with file mentions colored."""
    pattern = r"(@[\w./-]+)"
    parts = re.split(pattern, text)
    rich_text = Text()
    
    for part in parts:
        if part.startswith("@"):
            rich_text.append(part, style="magenta")
        else:
            rich_text.append(part, style=COLORS["user"])
    
    console.print(rich_text)


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


def show_help():
    """Display help information."""
    help_text = """
Slash Commands:
  /help              Show this help message
  /clear, /reset     Clear screen and reset conversation
  /compact           Compact context (summarize history)
  /debug             Show debug info (model, history count)
  /model [p/m]       Show or switch model (e.g., /model deepseek/deepseek-chat)
  /newchat           Start a new chat session
  /tools             List available additional tools
  /skills            List available skills and triggers
  /memory            Show conversation history
  /workspace         Show workspace status and files
  /exit, /quit       Exit the CLI

Keyboard Shortcuts:
  Enter              Submit your message
  Ctrl+X             Toggle Agent/Shell mode ($ = shell, > = agent)
  Ctrl+J, Alt+Enter  Insert newline for multi-line input
  Ctrl+D             Exit
  Ctrl+C             Interrupt current operation

Input Features:
  @filename          Reference a file - content will be injected into prompt
  /command           Type / to trigger slash commands (auto-complete)

Shell Mode (Ctrl+X to toggle):
  When in shell mode ($ prompt), commands execute directly without AI.
  Useful for quick file operations, git commands, etc.

Tips:
  - Type @ followed by a filename to reference files
  - DeepAgent has built-in tools: ls, read_file, write_file, edit_file, glob, grep,
    execute, web_search, fetch_url, write_todos, read_todos, task, save_memory
  - Use --tools to add extra tools, e.g.: --tools calculator shell wikipedia
  - Say "remember this" or "save this" to trigger memory saving
"""
    console.print(help_text, style="yellow")


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
    
    # Execute command - shorten absolute paths in command
    if tool_name == "execute":
        command = tool_args.get("command", "")
        command = _shorten_paths_in_command(command)
        if len(command) > 300:
            return command[:297] + "..."
        return command
    
    # Todo tools - list the todo items
    if tool_name == "write_todos":
        todos = tool_args.get("todos", [])
        if isinstance(todos, list) and todos:
            todo_lines = []
            for todo in todos[:5]:
                if isinstance(todo, dict):
                    content = todo.get("content", "")[:50]
                    status = todo.get("status", "pending")
                    status_icon = "✓" if status == "completed" else "○" if status == "pending" else "◐"
                    todo_lines.append(f"{status_icon} {content}")
                else:
                    todo_lines.append(f"○ {str(todo)[:50]}")
            if len(todos) > 5:
                todo_lines.append(f"  ... and {len(todos) - 5} more")
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
    
    # Special handling for write_todos - multi-line display
    if tool_name == "write_todos" and "\n" in display_str:
        console_instance.print(f"  {icon} ", end="")
        console_instance.print(f"{tool_name}", end="", style="bold magenta")
        console_instance.print(" tasks:", style="dim")
        console_instance.print(f"    {display_str}", style="dim")
    else:
        console_instance.print(f"  {icon} ", end="")
        console_instance.print(f"{tool_name}", end="", style="bold magenta")
        if display_str:
            console_instance.print(f" {display_str}", style="dim")
        else:
            console_instance.print()


class StreamDisplayManager:
    """Manages CLI output display state for streaming responses."""
    
    def __init__(self, console_instance):
        self.console = console_instance
        self.reset()
    
    def reset(self):
        """Reset state for a new response."""
        self.in_thinking = False
        self.thinking_shown = False
        self.tool_count = 0
        self.in_tool_section = False
        self.response_started = False
        self.has_content_output = False
    
    def start_thinking(self):
        """Start thinking section."""
        if not self.thinking_shown:
            self.console.print()
            self.console.print("[dim italic]💭 Thinking...[/dim italic]")
            self.thinking_shown = True
            self.in_thinking = True
    
    def stream_thinking(self, content: str):
        """Stream thinking content."""
        self.console.print(content, end="", style="dim")
    
    def end_thinking(self):
        """End thinking section."""
        if self.in_thinking:
            self.console.print()
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
            self.console.print("[bold cyan]🔧 Tool Calls[/bold cyan]")
            self.in_tool_section = True
    
    def display_tool(self, tool_name: str, tool_args: dict):
        """Display a single tool call."""
        self.start_tool_section()
        self.tool_count += 1
        _display_tool_impl(self.console, tool_name, tool_args, self.tool_count)
    
    def display_tool_result(self, tool_name: str, result_content: str,
                            is_error: bool = False, elapsed: float = None):
        """Display tool execution result as a compact preview."""
        if not result_content:
            if elapsed is not None:
                self.console.print(f"    [dim]completed in {elapsed:.1f}s[/dim]")
            return

        # Special handling for task (subagent) - show inner tool calls
        if tool_name == "task":
            self._display_task_result(result_content, is_error)
            return

        result_str = str(result_content)

        # Shorten absolute paths in results for grep/glob/execute
        if tool_name in ("grep", "glob", "execute", "ls"):
            cwd = str(Path.cwd())
            if cwd in result_str:
                result_str = result_str.replace(cwd + "/", "").replace(cwd, ".")

        lines = result_str.splitlines()

        # grep results: fewer preview lines, they can be very long
        if tool_name == "grep":
            max_lines = 3
            max_line_width = 100
        else:
            max_lines = 4
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
            self.console.print(f"{cont_prefix}... ({remaining} more lines)", style="dim italic")
    
    def _display_task_result(self, result_content: str, is_error: bool = False):
        """Display subagent task result with inner tool calls and execution summary."""
        try:
            data = json.loads(result_content)
        except (ValueError, TypeError):
            self.console.print(f"    ⎿ {str(result_content)[:120]}", style="dim")
            return
        
        success = data.get("success", False)
        tool_summary = data.get("tool_calls_summary", [])
        exec_time = data.get("execution_time")
        tool_count = data.get("tool_count", len(tool_summary))
        subagent_name = data.get("subagent_name", "Subagent")
        
        if not success:
            error_msg = data.get("error", "Unknown error")
            self.console.print(f"    ⎿ ⚠ {error_msg[:120]}", style="dim red")
            return
        
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
    
    def end_tool_section(self):
        """End tool section."""
        if self.in_tool_section:
            self.in_tool_section = False
            self.response_started = False
    
    def start_response(self):
        """Start response section."""
        if not self.response_started:
            if self.in_thinking:
                self.end_thinking()
            if self.in_tool_section:
                self.end_tool_section()
            self.console.print()
            self.response_started = True
    
    def stream_response(self, content: str):
        """Stream response content."""
        self.start_response()
        self.console.print(content, end="", style=COLORS["agent"])
        self.has_content_output = True
    
    def finalize(self):
        """Finalize output, close any open sections."""
        if self.in_thinking:
            self.end_thinking()
        if self.in_tool_section:
            self.end_tool_section()
        if self.has_content_output:
            self.console.print()


def display_tool_call(tool_name: str, tool_args: dict) -> None:
    """Display a tool call with icon and colored tool name."""
    _display_tool_impl(console, tool_name, tool_args)
