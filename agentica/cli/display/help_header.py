# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI header, /help text, and session summary rendering
"""

import os
from typing import Any, List, Optional

from rich.markup import escape
from rich.text import Text

from agentica.cli.runtime import BUILTIN_TOOLS, get_console
from agentica.model.usage import Usage
from agentica.version import __version__

def print_header(model_provider: str, model_name: str, work_dir: Optional[str] = None,
                 extra_tools: Optional[List[str]] = None):
    """Print the application header with version and model information"""
    console = get_console()
    box_width = min(console.width, 80)
    console.print("=" * box_width, style="bright_cyan")
    console.print(f"  Agentica CLI v{__version__} - Interactive AI Assistant")
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
    console.print(f"  Built-in Tools: [white]{', '.join(BUILTIN_TOOLS)}[/white]")

    # Extra tools info
    if extra_tools:
        tools_str = ", ".join(extra_tools)
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        console.print(f"  Extra Tools: [bright_green]{tools_str}[/bright_green]")

    # Log file location (helps users find logs when debugging)
    from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL
    if AGENTICA_LOG_FILE:
        log_path = AGENTICA_LOG_FILE
        if log_path.startswith(home):
            log_path = "~" + log_path[len(home):]
        console.print(f"  Log File ({AGENTICA_LOG_LEVEL}): [white]{log_path}[/white]")

    console.print("=" * box_width, style="bright_cyan")
    console.print()
    # Keyboard shortcuts
    console.print("  [bright_green]Enter[/bright_green]       Submit your message")
    console.print("  [bright_green]Ctrl+J[/bright_green]      Insert newline (Alt+Enter also works)")
    console.print("  [bright_green]Ctrl+D[/bright_green]      Exit and show session summary")
    console.print("  [bright_green]Ctrl+C[/bright_green]      Interrupt current operation (press twice to exit)")
    console.print("  [bright_green]Ctrl+V[/bright_green]      Paste image from clipboard (or just paste directly)")
    console.print("  [bright_green]Ctrl+O[/bright_green]      Expand truncated tool commands and output in pager (Ctrl+O or Esc to return)")
    console.print("  [bright_green]Alt+P[/bright_green]       Pause/resume live output while browsing terminal history")
    console.print()
    # Input features
    console.print("  [bright_green]@filename[/bright_green]   Type @ to auto-complete files (images auto-attach)")
    console.print("  [bright_green]/command[/bright_green]    Type / to see available commands (try /help)")
    console.print()


def format_session_summary(
    *, elapsed_seconds: float, usage: Usage, session_id: str | None, brief: bool = False
) -> Text:
    """Build the session summary block.

    ``brief`` renders only the "Worked for" rule — for mid-session interrupts
    (Ctrl+C), where the run continues afterwards so token totals and the
    resume hint are noise. The full block is for actually leaving a session
    (real exit, ``/new``, one-shot abort).
    """
    elapsed = max(0, int(elapsed_seconds))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"

    text = Text()
    text.append(f"Worked for {duration} ", style="dim")
    text.append("─" * 42, style="dim")
    if brief:
        return text
    text.append("\n\nToken usage: ", style="dim")
    text.append(f"total={usage.total_tokens:,} input={usage.input_tokens:,}")
    cached_tokens = usage.input_tokens_details.cache_read_tokens
    if cached_tokens <= 0:
        cached_tokens = usage.input_tokens_details.cached_tokens
    if cached_tokens > 0:
        text.append(f" (+ {cached_tokens:,} cached)", style="dim")
    text.append(f" output={usage.output_tokens:,}")
    reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    if reasoning_tokens > 0:
        text.append(f" (reasoning {reasoning_tokens:,})", style="dim")
    if session_id:
        text.append("\nTo continue this session, run ", style="dim")
        text.append(f"agentica resume {session_id}", style="bold")
    return text


def resumable_session_id(agent: Any) -> str | None:
    """Return the session id only when its JSONL log exists on disk."""
    session_log = agent._session_log
    if session_log is None or not session_log.exists():
        return None
    return agent.session_id

def show_help(skills_registry=None):
    """Display categorized help information."""
    categories = {
        "Session": {
            "/new":             "Start a new chat session",
            "/clear, /reset":   "Clear screen and reset conversation",
            "/rename <name>":   "Name current session for easy resume",
            "/resume [target]": "Resume by number, name, or id prefix ('all' lists every project, 'at <uuid>' forks)",
            "/fork [n|uuid]":   "Branch into a new session ('list' shows earlier points)",
            "/history":         "Show conversation history or full tool details",
            "/save, /export":   "Save conversation to JSON (no system prompts)",
            "/retry":           "Retry the last message (resend to agent)",
            "/undo":            "Remove the last user/assistant exchange",
            "/compact":         "Compact context (summarize history)",
            "/btw <question>":  "Ephemeral side question (no tools, not saved)",
            "/queue":           "Run prompt as the NEXT turn (plain input steers the current run)",
            "/steer <text>":    "Guide the running agent mid-task (plain input already steers)",
            "/checkpoint":      "Durable file snapshots: list | create | diff | restore",
            "/background":      "Run prompt in background (/bg alias)",
            "/ps":              "List background agents and terminals",
            "/stop <id|all>":   "Stop background tasks (needs a target; Ctrl+C stops the current run)",
            "/list-agents":     "List your other live sessions (/peers alias)",
            "/send-message":    "Message another session yourself (/send alias)",
        },
        "Configure": {
            "/model [p/m]":     "Show or switch model",
            "/config":          "Show current configuration",
            "/usage":           "Token usage, cost, and what fills the context",
            "/debug [on|off]":  "Toggle verbose debug logging",
            "/reasoning":       "Toggle reasoning display: on | off",
            "/statusbar, /sb":  "Toggle the status bar",
        },
        "Tools & Skills": {
            "/tools":           "Manage tools: add | remove | info | search",
            "/skills":          "Manage skills: search | browse | install | remove | inspect | tap",
        },
        "Automation": {
            "/cron":            "Scheduled jobs: list | add \"<prompt>\" <schedule> | edit | pause | resume | remove | runs | run | daemon on/off",
        },
        "Permissions": {
            "/permissions":     "View or set mode (ask/auto/allow-all)",
        },
        "Other": {
            "/help":            "Show this help message",
            "/exit, /quit":     "Exit and show session summary",
        },
    }

    console = get_console()
    console.print()
    console.print("  [bold]Available Commands[/bold]")
    console.print()

    # Command names and descriptions are data, not markup: a bracketed
    # placeholder like "/model [p/m]" parses as a style tag and rich drops it
    # silently. Pad first, then escape, so the backslashes rich strips at
    # render time don't skew column alignment.
    for category, commands in categories.items():
        console.print(f"  [bold]-- {category} --[/bold]")
        for cmd, desc in commands.items():
            console.print(f"    [bright_green]{escape(f'{cmd:<18}')}[/bright_green] [dim]{escape(desc)}[/dim]")
        console.print()

    # Skill auto-commands
    if skills_registry and len(skills_registry) > 0:
        skill_cmds = skills_registry.auto_commands()
        if skill_cmds:
            console.print("  [bold]-- Skill Commands --[/bold]")
            for slug, skill in skill_cmds.items():
                desc = skill.description[:50] if skill.description else ""
                console.print(f"    [bright_green]{escape(f'{slug:<18}')}[/bright_green] [dim]{escape(desc)}[/dim]")
            console.print()

    console.print("  [bold]Keyboard Shortcuts[/bold]")
    console.print()
    shortcuts = {
        "Enter":             "Submit your message",
        "Ctrl+J, Alt+Enter": "Insert newline for multi-line input",
        "Ctrl+D":            "Exit and show session summary",
        "Ctrl+C":            "Interrupt the current task; press twice to exit",
        "Tab, Right Arrow":  "Accept completion / auto-suggestion",
        "Ctrl+V":            "Paste image from clipboard",
        "Ctrl+O":            "Expand truncated tool commands and output in pager",
    }
    for key, desc in shortcuts.items():
        console.print(f"    [bright_green]{escape(f'{key:<20}')}[/bright_green] [dim]{escape(desc)}[/dim]")
    console.print()

    console.print("  [bold]Input Features[/bold]")
    console.print()
    console.print("    [bright_green]@filename[/bright_green]           Reference a file - content injected into prompt")
    console.print("    [bright_green]/command[/bright_green]            Type / to see slash commands with auto-complete")
    console.print()
    console.print("  [dim]Tip: type your message and press Enter to chat![/dim]")
    console.print()
