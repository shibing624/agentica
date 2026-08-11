# -*- coding: utf-8 -*-
"""Slash-command registry wiring."""

from __future__ import annotations

from agentica.cli.runtime import get_console

from agentica.cli.commands.cron_cmd import _cmd_cron
from agentica.cli.commands.goal import _cmd_goal, _cmd_subgoal
from agentica.cli.commands.model_config import (
    _cmd_config,
    _cmd_debug,
    _cmd_model,
    _cmd_reasoning,
    _cmd_status,
    _cmd_statusbar,
    _cmd_upgrade,
    _cmd_usage,
)
from agentica.cli.commands.runtime import (
    _cmd_background,
    _cmd_btw,
    _cmd_checkpoint,
    _cmd_exit,
    _cmd_fork,
    _cmd_help,
    _cmd_image,
    _cmd_list_agents,
    _cmd_paste,
    _cmd_ps,
    _cmd_queue,
    _cmd_send_message,
    _cmd_steer,
    _cmd_stop,
)
from agentica.cli.commands.session import (
    _cmd_clear,
    _cmd_compact,
    _cmd_export,
    _cmd_history,
    _cmd_newchat,
    _cmd_rename,
    _cmd_resume,
    _cmd_retry,
    _cmd_undo,
)
from agentica.cli.commands.tools_skills import (
    _cmd_agents,
    _cmd_permissions,
    _cmd_skills,
    _cmd_tools,
)



# ==================== Command Registry ====================

COMMAND_REGISTRY = {
    # Session
    "/new": (_cmd_newchat, "Start a new chat session"),
    "/clear": (_cmd_clear, "Clear screen and reset"),
    "/reset": (_cmd_clear, "Clear screen and reset (alias)"),
    "/history": (_cmd_history, "Show conversation history or full tool details"),
    "/export": (_cmd_export, "Save conversation to JSON"),
    "/save": (_cmd_export, "Save conversation to JSON (alias)"),
    "/retry": (_cmd_retry, "Retry the last message (resend to agent)"),
    "/undo": (_cmd_undo, "Remove the last user/assistant exchange"),
    "/compact": (_cmd_compact, "Compact context (summarize history)"),
    "/rename": (_cmd_rename, "Rename the current session for easy resume"),
    "/resume": (_cmd_resume, "Resume by number, name, or id prefix"),
    "/goal": (_cmd_goal, "Set or manage a standing goal (auto-continues until done; --tokens -1 = unlimited)"),
    "/subgoal": (_cmd_subgoal, "Add or manage acceptance criteria on the active goal"),
    "/btw": (_cmd_btw, "Quick aside answered in parallel \u2014 no tools, not persisted"),
    "/queue": (
        _cmd_queue,
        "Run as the NEXT turn after the current run finishes | list | edit <n> | insert <n> | remove <n> | clear",
    ),
    "/q": (_cmd_queue, "Run as the next turn after current run (alias)"),
    "/background": (_cmd_background, "Run NOW in a parallel independent agent (own session)"),
    "/bg": (_cmd_background, "Run now in a parallel independent agent (alias)"),
    "/ps": (_cmd_ps, "List background agents and terminal commands"),
    "/stop": (_cmd_stop, "Stop background tasks: /stop <id|#n|pid> | all (Ctrl+C stops the current run)"),
    "/steer": (_cmd_steer, "Course-correct the CURRENT run mid-task (plain text typed mid-run steers by default)"),
    "/list-agents": (_cmd_list_agents, "List your other live CLI sessions this one can message"),
    "/peers": (_cmd_list_agents, "List messageable live sessions (alias for /list-agents)"),
    "/send-message": (_cmd_send_message, "Send a message yourself: /send-message <session> <text>"),
    "/send": (_cmd_send_message, "Send a message to a session (alias for /send-message)"),
    "/fork": (_cmd_fork, "Branch into a new session: /fork [list|n|uuid]"),
    "/checkpoint": (
        _cmd_checkpoint,
        "Durable file snapshots: list | create <label> <path...> | diff <id> | restore <id>",
    ),
    # Model & Config
    "/model": (_cmd_model, "View or switch model"),
    "/config": (_cmd_config, "Show config | set <field> <value> | env <KEY> <value> | path"),
    "/upgrade": (_cmd_upgrade, "Self-upgrade agentica via pip (check | --pre)"),
    "/cron": (_cmd_cron, "Scheduled jobs: list | add | edit | pause | resume | remove | runs | run | daemon"),
    "/usage": (_cmd_usage, "Show token usage, cost, and context breakdown"),
    "/debug": (_cmd_debug, "Toggle verbose debug logging: on | off"),
    "/reasoning": (_cmd_reasoning, "Toggle reasoning display: on | off"),
    "/statusbar": (_cmd_statusbar, "Toggle the status bar visibility"),
    "/sb": (_cmd_statusbar, "Toggle the status bar (alias)"),
    "/status": (_cmd_status, "Show session status overview"),
    # Tools & Skills
    "/tools": (_cmd_tools, "Manage tools: add | remove | info | search"),
    "/skills": (_cmd_skills, "Manage skills: search | browse | install | remove | inspect | tap"),
    "/extensions": (_cmd_skills, "Manage skills (alias for /skills)"),
    "/agents": (
        _cmd_agents,
        "Manage subagents: list | create <name> | reload | remove <name>",
    ),
    "/agent": (_cmd_agents, "Manage subagents (alias for /agents)"),
    # Permissions
    "/permissions": (_cmd_permissions, "View or set permission mode (ask/auto/allow-all)"),
    # Media (hidden aliases; prefer Ctrl+V for clipboard paste and
    # @path completion / drag-and-drop for local files. Not shown in /help.)
    "/paste": (_cmd_paste, "Paste image from clipboard"),
    "/image": (_cmd_image, "Attach a local image file"),
    # Other
    "/help": (_cmd_help, "Show available commands"),
    "/exit": (_cmd_exit, "Exit the CLI"),
    "/quit": (_cmd_exit, "Exit the CLI (alias)"),
}


COMMAND_HANDLERS = {cmd: handler for cmd, (handler, _) in COMMAND_REGISTRY.items()}



# ---- Slash-command invocation echo ----
# Commands that produce their own conversational output (so an extra header
# would just add visual noise) or that are silently no-op control verbs.
_SILENT_CMDS: set = {
    "/exit",
    "/quit",
    "/clear",
    "/reset",  # these wipe the screen — the echo would vanish anyway
    "/btw",  # has its own concurrent UI
    "/paste",
    "/image",  # part of the user's message construction, not a query
}



def echo_command_invocation(cmd: str, cmd_args: str = "") -> None:
    """Print a single, consistent header for every slash-command invocation.

    Centralizing this means individual handlers no longer need to print their
    own per-command titles, and the user sees uniformly-formatted output
    regardless of whether the command was typed, replayed, or invoked
    programmatically (where the prompt's natural echo is absent).

    Silent commands (see ``_SILENT_CMDS``) skip the echo to avoid noise.

    Single call site: ``agentica/cli/interactive.py`` calls this once per slash
    command at the dispatch entrypoint, immediately before invoking the
    handler. Do not call this from inside individual command handlers or you
    will double-print the header.
    """
    if cmd in _SILENT_CMDS:
        return
    con = get_console()
    rendered = f"{cmd} {cmd_args}".rstrip()
    con.print(f"[bold dim]> {rendered}[/bold dim]")
