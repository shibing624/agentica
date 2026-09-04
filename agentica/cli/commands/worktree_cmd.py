# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: /worktree — human surface for WorktreeBinder (same actions as the tool).
"""
from __future__ import annotations

from agentica.cli.runtime import get_console
from agentica.cli.commands.context import CommandContext
from agentica.worktrees import WorktreeError

_USAGE = (
    "  [dim]Usage: /worktree [status] | use <name> [--base <branch>] | "
    "merge | remove[/dim]\n"
    "  [dim]Same as the worktree tool. Launch with --worktree <name> to enter "
    "one at start.[/dim]"
)


def _parse_use(tokens: list[str]) -> tuple[str, str | None]:
    name = ""
    base = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("--base", "-b") and i + 1 < len(tokens):
            base = tokens[i + 1]
            i += 2
            continue
        if not name:
            name = token
        i += 1
    return name, base


def _cmd_worktree(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    binder = ctx.worktree_binder
    if binder is None:
        con.print("  [yellow]Worktree binding is not available in this session.[/yellow]")
        return

    parts = cmd_args.split()
    action = parts[0].lower() if parts else "status"
    rest = parts[1:]

    try:
        if action in ("status", "list", "info"):
            con.print(binder.status())
            return
        if action in ("use", "switch"):
            name, base = _parse_use(rest)
            if not name:
                con.print("  [dim]Usage: /worktree use <name> [--base <branch>][/dim]")
                return
            con.print(binder.switch(name, base=base))
            return
        if action == "merge":
            con.print(binder.merge())
            return
        if action in ("remove", "delete"):
            con.print(binder.remove())
            return
    except WorktreeError as exc:
        con.print(f"  [red]{exc}[/red]")
        return

    con.print(_USAGE)
