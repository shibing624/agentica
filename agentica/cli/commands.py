# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI slash command handlers and dispatch registry.

All _cmd_* functions live here. interactive.py imports COMMAND_REGISTRY
and COMMAND_HANDLERS to wire them into the TUI process_loop.
"""
import asyncio
import collections
import json
import os
import queue
import shlex
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.cli.config import (
    get_console,
    BUILTIN_TOOLS,
    TOOL_REGISTRY,
    MODEL_REGISTRY,
    EXAMPLE_MODELS,
    configure_tools,
    create_agent,
    get_model,
    _generate_session_id,
)
from agentica.cli.display import (
    print_header,
    show_help,
)
from agentica.cli.setup import (
    default_base_url,
    load_cli_config,
    save_cli_config,
)
from agentica.goals import GoalManager
from agentica.memory.models import AgentRun
from agentica.model.message import Message
from agentica.run_context import TaskAnchor
from agentica.run_response import RunResponse
from agentica.tools.goal_tool import GoalTool
from agentica.skills import (
    get_skill_registry,
    install_skills,
    list_installed_skills,
    load_skills,
    remove_skill,
)
from agentica.skills.skill_registry import reset_skill_registry


# ==================== CommandContext ====================

@dataclass
class CommandContext:
    """Shared context passed to all command handlers.

    Replaces the scattered **kwargs parameter bags with a single,
    type-checkable object.
    """
    agent_config: dict
    current_agent: Any  # Agent instance
    extra_tools: Optional[List] = None
    extra_tool_names: Optional[List[str]] = None
    workspace: Any = None  # Optional[Workspace]
    skills_registry: Any = None
    permission_manager: Any = None
    shell_mode: bool = False
    tui_state: Optional[dict] = None
    pending_queue: Any = None  # PendingQueue
    agent_running: bool = False
    attached_images: Optional[list] = None
    image_counter: Optional[list] = None
    # Background tasks — instance-level, not module-global
    bg_tasks: Dict[str, dict] = field(default_factory=dict)
    bg_task_counter: int = 0
    # Persistent goal loop (see agentica/goals.py). Same instance is shared
    # between the post-turn hook and /goal handlers, guarded by goal_lock.
    goal_manager: Any = None  # Optional[GoalManager]
    goal_lock: Any = None     # Optional[threading.Lock]


# ==================== PendingQueue ====================

class PendingQueue:
    """Thread-safe observable queue with list/clear/remove support.

    Each enqueued item is paired with a wall-clock timestamp so the TUI
    queue bar can show when each pending message was submitted.
    """

    def __init__(self):
        self._deque = collections.deque()
        self._timestamps = collections.deque()
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._deque.append(item)
            self._timestamps.append(time.time())

    def get(self, timeout: float = 0.1):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._deque:
                    self._timestamps.popleft()
                    return self._deque.popleft()
            if time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(0.02)

    def peek_all(self) -> list:
        with self._lock:
            return list(self._deque)

    def peek_all_with_timestamps(self) -> list:
        """Return ``[(item, ts_epoch_seconds), ...]`` snapshot."""
        with self._lock:
            return list(zip(self._deque, self._timestamps))

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def empty(self) -> bool:
        with self._lock:
            return len(self._deque) == 0

    def clear(self):
        with self._lock:
            self._deque.clear()
            self._timestamps.clear()

    def remove_index(self, idx: int) -> bool:
        with self._lock:
            if 0 <= idx < len(self._deque):
                del self._deque[idx]
                del self._timestamps[idx]
                return True
            return False


# ==================== Concurrent commands ====================

# Commands that can execute while the agent is streaming (non-blocking).
# Readonly info commands + queue/bg management.
CONCURRENT_CMDS = frozenset({
    "/bg", "/background", "/stop",
    "/q", "/queue", "/steer",
    "/cost", "/usage", "/config", "/debug",
    "/history", "/help", "/tools", "/skills",
    "/permissions", "/statusbar", "/sb",
    "/reasoning",
    # /goal and /subgoal: status/pause/clear/list subcommands are concurrent-safe.
    # Handlers reject "set new objective" when agent_running.
    "/goal", "/subgoal",
})


# ==================== Helpers ====================

IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})


def _cmd_title(name: str):
    """Print a command header."""
    get_console().print(f"\n  [bold]{name}[/bold]")


def _sanitize_history_for_model_switch(agent) -> None:
    """Strip tool_calls and tool messages from working memory history."""
    wm = agent.working_memory
    for run in wm.runs:
        if not run.response or not run.response.messages:
            continue
        cleaned = []
        for msg in run.response.messages:
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and msg.tool_calls:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    cleaned.append(Message(role="assistant", content=text))
                continue
            if msg.role == "system":
                continue
            cleaned.append(msg)
        run.response.messages = cleaned


def _refresh_skills_session(ctx: CommandContext):
    """Reload skill registry from disk and rebuild the current agent."""
    reset_skill_registry()
    load_skills()
    new_registry = get_skill_registry()
    new_agent = create_agent(ctx.agent_config, ctx.extra_tools, ctx.workspace, new_registry)
    return {
        "skills_registry": new_registry,
        "current_agent": new_agent,
    }


def _run_async_safe(coro):
    """Run an async coroutine safely from a sync context.

    Uses asyncio.run() in threads without an event loop.
    Falls back to loop.run_until_complete() if a loop already exists.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ==================== Command Handlers ====================

def _cmd_help(ctx: CommandContext, cmd_args: str = ""):
    _cmd_title(f"/help {cmd_args.strip()}" if cmd_args.strip() else "/help")
    show_help(skills_registry=ctx.skills_registry)


def _cmd_exit(ctx: CommandContext, cmd_args: str = ""):
    return "EXIT"


def _cmd_tools(ctx: CommandContext, cmd_args: str = ""):
    """Manage tools: list, add, remove, info, search."""
    con = get_console()
    args_str = cmd_args.strip()
    parts = args_str.split(None, 1) if args_str else []
    subcmd = parts[0].lower() if parts else ""
    sub_args = parts[1].strip() if len(parts) > 1 else ""

    _cmd_title(f"/tools {args_str}" if args_str else "/tools")

    # ── /tools add <name> ──
    if subcmd == "add":
        tool_names = sub_args.split()
        if not tool_names:
            con.print("  [dim]Usage: /tools add <name> [name2 ...][/dim]")
            return
        agent = ctx.current_agent
        if not agent:
            con.print("  [red]No active agent.[/red]")
            return
        for name in tool_names:
            if name not in TOOL_REGISTRY:
                con.print(f"  [red]Unknown tool: {name}[/red]")
                continue
            active_names = _get_active_tool_names(agent)
            if name in active_names:
                con.print(f"  [dim]{name} is already active.[/dim]")
                continue
            new_tools = configure_tools([name])
            if new_tools:
                if agent.tools is None:
                    agent.tools = []
                agent.tools.extend(new_tools)
                if ctx.extra_tool_names is None:
                    ctx.extra_tool_names = []
                if name not in ctx.extra_tool_names:
                    ctx.extra_tool_names.append(name)
                con.print(f"  [green]{name} loaded.[/green]")
        return {"extra_tool_names": ctx.extra_tool_names}

    # ── /tools remove <name> ──
    if subcmd in ("remove", "rm"):
        tool_names = sub_args.split()
        if not tool_names:
            con.print("  [dim]Usage: /tools remove <name> [name2 ...][/dim]")
            return
        agent = ctx.current_agent
        if not agent:
            con.print("  [red]No active agent.[/red]")
            return
        builtin_set = set(BUILTIN_TOOLS)
        for name in tool_names:
            if name in builtin_set:
                con.print(f"  [yellow]{name} is a built-in tool and cannot be removed.[/yellow]")
                continue
            # Find and remove the tool instance from agent.tools
            removed = False
            if agent.tools:
                for i, tool in enumerate(agent.tools):
                    cls_name = type(tool).__name__
                    reg_entry = TOOL_REGISTRY.get(name)
                    if reg_entry and cls_name == reg_entry[1]:
                        agent.tools.pop(i)
                        removed = True
                        break
            if removed:
                if ctx.extra_tool_names and name in ctx.extra_tool_names:
                    ctx.extra_tool_names.remove(name)
                con.print(f"  [green]{name} removed.[/green]")
            else:
                con.print(f"  [dim]{name} is not currently active.[/dim]")
        return {"extra_tool_names": ctx.extra_tool_names}

    # ── /tools info <name> ──
    if subcmd == "info":
        name = sub_args.strip()
        if not name:
            con.print("  [dim]Usage: /tools info <name>[/dim]")
            return
        if name in set(BUILTIN_TOOLS):
            con.print(f"  [bold]{name}[/bold]  [green]built-in, always active[/green]")
            return
        reg_entry = TOOL_REGISTRY.get(name)
        if not reg_entry:
            con.print(f"  [red]Unknown tool: {name}[/red]")
            return
        _mod, _cls, _cat, desc = reg_entry
        agent = ctx.current_agent
        is_active = name in _get_active_tool_names(agent) if agent else False
        status = "[green]active[/green]" if is_active else "[dim]inactive[/dim]"
        con.print(f"  [bold]{name}[/bold]  {status}")
        con.print(f"  Category:  {_cat}")
        con.print(f"  Class:     {_cls}")
        con.print(f"  Module:    agentica.tools.{_mod}_tool")
        con.print(f"  {desc}")
        # Show registered functions if tool is active
        if is_active and agent and agent.tools:
            for tool in agent.tools:
                if type(tool).__name__ == _cls:
                    funcs = tool.functions if hasattr(tool, 'functions') else {}
                    if funcs:
                        con.print(f"  Functions: {', '.join(funcs.keys())}")
                    break
        return

    # ── /tools search <keyword> ──
    if subcmd in ("search", "find"):
        keyword = sub_args.lower()
        if not keyword:
            con.print("  [dim]Usage: /tools search <keyword>[/dim]")
            return
        matches = []
        for name in BUILTIN_TOOLS:
            if keyword in name:
                matches.append((name, "built-in", True))
        for name, (_mod, _cls, _cat, desc) in TOOL_REGISTRY.items():
            if keyword in name or keyword in desc.lower() or keyword in _cat.lower():
                agent = ctx.current_agent
                is_active = name in _get_active_tool_names(agent) if agent else False
                matches.append((name, desc, is_active))
        if matches:
            con.print(f"  Found {len(matches)} tool(s):")
            for name, desc, is_active in matches:
                marker = "[green]●[/green]" if is_active else "[dim]○[/dim]"
                con.print(f"    {marker} [bold]{name:<20}[/bold] {desc}")
        else:
            con.print(f"  [dim]No tools matching '{keyword}'.[/dim]")
        return

    # ── /tools (no args) — list all ──
    active_names = set()
    agent = ctx.current_agent
    if agent:
        active_names = _get_active_tool_names(agent)
    if ctx.extra_tool_names:
        active_names.update(ctx.extra_tool_names)

    all_tools = {}
    for name in BUILTIN_TOOLS:
        all_tools[name] = ("built-in", True)
    for name, (_mod, _cls, _cat, desc) in TOOL_REGISTRY.items():
        is_active = name in active_names
        all_tools[name] = (desc, is_active)

    con.print()
    for name in sorted(all_tools.keys()):
        desc, is_active = all_tools[name]
        if is_active:
            con.print(f"    [green]●[/green] [bold]{name:<20}[/bold] {desc}")
        else:
            con.print(f"    [dim]○[/dim] [dim]{name:<20}[/dim] [dim]{desc}[/dim]")
    con.print()
    active_count = sum(1 for _, (_, a) in all_tools.items() if a)
    con.print(f"  [green]● = active ({active_count})[/green]  [dim]○ = available ({len(all_tools) - active_count})[/dim]")
    con.print(f"  [dim]Commands: /tools add <name> | remove <name> | info <name> | search <keyword>[/dim]")
    con.print()


def _get_active_tool_names(agent) -> set:
    """Get names of tools currently active on the agent."""
    names = set()
    if not agent or not agent.tools:
        return names
    for tool in agent.tools:
        cls_name = type(tool).__name__
        # Match against TOOL_REGISTRY class names
        for reg_name, (_mod, reg_cls, _cat, _desc) in TOOL_REGISTRY.items():
            if cls_name == reg_cls:
                names.add(reg_name)
                break
    return names


def _cmd_skills(ctx: CommandContext, cmd_args: str = ""):
    """Unified skill management: list, search, browse, install, remove, inspect, reload, tap."""
    con = get_console()
    args_str = cmd_args.strip()
    parts = shlex.split(args_str) if args_str else []
    subcommand = parts[0].lower() if parts else ""
    sub_args = parts[1:]

    # Title shows the full command as user typed it
    _cmd_title(f"/skills {args_str}" if args_str else "/skills")

    # ── /skills search <query> — search hub registries ──
    if subcommand == "search":
        query = " ".join(sub_args)
        if not query:
            con.print("  [dim]Usage: /skills search <query>[/dim]")
            return
        from agentica.skills.hub import unified_search
        con.print(f"  Searching for: {query}...")
        results = unified_search(query, limit=15, deduplicate=False)
        if not results:
            con.print("  [dim]No skills found matching your query.[/dim]")
            return
        con.print(f"  [bold cyan]Found {len(results)} skill(s)[/bold cyan]")
        con.print()
        for r in results:
            trust_style = {"trusted": "green"}.get(r.trust_level, "yellow")
            con.print(f"    [bold]{r.name:<25}[/bold] [{trust_style}]{r.trust_level:<10}[/{trust_style}] [dim]{r.source}[/dim]")
            if r.description:
                con.print(f"      [dim]{r.description[:70]}{'...' if len(r.description) > 70 else ''}[/dim]")
            con.print(f"      [dim]identifier: {r.identifier}[/dim]")
        con.print()
        con.print("  [dim]Install: /skills install <name-or-identifier>  |  Preview: /skills inspect <name>[/dim]")
        return

    # ── /skills browse [query] — paginated listing, optional filter ──
    if subcommand == "browse":
        page = 1
        page_size = 20
        query_parts = []
        for i, a in enumerate(sub_args):
            if a == "--page" and i + 1 < len(sub_args):
                page = int(sub_args[i + 1])
            elif a == "--size" and i + 1 < len(sub_args):
                page_size = int(sub_args[i + 1])
            elif not a.startswith("--") and (i == 0 or sub_args[i - 1] not in ("--page", "--size")):
                query_parts.append(a)
        query = " ".join(query_parts)

        from agentica.skills.hub import unified_search
        con.print(f"  Loading skills from all sources{f' (filter: {query})' if query else ''}...")
        results = unified_search(query, limit=500)
        if not results:
            con.print("  [dim]No skills found.[/dim]")
            return

        # If query is given, score and filter by relevance
        if query:
            query_lower = query.lower()

            def _relevance(r):
                name = r.name.lower()
                if name == query_lower:
                    return 100
                if name.startswith(query_lower):
                    return 80
                if query_lower in name:
                    return 60
                desc = (r.description or "").lower()
                if query_lower in desc:
                    return 20
                tags = " ".join(r.tags).lower() if r.tags else ""
                if query_lower in tags:
                    return 10
                return 0

            scored = [(r, _relevance(r)) for r in results]
            scored = [(r, s) for r, s in scored if s > 0]
            scored.sort(key=lambda x: (-x[1], x[0].name.lower()))
            results = [r for r, _ in scored]

        total = len(results)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        page_items = results[start:start + page_size]
        con.print(f"  [bold cyan]Skills Hub ({total} skills, page {page}/{total_pages})[/bold cyan]")
        con.print()
        for i, r in enumerate(page_items, start=start + 1):
            trust_style = {"trusted": "green"}.get(r.trust_level, "yellow")
            con.print(f"    {i:>3}. [bold]{r.name:<25}[/bold] [{trust_style}]{r.trust_level:<10}[/{trust_style}] [dim]{r.source}[/dim]")
        con.print()
        nav = []
        browse_cmd = f"/skills browse {query}" if query else "/skills browse"
        if page > 1:
            nav.append(f"{browse_cmd} --page {page - 1}")
        if page < total_pages:
            nav.append(f"{browse_cmd} --page {page + 1}")
        if nav:
            con.print(f"  [dim]{' | '.join(nav)}[/dim]")
        return

    # ── /skills install — supports hub identifier, short name, git URL, local path ──
    if subcommand == "install":
        if not sub_args:
            con.print("  [dim]Usage: /skills install <name-or-identifier> [--force][/dim]")
            return
        source = None
        force = False
        category = ""
        for i, arg in enumerate(sub_args):
            if arg == "--force":
                force = True
            elif arg == "--category" and i + 1 < len(sub_args):
                category = sub_args[i + 1]
            elif source is None and not arg.startswith("--"):
                source = arg
        if source is None:
            con.print("  [dim]Missing install source.[/dim]")
            return

        # Determine source type
        is_git_url = source.startswith(("http://", "https://", "git@"))
        is_local = Path(source).expanduser().exists()

        if is_git_url or is_local:
            replaced = []
            installed = install_skills(source, force=force, replaced_symlinked_skills=replaced)
            for skill in installed:
                con.print(f"  [green]Installed '{skill.name}' (user-level)[/green]")
                con.print(f"  Path: {skill.path}")
            for name in replaced:
                con.print(f"  [green]Replaced existing: {name}[/green]")
            return _refresh_skills_session(ctx)

        # Hub identifier or short name: use hub pipeline
        from agentica.skills.hub import hub_install
        con.print(f"  Fetching: {source}...")
        success, msg = hub_install(source, category=category, force=force)
        if success:
            con.print(f"  [green]{msg}[/green]")
            return _refresh_skills_session(ctx)
        con.print(f"  [red]{msg}[/red]")
        return

    # ── /skills uninstall <name> — hub-aware uninstall ──
    if subcommand == "uninstall":
        if not sub_args:
            con.print("  [dim]Usage: /skills uninstall <name>[/dim]")
            return
        from agentica.skills.hub import uninstall_skill as hub_uninstall
        success, msg = hub_uninstall(sub_args[0])
        if success:
            con.print(f"  [green]{msg}[/green]")
            return _refresh_skills_session(ctx)
        # Fallback to local remove
        removed_path = remove_skill(sub_args[0])
        con.print(f"  [green]Removed skill '{sub_args[0]}' from {removed_path}[/green]")
        return _refresh_skills_session(ctx)

    # ── /skills remove <name> — local remove ──
    if subcommand == "remove":
        if not sub_args:
            con.print("  [dim]Usage: /skills remove <skill-name>[/dim]")
            return
        removed_path = remove_skill(sub_args[0])
        con.print(f"  [green]Removed skill '{sub_args[0]}' from {removed_path}[/green]")
        return _refresh_skills_session(ctx)

    if subcommand == "reload":
        return _cmd_reload_skills(ctx)

    # ── /skills inspect <name-or-identifier> — local or hub preview ──
    if subcommand == "inspect":
        query = " ".join(sub_args).strip()
        if not query:
            con.print("  [dim]Usage: /skills inspect <skill-name-or-identifier>[/dim]")
            return
        # Try local first
        found = None
        query_lower = query.lower()
        if ctx.skills_registry:
            for skill in ctx.skills_registry.list_all():
                if skill.name.lower() == query_lower:
                    found = skill
                    break
        if not found:
            for skill in list_installed_skills():
                if skill.name.lower() == query_lower:
                    found = skill
                    break
        if found:
            con.print(f"  [bold cyan]{found.name}[/bold cyan]")
            con.print(f"  [dim]Path: {found.path}[/dim]")
            con.print(f"  [dim]Location: {found.location}[/dim]")
            if found.description:
                con.print(f"  {found.description}")
            if found.trigger:
                con.print(f"  Trigger: [green]{found.trigger}[/green]")
            if found.requires:
                con.print(f"  Requires: {', '.join(found.requires)}")
            content = found.content
            if content:
                lines = content.splitlines()[:10]
                con.print()
                for line in lines:
                    con.print(f"  [dim]{line}[/dim]")
                if len(content.splitlines()) > 10:
                    con.print(f"  [dim]... ({len(content.splitlines()) - 10} more lines)[/dim]")
            return

        # Try hub inspect
        from agentica.skills.hub import create_source_router, resolve_short_name
        sources = create_source_router()
        identifier = query
        if "/" not in identifier:
            identifier = resolve_short_name(identifier, sources) or query
        for src in sources:
            meta = src.inspect(identifier)
            if meta:
                con.print(f"  [bold cyan]{meta.name}[/bold cyan]  [dim]({meta.source})[/dim]")
                con.print(f"  {meta.description}")
                con.print(f"  Identifier: [dim]{meta.identifier}[/dim]")
                con.print(f"  Trust: {meta.trust_level}")
                if meta.tags:
                    con.print(f"  Tags: {', '.join(meta.tags)}")
                con.print()
                con.print(f"  [dim]Install: /skills install {meta.identifier}[/dim]")
                return
        con.print(f"  [yellow]Skill '{query}' not found locally or in hub.[/yellow]")
        return

    # ── /skills tap — manage custom GitHub sources ──
    if subcommand == "tap":
        from agentica.skills.hub import TapsManager
        mgr = TapsManager()
        tap_action = sub_args[0].lower() if sub_args else "list"
        tap_repo = sub_args[1] if len(sub_args) > 1 else ""

        if tap_action == "list":
            taps = mgr.list_taps()
            if not taps:
                con.print("  [dim]No custom taps. Using default sources only.[/dim]")
            else:
                con.print(f"  [bold cyan]Taps ({len(taps)})[/bold cyan]")
                for t in taps:
                    con.print(f"    {t.get('repo', 'unknown')}  [dim]{t.get('path', 'skills/')}[/dim]")
            con.print()
            con.print("  [dim]Commands: /skills tap add <owner/repo> | remove <owner/repo>[/dim]")
        elif tap_action == "add":
            if not tap_repo:
                con.print("  [dim]Usage: /skills tap add <owner/repo>[/dim]")
                return
            if mgr.add(tap_repo):
                con.print(f"  [green]Added tap: {tap_repo}[/green]")
            else:
                con.print(f"  [dim]Tap already exists: {tap_repo}[/dim]")
        elif tap_action == "remove":
            if not tap_repo:
                con.print("  [dim]Usage: /skills tap remove <owner/repo>[/dim]")
                return
            if mgr.remove(tap_repo):
                con.print(f"  [green]Removed tap: {tap_repo}[/green]")
            else:
                con.print(f"  [red]Tap not found: {tap_repo}[/red]")
        return

    # ── /skills list (or /skills with no subcommand) — show installed ──
    all_skills = []
    if ctx.skills_registry and len(ctx.skills_registry) > 0:
        for skill in ctx.skills_registry.list_all():
            all_skills.append(("loaded", skill))

    if ctx.current_agent and ctx.current_agent.tools:
        from agentica.tools.skill_tool import SkillTool
        for tool in ctx.current_agent.tools:
            if isinstance(tool, SkillTool):
                for skill in tool._get_enabled_skills():
                    all_skills.append(("agent", skill))
                break

    installed = list_installed_skills()
    loaded_names = {s.name for _, s in all_skills}
    for skill in installed:
        if skill.name not in loaded_names:
            all_skills.append(("installed", skill))

    if not all_skills and subcommand not in ("list", ""):
        con.print("  No skills found.")
        con.print()

    if all_skills:
        con.print(f"  [bold cyan]Installed Skills ({len(all_skills)})[/bold cyan]")
        con.print()
        for source_type, skill in all_skills:
            trigger_str = f" [green]{skill.trigger}[/green]" if skill.trigger else ""
            loc = f"[dim]({source_type})[/dim]"
            con.print(f"    [bold]{skill.name}[/bold]{trigger_str} {loc}")
            if skill.description:
                desc = skill.description[:70] + ("..." if len(skill.description) > 70 else "")
                con.print(f"      [dim]{desc}[/dim]")
        con.print()
    else:
        con.print("  No installed skills.")
        con.print()

    con.print("  [dim]Commands: search <q> | browse | install <name> | remove <name> | inspect <name> | tap | reload[/dim]")


def _cmd_history(ctx: CommandContext, cmd_args: str = ""):
    """Display conversation history in compact format."""
    _cmd_title(f"/history {cmd_args.strip()}" if cmd_args.strip() else "/history")
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation history yet.[/yellow]")
        return
    messages = agent.working_memory.messages
    if not messages:
        con.print("[yellow]No conversation history yet.[/yellow]")
        return

    preview_limit = 400
    visible_index = 0
    hidden_tool_msgs = 0

    def _flush_tools():
        nonlocal hidden_tool_msgs
        if hidden_tool_msgs == 0:
            return
        noun = "message" if hidden_tool_msgs == 1 else "messages"
        con.print(f"\n  [dim]\\[Tools] ({hidden_tool_msgs} tool {noun} hidden)[/dim]")
        hidden_tool_msgs = 0

    con.print()
    con.print("  [bold cyan]Conversation History[/bold cyan]")

    for msg in messages:
        role = msg.role
        if role == "system":
            continue
        if role == "tool":
            hidden_tool_msgs += 1
            continue

        _flush_tools()
        visible_index += 1

        content = msg.content or ""
        if isinstance(content, list):
            content = str(content)
        content_text = content if isinstance(content, str) else str(content)

        if role == "user":
            preview = content_text[:preview_limit]
            suffix = "..." if len(content_text) > preview_limit else ""
            con.print(f"\n  [cyan]\\[You #{visible_index}][/cyan]")
            con.print(f"    {preview}{suffix}")
            continue

        tool_calls = msg.tool_calls or []
        if content_text:
            preview = content_text[:preview_limit]
            suffix = "..." if len(content_text) > preview_limit else ""
        elif tool_calls:
            n = len(tool_calls)
            preview = f"(requested {n} tool call{'s' if n > 1 else ''})"
            suffix = ""
        else:
            preview = "(no text response)"
            suffix = ""
        con.print(f"\n  [green]\\[Agent #{visible_index}][/green]")
        con.print(f"    {preview}{suffix}")

    _flush_tools()
    con.print()


def _cmd_config(ctx: CommandContext, cmd_args: str = ""):
    """Display current configuration and workspace status."""
    _cmd_title(f"/config {cmd_args.strip()}" if cmd_args.strip() else "/config")
    con = get_console()

    con.print()
    con.print("  [bold]-- Model --[/bold]")
    con.print(f"  Model:       {ctx.agent_config.get('model_provider', '')}/{ctx.agent_config.get('model_name', '')}")
    if ctx.current_agent and ctx.current_agent.model:
        model = ctx.current_agent.model
        if model.base_url:
            con.print(f"  Base URL:    {model.base_url}")
        api_key = model.api_key or ""
        key_display = "********" + api_key[-4:] if len(api_key) > 4 else "(not set)"
        con.print(f"  API Key:     {key_display}")
        con.print(f"  Context:     {model.context_window:,} tokens")

    con.print()
    con.print("  [bold]-- Terminal --[/bold]")
    con.print(f"  Working Dir: {os.getcwd()}")
    con.print(f"  Mode:        {'Shell' if ctx.shell_mode else 'Agent'}")
    if ctx.permission_manager:
        con.print(f"  Permissions: {ctx.permission_manager.mode}")

    con.print()
    con.print("  [bold]-- Agent --[/bold]")
    all_active = list(BUILTIN_TOOLS)
    if ctx.extra_tool_names:
        all_active.extend(ctx.extra_tool_names)
    con.print(f"  Tools:       {', '.join(all_active)}")
    if ctx.skills_registry and len(ctx.skills_registry) > 0:
        con.print(f"  Skills:      {len(ctx.skills_registry)} loaded")
    show_reasoning = ctx.tui_state.get("show_reasoning", True) if ctx.tui_state else True
    con.print(f"  Reasoning:   {'on' if show_reasoning else 'off'}")

    con.print()
    con.print("  [bold]-- Session --[/bold]")
    if ctx.current_agent:
        con.print(f"  Session ID:  {ctx.current_agent.session_id}")
    started = ctx.tui_state.get("session_start") if ctx.tui_state else None
    if started:
        con.print(f"  Started:     {started}")
    msg_count = 0
    if ctx.current_agent:
        msg_count = len(ctx.current_agent.working_memory.messages)
    con.print(f"  Messages:    {msg_count}")

    if ctx.workspace and ctx.workspace.exists():
        con.print(f"  Workspace:   {ctx.workspace.path}")
        memory_files = ctx.workspace.get_all_memory_files()
        if memory_files:
            paths = ", ".join(str(mf) for mf in memory_files)
            con.print(f"  Memory:      {paths}")
        else:
            con.print("  Memory:      (none)")
    elif ctx.workspace:
        con.print(f"  Workspace:   {ctx.workspace.path} (not initialized)")
    else:
        con.print("  Workspace:   (not configured)")
    con.print()


def _cmd_newchat(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    current_agent = create_agent(ctx.agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)
    con.print("[green]New chat session created.[/green]")
    con.print("[dim]Conversation history cleared.[/dim]")
    # Drop any goal manager — the new session has a new SessionLog.
    return {"current_agent": current_agent, "goal_manager": None}


def _cmd_resume(ctx: CommandContext, cmd_args: str = ""):
    """Resume a previous session from JSONL log."""
    from agentica.memory.session_log import SessionLog
    con = get_console()

    sessions = SessionLog.list_sessions()
    if not sessions:
        con.print("[yellow]No sessions found to resume.[/yellow]")
        return

    args_str = (cmd_args or "").strip()
    resume_at_uuid = None
    if " at " in args_str:
        parts = args_str.split(" at ", 1)
        args_str = parts[0].strip()
        resume_at_uuid = parts[1].strip()

    if args_str:
        try:
            idx = int(args_str) - 1
            if 0 <= idx < len(sessions):
                chosen = sessions[idx]
            else:
                con.print("[red]Invalid number.[/red]")
                return
        except ValueError:
            matching = [s for s in sessions if args_str in s["session_id"]]
            if not matching:
                con.print(f"[red]No session matching '{args_str}'[/red]")
                return
            chosen = matching[0]

        if resume_at_uuid is None:
            log = SessionLog(chosen["session_id"])
            user_msgs = log.list_user_messages(limit=10)
            if user_msgs:
                con.print(f"\n[bold]Session: {chosen['session_id']}[/bold]")
                con.print("[dim]Recent queries (resume from any point):[/dim]\n")
                for i, m in enumerate(user_msgs, 1):
                    ts = m.get("timestamp", "")[:19].replace("T", " ") if m.get("timestamp") else ""
                    con.print(f"  {i}. [dim]{ts}[/dim] {m['content']}")
                con.print(f"\n[dim]Usage: /resume {chosen['session_id']} at <uuid>[/dim]")

        agent_config = dict(ctx.agent_config)
        agent_config["session_id"] = chosen["session_id"]
        agent_config["_resume_at_uuid"] = resume_at_uuid
        current_agent = create_agent(agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)

        if resume_at_uuid and current_agent._session_log:
            current_agent.working_memory.clear()
            for rm in current_agent._session_log.load(resume_at=resume_at_uuid):
                current_agent.working_memory.add_message(
                    Message(role=rm["role"], content=rm.get("content", ""))
                )

        con.print(f"[green]Resumed session: {chosen['session_id']}"
                  f"{f' at {resume_at_uuid[:8]}...' if resume_at_uuid else ''}[/green]")

        # If the resumed session had an active goal, demote to paused for
        # safety — automatic continuation on resume is too surprising
        # without token-budget guards (P0).
        resumed_goal_manager = None
        if current_agent._session_log is not None:
            judge_model = current_agent.auxiliary_model or current_agent.model
            resumed_goal_manager = GoalManager(
                current_agent._session_log, judge_model=judge_model
            )
            state = resumed_goal_manager.load()
            if state is not None:
                if state.status == "active":
                    resumed_goal_manager.force_pause_on_resume()
                    con.print(
                        f"  [yellow]⊙ Standing goal detected and paused for safety:[/yellow] "
                        f"{state.objective}"
                    )
                    con.print("  [dim]Use /goal resume to continue working on it.[/dim]")
                elif state.status in ("paused", "complete"):
                    con.print(
                        f"  [dim]⊙ Previous goal ({state.status}): {state.objective}[/dim]"
                    )

        return {"current_agent": current_agent, "goal_manager": resumed_goal_manager}
    else:
        con.print("\n[bold]Available sessions:[/bold]\n")
        for i, s in enumerate(sessions[:10], 1):
            ts_str = s.get("last_timestamp", "") or ""
            if ts_str:
                ts_str = ts_str[:19].replace("T", " ")
            size_kb = s["size_bytes"] / 1024
            sid = s["session_id"]
            display_id = f"{sid[:8]}...{sid[-4:]}" if len(sid) > 20 else sid
            con.print(f"  {i}. [cyan]{display_id}[/cyan]  {ts_str}  ({size_kb:.0f}KB)")
        con.print(f"\n[dim]Usage: /resume <number> or /resume <session_id>[/dim]")
        return


def _cmd_clear(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    os.system('clear' if os.name != 'nt' else 'cls')
    current_agent = create_agent(ctx.agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)
    print_header(
        ctx.agent_config["model_provider"],
        ctx.agent_config["model_name"],
        work_dir=ctx.agent_config.get("work_dir"),
        extra_tools=ctx.extra_tool_names,
        shell_mode=ctx.shell_mode,
    )
    con.print("[info]Screen cleared and conversation reset.[/info]")
    return {"current_agent": current_agent, "goal_manager": None}


def _persist_model_choice(provider: str, model_name: str, base_url: Optional[str]) -> None:
    """Remember a live /model switch so it survives the next launch."""
    config = load_cli_config()
    config["onboarded"] = True
    config["model_provider"] = provider
    config["model_name"] = model_name
    config["base_url"] = base_url
    save_cli_config(config)


def _cmd_model(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    supported_providers = set(MODEL_REGISTRY.keys())
    if cmd_args:
        if "/" in cmd_args:
            new_provider, new_model = cmd_args.split("/", 1)
            new_provider = new_provider.strip().lower()
            new_model = new_model.strip()
        else:
            new_model = cmd_args.strip()
            new_provider = ctx.agent_config["model_provider"]

        if new_provider not in supported_providers:
            con.print(f"[red]Unknown provider: {new_provider}[/red]")
            con.print(f"Supported: {', '.join(sorted(supported_providers))}", style="dim")
            return

        old_provider = ctx.agent_config["model_provider"]
        if new_provider == old_provider:
            new_base_url = ctx.agent_config.get("base_url") or default_base_url(new_provider)
        else:
            new_base_url = default_base_url(new_provider)

        ctx.agent_config["model_provider"] = new_provider
        ctx.agent_config["model_name"] = new_model
        ctx.agent_config["base_url"] = new_base_url
        _persist_model_choice(new_provider, new_model, new_base_url)

        new_model_obj = get_model(
            model_provider=new_provider,
            model_name=new_model,
            base_url=new_base_url,
            api_key=ctx.agent_config.get("api_key"),
            max_tokens=ctx.agent_config.get("max_tokens"),
            temperature=ctx.agent_config.get("temperature"),
        )
        if ctx.current_agent is not None:
            ctx.current_agent.model = new_model_obj
            _sanitize_history_for_model_switch(ctx.current_agent)
            con.print(f"[green]Switched to: {new_provider}/{new_model} (session preserved)[/green]")
            return {"model_switched": True}
        else:
            current_agent = create_agent(ctx.agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)
            con.print(f"[green]Switched to: {new_provider}/{new_model}[/green]")
            return {"current_agent": current_agent}
    else:
        con.print(f"Current model: [bold cyan]{ctx.agent_config['model_provider']}/{ctx.agent_config['model_name']}[/bold cyan]")
        con.print()
        con.print("Supported providers and example models:", style="cyan")
        for provider in sorted(EXAMPLE_MODELS.keys()):
            marker = " [current]" if provider == ctx.agent_config["model_provider"] else ""
            models_str = ", ".join(EXAMPLE_MODELS[provider][:3]) + ", ..."
            con.print(f"  {provider}{marker}: [dim]{models_str}[/dim]")
        con.print()
        con.print("Usage: /model <provider>/<model>  (any model name accepted)", style="dim")
        con.print("Examples: /model openai/gpt-5, /model deepseek/deepseek-chat", style="dim")


def _cmd_compact(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent or not agent.working_memory:
        con.print("[yellow]No conversation history to compact.[/yellow]")
        return

    messages = agent.working_memory.messages
    msg_count = len(messages)
    if msg_count == 0:
        con.print("[yellow]No messages to compact.[/yellow]")
        return

    custom_instructions = cmd_args.strip() if cmd_args else None
    cm = agent.tool_config.compression_manager if agent.tool_config else None

    if cm is not None:
        con.print(f"[dim]Compacting {msg_count} messages with LLM summary...[/dim]")
        model = agent.model
        wm = agent.working_memory

        compacted = _run_async_safe(cm.auto_compact(
            messages, model=model, force=True, working_memory=wm,
            custom_instructions=custom_instructions,
        ))
        if compacted:
            con.print(f"[green]Context compacted: {msg_count} messages -> {len(messages)} summary.[/green]")
        else:
            con.print("[yellow]Compaction failed. Falling back to rule-based.[/yellow]")
            _rule_based_compact(agent, messages, msg_count)
    else:
        con.print(f"[dim]Compacting {msg_count} messages (rule-based)...[/dim]")
        _rule_based_compact(agent, messages, msg_count)

    con.print("[dim]Workspace memory preserved.[/dim]")


def _rule_based_compact(current_agent, messages, msg_count):
    con = get_console()
    keep_recent = 6
    if msg_count <= keep_recent:
        con.print("[yellow]Too few messages to compact.[/yellow]")
        return

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    summary_parts = []
    for msg in old_messages:
        content = msg.content or ''
        if isinstance(content, str) and content:
            preview = content[:300] + "..." if len(content) > 300 else content
            summary_parts.append(f"[{msg.role}] {preview}")

    if summary_parts:
        summary = "Previous conversation summary:\n" + "\n".join(summary_parts)
        messages.clear()
        messages.append(Message(role="user", content=f"[Context compressed]\n\n{summary}"))
        messages.append(Message(role="assistant", content="Understood. I have the conversation context. Continuing."))
        messages.extend(recent_messages)
        con.print(f"[green]Context compacted: {msg_count} messages -> {len(messages)} messages.[/green]")
    else:
        messages.clear()
        con.print(f"[green]Context cleared ({msg_count} messages).[/green]")


def _cmd_debug(ctx: CommandContext, cmd_args: str = ""):
    _cmd_title(f"/debug {cmd_args.strip()}" if cmd_args.strip() else "/debug")
    con = get_console()
    con.print("[bold cyan]Debug Info[/bold cyan]")
    con.print(f"  Model: {ctx.agent_config['model_provider']}/{ctx.agent_config['model_name']}")
    con.print(f"  Shell Mode: {'[green]ON[/green]' if ctx.shell_mode else '[dim]OFF[/dim]'}")
    con.print(f"  Work Dir: {ctx.agent_config.get('work_dir') or os.getcwd()}")
    agent = ctx.current_agent
    if agent and agent.working_memory:
        msg_count = len(agent.working_memory.messages)
        con.print(f"  History Messages: {msg_count}")
    if agent and agent.tools:
        con.print(f"  Extra Tools: {len(agent.tools)}")
    if ctx.workspace:
        con.print(f"  Workspace: {ctx.workspace.path}")
    if ctx.skills_registry:
        con.print(f"  Skills Loaded: {len(ctx.skills_registry)}")


def _cmd_reload_skills(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    result = _refresh_skills_session(ctx)
    con.print(f"Reloaded {len(result['skills_registry'])} skills from disk.", style="green")
    return result


def _cmd_cost(ctx: CommandContext, cmd_args: str = ""):
    """Display detailed token usage and cost for the current session."""
    _cmd_title(f"/usage {cmd_args.strip()}" if cmd_args.strip() else "/usage")
    con = get_console()
    tracker = ctx.current_agent.run_response.cost_tracker if ctx.current_agent else None

    if tracker is None or tracker.turns == 0:
        con.print("[yellow]No usage data yet.[/yellow]")
        return

    model_name = f"{ctx.agent_config.get('model_provider', '')}/{ctx.agent_config.get('model_name', '')}"

    total_cache_read = 0
    total_cache_write = 0
    for stat in tracker.model_usage.values():
        total_cache_read += stat.cache_read_tokens
        total_cache_write += stat.cache_write_tokens

    prompt_total = tracker.total_input_tokens + total_cache_read + total_cache_write
    total_all = prompt_total + tracker.total_output_tokens

    ts = ctx.tui_state or {}
    ctx_tokens = ts.get("context_tokens", 0)
    ctx_window = ts.get("context_window", 128000)
    ctx_pct = (ctx_tokens / ctx_window * 100) if ctx_window > 0 else 0
    active_secs = ts.get("active_seconds", 0)

    msg_count = 0
    if ctx.current_agent:
        msg_count = len(ctx.current_agent.working_memory.messages)

    if active_secs < 60:
        duration_str = f"{active_secs:.0f}s"
    elif active_secs < 3600:
        m, s = divmod(int(active_secs), 60)
        duration_str = f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(active_secs), 3600)
        m, _ = divmod(rem, 60)
        duration_str = f"{h}h {m:02d}m"

    session_cost = ts.get("cost_usd", 0.0) if ts else tracker.total_cost_usd
    cost_str = f"~${session_cost:.4f}"

    sep = "─" * 42
    con.print()
    con.print("  [bold cyan]Session Token Usage[/bold cyan]")
    con.print(f"  {sep}")
    con.print(f"  {'Model:':<30} {model_name}")
    con.print(f"  {'Input tokens:':<30} {tracker.total_input_tokens:>12,}")
    if total_cache_read > 0:
        con.print(f"  {'Cache read tokens:':<30} {total_cache_read:>12,}")
    if total_cache_write > 0:
        con.print(f"  {'Cache write tokens:':<30} {total_cache_write:>12,}")
    con.print(f"  {'Output tokens:':<30} {tracker.total_output_tokens:>12,}")
    con.print(f"  {'Prompt tokens (total):':<30} {prompt_total:>12,}")
    con.print(f"  {'Total tokens:':<30} {total_all:>12,}")
    con.print(f"  {'API calls:':<30} {ts.get('total_api_calls', tracker.turns):>12}")
    con.print(f"  {'Session duration:':<30} {duration_str:>12}")
    con.print(f"  {'Total cost:':<30} {cost_str}")
    con.print(f"  {sep}")
    con.print(f"  Current context:  {ctx_tokens:,} / {ctx_window:,} ({ctx_pct:.0f}%)")
    con.print(f"  Messages:         {msg_count}")
    con.print()


def _cmd_export(ctx: CommandContext, cmd_args: str = ""):
    """Save conversation history to a JSON file (excludes system prompts)."""
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation to save.[/yellow]")
        return

    messages = agent.working_memory.messages
    export_msgs = []
    for msg in messages:
        if msg.role == "system":
            continue
        content = msg.content or ""
        if isinstance(content, list):
            content = str(content)
        if isinstance(content, str):
            content = content.strip()
        entry = {"role": msg.role, "content": content}
        if msg.tool_calls:
            entry["tool_calls"] = len(msg.tool_calls)
        export_msgs.append(entry)

    if not export_msgs:
        con.print("[yellow]No messages to save.[/yellow]")
        return

    filename = cmd_args.strip() if cmd_args.strip() else f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if not filename.endswith(".json"):
        filename += ".json"

    model_name = f"{ctx.agent_config.get('model_provider', '')}/{ctx.agent_config.get('model_name', '')}"

    data = {
        "model": model_name,
        "session_id": agent.session_id,
        "exported_at": datetime.now().isoformat(),
        "messages": export_msgs,
    }
    Path(filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    con.print(f"  [green]Saved {len(export_msgs)} messages to {filename}[/green]")


def _cmd_permissions(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if cmd_args.strip():
        new_mode = cmd_args.strip().lower()
        if new_mode not in ("allow-all", "auto", "strict"):
            con.print(f"[red]Invalid mode: {new_mode}. Use: allow-all, auto, strict[/red]")
            return
        if ctx.permission_manager:
            ctx.permission_manager.mode = new_mode
            ctx.permission_manager.session_allowed.clear()
            con.print(f"[green]Permission mode set to: {new_mode}[/green]")
        return

    if ctx.permission_manager:
        con.print(f"[bold cyan]Permission Mode: {ctx.permission_manager.mode}[/bold cyan]")
        if ctx.permission_manager.session_allowed:
            con.print(f"  Session-allowed tools: {', '.join(sorted(ctx.permission_manager.session_allowed))}")
        con.print()
        con.print("  [dim]allow-all[/dim]  - auto-approve everything")
        con.print("  [dim]auto[/dim]      - prompt for write/execute, auto-approve reads")
        con.print("  [dim]strict[/dim]    - prompt for every tool call")
        con.print()
        con.print("Usage: /permissions <mode>", style="dim")


def _cmd_yolo(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if not ctx.permission_manager:
        return
    if ctx.permission_manager.mode == "allow-all":
        ctx.permission_manager.mode = "auto"
        ctx.permission_manager.session_allowed.clear()
        con.print("[cyan]YOLO OFF[/cyan] -- back to auto-approve mode")
    else:
        ctx.permission_manager.mode = "allow-all"
        con.print("[bold yellow]YOLO ON[/bold yellow] -- all tool calls auto-approved")


def _cmd_paste(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.attached_images is None or ctx.image_counter is None:
        con.print("[dim]Image paste not available.[/dim]")
        return
    from agentica.cli.clipboard import has_clipboard_image
    from agentica.cli.interactive import _try_attach_clipboard_image
    if has_clipboard_image():
        if _try_attach_clipboard_image(ctx.attached_images, ctx.image_counter):
            img = ctx.attached_images[-1]
            size_kb = img.stat().st_size // 1024 if img.exists() else 0
            con.print(f"  [green]Image #{len(ctx.attached_images)} attached: {img.name} ({size_kb}KB)[/green]")
        else:
            con.print("  [dim]Clipboard has an image but extraction failed.[/dim]")
    else:
        con.print("  [dim]No image found in clipboard.[/dim]")


def _cmd_image(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.attached_images is None or ctx.image_counter is None:
        con.print("[dim]Image attachment not available.[/dim]")
        return
    raw_args = cmd_args.strip()
    if not raw_args:
        con.print("  [dim]Usage: /image <path>  e.g. /image /path/to/image.png[/dim]")
        return

    from agentica.cli.interactive import _split_path_input, _resolve_attachment_path
    path_token, _ = _split_path_input(raw_args)
    image_path = _resolve_attachment_path(path_token)
    if image_path is None:
        con.print(f"  [dim]File not found: {path_token}[/dim]")
        return
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        con.print(f"  [dim]Not a supported image file: {image_path.name}[/dim]")
        return

    ctx.attached_images.append(image_path)
    ctx.image_counter[0] += 1
    con.print(f"  [green]Attached image: {image_path.name}[/green]")


def _extract_queue_text(item) -> str:
    """Extract display text from a queue payload (str, tuple, etc.)."""
    if isinstance(item, tuple):
        if item[0] == "__BTW__":
            return str(item[1])
        return str(item[0])  # (text, images)
    return str(item)


def _cmd_queue(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    pq = ctx.pending_queue
    args = cmd_args.strip()

    if not args:
        items = pq.peek_all() if pq else []
        if items:
            con.print(f"  [cyan]Queued messages ({len(items)}):[/cyan]")
            for i, item in enumerate(items):
                preview = _extract_queue_text(item)[:80]
                con.print(f"    {i + 1}. [dim]{preview}[/dim]")
            con.print()
        con.print("  [dim]Usage: /queue <prompt>  |  /queue list  |  /queue clear  |  /queue remove <n>[/dim]")
        con.print("  [dim]See also: /steer (nudge current run) · /background (run in parallel)[/dim]")
        return

    sub = args.split(maxsplit=1)
    subcommand = sub[0].lower()

    if subcommand == "list":
        if pq is None or pq.empty():
            con.print("  [dim]Queue is empty.[/dim]")
            return
        items = pq.peek_all()
        con.print(f"  [cyan]Queued messages ({len(items)}):[/cyan]")
        for i, item in enumerate(items):
            con.print(f"    {i + 1}. [dim]{_extract_queue_text(item)[:80]}[/dim]")
        return

    if subcommand == "clear":
        if pq is None:
            return
        n = pq.qsize()
        pq.clear()
        con.print(f"  [green]Cleared {n} queued message(s).[/green]")
        return

    if subcommand == "remove":
        if pq is None:
            return
        idx_str = sub[1].strip() if len(sub) > 1 else ""
        if not idx_str.isdigit():
            con.print("  [dim]Usage: /queue remove <number>[/dim]")
            return
        idx = int(idx_str) - 1
        if pq.remove_index(idx):
            con.print(f"  [green]Removed queued message #{idx + 1}.[/green]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1}[/red]")
        return

    # Default: queue a prompt
    pq.put(args)
    preview = args[:80] + ("..." if len(args) > 80 else "")
    if not ctx.agent_running:
        con.print(f"  Queued: {preview}")


def _cmd_steer(ctx: CommandContext, cmd_args: str = ""):
    """Inject guidance into the running agent's tool loop (mid-task).

    Unlike /queue (runs as a fresh turn after the current run finishes), /steer
    is consumed between tool batches of the CURRENT run, so the agent can course-
    correct without being interrupted.
    """
    con = get_console()
    guidance = cmd_args.strip()
    if not guidance:
        con.print("  [dim]Usage: /steer <guidance>  (e.g. /steer don't change the API, keep it compatible)[/dim]")
        return
    if not ctx.agent_running:
        con.print("  [yellow]Agent isn't running — use /queue to send this as the next message instead.[/yellow]")
        return
    if ctx.current_agent.steer(guidance):
        con.print("  [green]Steering queued — the agent will see it on its next step.[/green]")


def _checkpoint_manager(ctx: CommandContext):
    """Build a disk-backed CheckpointManager scoped to the current session."""
    from agentica.checkpoint import CheckpointManager
    session_id = ctx.current_agent.session_id or "default"
    return CheckpointManager(session_id=session_id)


def _resolve_ckpt_path(ctx: CommandContext, raw: str) -> str:
    """Resolve a user-supplied path against the agent's work_dir."""
    p = os.path.expanduser(raw)
    if os.path.isabs(p):
        return p
    base = ctx.current_agent.work_dir or os.getcwd()
    return os.path.join(str(base), p)


def _work_dir_root(ctx: CommandContext) -> Path:
    return Path(ctx.current_agent.work_dir or os.getcwd()).expanduser().resolve()


def _is_inside_work_dir(path: str, root: Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _cmd_checkpoint(ctx: CommandContext, cmd_args: str = ""):
    """Manual, durable, multi-file checkpoints for the current session.

    /checkpoint [list]                 -> list checkpoints (newest first)
    /checkpoint create <label> <path...> -> snapshot files' current content
    /checkpoint diff <id>              -> unified diff snapshot -> current
    /checkpoint restore <id>           -> roll files back to the snapshot
    """
    con = get_console()
    cm = _checkpoint_manager(ctx)
    args = cmd_args.strip()
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        con.print(f"  [red]Invalid checkpoint command: {exc}[/red]")
        return
    sub = parts[0].lower() if parts else "list"

    def _find(cid_prefix: str):
        ck = cm.get(cid_prefix)
        if ck is not None:
            return ck
        matches = [c for c in cm.list() if c.id.startswith(cid_prefix)]
        return matches[0] if len(matches) == 1 else None

    if sub == "list":
        items = cm.list()
        if not items:
            con.print("  [dim]No checkpoints. Create one: /checkpoint create <label> <path...>[/dim]")
            return
        con.print(f"  [cyan]Checkpoints ({len(items)}):[/cyan]")
        for c in items:
            con.print(f"    {c.id[:18]}  [dim]{c.created_at}[/dim]  {c.label}  ([dim]{len(c.files)} file(s)[/dim])")
        return

    if sub == "create":
        if len(parts) < 3:
            con.print("  [dim]Usage: /checkpoint create <label> <path> [more paths...][/dim]")
            return
        label = parts[1]
        paths = [_resolve_ckpt_path(ctx, p) for p in parts[2:]]
        ck = cm.create(label, paths)
        con.print(f"  [green]Created checkpoint {ck.id[:18]} ({label}) with {len(ck.files)} file(s).[/green]")
        return

    if sub in ("diff", "restore"):
        if len(parts) < 2:
            con.print(f"  [dim]Usage: /checkpoint {sub} <id>{' --yes' if sub == 'restore' else ''}[/dim]")
            return
        ck = _find(parts[1])
        if ck is None:
            con.print(f"  [red]No checkpoint matching '{parts[1]}'.[/red]")
            return
        if sub == "diff":
            con.print(cm.diff(ck.id))
            return

        root = _work_dir_root(ctx)
        outside = [f.path for f in ck.files if not _is_inside_work_dir(f.path, root)]
        if outside:
            con.print(
                f"  [red]Refusing to restore checkpoint files outside work_dir: {root}[/red]"
            )
            for path in outside[:5]:
                con.print(f"    [dim]{path}[/dim]")
            return

        diff_text = cm.diff(ck.id)
        deletions = [f.path for f in ck.files if not f.existed and Path(f.path).exists()]
        if "--yes" not in parts:
            con.print(diff_text)
            if deletions:
                con.print("  [yellow]Restore will delete file(s) created after the checkpoint:[/yellow]")
                for path in deletions:
                    con.print(f"    [dim]{path}[/dim]")
            con.print("  [yellow]Re-run with --yes to restore this checkpoint.[/yellow]")
            return

        restored = cm.restore(ck.id)
        con.print(f"  [green]Restored {len(restored)} file(s) from {ck.id[:18]} ({ck.label}).[/green]")
        return

    con.print("  [dim]Usage: /checkpoint list | create <label> <path...> | diff <id> | restore <id> --yes[/dim]")


def _cmd_reasoning(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.tui_state is None:
        return
    arg = cmd_args.strip().lower()
    if not arg:
        state = "ON" if ctx.tui_state.get("show_reasoning", True) else "OFF"
        con.print(f"  Reasoning display: [bold]{state}[/bold]")
        con.print("  [dim]Usage: /reasoning on|off[/dim]")
        return
    if arg in ("show", "on", "true", "1"):
        ctx.tui_state["show_reasoning"] = True
        con.print("  [green]Reasoning display: ON[/green]")
    elif arg in ("hide", "off", "false", "0"):
        ctx.tui_state["show_reasoning"] = False
        con.print("  [green]Reasoning display: OFF[/green]")
    else:
        con.print(f"  [dim]Unknown argument: {arg}. Use: on, off[/dim]")


def _cmd_retry(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation to retry.[/yellow]")
        return
    wm = agent.working_memory
    last_user_msg = None
    for msg in reversed(wm.messages):
        if msg.role == "user":
            last_user_msg = msg
            break
    if last_user_msg is None or not last_user_msg.content:
        con.print("[yellow]No user message found to retry.[/yellow]")
        return
    user_text = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)
    if wm.runs:
        wm.runs.pop()
    while wm.messages and wm.messages[-1].role in ("assistant", "tool"):
        wm.messages.pop()
    if wm.messages and wm.messages[-1].role == "user":
        wm.messages.pop()
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(user_text)
        preview = user_text[:60] + ("..." if len(user_text) > 60 else "")
        con.print(f"  [green]Retrying: {preview}[/green]")


def _cmd_undo(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation history.[/yellow]")
        return
    wm = agent.working_memory
    if not wm.messages:
        con.print("[yellow]No messages to undo.[/yellow]")
        return
    if wm.runs:
        wm.runs.pop()
    removed = 0
    while wm.messages and wm.messages[-1].role in ("assistant", "tool"):
        wm.messages.pop()
        removed += 1
    if wm.messages and wm.messages[-1].role == "user":
        wm.messages.pop()
        removed += 1
    if removed > 0:
        con.print(f"  [green]Undone last exchange ({removed} messages removed).[/green]")
    else:
        con.print("[yellow]Nothing to undo.[/yellow]")


def _cmd_btw(ctx: CommandContext, cmd_args: str = ""):
    """Ephemeral side question — dispatched as concurrent task, no tools, not persisted."""
    con = get_console()
    question = cmd_args.strip()
    if not question:
        con.print("  [dim]Usage: /btw <question>   (quick aside; for a persisted parallel task use /background)[/dim]")
        return
    if ctx.current_agent is None:
        con.print("[yellow]No active agent.[/yellow]")
        return
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(("__BTW__", question))
        con.print(f"  [dim]Side question: {question[:60]}{'...' if len(question) > 60 else ''}[/dim]")


def _cmd_statusbar(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.tui_state is None:
        return
    current = ctx.tui_state.get("statusbar_visible", True)
    ctx.tui_state["statusbar_visible"] = not current
    state = "OFF" if current else "ON"
    con.print(f"  [green]Status bar: {state}[/green]")


def _cmd_background(ctx: CommandContext, cmd_args: str = ""):
    """Run a prompt in the background (independent agent with context snapshot)."""
    con = get_console()
    prompt = cmd_args.strip()
    if not prompt:
        if ctx.bg_tasks:
            con.print(f"  [cyan]Active background tasks ({len(ctx.bg_tasks)}):[/cyan]")
            for tid, info in ctx.bg_tasks.items():
                con.print(f"    #{info['num']} [dim]{tid}[/dim] {info['prompt'][:60]}")
        else:
            con.print("  [dim]No active background tasks.[/dim]")
        con.print("  [dim]Usage: /background <prompt>  |  /stop to kill all[/dim]")
        con.print("  [dim]See also: /queue (next turn, same session) · /btw (quick aside, not persisted)[/dim]")
        return

    ctx.bg_task_counter += 1
    task_num = ctx.bg_task_counter
    task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{task_num}"

    ctx.bg_tasks[task_id] = {"thread": None, "agent": None, "prompt": prompt, "num": task_num}

    # Capture references needed by the background thread
    agent_config = ctx.agent_config
    extra_tools = ctx.extra_tools
    workspace = ctx.workspace
    skills_registry = ctx.skills_registry
    bg_tasks = ctx.bg_tasks

    # Snapshot current conversation context for the background agent.
    # History is loaded via working_memory.runs (not .messages) by the runner,
    # so we must inject a synthetic AgentRun with the snapshot messages.
    context_snapshot = []
    main_agent = ctx.current_agent
    if main_agent and main_agent.working_memory and main_agent.working_memory.messages:
        for msg in main_agent.working_memory.messages:
            if msg.role in ("user", "assistant") and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if len(content) > 500:
                    content = content[:500] + "..."
                context_snapshot.append(Message(role=msg.role, content=content))
        # Keep only last 10 messages to avoid blowing up context
        context_snapshot = context_snapshot[-10:]

    def _run_bg():
        bg_config = dict(agent_config)
        bg_config["session_id"] = _generate_session_id()
        bg_config["debug"] = False
        bg_agent = create_agent(bg_config, extra_tools, workspace, skills_registry)
        bg_tasks[task_id]["agent"] = bg_agent

        # Inject context snapshot as a synthetic AgentRun so the runner
        # picks it up via get_messages_from_last_n_runs().
        if context_snapshot:
            synthetic_run = AgentRun(
                response=RunResponse(messages=context_snapshot),
            )
            bg_agent.working_memory.runs.append(synthetic_run)

        result_text = ""
        try:
            response = bg_agent.run_sync(prompt)
            result_text = response.content if response else ""
        except Exception as e:
            if bg_agent._cancelled:
                result_text = "(cancelled)"
            else:
                result_text = f"Error: {e}"
        finally:
            bg_tasks.pop(task_id, None)

        # Use shared box printer from interactive
        from agentica.cli.interactive import _print_boxed_result
        _print_boxed_result(
            f"Background #{task_num}", prompt, result_text or "",
            color="bright_magenta",
        )

    thread = threading.Thread(target=_run_bg, daemon=True, name=task_id)
    bg_tasks[task_id]["thread"] = thread
    thread.start()
    preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
    con.print(f"  [green]Background #{task_num} started:[/green] {preview}")


def _cmd_stop(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if not ctx.bg_tasks:
        con.print("  [dim]No running background tasks.[/dim]")
        return
    count = len(ctx.bg_tasks)
    for tid, info in list(ctx.bg_tasks.items()):
        agent = info.get("agent")
        if agent is not None:
            agent.cancel()
    con.print(f"  [green]Stopped {count} background task(s).[/green]")


# ==================== /goal & /subgoal ====================

def _attach_goal_tool(agent: Any) -> None:
    """Idempotently attach the GoalTool so the model can break the loop.

    Delegates to ``Agent.enable_goal_tool()`` (single source of truth shared
    with the SDK entry point ``Agent.run_goal``).
    """
    if agent is None or agent._session_log is None:
        return
    agent.enable_goal_tool()


def _detach_goal_tool(agent: Any) -> None:
    if agent is None or not agent.tools:
        return
    agent.tools = [t for t in agent.tools if not isinstance(t, GoalTool)]


def _ensure_goal_manager(ctx: CommandContext) -> Optional[GoalManager]:
    """Return existing manager, or build one bound to the current agent's
    SessionLog. Returns None if the agent has no session_log (impossible in
    normal CLI flow, but keep defensive).
    """
    if ctx.goal_manager is not None:
        return ctx.goal_manager
    agent = ctx.current_agent
    if agent is None or agent._session_log is None:
        return None
    # Delegate to Agent.get_goal_manager() so CLI and SDK share one
    # construction path; the agent also caches the manager on itself.
    return agent.get_goal_manager()


def _parse_goal_set_args(raw: str) -> tuple[str, Dict[str, Any], Optional[str]]:
    """Parse /goal budget flags while keeping plain text fully compatible."""
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return "", {}, str(exc)

    budgets: Dict[str, Any] = {}
    objective: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"--turns", "--turn-budget"}:
            if i + 1 >= len(tokens):
                return "", {}, f"{token} requires an integer value"
            try:
                budgets["turn_budget"] = int(tokens[i + 1])
            except ValueError:
                return "", {}, f"{token} must be an integer"
            i += 2
            continue
        if token.startswith("--turns=") or token.startswith("--turn-budget="):
            value = token.split("=", 1)[1]
            try:
                budgets["turn_budget"] = int(value)
            except ValueError:
                return "", {}, "--turns must be an integer"
            i += 1
            continue
        if token == "--tokens":
            if i + 1 >= len(tokens):
                return "", {}, "--tokens requires an integer value"
            try:
                budgets["token_budget"] = int(tokens[i + 1])
            except ValueError:
                return "", {}, "--tokens must be an integer"
            i += 2
            continue
        if token.startswith("--tokens="):
            try:
                budgets["token_budget"] = int(token.split("=", 1)[1])
            except ValueError:
                return "", {}, "--tokens must be an integer"
            i += 1
            continue
        if token == "--wall":
            if i + 1 >= len(tokens):
                return "", {}, "--wall requires a number of seconds"
            try:
                budgets["wall_clock_budget_sec"] = float(tokens[i + 1])
            except ValueError:
                return "", {}, "--wall must be a number of seconds"
            i += 2
            continue
        if token.startswith("--wall="):
            try:
                budgets["wall_clock_budget_sec"] = float(token.split("=", 1)[1])
            except ValueError:
                return "", {}, "--wall must be a number of seconds"
            i += 1
            continue
        objective.append(token)
        i += 1

    for key, value in budgets.items():
        if value <= 0:
            return "", {}, f"{key} must be positive"
    return " ".join(objective).strip(), budgets, None


def _cmd_goal(ctx: CommandContext, cmd_args: str = ""):
    """
    /goal                  -> show status
    /goal status           -> show status (alias)
    /goal <objective>      -> set new objective + enqueue first turn
    /goal --turns 5 <objective>       -> set turn budget
    /goal --tokens 80000 <objective>  -> set token budget
    /goal --wall 1800 <objective>     -> set wall-clock budget seconds
    /goal pause            -> pause auto-continuation
    /goal resume           -> resume + enqueue continuation
    /goal clear            -> clear current goal
    """
    con = get_console()
    arg = (cmd_args or "").strip()
    _cmd_title(f"/goal {arg}" if arg else "/goal")

    mgr = _ensure_goal_manager(ctx)
    if mgr is None:
        con.print("  [yellow]No active agent / session log unavailable.[/yellow]")
        return

    sub = arg.lower()

    # ── status (default) ──
    if not arg or sub == "status":
        con.print(f"  {mgr.status_line()}")
        if mgr.load() is None:
            con.print("  [dim]Usage: /goal <objective>  |  pause | resume | clear[/dim]")
        return {"goal_manager": mgr}

    # ── pause / resume / clear are safe while agent is running ──
    if sub == "pause":
        state = mgr.pause("user")
        if state is None:
            con.print("  [dim]No goal to pause.[/dim]")
        else:
            con.print(f"  ⊙ Goal paused: {state.objective}")
        return {"goal_manager": mgr}

    if sub == "resume":
        if ctx.agent_running:
            con.print("  [yellow]Agent is currently running; goal will continue automatically.[/yellow]")
            return {"goal_manager": mgr}
        state = mgr.resume()
        if state is None:
            con.print("  [dim]No goal to resume.[/dim]")
            return {"goal_manager": mgr}
        if state.status != "active":
            con.print(f"  [dim]Goal status is {state.status}, cannot resume.[/dim]")
            return {"goal_manager": mgr}
        # Re-attach the tool (might have been detached by /clear race).
        _attach_goal_tool(ctx.current_agent)
        # Re-prime the loop with a continuation prompt.
        if ctx.pending_queue is not None:
            ctx.pending_queue.put(mgr.next_continuation_prompt())
        con.print(f"  ↻ Goal resumed: {state.objective}")
        return {"goal_manager": mgr}

    if sub == "clear":
        mgr.clear()
        _detach_goal_tool(ctx.current_agent)
        con.print("  ✗ Goal cleared.")
        return {"goal_manager": mgr}

    # ── set new objective ──
    if ctx.agent_running:
        con.print("  [yellow]Cannot set a new goal while the agent is running. "
                  "Wait for the current turn or use /goal pause/clear.[/yellow]")
        return {"goal_manager": mgr}

    objective, budgets, parse_error = _parse_goal_set_args(arg)
    if parse_error:
        con.print(f"  [red]Invalid goal options: {parse_error}[/red]")
        con.print("  [dim]Usage: /goal [--turns N] [--tokens N] [--wall SECONDS] <objective>[/dim]")
        return {"goal_manager": mgr}
    try:
        state = mgr.set(objective, **budgets)
    except ValueError as exc:
        con.print(f"  [red]Invalid goal: {exc}[/red]")
        return {"goal_manager": mgr}

    # Overwrite the session's TaskAnchor so prompts.py + workspace retrieval
    # bind to the standing goal, not the latest user message.
    agent = ctx.current_agent
    if agent is not None:
        agent.task_anchor = TaskAnchor(
            goal=state.objective, source_query=state.objective,
            source="goal",
        )
        agent._anchor_session_id = agent.session_id

    # Workspace freeze timing: if workspace was already frozen on an earlier
    # query, retrieval will NOT re-bind to the new goal. Document the limit.
    if agent is not None and agent.workspace is not None:
        try:
            already_frozen = agent.workspace.get_frozen_context() is not None
        except Exception:
            already_frozen = False
        if already_frozen:
            con.print(
                "  [dim]ℹ Workspace memory was already frozen against an earlier query. "
                "Goal-bound retrieval will activate from the next /new session.[/dim]"
            )

    # Attach the update_goal tool so the model can break the loop.
    _attach_goal_tool(agent)

    budget_bits = [f"{state.turn_budget} turns"]
    if state.token_budget is not None:
        budget_bits.append(f"{state.token_budget:,} tokens")
    if state.wall_clock_budget_sec is not None:
        budget_bits.append(f"{state.wall_clock_budget_sec:.0f}s wall")
    con.print(f"  ⊙ Goal set ({', '.join(budget_bits)}): {state.objective}")

    # Kick off the first turn so the user doesn't need to send a follow-up.
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(state.objective)

    return {"goal_manager": mgr}


def _cmd_subgoal(ctx: CommandContext, cmd_args: str = ""):
    """
    /subgoal               -> list subgoals
    /subgoal <text>        -> add a subgoal to active goal
    /subgoal remove <n>    -> remove the n-th subgoal (1-based)
    /subgoal clear         -> drop all subgoals
    """
    con = get_console()
    arg = (cmd_args or "").strip()
    _cmd_title(f"/subgoal {arg}" if arg else "/subgoal")

    mgr = _ensure_goal_manager(ctx)
    if mgr is None or mgr.load() is None:
        con.print("  [yellow]No active goal — set one with /goal first.[/yellow]")
        return

    if not arg:
        state = mgr.load()
        if not state.subgoals:
            con.print("  [dim]No subgoals.[/dim]")
        else:
            con.print(f"  Subgoals ({len(state.subgoals)}):")
            for i, sg in enumerate(state.subgoals, 1):
                con.print(f"    {i}. {sg}")
        con.print("  [dim]Usage: /subgoal <text>  |  remove <n>  |  clear[/dim]")
        return {"goal_manager": mgr}

    parts = arg.split(maxsplit=1)
    sub = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "remove" or sub == "rm":
        if not rest.isdigit():
            con.print("  [dim]Usage: /subgoal remove <number>[/dim]")
            return {"goal_manager": mgr}
        removed = mgr.remove_subgoal(int(rest))
        if removed is None:
            con.print(f"  [red]Invalid subgoal index: {rest}[/red]")
        else:
            con.print(f"  ✗ Removed subgoal: {removed}")
        return {"goal_manager": mgr}

    if sub == "clear":
        n = mgr.clear_subgoals()
        con.print(f"  ✗ Cleared {n} subgoal(s).")
        return {"goal_manager": mgr}

    # Default: add the whole argument as a subgoal.
    try:
        text = mgr.add_subgoal(arg)
    except ValueError as exc:
        con.print(f"  [red]{exc}[/red]")
        return {"goal_manager": mgr}
    con.print(f"  + Subgoal added: {text}")
    return {"goal_manager": mgr}


# ==================== Command Registry ====================

COMMAND_REGISTRY = {
    # Session
    "/new":           (_cmd_newchat,       "Start a new chat session"),
    "/clear":         (_cmd_clear,         "Clear screen and reset"),
    "/reset":         (_cmd_clear,         "Clear screen and reset (alias)"),
    "/history":       (_cmd_history,       "Show conversation history"),
    "/export":        (_cmd_export,        "Save conversation to JSON"),
    "/save":          (_cmd_export,        "Save conversation to JSON (alias)"),
    "/retry":         (_cmd_retry,         "Retry the last message (resend to agent)"),
    "/undo":          (_cmd_undo,          "Remove the last user/assistant exchange"),
    "/compact":       (_cmd_compact,       "Compact context (summarize history)"),
    "/resume":        (_cmd_resume,        "Resume a previous session"),
    "/goal":          (_cmd_goal,          "Set or manage a standing goal (auto-continues until done)"),
    "/subgoal":       (_cmd_subgoal,       "Add or manage acceptance criteria on the active goal"),
    "/btw":           (_cmd_btw,           "Quick aside answered in parallel \u2014 no tools, not persisted"),
    "/queue":         (_cmd_queue,         "Run as the NEXT turn after the current run finishes | list | clear | remove <n>"),
    "/q":             (_cmd_queue,         "Run as the next turn after current run (alias)"),
    "/background":    (_cmd_background,    "Run NOW in a parallel independent agent (own session)"),
    "/bg":            (_cmd_background,    "Run now in a parallel independent agent (alias)"),
    "/stop":          (_cmd_stop,          "Kill all running background tasks"),
    "/steer":         (_cmd_steer,         "Course-correct the CURRENT run mid-task (injected between tool batches)"),
    "/checkpoint":    (_cmd_checkpoint,    "Durable file snapshots: list | create <label> <path...> | diff <id> | restore <id>"),
    # Model & Config
    "/model":         (_cmd_model,         "View or switch model"),
    "/config":        (_cmd_config,        "Show current configuration"),
    "/cost":          (_cmd_cost,          "Show token usage and cost"),
    "/usage":         (_cmd_cost,          "Show token usage and cost (alias)"),
    "/debug":         (_cmd_debug,         "Show debug info"),
    "/reasoning":     (_cmd_reasoning,     "Toggle reasoning display: on | off"),
    "/statusbar":     (_cmd_statusbar,     "Toggle the status bar visibility"),
    "/sb":            (_cmd_statusbar,     "Toggle the status bar (alias)"),
    # Tools & Skills
    "/tools":         (_cmd_tools,         "Manage tools: add | remove | info | search"),
    "/skills":        (_cmd_skills,        "Manage skills: search | browse | install | remove | inspect | tap"),
    "/extensions":    (_cmd_skills,        "Manage skills (alias for /skills)"),
    # Permissions
    "/permissions":   (_cmd_permissions,   "View or set permission mode"),
    "/yolo":          (_cmd_yolo,          "Toggle YOLO mode (auto-approve all)"),
    # Media
    "/paste":         (_cmd_paste,         "Paste image from clipboard"),
    "/image":         (_cmd_image,         "Attach a local image file"),
    # Other
    "/help":          (_cmd_help,          "Show available commands"),
    "/exit":          (_cmd_exit,          "Exit the CLI"),
    "/quit":          (_cmd_exit,          "Exit the CLI (alias)"),
}

COMMAND_HANDLERS = {cmd: handler for cmd, (handler, _) in COMMAND_REGISTRY.items()}
