# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tools, skills, agents, and permissions slash commands
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from agentica.cli.runtime import (
    get_console,
    BUILTIN_TOOLS,
    TOOL_REGISTRY,
    configure_tools,
)
from agentica.skills import (
    install_skills,
    list_installed_skills,
    remove_skill,
)

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.helpers import (
    _get_defined_agents_for_display,
    _get_subagent_loader,
    _load_custom_tool_module,
    _refresh_skills_session,
    _safe_tool_module_name,
    _set_skill_runtime_state,
)
from agentica.cli.commands.cron_cmd import _ask_text_via_tui, _confirm_via_tui




def _cmd_agents(ctx: CommandContext, cmd_args: str = ""):
    """Manage subagents: list, create, reload, remove.

    All types come from Markdown definitions. Project and user files override
    package defaults with the same file stem.
    """
    con = get_console()
    args_str = cmd_args.strip()
    parts = shlex.split(args_str) if args_str else []
    subcmd = parts[0].lower() if parts else ""
    sub_args = parts[1:]

    # ── /agents create <name> — interactive, writes .agentica/agents/<name>.md ──
    if subcmd == "create":
        if not sub_args:
            con.print("  [dim]Usage: /agents create <name>[/dim]")
            return
        name = sub_args[0]
        loader = _get_subagent_loader()
        description = _ask_text_via_tui(ctx, "  Description: ")
        if not description:
            con.print("  [red]Description is required.[/red]")
            return
        tools_raw = _ask_text_via_tui(ctx, "  Allowed tools (comma-separated, blank to inherit parent): ")
        allowed_tools = None
        if tools_raw:
            allowed_tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        system_prompt = f"You are a {name} specialist. {description}\n\n(Describe how this subagent should behave.)"
        try:
            path = loader.create_agent_file(
                name=name,
                description=description,
                system_prompt=system_prompt,
                allowed_tools=allowed_tools,
            )
        except Exception as exc:
            con.print(f"  [red]Failed to create agent: {exc}[/red]")
            return
        con.print(f"  [green]Created subagent '{name}' at {path}[/green]")
        con.print("  [dim]Edit the .md file to customize its system prompt, then /agents reload.[/dim]")
        return

    # ── /agents reload — rescan disk and re-register ──
    if subcmd == "reload":
        loader = _get_subagent_loader()
        count = loader.load_all_agents()
        con.print(f"  [green]Loaded {count} subagent(s) from disk.[/green]")
        return

    # ── /agents remove <name> — delete file + unregister ──
    if subcmd in ("remove", "rm"):
        if not sub_args:
            con.print("  [dim]Usage: /agents remove <name>[/dim]")
            return
        name = sub_args[0]
        loader = _get_subagent_loader()
        removed = loader.remove_agent_file(name)
        if removed:
            con.print(f"  [green]Removed subagent '{name}'.[/green]")
        else:
            con.print(f"  [dim]No agent file found for '{name}'.[/dim]")
        return

    # ── /agents (no args) or /agents list ──
    if subcmd and subcmd != "list":
        con.print(f"  [red]Unknown subcommand: {subcmd}[/red]")
        con.print("  [dim]Usage: /agents [list | create <name> | reload | remove <name>][/dim]")
        return

    defined_agents = _get_defined_agents_for_display()
    if defined_agents:
        con.print(f"  [bold]Available subagents ({len(defined_agents)}):[/bold]")
        for agent in defined_agents:
            agent_id = agent.get("id", "?")
            display_name = agent.get("name") or agent_id
            desc = agent.get("description") or ""
            desc_first = desc.split("\n")[0].strip()
            tools = agent.get("allowed_tools")
            if tools is None:
                tools_str = "(inherit parent)"
            elif tools:
                tools_str = ", ".join(tools)
            else:
                tools_str = "(none)"
            source = agent.get("source") or "runtime"
            model_tier = agent.get("model_tier") or "auxiliary"
            con.print(
                f"    [green]●[/green] [bold]{agent_id:<12}[/bold] "
                f"{display_name} — {desc_first}"
            )
            con.print(
                f"      [dim]source: {source} | model: {model_tier} | tools: {tools_str}[/dim]"
            )
            path = agent.get("path")
            if path:
                con.print(f"      [dim]file: {path}[/dim]")
    else:
        con.print("  [dim]No subagents found. Create one with /agents create <name>.[/dim]")
    con.print()
    con.print("  [dim]Commands: /agents [list] | create <name> | reload | remove <name>[/dim]")



def _cmd_tools(ctx: CommandContext, cmd_args: str = ""):
    """Manage tools: list, add, remove, info, search."""
    con = get_console()
    args_str = cmd_args.strip()
    parts = args_str.split(None, 1) if args_str else []
    subcmd = parts[0].lower() if parts else ""
    sub_args = parts[1].strip() if len(parts) > 1 else ""

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

    # ── /tools add-from <name> — load a custom tool from .agentica/tools/<name>.py ──
    if subcmd == "add-from":
        raw_name = sub_args.strip()
        if not raw_name:
            con.print("  [dim]Usage: /tools add-from <name>  (loads .agentica/tools/<name>.py)[/dim]")
            return
        name = _safe_tool_module_name(raw_name)
        if name is None:
            con.print(
                f"  [red]Invalid tool name: {raw_name!r}. Use a plain filename (no path, no ~, no module:attr).[/red]"
            )
            return
        agent = ctx.current_agent
        if not agent:
            con.print("  [red]No active agent.[/red]")
            return
        work_dir = agent.work_dir or os.getcwd()
        file_path = Path(work_dir) / ".agentica" / "tools" / f"{name}.py"
        if not file_path.is_file():
            con.print(f"  [red]Tool file not found: {file_path}[/red]")
            return
        con.print(f"  [yellow]About to load and execute: {file_path}[/yellow]")
        con.print("  [yellow]This runs the module's top-level code (arbitrary code execution).[/yellow]")
        if not _confirm_via_tui(ctx, "Proceed?"):
            con.print("  [dim]Aborted.[/dim]")
            return
        loaded = _load_custom_tool_module(name, file_path)
        if loaded is None:
            return
        if agent.tools is None:
            agent.tools = []
        agent.tools.append(loaded)
        con.print(f"  [green]Loaded tool '{name}' from {file_path}[/green]")
        return

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
                    funcs = tool.functions if hasattr(tool, "functions") else {}
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
    con.print(
        f"  [green]● = active ({active_count})[/green]  [dim]○ = available ({len(all_tools) - active_count})[/dim]"
    )
    con.print(
        f"  [dim]Commands: /tools add <name> | add-from <name> | remove <name> | info <name> | search <keyword>[/dim]"
    )
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
            con.print(
                f"    [bold]{r.name:<25}[/bold] [{trust_style}]{r.trust_level:<10}[/{trust_style}] [dim]{r.source}[/dim]"
            )
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
        page_items = results[start : start + page_size]
        con.print(f"  [bold cyan]Skills Hub ({total} skills, page {page}/{total_pages})[/bold cyan]")
        con.print()
        for i, r in enumerate(page_items, start=start + 1):
            trust_style = {"trusted": "green"}.get(r.trust_level, "yellow")
            con.print(
                f"    {i:>3}. [bold]{r.name:<25}[/bold] [{trust_style}]{r.trust_level:<10}[/{trust_style}] [dim]{r.source}[/dim]"
            )
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

    # ── /skills enable|disable <name> — runtime enable/disable via runtime_config.yaml ──
    if subcommand in ("enable", "disable"):
        if not sub_args:
            con.print(f"  [dim]Usage: /skills {subcommand} <name>[/dim]")
            return
        name = sub_args[0]
        path = _set_skill_runtime_state(ctx, name, subcommand == "enable")
        if path is None:
            return
        state = "enabled" if subcommand == "enable" else "disabled"
        con.print(f"  [green]Skill '{name}' {state} in {path}[/green]")
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

    con.print(
        "  [dim]Commands: search <q> | browse | install <name> | remove <name> | inspect <name> | tap | reload | enable <name> | disable <name>[/dim]"
    )



def _cmd_reload_skills(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    result = _refresh_skills_session(ctx)
    con.print(f"Reloaded {len(result['skills_registry'])} skills from disk.", style="green")
    return result



def _cmd_permissions(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    from agentica.agent.permissions import PERMISSION_MODES

    if cmd_args.strip():
        new_mode = cmd_args.strip().lower()
        if new_mode not in PERMISSION_MODES:
            con.print(f"[red]Invalid mode: {new_mode}. Use: {', '.join(PERMISSION_MODES)}[/red]")
            return
        if ctx.current_agent:
            ctx.current_agent.set_permission_mode(new_mode)
            con.print(f"[green]Permission mode set to: {new_mode}[/green]")
        return

    if ctx.current_agent:
        con.print(f"[bold cyan]Permission Mode: {ctx.current_agent.tool_config.permission_mode}[/bold cyan]")
        con.print()
        con.print("  [dim]ask[/dim]        - Ask for approval: reads (including outside the workspace), read-only shell, memory, task/delegate, and builtins run; file writes, mutating shell, and network need confirmation")
        con.print("  [dim]auto[/dim]       - Approve for me: ask about file writes outside the workspace, sensitive paths, and hard-unsafe commands")
        con.print("  [dim]allow-all[/dim]  - Full Access: never ask, never deny (project deny-similar applies only in ask/auto; hard-unsafe is logged and still run)")
        con.print()
        con.print(
            "Tools stay in the schema in every tier. ask does not hide write tools. "
            "Allow-similar and deny-similar grants are stored in this project's project.json.",
            style="dim",
        )
        con.print()
        con.print("Usage: /permissions <mode>", style="dim")
