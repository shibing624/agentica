# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model/profile/config slash commands and profile switching
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from agentica.cli.runtime import (
    get_console,
    active_tool_names,
    BUILTIN_TOOLS,
    create_agent,
    get_model,
    _build_environment_context,
    _build_fallback_models,
    _build_sibling_model,
    _resolve_compact_token_limit,
)
from agentica.cli.setup import (
    apply_named_profile_to_agent_config,
    session_profile,
)
from agentica.global_config import (
    get_profile,
    get_profiles,
    get_active_profile_name,
    resolve_active_profile_name,
    set_project_profile,
    clear_project_profile,
    get_project_profile,
)
from agentica.subagents import get_subagent_configs
from agentica.utils.log import (
    restore_console_logging,
    set_log_level_to_debug,
    set_log_level_to_info,
    suppress_console_logging,
)
from agentica.cli import self_manage
from agentica.cli.context_usage import measure_context
from agentica.cli.usage_display import ProviderUsageSummary, format_cost_usd
from agentica.project_store import project_base_dir

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.helpers import (
    _count_enabled_skills,
    _run_async_safe,
    _sanitize_history_for_model_switch,
    _update_task_tool_auxiliary_model,
    format_cli_log_location,
    format_path_for_display,
)
from agentica.cli.commands.cron_cmd import _confirm_via_tui


def _cmd_status(ctx: CommandContext, cmd_args: str = ""):
    """Print a compact one-screen overview of the current session."""
    con = get_console()
    ac = ctx.agent_config

    try:
        from agentica.version import __version__
    except ImportError:
        __version__ = "unknown"

    provider = ac.get("model_provider")
    model_name = ac.get("model_name")
    base_url = ac.get("base_url")
    profile, profile_source = session_profile(ac, ac.get("work_dir") or os.getcwd())

    auxiliary_provider = ac.get("auxiliary_model_provider")
    auxiliary_model_name = ac.get("auxiliary_model_name")

    agent = ctx.current_agent
    tools_count = len(agent.tools) if agent and agent.tools else 0
    skills_count = _count_enabled_skills(agent)
    subagent_configs = get_subagent_configs()
    package_subagents = sum(
        1 for cfg in subagent_configs.values() if cfg.source == "package"
    )
    custom_subagents = len(subagent_configs) - package_subagents
    subagent_total = len(subagent_configs)

    perm_mode = agent.tool_config.permission_mode if agent else None

    # Context usage and session cost are best-effort from the TUI state.
    ts = ctx.tui_state or {}
    ctx_tokens = ts.get("context_tokens")
    ctx_window = ts.get("context_window")
    ctx_pct = None
    ctx_tokens_value = 0
    ctx_window_value = 0
    if isinstance(ctx_tokens, (int, float)) and isinstance(ctx_window, (int, float)) and ctx_window > 0:
        ctx_pct = ctx_tokens / ctx_window * 100
        ctx_tokens_value = int(ctx_tokens)
        ctx_window_value = int(ctx_window)

    session_cost = ts.get("cost_usd") if ts else None
    if session_cost is None and agent is not None:
        cost_tracker = agent.run_response.cost_tracker
        if cost_tracker is not None:
            session_cost = cost_tracker.total_cost_usd

    model_str = f"{provider}/{model_name}" if provider and model_name else "(unset)"
    auxiliary_str = (
        f"{auxiliary_provider}/{auxiliary_model_name}"
        if auxiliary_provider and auxiliary_model_name
        else "(reuse main)"
    )
    skills_str = str(skills_count) if skills_count is not None else "n/a"
    perm_str = perm_mode or "(default)"
    ctx_str = f"{ctx_pct:.0f}% ({ctx_tokens_value:,}/{ctx_window_value:,})" if ctx_pct is not None else "n/a"
    cost_str = f"${session_cost:.4f}" if isinstance(session_cost, (int, float)) else "n/a"

    profile_label = profile or "(none)"
    if profile and profile_source in ("flag", "session", "project", "global", "default"):
        profile_label = f"{profile} ({profile_source})"
    con.print(f"  [bold]Agentica[/bold] [dim]v{__version__}[/dim]  profile: [cyan]{profile_label}[/cyan]")
    con.print(f"  Model:     [bold]{model_str}[/bold]")
    if base_url:
        con.print(f"  Endpoint:  [dim]{base_url}[/dim]")
    con.print(f"  Auxiliary model: {auxiliary_str}")
    con.print(
        f"  Tools: {tools_count}  |  Skills: {skills_str}  |  "
        f"Subagents: {subagent_total} ({package_subagents} package + {custom_subagents} override/custom)"
    )
    con.print(f"  Permissions: {perm_str}  |  Context: {ctx_str}  |  Cost: {cost_str}")
    if agent is not None and agent.session_id:
        session_name = agent._session_log.get_name() if agent._session_log is not None else None
        session_label = f"{session_name} ({agent.session_id})" if session_name else agent.session_id
        con.print(f"  Session:    [cyan]{session_label}[/cyan]")
        forked_from = agent._session_log.get_forked_from() if agent._session_log is not None else None
        if forked_from:
            con.print(f"  Forked from: [dim]{forked_from}[/dim]")
        slog = agent._session_log
        if slog is not None:
            path_str = str(slog.path)
            size_bit = ""
            try:
                p = Path(path_str)
                if p.is_file():
                    size_bit = f"  [dim]({p.stat().st_size:,} B)[/dim]"
            except (OSError, TypeError, ValueError):
                pass
            con.print(
                f"  Session log: [cyan]{format_path_for_display(path_str)}[/cyan]{size_bit}"
            )
    peers = ctx.peer_session
    if peers is not None:
        con.print(
            f"  Peer:       [cyan]{peers.name}[/cyan] "
            f"[dim]peer={peers.peer_id}[/dim]  "
            f"[dim](other sessions address you as this name)[/dim]"
        )
    log_location = format_cli_log_location()
    if log_location:
        con.print(f"  Debug log:  [dim]{log_location}[/dim]")



def _cmd_config(ctx: CommandContext, cmd_args: str = ""):
    """Display or edit configuration.

    Subcommands:
      /config                 show current config + workspace status
      /config show            same as above, with config.yaml + .env summary
      /config path            print config file locations
      /config set <field> <value> [profile]   edit a config.yaml profile field
      /config env <KEY> <value|->             set (or delete with '-') a .env var
    """
    args = cmd_args.strip()
    sub = args.split()[0].lower() if args else ""
    if sub == "set":
        return _cmd_config_set(ctx, args[len(sub) :].strip())
    if sub == "env":
        return _cmd_config_env(ctx, args[len(sub) :].strip())
    if sub == "path":
        con = get_console()
        con.print(f"  config.yaml: [cyan]{self_manage.config_file_path()}[/cyan]")
        con.print(f"  .env:        [cyan]{self_manage.dotenv_path()}[/cyan]")
        return
    if sub == "show":
        _cmd_config_show_files(ctx)
        # fall through to the regular status display too
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
        cap = ctx.agent_config.get("compact_token_limit")
        if not cap and ctx.current_agent is not None:
            cap = ctx.current_agent.tool_config.compact_token_limit
        if cap:
            con.print(f"  Compact at:  {int(cap):,} tokens")

    con.print()
    con.print("  [bold]-- Terminal --[/bold]")
    work_dir = ctx.agent_config.get("work_dir") or os.getcwd()
    con.print(f"  Working Dir: {work_dir}")
    user_id = None
    if ctx.current_agent is not None:
        user_id = ctx.current_agent.user_id
    elif ctx.workspace is not None:
        user_id = ctx.workspace.user_id
    else:
        user_id = ctx.agent_config.get("user_id")
    con.print(f"  Project Dir: {project_base_dir(work_dir, user_id=user_id)}")
    if ctx.current_agent:
        con.print(f"  Permissions: {ctx.current_agent.tool_config.permission_mode}")

    con.print()
    con.print("  [bold]-- Agent --[/bold]")
    # Prefer the live agent's actual toolset — it reflects /tools load/remove,
    # conditional tools (peer, worktree, self_manage, cron) and any runtime
    # stripping, none of which the static list can know. Fall back to the
    # static list + session extras when no agent is built yet.
    all_active = active_tool_names(ctx.current_agent)
    if not all_active:
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
        # Surface the user-set session name (via /rename), if any.
        # Quiet line — only render when a name exists, to keep /status
        # output minimal for unnamed sessions. get_name() never raises
        # (corrupt sidecar == no name), so no defensive try/except needed.
        _slog = ctx.current_agent._session_log
        if _slog is not None:
            _sname = _slog.get_name()
            if _sname:
                con.print(f"  Session name: {_sname}")
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



def _cmd_config_show_files(ctx: CommandContext):
    """Print a masked summary of config.yaml profiles and .env vars."""
    con = get_console()
    summary = self_manage.read_config_summary()
    if summary:
        con.print(f"[bold]config.yaml[/bold] [dim]({self_manage.config_file_path()})[/dim]")
        active_name, active_source = resolve_active_profile_name(
            work_dir=ctx.agent_config.get("work_dir") or os.getcwd()
        )
        source_label = f" [dim]({active_source})[/dim]" if active_source else ""
        con.print(f"  active profile: [cyan]{active_name}[/cyan]{source_label}")
        if active_source == "project":
            con.print(f"  [dim](global default: {summary.get('active_profile')})[/dim]")
        # This view describes config.yaml; --profile picks a profile for one
        # run without touching it, so say when the two disagree.
        session_name, session_source = session_profile(
            ctx.agent_config, ctx.agent_config.get("work_dir") or os.getcwd()
        )
        if session_source == "flag" and session_name != active_name:
            con.print(f"  [dim](this session: {session_name}, from --profile)[/dim]")
        for pname, profile in (summary.get("profiles") or {}).items():
            marker = "*" if pname == active_name else " "
            con.print(f"  {marker} [yellow]{pname}[/yellow]")
            for k, v in profile.items():
                con.print(f"      {k} = [green]{v}[/green]")
    env_vars = self_manage.read_dotenv()
    con.print(f"[bold].env[/bold] [dim]({self_manage.dotenv_path()})[/dim]")
    if env_vars:
        for k, v in env_vars.items():
            con.print(f"  {k} = [green]{v}[/green]")
    else:
        con.print("  [dim](empty)[/dim]")



def _rebuild_live_model(ctx: CommandContext):
    """Rebuild the running agent's model from the current ctx.agent_config.

    Used after a config edit so changes take effect without restarting. Mirrors
    the rebuild path in _cmd_model's profile-apply branch.
    """
    if ctx.current_agent is None:
        return
    model_kwargs = {
        "model_provider": ctx.agent_config.get("model_provider"),
        "model_name": ctx.agent_config.get("model_name"),
        "base_url": ctx.agent_config.get("base_url"),
        "api_key": ctx.agent_config.get("api_key"),
        "wire_api": ctx.agent_config.get("wire_api"),
        "max_tokens": ctx.agent_config.get("max_tokens"),
        "temperature": ctx.agent_config.get("temperature"),
        "reasoning_effort": ctx.agent_config.get("reasoning_effort"),
        "reasoning": ctx.agent_config.get("reasoning"),
        "top_p": ctx.agent_config.get("top_p"),
        "context_window": ctx.agent_config.get("context_window"),
        "enable_cache_control": ctx.agent_config.get("enable_cache_control"),
        "cache_control_messages": ctx.agent_config.get("cache_control_messages"),
        "cache_control_session_header": ctx.agent_config.get("cache_control_session_header"),
        "cache_keepalive": ctx.agent_config.get("cache_keepalive"),
        "extra_body": ctx.agent_config.get("extra_body"),
        "extra_headers": ctx.agent_config.get("extra_headers"),
        "default_headers": ctx.agent_config.get("default_headers"),
    }
    previous = ctx.current_agent.model
    previous_key = (type(previous), previous.id if previous is not None else None)
    ctx.current_agent.model = get_model(**model_kwargs)
    new_model = ctx.current_agent.model
    if (type(new_model), new_model.id) != previous_key:
        _sanitize_history_for_model_switch(ctx.current_agent)
    ctx.current_agent.environment_context = _build_environment_context(ctx.current_agent, ctx.agent_config)
    cap = _resolve_compact_token_limit(ctx.agent_config)
    ctx.current_agent.tool_config.compact_token_limit = cap
    cm = ctx.current_agent.tool_config.compression_manager
    if cm is not None:
        cm.compact_token_limit = cap



def _cmd_config_set(ctx: CommandContext, cmd_args: str = ""):
    """Edit a config.yaml profile field at runtime: set <field> <value> [profile]."""
    con = get_console()
    parts = cmd_args.split()
    if len(parts) < 2:
        con.print("[red]Usage: /config set <field> <value> [profile][/red]")
        con.print(f"[dim]Editable fields: {', '.join(sorted(self_manage._EDITABLE_PROFILE_FIELDS))}[/dim]")
        return
    field = parts[0]
    # Value may contain spaces only if a profile is NOT given; keep it simple:
    # last token is treated as profile only when 3+ tokens AND it names a profile.
    profile_name = None
    if len(parts) >= 3:
        candidate = parts[-1]
        from agentica.global_config import load_global_config

        cfg = load_global_config() or {}
        if candidate in (cfg.get("profiles") or {}):
            profile_name = candidate
            value = " ".join(parts[1:-1])
        else:
            value = " ".join(parts[1:])
    else:
        value = parts[1]
    # Resolve target profile up front: respect project-scoped override so
    # `/config set` in a project with an override edits the profile the user
    # is actually *using*, not the global default. Passing an explicit
    # profile_name into set_profile_field also avoids self_manage having to
    # know about work_dir.
    effective_active, _src = resolve_active_profile_name(work_dir=ctx.agent_config.get("work_dir") or os.getcwd())
    target = profile_name or effective_active
    try:
        self_manage.set_profile_field(field, value, target)
    except ValueError as e:
        con.print(f"[red]{e}[/red]")
        return
    con.print(f"[green]Updated profile '{target}': {field} = {self_manage.mask_secret(field, value)}[/green]")
    # If editing the active profile, sync ctx + rebuild the live model.
    if target == effective_active:
        coerced = self_manage._coerce_profile_value(field, value)
        ctx.agent_config[field] = coerced
        try:
            _rebuild_live_model(ctx)
            con.print("[dim]Applied to running agent (no restart needed).[/dim]")
            return {"model_switched": True}
        except Exception as e:
            con.print(f"[yellow]Saved, but live-apply failed: {e}. Restart to take effect.[/yellow]")
    return



def _cmd_config_env(ctx: CommandContext, cmd_args: str = ""):
    """Set or delete a .env variable: env <KEY> <value>  |  env <KEY> -"""
    con = get_console()
    parts = cmd_args.split(maxsplit=1)
    if len(parts) < 2:
        con.print("[red]Usage: /config env <KEY> <value>   (use '-' as value to delete)[/red]")
        return
    key, value = parts[0], parts[1].strip()
    try:
        if value == "-":
            self_manage.set_dotenv_var(key, None)
            con.print(f"[green]Deleted .env var {key}[/green]")
        else:
            self_manage.set_dotenv_var(key, value)
            con.print(f"[green]Set .env var {key} = {self_manage.mask_secret(key, value)}[/green]")
        con.print("[dim]Applied to current process environment.[/dim]")
    except ValueError as e:
        con.print(f"[red]{e}[/red]")



def _cmd_upgrade(ctx: CommandContext, cmd_args: str = ""):
    """Self-upgrade the agentica package via pip.

    /upgrade           check + (after confirm) upgrade to latest PyPI release
    /upgrade check     only report current vs latest, do not install
    /upgrade --pre     allow pre-release versions
    """
    con = get_console()
    args = cmd_args.strip().lower()
    pre = "--pre" in args
    check_only = "check" in args

    current = self_manage.get_current_version()
    con.print(f"  current version: [cyan]{current}[/cyan]")
    con.print("  checking PyPI for latest...")
    latest = self_manage.get_latest_version()
    if latest is None:
        con.print("[yellow]  could not reach PyPI (offline?). Try again later.[/yellow]")
        return
    con.print(f"  latest version:  [cyan]{latest}[/cyan]")

    if not self_manage.is_upgrade_available(current, latest):
        con.print("[green]  already up to date.[/green]")
        return
    con.print(f"[bold yellow]  upgrade available: {current} -> {latest}[/bold yellow]")
    if check_only:
        con.print("[dim]  run /upgrade to install.[/dim]")
        return

    if not _confirm_via_tui(ctx, f"Upgrade agentica {current} -> {latest}?"):
        con.print("[dim]  cancelled.[/dim]")
        return

    con.print("  running pip install -U agentica ...")
    code, output = self_manage.run_pip_upgrade("agentica", pre=pre)
    # Surface the real pip output rather than swallowing it.
    con.print(output.strip() or "[dim](no output)[/dim]")
    if code == 0:
        con.print(f"[green]  upgraded. Restart the CLI to load {latest}.[/green]")
    else:
        con.print(f"[red]  pip exited with code {code}. See output above.[/red]")



def _apply_profile(
    ctx: CommandContext,
    name: str,
    *,
    persist_project: bool = True,
    profile_source: str = "project",
):
    """Switch the live agent to a named config.yaml profile."""
    con = get_console()
    profile = get_profile(name)
    if not profile or not profile.get("model_provider"):
        con.print(f"[red]Profile not found or incomplete: {name}[/red]")
        names = list(get_profiles().keys())
        if names:
            con.print(f"Available profiles: {', '.join(names)}", style="dim")
        return

    apply_named_profile_to_agent_config(ctx.agent_config, name, source=profile_source)
    new_provider = ctx.agent_config["model_provider"]
    new_model = ctx.agent_config["model_name"]
    new_base_url = ctx.agent_config["base_url"]
    new_key = ctx.agent_config["api_key"]
    new_wire_api = ctx.agent_config.get("wire_api")
    new_max_tokens = ctx.agent_config.get("max_tokens")
    new_temperature = ctx.agent_config.get("temperature")
    new_reasoning_effort = ctx.agent_config.get("reasoning_effort")
    new_reasoning = ctx.agent_config.get("reasoning")
    new_top_p = ctx.agent_config.get("top_p")
    new_context_window = ctx.agent_config.get("context_window")
    new_extra_body = ctx.agent_config.get("extra_body")
    new_extra_headers = ctx.agent_config.get("extra_headers")
    new_default_headers = ctx.agent_config.get("default_headers")

    # Auxiliary model: a profile switch fully replaces the auxiliary model too. An
    # auxiliary_model block rebuilds the sibling (background calls + task subagent);
    # a profile without one clears the auxiliary fields so they fall back to the
    # main model. _build_sibling_model handles same-provider base_url/api_key
    # inheritance; cross-provider reads the block's own key (or env). The auxiliary
    # rebuild is a tolerance boundary — a broken auxiliary config falls back to the
    # main model with a warning instead of blocking the core model switch.
    if ctx.agent_config.get("auxiliary_model_name"):
        try:
            new_auxiliary_model = _build_sibling_model(ctx.agent_config, "auxiliary")
        except Exception as exc:
            con.print(f"[yellow]Auxiliary model build failed, falling back to main model: {exc}[/yellow]")
            ctx.agent_config["auxiliary_model_provider"] = None
            ctx.agent_config["auxiliary_model_name"] = None
            ctx.agent_config["auxiliary_base_url"] = None
            ctx.agent_config["auxiliary_api_key"] = None
            ctx.agent_config["auxiliary_wire_api"] = None
            ctx.agent_config["auxiliary_extra_body"] = None
            ctx.agent_config["auxiliary_extra_headers"] = None
            ctx.agent_config["auxiliary_reasoning"] = None
            ctx.agent_config["auxiliary_reasoning_effort"] = None
            new_auxiliary_model = None
    else:
        ctx.agent_config["auxiliary_model_provider"] = None
        ctx.agent_config["auxiliary_model_name"] = None
        ctx.agent_config["auxiliary_base_url"] = None
        ctx.agent_config["auxiliary_api_key"] = None
        ctx.agent_config["auxiliary_wire_api"] = None
        ctx.agent_config["auxiliary_extra_body"] = None
        ctx.agent_config["auxiliary_extra_headers"] = None
        ctx.agent_config["auxiliary_reasoning"] = None
        ctx.agent_config["auxiliary_reasoning_effort"] = None
        new_auxiliary_model = None
    ctx.agent_config["auxiliary_model"] = new_auxiliary_model

    # Persist an explicit switch as the project-scoped recent active profile in
    # project.json (``active_profile``). config.yaml's global `active_profile:`
    # pointer is untouched — that stays the machine-wide default, and other
    # projects keep whatever they had.
    #
    # Profile bodies (model_*, auxiliary_*, tuning) are write-only-by-setup
    # — never touched here. That separation is what fixed the original
    # "config.yaml 乱掉" bug, where the old free-form `/model provider/name`
    # path rewrote fields of the active profile in place.
    work_dir = ctx.agent_config.get("work_dir") or os.getcwd()
    if persist_project:
        set_project_profile(work_dir, name)

    # Record the session profile separately from project.json so resume can
    # restore this session's provider/model even if another session in the same
    # work_dir switches the project active profile later.
    if ctx.current_agent is not None and ctx.current_agent._session_log is not None:
        ctx.current_agent._session_log.set_profile(name, ctx.agent_config["profile_source"])

    model_kwargs = {
        "model_provider": new_provider,
        "model_name": new_model,
        "base_url": new_base_url,
        "api_key": new_key,
        "wire_api": new_wire_api,
        "max_tokens": new_max_tokens,
        "temperature": new_temperature,
        "reasoning_effort": new_reasoning_effort,
        "reasoning": new_reasoning,
        "top_p": new_top_p,
        "context_window": new_context_window,
        "enable_cache_control": ctx.agent_config.get("enable_cache_control"),
        "cache_control_messages": ctx.agent_config.get("cache_control_messages"),
        "cache_control_session_header": ctx.agent_config.get("cache_control_session_header"),
        "cache_keepalive": ctx.agent_config.get("cache_keepalive"),
        "extra_body": new_extra_body,
        "extra_headers": new_extra_headers,
        "default_headers": new_default_headers,
    }
    new_model_obj = get_model(**model_kwargs)
    if ctx.current_agent is not None:
        ctx.current_agent.model = new_model_obj
        ctx.current_agent.auxiliary_model = new_auxiliary_model
        # Fallback chain + retry count follow the profile switch too, so the
        # resilience config never goes stale (a profile without fallback_models
        # clears the chain back to []).
        try:
            ctx.current_agent.fallback_models = _build_fallback_models(ctx.agent_config)
        except Exception as exc:
            con.print(f"[yellow]Fallback model build failed, clearing fallback chain: {exc}[/yellow]")
            ctx.current_agent.fallback_models = []
        _mar = ctx.agent_config.get("max_api_retry")
        ctx.current_agent.max_api_retry = _mar if _mar is not None else 2
        # Repoint the cheap tier of the task subagent tool onto the new auxiliary
        # model (None = fall back to the parent's main model, matching
        # create_agent's default).
        _update_task_tool_auxiliary_model(ctx.current_agent, new_auxiliary_model)
        _sanitize_history_for_model_switch(ctx.current_agent)
        # Refresh the self-description block so the agent reports its new
        # model/auxiliary model on the next turn.
        ctx.current_agent.environment_context = _build_environment_context(ctx.current_agent, ctx.agent_config)
        auxiliary_provider = ctx.agent_config.get("auxiliary_model_provider")
        auxiliary_model_name = ctx.agent_config.get("auxiliary_model_name")
        auxiliary_str = (
            f"{auxiliary_provider}/{auxiliary_model_name}"
            if auxiliary_provider and auxiliary_model_name
            else "reuse main"
        )
        con.print(f"[green]Switched to profile '{name}': {new_provider}/{new_model} (session preserved)[/green]")
        con.print(f"[dim]Auxiliary model: {auxiliary_str}[/dim]")
        return {"model_switched": True}
    current_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
        background_process_registry=ctx.background_processes,
        peer_session=ctx.peer_session,
        worktree_binder=ctx.worktree_binder,
        approve=ctx.approve,
    )
    con.print(f"[green]Switched to profile '{name}': {new_provider}/{new_model}[/green]")
    return {"current_agent": current_agent}



def _clear_project_profile_override(ctx: CommandContext) -> Any:
    """Handle `/model --clear`: drop the project override and re-apply global default."""
    con = get_console()
    work_dir = ctx.agent_config.get("work_dir") or os.getcwd()
    override = get_project_profile(work_dir)
    if not override:
        con.print("[yellow]No project-scoped profile override to clear.[/yellow]")
        default_name, source = resolve_active_profile_name(work_dir=work_dir)
        con.print(f"[dim]Current profile: {default_name} ({source})[/dim]")
        return
    clear_project_profile(work_dir)
    default_name, source = resolve_active_profile_name(work_dir=work_dir)
    con.print(f"[green]Cleared project profile override.[/green]")
    con.print(f"[dim]Falling back to {default_name} ({source}).[/dim]")
    # Actually apply the fallback profile to the live session so the state
    # matches the message the user just saw.
    return _apply_profile(ctx, default_name, persist_project=False, profile_source=source)



def _list_profiles(active_name: Optional[str] = None, active_source: Optional[str] = None):
    """Print all configured config.yaml profiles with main/auxiliary full names.

    If ``active_name`` is provided, it labels that profile as ``[active]`` and
    annotates the source (project/global/default). Otherwise falls back to the
    global default from config.yaml.
    """
    con = get_console()
    profiles = get_profiles()
    active = active_name if active_name is not None else get_active_profile_name()
    if not profiles:
        con.print("[yellow]No profiles configured in ~/.agentica/config.yaml[/yellow]")
        con.print("Create one with: agentica setup", style="dim")
        return
    con.print("Configured profiles:", style="cyan")
    for name, p in profiles.items():
        if name == active:
            source_label = f" [dim]({active_source})[/dim]" if active_source else ""
            marker = f" [bold green][active][/bold green]{source_label}"
        else:
            marker = ""
        provider = p.get("model_provider", "?")
        model = p.get("model_name", "?")
        has_key = "key set" if p.get("api_key") else "no key"
        con.print(f"  [bold]{name}[/bold]{marker}")
        con.print(f"      main: [cyan]{provider}/{model}[/cyan] [dim]({has_key})[/dim]")
        auxiliary_block = p.get("auxiliary_model")
        if isinstance(auxiliary_block, dict) and auxiliary_block.get("model_name"):
            auxiliary_provider = auxiliary_block.get("model_provider") or provider
            auxiliary_model = auxiliary_block.get("model_name")
            auxiliary_has_key = "key set" if auxiliary_block.get("api_key") else "inherits main"
            con.print(
                f"      auxiliary:  [cyan]{auxiliary_provider}/{auxiliary_model}[/cyan] [dim]({auxiliary_has_key})[/dim]"
            )
        else:
            con.print("      auxiliary:  [dim]reuse main[/dim]")
        tuning = []
        if p.get("wire_api"):
            tuning.append(f"wire_api={p['wire_api']}")
        if p.get("reasoning"):
            tuning.append(f"responses_reasoning={p['reasoning']}")
        if p.get("reasoning_effort"):
            tuning.append(f"effort={p['reasoning_effort']}")
        if p.get("max_tokens"):
            tuning.append(f"max_tokens={p['max_tokens']}")
        if p.get("context_window"):
            tuning.append(f"context={p['context_window']}")
        if p.get("compact_token_limit"):
            tuning.append(f"compact={p['compact_token_limit']}")
        if p.get("temperature") is not None:
            tuning.append(f"temp={p['temperature']}")
        if p.get("top_p") is not None:
            tuning.append(f"top_p={p['top_p']}")
        if p.get("extra_body"):
            tuning.append("extra_body=set")
        if p.get("extra_headers"):
            tuning.append("extra_headers=set")
        if tuning:
            con.print(f"      [dim]tuning: {', '.join(tuning)}[/dim]")
    con.print()
    con.print("Switch with: /model <profile_name>", style="dim")
    con.print("Add a new model: agentica setup", style="dim")



def _cmd_model(ctx: CommandContext, cmd_args: str = ""):
    # Profile architecture: each profile in ~/.agentica/config.yaml is a fully
    # self-contained model setup (main + optional auxiliary + tuning). `/model`
    # is a *read + switch* command: it lists profiles, and switching persists
    # only the top-level `active:` pointer — profile bodies are never rewritten
    # here (that is `agentica setup`'s job).
    #
    # Why no free-form `/model openai/gpt-5` path: the old behaviour rewrote
    # the *currently active* profile's fields in place, silently destroying
    # whatever main/auxiliary/tuning the user had saved. Mutating profile
    # bodies is the job of `agentica setup`, which collects a complete profile.
    con = get_console()

    stripped = cmd_args.strip()
    if not stripped:
        return _model_list_overview(ctx)

    # /model --clear (or --reset) drops the project-scoped override and
    # re-applies the global default. Deliberately no `--global` flag: writing
    # global defaults is `agentica setup` / `/config` territory.
    if stripped in ("--clear", "--reset", "clear", "reset"):
        return _clear_project_profile_override(ctx)

    # Tolerate (and gently redirect) the legacy `profile <name>` form so users
    # with muscle memory still get a working switch.
    parts = stripped.split(None, 1)
    if parts[0].lower() in ("profile", "profiles"):
        if len(parts) == 1:
            return _model_list_overview(ctx)
        rest = parts[1].strip()
        if rest in ("--clear", "--reset", "clear", "reset"):
            return _clear_project_profile_override(ctx)
        return _apply_profile(ctx, rest)

    # Anything containing "/" is the old "<provider>/<model>" free-form path.
    # Reject it with an actionable pointer rather than silently mutating config.
    if "/" in stripped:
        con.print(f"[yellow]/model no longer accepts free-form '{stripped}'.[/yellow]")
        con.print(
            "This used to overwrite the active profile in config.yaml. Run [bold]agentica setup[/bold] to add or edit a profile.",
            style="dim",
        )
        con.print("To switch between saved profiles: /model <profile_name>", style="dim")
        return

    # Single token => treat as a profile name to switch to.
    return _apply_profile(ctx, stripped)



def _model_list_overview(ctx: CommandContext) -> None:
    """Readonly overview: current live model + every saved profile (rich detail).

    Reuses ``_list_profiles`` for the per-profile rendering so the two views
    stay in lockstep; this function only adds the live-session header and the
    "how to use /model" footer.
    """
    con = get_console()
    con.print(
        f"Current model: [bold cyan]{ctx.agent_config['model_provider']}/{ctx.agent_config['model_name']}[/bold cyan]"
    )
    con.print()
    active_name, active_source = resolve_active_profile_name(work_dir=ctx.agent_config.get("work_dir") or os.getcwd())
    _list_profiles(active_name=active_name, active_source=active_source)
    con.print("Usage:", style="cyan")
    con.print("  /model                  list saved profiles (this view)", style="dim")
    con.print(
        "  /model <profile_name>   switch to a saved profile (project-scoped; config.yaml untouched)", style="dim"
    )
    con.print("  /model --clear          drop project override, fall back to global default", style="dim")
    con.print("To add or edit a profile, run [bold]agentica setup[/bold] outside the session.", style="dim")



def _cmd_debug(ctx: CommandContext, cmd_args: str = ""):
    """Turn verbose debug logging on or off for the rest of the session.

    Runtime equivalent of the ``--debug`` startup flag: DEBUG records reach the
    console (the file log already gets them) and subagent tool output switches
    to the verbose form on the next turn. The flag is written back to
    ``agent_config`` so it survives an agent rebuild (``/model``, ``/resume``).

    ``/debug`` with no argument flips the current state; ``on`` / ``off`` set it
    explicitly. The session facts this command used to print live in ``/status``.
    """
    con = get_console()
    arg = cmd_args.strip().lower()
    current = bool(ctx.agent_config.get("debug"))

    if arg == "":
        enable = not current
    elif arg in ("on", "true", "1"):
        enable = True
    elif arg in ("off", "false", "0"):
        enable = False
    else:
        con.print(f"  [dim]Unknown argument: {arg}. Use: /debug on|off[/dim]")
        return

    ctx.agent_config["debug"] = enable
    if ctx.tui_state is not None:
        ctx.tui_state["debug"] = enable
    if ctx.current_agent is not None:
        ctx.current_agent.debug = enable

    if enable:
        restore_console_logging("DEBUG", color=False)
        set_log_level_to_debug()
        con.print("  [green]Debug logging: ON[/green]")
        con.print("  [dim]DEBUG records now print to the console; subagent output turns verbose next turn.[/dim]")
    else:
        set_log_level_to_info()
        suppress_console_logging()
        con.print("  [green]Debug logging: OFF[/green]")



def _render_context_breakdown(con, agent) -> None:
    """Print what is currently occupying the context window, by origin.

    The split is measured locally from the same inputs used to build the next
    main-agent request. Provider usage is deliberately absent: it describes API
    consumption for a completed call, not the current session context state.
    """
    try:
        breakdown = _run_async_safe(measure_context(agent))
    except Exception as e:
        con.print(f"  [yellow]Could not measure context breakdown: {e}[/yellow]")
        return

    sections = breakdown.visible_sections()
    if not sections:
        return

    total = breakdown.total
    sep = "─" * 46

    con.print()
    con.print(
        f"  [bold cyan]Context Window[/bold cyan]  "
        f"[dim]{_fmt_tokens(total)} / {_fmt_tokens(breakdown.window)}[/dim]"
    )
    con.print(f"  {sep}")
    con.print(f"  {'Messages:':<24} {len(agent.working_memory.messages):>7}")
    for label, tokens in sections:
        con.print(f"  {label:<24} {_fmt_tokens(tokens):>7}")
    con.print(f"  {sep}")

    con.print()



def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)



def _cmd_usage(ctx: CommandContext, cmd_args: str = ""):
    """Show latest-turn API usage and current session context occupancy."""
    con = get_console()
    agent = ctx.current_agent
    tracker = agent.run_response.cost_tracker if agent else None

    if tracker is None or tracker.turns == 0:
        # No API call yet, but the window is not empty — the prompt is already
        # assembled. Show what is loaded rather than a bare "no data".
        con.print("[yellow]No API calls yet this session.[/yellow]")
        if agent is not None:
            _render_context_breakdown(con, agent)
        return

    model_name = f"{ctx.agent_config.get('model_provider', '')}/{ctx.agent_config.get('model_name', '')}"

    ts = ctx.tui_state or {}
    entries = agent.model.usage.request_usage_entries
    baseline = ts.get("_turn_usage_entry_baseline")
    if baseline is None:
        baseline = max(len(entries) - tracker.turns, 0)
    usage = ProviderUsageSummary.from_request_entries(
        entries[baseline:],
        cost_usd=tracker.total_cost_usd,
    )
    cache_hit_percent = usage.cache_hit_percent

    active_secs = ts.get("active_seconds", 0)

    if active_secs < 60:
        duration_str = f"{active_secs:.0f}s"
    elif active_secs < 3600:
        m, s = divmod(int(active_secs), 60)
        duration_str = f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(active_secs), 3600)
        m, _ = divmod(rem, 60)
        duration_str = f"{h}h {m:02d}m"

    session_cost = ts.get("cost_usd", tracker.total_cost_usd)

    sep = "─" * 42
    con.print()
    con.print("  [bold cyan]Latest Turn API Usage[/bold cyan]")
    con.print(f"  {sep}")
    con.print(f"  {'Model:':<30} {model_name}")
    con.print(
        f"  {'API calls this turn:':<30} {usage.api_calls:>12}"
        f"  [dim](tool rounds + final answer)[/dim]"
    )
    tool_calls = ts.get("last_turn_tool_count", 0)
    if tool_calls > 0:
        con.print(f"  {'Tool calls this turn:':<30} {tool_calls:>12}")
    if usage.api_calls > 0:
        avg = round(usage.prompt_tokens / usage.api_calls)
        con.print(
            f"  {'Input tokens:':<30} {usage.prompt_tokens:>12,}  [dim](avg {_fmt_tokens(avg)}/call)[/dim]"
        )
    else:
        con.print(f"  {'Input tokens:':<30} {usage.prompt_tokens:>12,}")
    if usage.cache_read_tokens > 0 or usage.cache_write_tokens > 0:
        con.print(f"  {'  Fresh input tokens:':<30} {usage.fresh_input_tokens:>12,}")
    if usage.cache_read_tokens > 0:
        cache_suffix = f" / {cache_hit_percent:.1f}% hit" if cache_hit_percent is not None else ""
        con.print(f"  {'  Cached input tokens:':<30} {usage.cache_read_tokens:>12,}{cache_suffix}")
    if usage.cache_write_tokens > 0:
        con.print(f"  {'  Cache write tokens:':<30} {usage.cache_write_tokens:>12,}")
    con.print(f"  {'Output tokens:':<30} {usage.output_tokens:>12,}")
    # Net new == billed total whenever nothing was re-read (cache_read == 0),
    # so the extra row only earns its place on cache-hit turns.
    if usage.cache_read_tokens > 0:
        con.print(
            f"  {'Net new tokens:':<30} {usage.net_new_tokens:>12,}"
            f"  [dim](fresh + cache write + output)[/dim]"
        )
    con.print(f"  {'Total tokens (billed):':<30} {usage.total_tokens:>12,}")
    con.print(f"  {'Turn cost:':<30} ~{format_cost_usd(usage.cost_usd)}")
    con.print(f"  {sep}")
    con.print("  [bold cyan]Session[/bold cyan]")
    con.print(f"  {sep}")
    con.print(f"  {'API calls:':<30} {ts.get('total_api_calls', usage.api_calls):>12}")
    con.print(f"  {'Active time:':<30} {duration_str:>12}")
    con.print(f"  {'Cost:':<30} ~{format_cost_usd(session_cost)}")
    con.print(f"  {sep}")
    _render_context_breakdown(con, agent)



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



def _cmd_statusbar(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.tui_state is None:
        return
    current = ctx.tui_state.get("statusbar_visible", True)
    ctx.tui_state["statusbar_visible"] = not current
    state = "OFF" if current else "ON"
    con.print(f"  [green]Status bar: {state}[/green]")
