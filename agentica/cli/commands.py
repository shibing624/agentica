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
from uuid import UUID

from agentica.cli.runtime import (
    get_console,
    BUILTIN_TOOLS,
    TOOL_REGISTRY,
    MODEL_REGISTRY,
    EXAMPLE_MODELS,
    configure_tools,
    create_agent,
    get_model,
    _generate_session_id,
    _build_environment_context,
    _build_sibling_model,
)
from agentica.cli.display import (
    print_header,
    show_help,
)
from agentica.cli.setup import (
    default_base_url,
    default_model_name,
    get_profile_api_key,
)
from agentica.global_config import (
    get_profile,
    get_profiles,
    get_active_profile_name,
    resolve_active_profile_name,
    set_active_profile,
    set_project_profile,
    clear_project_profile,
    get_project_profile,
    upsert_profile,
)
from agentica.goals import GoalManager
from agentica.memory.models import AgentRun
from agentica.memory.session_log import SessionLog
from agentica.model.message import Message
from agentica.run_context import TaskAnchor
from agentica.run_response import RunResponse, AgentCancelledError
from agentica.subagent import (
    CODE_SUBAGENT_CONFIG,
    EXPLORE_SUBAGENT_CONFIG,
    RESEARCH_SUBAGENT_CONFIG,
    get_custom_subagent_configs,
)
from agentica.tools.goal_tool import GoalTool
from agentica.skills import (
    get_skill_registry,
    install_skills,
    list_installed_skills,
    load_skills,
    remove_skill,
)
from agentica.skills.skill_registry import reset_skill_registry
from agentica.cli import self_manage


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
    goal_lock: Any = None  # Optional[threading.Lock]
    # Callback the ask_user_question/confirm tools use to read via the TUI input box
    # instead of a blocking input(). Must be preserved across agent rebuilds
    # (/model, /newchat, /reload, …) or those paths reintroduce the deadlock.
    ask_user_question_callback: Any = None


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

    def replace_index(self, idx: int, item) -> bool:
        """Replace the item at ``idx`` in place and refresh its timestamp.

        Returns ``False`` if ``idx`` is out of range. Refreshing the timestamp
        makes the TUI queue bar treat the edit as a re-submission so the
        "x seconds ago" label reflects the latest user intent.
        """
        with self._lock:
            if 0 <= idx < len(self._deque):
                self._deque[idx] = item
                self._timestamps[idx] = time.time()
                return True
            return False

    def insert_index(self, idx: int, item) -> bool:
        """Insert ``item`` at position ``idx`` (0-based).

        ``idx == len(queue)`` is allowed and equivalent to ``put`` (append).
        Returns ``False`` for any other out-of-range index so callers can
        report the error with the same shape as ``remove_index``.
        """
        with self._lock:
            if 0 <= idx <= len(self._deque):
                self._deque.insert(idx, item)
                self._timestamps.insert(idx, time.time())
                return True
            return False


# ==================== Concurrent commands ====================

# Commands that can execute while the agent is streaming (non-blocking).
# Readonly info commands + queue/bg management.
CONCURRENT_CMDS = frozenset(
    {
        "/bg",
        "/background",
        "/stop",
        "/q",
        "/queue",
        "/steer",
        "/cost",
        "/usage",
        "/config",
        "/debug",
        "/history",
        "/help",
        "/tools",
        "/skills",
        "/permissions",
        "/statusbar",
        "/sb",
        "/reasoning",
        "/status",
        "/agents",
        "/agent",
        # /goal and /subgoal: status/pause/clear/list subcommands are concurrent-safe.
        # Handlers reject "set new objective" when agent_running.
        "/goal",
        "/subgoal",
    }
)


# ==================== Helpers ====================

IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".svg",
        ".ico",
    }
)


def _sanitize_history_for_model_switch(agent) -> None:
    """Strip tool artifacts from history so it replays on a different provider.

    Cross-provider switches (e.g. OpenAI chat/completions <-> Anthropic
    /v1/messages) fail because tool calls/results are serialised differently:
    OpenAI uses flat role="tool" messages + assistant.tool_calls, while
    Anthropic uses list content blocks (tool_use / tool_result with
    tool_use_id). Replaying one format on the other API 400s
    ("unexpected tool_use_id found in tool_result blocks"). We drop every
    tool artifact (both formats) and keep only plain user/assistant text.

    Both ``wm.runs[].response.messages`` (the source for future prompts) and
    the flat ``wm.messages`` list are sanitised.
    """
    from agentica.agent.history_filter import strip_all_tool_artifacts

    wm = agent.working_memory
    for run in wm.runs:
        if not run.response or not run.response.messages:
            continue
        run.response.messages = strip_all_tool_artifacts(run.response.messages, drop_system=True)
    if getattr(wm, "messages", None):
        wm.messages = strip_all_tool_artifacts(wm.messages, drop_system=False)


def _refresh_skills_session(ctx: CommandContext):
    """Reload skill registry from disk and rebuild the current agent."""
    reset_skill_registry()
    load_skills()
    new_registry = get_skill_registry()
    new_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        new_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
    )
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


# ==================== Capability helpers ====================


def _count_enabled_skills(agent) -> Optional[int]:
    """Count enabled skills on the agent's SkillTool, or None when no SkillTool.

    External boundary: SkillTool registry / disk-backed usage loading can fail
    in odd environments, and /status must never crash on it, so any failure
    reports "unavailable" (None) instead of raising.
    """
    from agentica.tools.skill_tool import SkillTool

    if not agent or not agent.tools:
        return None
    for tool in agent.tools:
        if isinstance(tool, SkillTool):
            try:
                return len(tool._get_enabled_skills())
            except Exception:
                return None
    return None


def _get_subagent_loader():
    """Return the subagent_loader module, or None if not installed yet.

    The loader is built by a parallel work stream; importing it lazily here
    keeps commands.py loadable before it exists. An ImportError is the only
    swallowed case — once the module is present, errors propagate normally.
    """
    try:
        import agentica.subagent_loader as loader  # noqa: PLC0415

        return loader
    except ImportError:
        return None


def _get_defined_agents_for_display() -> list:
    """Custom subagent descriptors for /agents listing.

    Prefers the loader's on-disk view (includes file paths); falls back to the
    in-memory custom registry when the loader is not installed yet.
    """
    loader = _get_subagent_loader()
    if loader is not None:
        return loader.list_defined_agents()
    return [
        {
            "name": name,
            "description": cfg.description,
            "allowed_tools": cfg.allowed_tools,
            "denied_tools": cfg.denied_tools,
            "tool_call_limit": cfg.tool_call_limit,
            "path": None,
        }
        for name, cfg in get_custom_subagent_configs().items()
    ]


def _runtime_config_path(ctx: CommandContext) -> Path:
    """Resolve the runtime_config.yaml path used for skill enable/disable.

    Mirrors Agent._load_runtime_config's read priority: an existing file under
    the workspace wins, then cwd; when neither exists we create it at the
    workspace location (or cwd) so the next read picks up the change.
    """
    config_name = ".agentica/runtime_config.yaml"
    agent = ctx.current_agent
    if agent is not None and agent.workspace is not None:
        candidate = agent.workspace.path / config_name
        if candidate.exists():
            return candidate
    cwd_candidate = Path(os.getcwd()) / config_name
    if cwd_candidate.exists():
        return cwd_candidate
    if agent is not None and agent.workspace is not None:
        return agent.workspace.path / config_name
    return cwd_candidate


def _set_skill_runtime_state(ctx: CommandContext, name: str, enabled: bool) -> Optional[Path]:
    """Write a skill's enabled state into runtime_config.yaml (schema: skills.<name>.enabled).

    External I/O boundary: YAML read/write failures return None so /skills can
    print a clear error instead of crashing. Returns the path written on success.
    """
    con = get_console()
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        con.print("  [red]PyYAML not installed; cannot edit runtime_config.yaml.[/red]")
        return None

    path = _runtime_config_path(ctx)
    data: dict = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, yaml.YAMLError) as exc:
            con.print(f"  [red]Cannot read {path}: {exc}[/red]")
            return None

    skills = data.get("skills")
    if not isinstance(skills, dict):
        skills = {}
        data["skills"] = skills
    skills[name] = {"enabled": enabled}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
    except OSError as exc:
        con.print(f"  [red]Cannot write {path}: {exc}[/red]")
        return None
    return path


def _update_task_tool_model_override(agent, model_override) -> None:
    """Repoint the BuiltinTaskTool's model override on the live agent."""
    from agentica.tools.builtin_task_tool import BuiltinTaskTool

    if not agent or not agent.tools:
        return
    for tool in agent.tools:
        if isinstance(tool, BuiltinTaskTool):
            tool._model_override = model_override
            return


def _safe_tool_module_name(name: str) -> Optional[str]:
    """Sanitize a /tools add-from name to a plain module basename.

    Rejects anything path-like (separators, ~, leading dot, traversal, colon)
    so the resolved path can never escape .agentica/tools/. Strips a trailing
    .py the user may have added.
    """
    if not name:
        return None
    if name.endswith(".py"):
        name = name[:-3]
    if not name or "/" in name or "\\" in name or "~" in name or ":" in name or ".." in name:
        return None
    if name in (".", "") or name.startswith("."):
        return None
    return name


def _load_custom_tool_module(name: str, file_path: Path):
    """Import a .agentica/tools/<name>.py module and extract its exported tool.

    Looks for, in order: a ``tool`` Tool/Function (or @tool-decorated callable),
    a ``get_tool`` callable returning one, then any @tool-decorated function
    exposed on the module. Returns the addable tool, or None after printing a
    clear error. Module execution is an explicit boundary — arbitrary user code
    failures are reported, not raised.
    """
    con = get_console()
    import importlib.util  # noqa: PLC0415
    from agentica.tools.base import Function, Tool  # noqa: PLC0415

    def _is_export(obj) -> bool:
        return isinstance(obj, (Tool, Function)) or (
            callable(obj) and not isinstance(obj, type) and hasattr(obj, "_tool_metadata")
        )

    spec = importlib.util.spec_from_file_location(f"agentica_user_tool_{name}", file_path)
    if spec is None or spec.loader is None:
        con.print(f"  [red]Cannot load module spec for {file_path}[/red]")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        con.print(f"  [red]Error executing {file_path}: {exc}[/red]")
        return None

    candidate = None
    explicit_tool = getattr(module, "tool", None)
    if _is_export(explicit_tool):
        candidate = explicit_tool
    if candidate is None:
        factory = getattr(module, "get_tool", None)
        if callable(factory) and not isinstance(factory, type):
            try:
                produced = factory()
            except Exception as exc:
                con.print(f"  [red]get_tool() failed: {exc}[/red]")
                return None
            if _is_export(produced):
                candidate = produced
    if candidate is None:
        for attr in vars(module).values():
            if _is_export(attr):
                candidate = attr
                break
    if candidate is None:
        con.print(
            f"  [red]No exported tool found in {file_path}. Export `tool` (Tool), "
            "`get_tool()` (callable), or use the @tool decorator.[/red]"
        )
        return None
    return candidate


# ==================== Command Handlers ====================


def _cmd_help(ctx: CommandContext, cmd_args: str = ""):
    show_help(skills_registry=ctx.skills_registry)


def _cmd_exit(ctx: CommandContext, cmd_args: str = ""):
    return "EXIT"


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
    profile, profile_source = resolve_active_profile_name(work_dir=ac.get("work_dir") or os.getcwd())

    auxiliary_provider = ac.get("auxiliary_model_provider")
    auxiliary_model_name = ac.get("auxiliary_model_name")

    agent = ctx.current_agent
    tools_count = len(agent.tools) if agent and agent.tools else 0
    skills_count = _count_enabled_skills(agent)
    custom_subagents = len(get_custom_subagent_configs())
    subagent_total = 3 + custom_subagents

    perm_mode = agent.tool_config.permission_mode if agent else None

    # Context usage and session cost are best-effort from the TUI state.
    ts = ctx.tui_state or {}
    ctx_tokens = ts.get("context_tokens")
    ctx_window = ts.get("context_window")
    ctx_pct = None
    if isinstance(ctx_tokens, (int, float)) and isinstance(ctx_window, (int, float)) and ctx_window > 0:
        ctx_pct = ctx_tokens / ctx_window * 100

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
    ctx_str = f"{ctx_pct:.0f}% ({int(ctx_tokens):,}/{int(ctx_window):,})" if ctx_pct is not None else "n/a"
    cost_str = f"${session_cost:.4f}" if isinstance(session_cost, (int, float)) else "n/a"

    profile_label = profile or "(none)"
    if profile and profile_source in ("project", "global", "default"):
        profile_label = f"{profile} ({profile_source})"
    con.print(f"  [bold]Agentica[/bold] [dim]v{__version__}[/dim]  profile: [cyan]{profile_label}[/cyan]")
    con.print(f"  Model:     [bold]{model_str}[/bold]")
    if base_url:
        con.print(f"  Endpoint:  [dim]{base_url}[/dim]")
    con.print(f"  Auxiliary model: {auxiliary_str}")
    con.print(
        f"  Tools: {tools_count}  |  Skills: {skills_str}  |  "
        f"Subagents: {subagent_total} (3 builtin + {custom_subagents} custom)"
    )
    con.print(f"  Permissions: {perm_str}  |  Context: {ctx_str}  |  Cost: {cost_str}")


def _cmd_agents(ctx: CommandContext, cmd_args: str = ""):
    """Manage subagents: list, create, reload, remove.

    Built-in types (explore/research/code) come from subagent.py defaults;
    custom types come from .agentica/agents/*.md via the subagent loader.
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
        if loader is None:
            con.print("  [red]Subagent loader not available.[/red]")
            return
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
        if loader is None:
            con.print("  [red]Subagent loader not available.[/red]")
            return
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
        if loader is None:
            con.print("  [red]Subagent loader not available.[/red]")
            return
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

    builtin_configs = [
        ("explore", EXPLORE_SUBAGENT_CONFIG),
        ("research", RESEARCH_SUBAGENT_CONFIG),
        ("code", CODE_SUBAGENT_CONFIG),
    ]
    con.print("  [bold]Built-in subagents:[/bold]")
    for type_name, cfg in builtin_configs:
        desc_first = cfg.description.split("\n")[0].strip()
        con.print(f"    [green]●[/green] [bold]{type_name:<12}[/bold] {desc_first}")
        con.print(f"      [dim]tools: {', '.join(cfg.allowed_tools or [])}[/dim]")

    custom_agents = _get_defined_agents_for_display()
    con.print()
    if custom_agents:
        con.print(f"  [bold]Custom subagents ({len(custom_agents)}):[/bold]")
        for agent in custom_agents:
            agent_name = agent.get("name", "?")
            desc = agent.get("description") or ""
            desc_first = desc.split("\n")[0].strip()
            tools = agent.get("allowed_tools")
            tools_str = ", ".join(tools) if tools else "(inherit parent)"
            con.print(f"    [green]●[/green] [bold]{agent_name:<12}[/bold] {desc_first}")
            con.print(f"      [dim]tools: {tools_str}[/dim]")
            path = agent.get("path")
            if path:
                con.print(f"      [dim]file: {path}[/dim]")
    else:
        con.print("  [dim]No custom subagents. Create one with /agents create <name>.[/dim]")
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


def _cmd_history(ctx: CommandContext, cmd_args: str = ""):
    """Display conversation history in compact format."""
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

    con.print()
    con.print("  [bold]-- Terminal --[/bold]")
    con.print(f"  Working Dir: {os.getcwd()}")
    con.print(f"  Mode:        {'Shell' if ctx.shell_mode else 'Agent'}")
    if ctx.current_agent:
        con.print(f"  Permissions: {ctx.current_agent.tool_config.permission_mode}")

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
        "max_tokens": ctx.agent_config.get("max_tokens"),
        "temperature": ctx.agent_config.get("temperature"),
        "reasoning_effort": ctx.agent_config.get("reasoning_effort"),
        "top_p": ctx.agent_config.get("top_p"),
        "context_window": ctx.agent_config.get("context_window"),
        "extra_body": ctx.agent_config.get("extra_body"),
        "extra_headers": ctx.agent_config.get("extra_headers"),
    }
    ctx.current_agent.model = get_model(**model_kwargs)
    ctx.current_agent.environment_context = _build_environment_context(ctx.current_agent, ctx.agent_config)


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
        updated = self_manage.set_profile_field(field, value, target)
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


def _fmt_ms(ms) -> str:
    """Format an epoch-millis timestamp as 'YYYY-MM-DD HH:MM:SS', or '-' if falsy."""
    import datetime as _dt

    if not ms:
        return "-"
    try:
        return _dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)


def _fmt_next_run(job) -> str:
    """Human-friendly 'next run' time for a job, or '-' when not scheduled."""
    return _fmt_ms(getattr(job, "next_run_at_ms", None))


def _confirm_via_tui(ctx: CommandContext, question: str) -> bool:
    """TUI-safe yes/no confirmation for a command handler.

    Command handlers run on the background ``process_loop`` thread while
    prompt_toolkit owns the main thread. A nested ``pt_prompt`` there spins up
    a second Application that fights the live one — it deadlocks and leaks CPR
    escapes (``^[[..R``) into the input line, so the user can never answer.
    Instead route through the SAME ``ask_user_question_callback`` the agent's
    ``ask_user_question`` uses: it arms ``state.input_request`` and the main
    thread renders the inline prompt + feeds the typed answer back.

    Returns False when there is no interactive channel (non-TUI / tests) — the
    safe default for a destructive confirmation.
    """
    cb = ctx.ask_user_question_callback
    if cb is None:
        return False
    try:
        ans = cb(f"{question} (yes/no)", ["yes", "no"])
    except AgentCancelledError:
        # Ctrl+C at the prompt == "no" (safe default for a destructive op).
        return False
    return str(ans).strip().lower() in ("yes", "y", "是", "确认", "确定")


def _ask_text_via_tui(ctx: CommandContext, prompt: str, default: str = "") -> str:
    """TUI-safe free-text input for a command handler.

    Same rationale as :func:`_confirm_via_tui`: a nested ``pt_prompt`` on the
    background thread deadlocks the live prompt_toolkit app. Route through the
    ``ask_user_question_callback`` instead. Returns ``default`` when there is no
    interactive channel (non-TUI / tests) or when the user cancels (Ctrl+C).
    """
    cb = ctx.ask_user_question_callback
    if cb is None:
        return default
    try:
        ans = cb(prompt, None)
    except AgentCancelledError:
        return default
    return (ans or "").strip()


_CRON_PROMPT_REFINE_SYSTEM = """You rewrite a user's rough scheduled-task description into a single, concrete, self-contained execution prompt that an autonomous agent will run UNATTENDED on every tick (no human is watching at run time).

Rules:
- Output ONLY the rewritten prompt text — no preamble, no quotes, no explanation.
- Keep the user's original language.
- Make it unambiguous and directly actionable in one run: spell out the exact action, target path/format, and any naming convention implied by the description.
- Do NOT re-implement scheduling/recurrence — the cron system already handles "every N minutes". Describe only what to do in ONE run.
- Never ask the user questions in the prompt; it must be executable without clarification.
- Keep it concise (1-3 sentences)."""


async def _refine_cron_prompt(model, raw_prompt: str, schedule_human: str) -> str:
    """One-shot LLM rewrite of a cron task prompt. Returns "" on any failure."""
    from agentica.model.message import Message

    user = (
        f"Rough task description: {raw_prompt}\n"
        f"Schedule (handled by the system, do not re-implement): {schedule_human}\n\n"
        "Rewrite it into the unattended per-run execution prompt."
    )
    resp = await model.response(
        messages=[
            Message(role="system", content=_CRON_PROMPT_REFINE_SYSTEM),
            Message(role="user", content=user),
        ]
    )
    return (resp.content or "").strip()


def _cmd_cron(ctx: CommandContext, cmd_args: str = ""):
    """Manage scheduled (cron) jobs.

    /cron                          list all jobs
    /cron add "<prompt>" <schedule>  create a job (schedule: cron expr / 'every 5m' / ISO time)
    /cron edit <id> prompt "<text>"  change a job's execution prompt
    /cron edit <id> schedule <sched> change a job's schedule
    /cron pause <id>               pause a job
    /cron resume <id>              resume a job
    /cron remove <id>              delete a job
    /cron runs [<id>]              show recent run history
    /cron run <id>                 run a job once now
    /cron daemon on|off|status     control the in-CLI scheduler thread
    """
    con = get_console()
    from agentica.cron import jobs as cronjobs

    args = cmd_args.strip()
    sub = args.split()[0].lower() if args else "list"
    rest = args[len(sub) :].strip() if args else ""

    if sub in ("list", "ls", ""):
        all_jobs = cronjobs.list_jobs()
        if not all_jobs:
            con.print('[dim]No cron jobs. Add one with: /cron add "<prompt>" <schedule>[/dim]')
            return
        con.print("[bold]Cron jobs[/bold]")
        for j in all_jobs:
            status = getattr(getattr(j, "status", None), "value", getattr(j, "status", "?"))
            human = cronjobs.schedule_to_human(j.schedule)
            con.print(
                f"  [cyan]{j.id}[/cyan]  [yellow]{j.name}[/yellow]  "
                f"[{'green' if status == 'active' else 'dim'}]{status}[/]  "
                f"next: {_fmt_next_run(j)}  [dim]({human})[/dim]"
            )
        return

    if sub == "add":
        # Parse: /cron add "prompt with spaces" <schedule>
        import shlex

        try:
            tokens = shlex.split(rest)
        except ValueError:
            con.print('[red]Could not parse. Use: /cron add "<prompt>" <schedule>[/red]')
            return
        if len(tokens) < 2:
            con.print('[red]Usage: /cron add "<prompt>" <schedule>[/red]')
            con.print("[dim]schedule examples: '0 9 * * *' (9am daily), 'every 30m', '2026-07-01T09:00:00'[/dim]")
            return
        prompt = tokens[0]
        schedule = " ".join(tokens[1:])
        # Validate the schedule up front so we don't burn an LLM refine call on
        # a job that can't be created anyway.
        try:
            parsed = cronjobs.parse_schedule(schedule)
        except Exception as e:
            con.print(f"[red]{e}[/red]")
            return
        human = cronjobs.schedule_to_human(parsed)

        # Add-time refinement + confirmation. A cron job runs UNATTENDED, so the
        # right moment to resolve ambiguity is NOW (the human is here), not at
        # first execution. Refine the rough prompt into a concrete per-run
        # execution prompt with the auxiliary model, then let the user confirm /
        # keep original / hand-write. Interactive path only (needs both a model
        # and the TUI input channel); non-TUI / tests fall straight through.
        final_prompt = prompt
        agent = ctx.current_agent
        if agent is not None and ctx.ask_user_question_callback is not None:
            con.print("[dim]Refining the task prompt with the model…[/dim]")
            refined = ""
            try:
                model = agent.resolve_auxiliary_model("cron_refine")
                refined = asyncio.run(_refine_cron_prompt(model, prompt, human))
            except Exception as e:
                con.print(f"[dim]Refine skipped ({e}); using the original prompt.[/dim]")
            if refined and refined != prompt:
                con.print(f"\n  [bold]Original:[/bold] {prompt}")
                con.print(f"  [bold]Refined :[/bold] {refined}\n")
                opt_refined = "Use the refined prompt (recommended)"
                opt_original = "Keep the original prompt"
                opt_manual = "Write it myself"
                try:
                    choice = str(
                        ctx.ask_user_question_callback(
                            "Which prompt should this scheduled job run?",
                            [opt_refined, opt_original, opt_manual],
                        )
                    ).strip()
                except AgentCancelledError:
                    con.print("[dim]cancelled — job not created[/dim]")
                    return
                if choice == opt_original:
                    final_prompt = prompt
                elif choice == opt_manual:
                    try:
                        typed = ctx.ask_user_question_callback("Enter the execution prompt:", None)
                    except AgentCancelledError:
                        con.print("[dim]cancelled — job not created[/dim]")
                        return
                    final_prompt = (typed or "").strip() or prompt
                else:
                    final_prompt = refined

        try:
            # Keep the user's own words as the display name; run the (possibly
            # refined) prompt under the hood.
            job = cronjobs.create_job(prompt=final_prompt, schedule=schedule, name=prompt[:50])
        except Exception as e:
            con.print(f"[red]Failed to create job: {e}[/red]")
            return
        con.print(f"[green]Created job [cyan]{job.id}[/cyan] '{job.name}'[/green] next: {_fmt_next_run(job)}")
        if final_prompt != prompt:
            con.print(f"[dim]execution prompt: {final_prompt}[/dim]")
        if not (ctx.tui_state and ctx.tui_state.get("cron_is_running", lambda: False)()):
            con.print("[yellow]Scheduler is OFF — job won't run until you enable it: /cron daemon on[/yellow]")
        return

    if sub in ("pause", "resume", "remove", "rm", "delete"):
        if not rest:
            con.print(f"[red]Usage: /cron {sub} <id>[/red]")
            return
        job_id = rest.split()[0]
        try:
            if sub == "pause":
                cronjobs.pause_job(job_id)
                con.print(f"[green]Paused {job_id}[/green]")
            elif sub == "resume":
                cronjobs.resume_job(job_id)
                con.print(f"[green]Resumed {job_id}[/green]")
            else:
                if _confirm_via_tui(ctx, f"Delete cron job {job_id}?"):
                    cronjobs.remove_job(job_id)
                    con.print(f"[green]Removed {job_id}[/green]")
                else:
                    con.print("[dim]cancelled[/dim]")
        except Exception as e:
            con.print(f"[red]{e}[/red]")
        return

    if sub == "edit":
        # /cron edit <id> prompt "<text>"   |   /cron edit <id> schedule <schedule>
        parts = rest.split(maxsplit=2)
        if len(parts) < 3 or parts[1].lower() not in ("prompt", "schedule"):
            con.print('[red]Usage: /cron edit <id> prompt "<text>"  |  /cron edit <id> schedule <schedule>[/red]')
            return
        job_id, field_name, value = parts[0], parts[1].lower(), parts[2].strip()
        if cronjobs.get_job(job_id) is None:
            con.print(f"[red]No job {job_id}[/red]")
            return
        try:
            if field_name == "prompt":
                new_prompt = value.strip('"').strip("'")
                cronjobs.update_job(job_id, {"prompt": new_prompt, "name": new_prompt[:50]})
                con.print(f"[green]Updated prompt for {job_id}[/green]")
                con.print(f"[dim]execution prompt: {new_prompt}[/dim]")
            else:  # schedule
                parsed = cronjobs.parse_schedule(value)
                next_run = cronjobs.compute_next_run_at_ms(parsed)
                cronjobs.update_job(job_id, {"schedule": parsed, "next_run_at_ms": next_run or 0})
                con.print(f"[green]Updated schedule for {job_id}: {cronjobs.schedule_to_human(parsed)}[/green]")
        except Exception as e:
            con.print(f"[red]{e}[/red]")
        return

    if sub == "runs":
        job_id = rest.split()[0] if rest else None
        runs = cronjobs.list_task_runs(job_id=job_id)
        if not runs:
            con.print("[dim]No run history.[/dim]")
            return
        con.print("[bold]Recent runs[/bold]")
        for r in runs[:20]:
            st = r.status.value if hasattr(r.status, "value") else str(r.status)
            when = _fmt_ms(r.started_at_ms)
            color = "green" if st == "ok" else ("yellow" if st == "timeout" else "red")
            con.print(f"  [cyan]{r.task_id}[/cyan]  [{color}]{st}[/]  {when}")
        return

    if sub == "run":
        if not rest:
            con.print("[red]Usage: /cron run <id>[/red]")
            return
        job_id = rest.split()[0]
        job = cronjobs.get_job(job_id)
        if not job:
            con.print(f"[red]No job {job_id}[/red]")
            return
        con.print(f"[dim]Running job {job_id} once now...[/dim]")
        try:
            from agentica.cron.scheduler import _execute_job
            from agentica.cron.cli_runner import CliAgentRunner, build_cli_agent_factory

            factory = build_cli_agent_factory(ctx.agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)
            runner = CliAgentRunner(factory)
            asyncio.run(_execute_job(job, agent_runner=runner, verbose=False))
            con.print(f"[green]Job {job_id} executed.[/green]")
        except Exception as e:
            con.print(f"[red]Run failed: {e}[/red]")
        return

    if sub == "daemon":
        action = rest.split()[0].lower() if rest else "status"
        ts = ctx.tui_state or {}
        is_running = ts.get("cron_is_running", lambda: False)()
        if action == "status":
            from agentica.global_config import get_setting

            enabled = bool(get_setting("cron.enabled", False))
            interval = int(get_setting("cron.interval", 60) or 60)
            con.print(
                f"  this session:   [{'green' if is_running else 'dim'}]{'RUNNING' if is_running else 'STOPPED'}[/]"
            )
            con.print(
                f"  config (persisted): cron.enabled="
                f"[{'green' if enabled else 'red'}]{enabled}[/]  interval={interval}s"
            )
            # The live thread only reflects THIS process. Explain a mismatch so
            # "status can't be seen" never looks like a silent failure.
            if enabled and not is_running:
                con.print(
                    "  [yellow]Enabled in config but no scheduler thread in this "
                    "session[/yellow] — a separate `agentica cron daemon` process may be "
                    "running it (the file lock prevents double execution), or the thread "
                    "failed to start. Run [cyan]/cron daemon on[/cyan] to start it here."
                )
            elif not enabled and is_running:
                con.print(
                    "  [yellow]Running in this session but disabled in config[/yellow] — "
                    "it will not auto-start next launch. Run [cyan]/cron daemon on[/cyan] to persist."
                )
            return
        if action == "on":
            from agentica.global_config import set_setting

            set_setting("cron.enabled", True)  # persist so it survives restart
            start = ts.get("cron_start")
            if start and start():
                con.print("[green]Scheduler started (and enabled in config).[/green]")
            else:
                con.print("[yellow]Enabled in config; could not start thread in this session. Restart CLI.[/yellow]")
            return
        if action == "off":
            from agentica.global_config import set_setting

            set_setting("cron.enabled", False)
            stop = ts.get("cron_stop")
            if stop:
                stop()
            con.print("[green]Scheduler stopped (and disabled in config).[/green]")
            return
        con.print("[red]Usage: /cron daemon on|off|status[/red]")
        return

    con.print(f"[red]Unknown subcommand '{sub}'. See /cron for usage.[/red]")


def _cmd_newchat(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    current_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
    )
    con.print("[green]New chat session created.[/green]")
    con.print("[dim]Conversation history cleared.[/dim]")
    # Drop any goal manager — the new session has a new SessionLog.
    return {"current_agent": current_agent, "goal_manager": None}


def _resume_base_dir(ctx: CommandContext) -> Optional[str]:
    """Resolve the sessions directory the CLI operates on.

    Prefer the active agent's live ``SessionLog.base_dir`` so the list is scoped
    by the same project (work_dir) + user the agent writes to. Falling back to
    ``None`` lets ``SessionLog`` derive the default from the process cwd, which
    for the CLI equals the current project.
    """
    agent = ctx.current_agent
    log = agent._session_log if agent is not None else None
    return str(log.base_dir) if log is not None else None


def _cmd_resume(ctx: CommandContext, cmd_args: str = ""):
    """Resume a previous session from JSONL log."""
    con = get_console()

    # Scope the session list by project (work_dir) + user, exactly like the Web
    # sidebar, so both entrypoints show a consistent set for the same project.
    # The running agent already carries the correctly-scoped work_dir/user_id;
    # fall back to the process cwd (which for the CLI equals the project).
    base_dir = _resume_base_dir(ctx)

    sessions = SessionLog.list_sessions(base_dir=base_dir)
    if not sessions:
        con.print("[yellow]No sessions found to resume.[/yellow]")
        return

    # Archived sessions are hidden from the picker (same "I don't want to see
    # this anymore" semantic as the Web UI sidebar), but an explicit id/prefix
    # match below still searches the full unfiltered `sessions` list so an
    # archived session remains directly resumable by id.
    visible_sessions = [s for s in sessions if not s.get("archived")]

    args_str = (cmd_args or "").strip()
    resume_at_uuid = None
    session_target, separator, at_candidate = args_str.rpartition(" at ")
    if separator and session_target.strip() and at_candidate.strip():
        try:
            UUID(at_candidate.strip())
        except ValueError:
            pass
        else:
            args_str = session_target.strip()
            resume_at_uuid = at_candidate.strip()

    if args_str:
        named_matches = [
            session
            for session in visible_sessions
            if isinstance(session.get("name"), str)
            and session["name"].casefold() == args_str.casefold()
        ]
        if args_str.isdecimal():
            index = int(args_str) - 1
            if 0 <= index < len(visible_sessions):
                chosen = visible_sessions[index]
            elif len(named_matches) == 1:
                chosen = named_matches[0]
            else:
                con.print("[red]Invalid number.[/red]")
                return
        elif len(named_matches) == 1:
            chosen = named_matches[0]
        elif len(named_matches) > 1:
            con.print(
                f"[red]Ambiguous: multiple sessions are named '{args_str}'. Use the number or id prefix.[/red]"
            )
            return
        else:
            # Accept the exact id, any unique prefix, or the truncated
            # "7154826e...0358" form printed by the picker.
            needle = args_str
            if "..." in needle:
                needle = needle.split("...", 1)[0].strip()
            matching = [
                session
                for session in sessions
                if needle and session["session_id"].startswith(needle)
            ]
            if not matching:
                con.print(f"[red]No session matching '{args_str}'[/red]")
                return
            if len(matching) > 1:
                con.print(
                    f"[red]Ambiguous: '{args_str}' matches {len(matching)} sessions. Use a longer prefix or the number.[/red]"
                )
                return
            chosen = matching[0]

        agent_config = dict(ctx.agent_config)
        agent_config["session_id"] = chosen["session_id"]
        agent_config["_resume_at_uuid"] = resume_at_uuid
        current_agent = create_agent(agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)

        # Eagerly load history into working_memory so /status, /context etc.
        # reflect the resumed state immediately (do not wait for the next _run
        # to lazily replay). Applies to both plain resume and `resume ... at <uuid>`.
        loaded_count = 0
        runs_built = 0
        if current_agent._session_log and current_agent._session_log.exists():
            current_agent.working_memory.clear()
            resumed = current_agent._session_log.load(resume_at=resume_at_uuid)
            if resumed:
                runs_built = current_agent.working_memory.hydrate_runs_from_history(resumed)
                loaded_count = len(resumed)

        # Show a preview of recent user queries so the human confirms the right
        # session was picked (also useful for finding an `at <uuid>` cut point).
        session_name = chosen.get("name")
        session_label = f"{session_name} ({chosen['session_id']})" if session_name else chosen["session_id"]
        if resume_at_uuid is None:
            log = SessionLog(chosen["session_id"], base_dir=base_dir)
            user_msgs = log.list_user_messages(limit=10)
            if user_msgs:
                con.print(f"\n[bold]Session: {session_label}[/bold]")
                con.print("[dim]Recent user queries in this session:[/dim]\n")
                for i, m in enumerate(user_msgs, 1):
                    ts = m.get("timestamp", "")[:19].replace("T", " ") if m.get("timestamp") else ""
                    con.print(f"  {i}. [dim]{ts}[/dim] {m['content']}")
                con.print(
                    f"\n[dim]Tip: fork from an earlier point with `/resume {chosen['session_id']} at <uuid>`[/dim]"
                )

        con.print(
            f"[green]Resumed session: {session_label}"
            f"{f' at {resume_at_uuid[:8]}...' if resume_at_uuid else ''}"
            f" — loaded {loaded_count} messages ({runs_built} runs) into context[/green]"
        )

        # If the resumed session had an active goal, demote to paused for
        # safety — automatic continuation on resume is too surprising
        # without token-budget guards (P0).
        resumed_goal_manager = None
        if current_agent._session_log is not None:
            judge_model = current_agent.auxiliary_model or current_agent.model
            resumed_goal_manager = GoalManager(current_agent._session_log, judge_model=judge_model)
            state = resumed_goal_manager.load()
            if state is not None:
                if state.status == "active":
                    resumed_goal_manager.force_pause_on_resume()
                    con.print(f"  [yellow]⊙ Standing goal detected and paused for safety:[/yellow] {state.objective}")
                    con.print("  [dim]Use /goal resume to continue working on it.[/dim]")
                elif state.status in ("paused", "complete"):
                    con.print(f"  [dim]⊙ Previous goal ({state.status}): {state.objective}[/dim]")

        return {"current_agent": current_agent, "goal_manager": resumed_goal_manager}
    else:
        if not visible_sessions:
            con.print("[yellow]No sessions found to resume (all sessions are archived).[/yellow]")
            return
        con.print("\n[bold]Available sessions:[/bold]\n")
        for i, s in enumerate(visible_sessions[:10], 1):
            ts_str = s.get("last_timestamp", "") or ""
            if ts_str:
                ts_str = ts_str[:16].replace("T", " ")
            size_kb = s["size_bytes"] / 1024
            sid = s["session_id"]
            # Show a clean, copy-pasteable 8-char prefix that /resume accepts
            # directly. Avoid the old "abc...wxyz" form which users would copy
            # verbatim (ellipsis included) and which then failed to match.
            short_id = sid if len(sid) <= 12 else sid[:8]
            # Prefer the user-set `/rename` label; otherwise show the first
            # user message that started the session.
            preview = SessionLog.session_preview(s["path"])
            turns = preview["user_count"]
            first_user = preview["first_user"]
            user_name = s.get("name")
            if user_name:
                # Named session: name is the headline, preview is the subline.
                summary = user_name[:80]
                subline = " ".join(first_user.split())[:80] if first_user else "(no messages yet)"
            elif first_user:
                # Unnamed session: keep the legacy single-line preview.
                summary = " ".join(first_user.split())[:80]
                subline = None
            else:
                summary = "(empty session)"
                subline = None
            con.print(f"  {i}. [cyan]{short_id}[/cyan]  {ts_str}  ({size_kb:.0f}KB, {turns} turns)")
            if user_name:
                con.print(f"     [bold]{summary}[/bold]")
                if subline:
                    con.print(f"     [dim]> {subline}[/dim]")
            else:
                con.print(f"     [dim]> {summary}[/dim]")
        con.print(
            f"\n[dim]Usage: /resume <number|name|id-prefix> (e.g. /resume {visible_sessions[0]['session_id'][:8]})[/dim]"
        )
        return


def _cmd_rename(ctx: CommandContext, cmd_args: str = ""):
    """Rename the active session so it is easy to identify in `/resume`."""
    con = get_console()
    new_name = (cmd_args or "").strip()
    if not new_name:
        con.print("  [dim]Usage: /rename <name>[/dim]")
        return

    agent = ctx.current_agent
    if agent is None or not agent.session_id or agent._session_log is None:
        con.print("[yellow]No active session to rename.[/yellow]")
        return

    try:
        agent._session_log.set_name(new_name)
    except OSError as error:
        con.print(f"  [red]Failed to rename session: {error}[/red]")
        return
    con.print(f"  [green]Renamed current session to[/green] [cyan]{new_name}[/cyan]")


def _cmd_clear(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    os.system("clear" if os.name != "nt" else "cls")
    current_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
    )
    print_header(
        ctx.agent_config["model_provider"],
        ctx.agent_config["model_name"],
        work_dir=ctx.agent_config.get("work_dir"),
        extra_tools=ctx.extra_tool_names,
        shell_mode=ctx.shell_mode,
    )
    con.print("[info]Screen cleared and conversation reset.[/info]")
    return {"current_agent": current_agent, "goal_manager": None}


def _apply_profile(ctx: CommandContext, name: str):
    """Switch the live agent to a named config.yaml profile."""
    con = get_console()
    profile = get_profile(name)
    if not profile or not profile.get("model_provider"):
        con.print(f"[red]Profile not found or incomplete: {name}[/red]")
        names = list(get_profiles().keys())
        if names:
            con.print(f"Available profiles: {', '.join(names)}", style="dim")
        return

    new_provider = profile["model_provider"]
    new_model = profile.get("model_name") or default_model_name(new_provider)
    new_base_url = profile.get("base_url") or default_base_url(new_provider)
    new_key = profile.get("api_key") or get_profile_api_key(new_provider, new_base_url)

    # Model tuning params: a profile-switch fully replaces the previous model's
    # tuning, so unset profile fields reset to None (the model factory default)
    # rather than leaking the old profile's values.
    new_max_tokens = profile.get("max_tokens")
    new_temperature = profile.get("temperature")
    new_reasoning_effort = profile.get("reasoning_effort")
    new_top_p = profile.get("top_p")
    new_context_window = profile.get("context_window")
    new_extra_body = profile.get("extra_body")
    new_extra_headers = profile.get("extra_headers")

    ctx.agent_config["model_provider"] = new_provider
    ctx.agent_config["model_name"] = new_model
    ctx.agent_config["base_url"] = new_base_url
    ctx.agent_config["api_key"] = new_key
    ctx.agent_config["max_tokens"] = new_max_tokens
    ctx.agent_config["temperature"] = new_temperature
    ctx.agent_config["reasoning_effort"] = new_reasoning_effort
    ctx.agent_config["top_p"] = new_top_p
    ctx.agent_config["context_window"] = new_context_window
    ctx.agent_config["extra_body"] = new_extra_body
    ctx.agent_config["extra_headers"] = new_extra_headers

    # Auxiliary model: a profile switch fully replaces the auxiliary model too. An
    # auxiliary_model block rebuilds the sibling (background calls + task subagent);
    # a profile without one clears the auxiliary fields so they fall back to the
    # main model. _build_sibling_model handles same-provider base_url/api_key
    # inheritance; cross-provider reads the block's own key (or env). The auxiliary
    # rebuild is a tolerance boundary — a broken auxiliary config falls back to the
    # main model with a warning instead of blocking the core model switch.
    auxiliary_block = profile.get("auxiliary_model")
    if isinstance(auxiliary_block, dict) and auxiliary_block.get("model_name"):
        ctx.agent_config["auxiliary_model_provider"] = auxiliary_block.get("model_provider") or new_provider
        ctx.agent_config["auxiliary_model_name"] = auxiliary_block.get("model_name")
        ctx.agent_config["auxiliary_base_url"] = auxiliary_block.get("base_url")
        ctx.agent_config["auxiliary_api_key"] = auxiliary_block.get("api_key")
        ctx.agent_config["auxiliary_extra_body"] = auxiliary_block.get("extra_body")
        ctx.agent_config["auxiliary_extra_headers"] = auxiliary_block.get("extra_headers")
        try:
            new_auxiliary_model = _build_sibling_model(ctx.agent_config, "auxiliary")
        except Exception as exc:
            con.print(f"[yellow]Auxiliary model build failed, falling back to main model: {exc}[/yellow]")
            ctx.agent_config["auxiliary_model_provider"] = None
            ctx.agent_config["auxiliary_model_name"] = None
            ctx.agent_config["auxiliary_base_url"] = None
            ctx.agent_config["auxiliary_api_key"] = None
            ctx.agent_config["auxiliary_extra_body"] = None
            ctx.agent_config["auxiliary_extra_headers"] = None
            new_auxiliary_model = None
    else:
        ctx.agent_config["auxiliary_model_provider"] = None
        ctx.agent_config["auxiliary_model_name"] = None
        ctx.agent_config["auxiliary_base_url"] = None
        ctx.agent_config["auxiliary_api_key"] = None
        ctx.agent_config["auxiliary_extra_body"] = None
        ctx.agent_config["auxiliary_extra_headers"] = None
        new_auxiliary_model = None
    ctx.agent_config["auxiliary_model"] = new_auxiliary_model

    # Persist the switch as a *project-scoped* override under
    # ~/.agentica/projects/<key>/profile. config.yaml's global
    # `active_profile:` pointer is untouched — that stays the machine-wide
    # default, and other projects keep whatever they had.
    #
    # Profile bodies (model_*, auxiliary_*, tuning) are write-only-by-setup
    # — never touched here. That separation is what fixed the original
    # "config.yaml 乱掉" bug, where the old free-form `/model provider/name`
    # path rewrote fields of the active profile in place.
    work_dir = ctx.agent_config.get("work_dir") or os.getcwd()
    set_project_profile(work_dir, name)

    model_kwargs = {
        "model_provider": new_provider,
        "model_name": new_model,
        "base_url": new_base_url,
        "api_key": new_key,
        "max_tokens": new_max_tokens,
        "temperature": new_temperature,
        "reasoning_effort": new_reasoning_effort,
        "top_p": new_top_p,
        "context_window": new_context_window,
        "extra_body": new_extra_body,
        "extra_headers": new_extra_headers,
    }
    new_model_obj = get_model(**model_kwargs)
    if ctx.current_agent is not None:
        ctx.current_agent.model = new_model_obj
        ctx.current_agent.auxiliary_model = new_auxiliary_model
        # Repoint the task subagent tool onto the new auxiliary model (None = clone
        # the parent's main model, matching create_agent's default).
        _update_task_tool_model_override(ctx.current_agent, new_auxiliary_model)
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
    )
    con.print(f"[green]Switched to profile '{name}': {new_provider}/{new_model}[/green]")
    return {"current_agent": current_agent}


def _clear_project_profile_override(ctx: CommandContext) -> None:
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
    return _apply_profile(ctx, default_name)


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
        if p.get("reasoning_effort"):
            tuning.append(f"effort={p['reasoning_effort']}")
        if p.get("max_tokens"):
            tuning.append(f"max_tokens={p['max_tokens']}")
        if p.get("context_window"):
            tuning.append(f"context={p['context_window']}")
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

        compacted = _run_async_safe(
            cm.auto_compact(
                messages,
                model=model,
                force=True,
                working_memory=wm,
                custom_instructions=custom_instructions,
            )
        )
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
        content = msg.content or ""
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
        con.print("  [dim]ask[/dim]        - only read-only tools are exposed; no writes at all")
        con.print("  [dim]auto[/dim]       - all tools exposed; writes restricted to work_dir")
        con.print("  [dim]allow-all[/dim]  - all tools exposed, no restriction (default)")
        con.print()
        con.print(
            "None of these are hard dead ends: if a write/read is blocked (outside work_dir "
            "in auto mode, or a sensitive path like ~/.ssh, ~/.aws/credentials, /etc — blocked "
            "in ANY mode including allow-all), the agent can call request_path_access(path, "
            "reason) to ask you for a one-time yes/no approval, which then whitelists that "
            "path for the rest of the session.",
            style="dim",
        )
        con.print()
        con.print("Usage: /permissions <mode>", style="dim")


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
        con.print(
            "  [dim]Usage: /queue <prompt>  |  /queue list  |  /queue edit <n> <text>  |  /queue insert <n> <text>  |  /queue remove <n>  |  /queue clear[/dim]"
        )
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

    if subcommand == "edit":
        if pq is None:
            return
        rest = sub[1].strip() if len(sub) > 1 else ""
        parts = rest.split(maxsplit=1)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].strip():
            con.print("  [dim]Usage: /queue edit <number> <new text>[/dim]")
            return
        idx = int(parts[0]) - 1
        new_text = parts[1]
        if pq.replace_index(idx, new_text):
            preview = new_text[:80] + ("..." if len(new_text) > 80 else "")
            con.print(f"  [green]Edited queued message #{idx + 1}:[/green] [dim]{preview}[/dim]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1}[/red]")
        return

    if subcommand == "insert":
        if pq is None:
            return
        rest = sub[1].strip() if len(sub) > 1 else ""
        parts = rest.split(maxsplit=1)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].strip():
            con.print("  [dim]Usage: /queue insert <number> <text>  (1 = front, qsize+1 = back)[/dim]")
            return
        idx = int(parts[0]) - 1
        new_text = parts[1]
        if pq.insert_index(idx, new_text):
            preview = new_text[:80] + ("..." if len(new_text) > 80 else "")
            con.print(f"  [green]Inserted at position #{idx + 1}:[/green] [dim]{preview}[/dim]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1} (valid range: 1..{pq.qsize() + 1})[/red]")
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
            con.print(f"  [red]Refusing to restore checkpoint files outside work_dir: {root}[/red]")
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
            f"Background #{task_num}",
            prompt,
            result_text or "",
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
        con.print(
            "  [yellow]Cannot set a new goal while the agent is running. "
            "Wait for the current turn or use /goal pause/clear.[/yellow]"
        )
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
            goal=state.objective,
            source_query=state.objective,
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

    # Attach the goal tool (verify_completion + update_goal) so the model can
    # verify completion with evidence and break the loop when actually done.
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
    "/new": (_cmd_newchat, "Start a new chat session"),
    "/clear": (_cmd_clear, "Clear screen and reset"),
    "/reset": (_cmd_clear, "Clear screen and reset (alias)"),
    "/history": (_cmd_history, "Show conversation history"),
    "/export": (_cmd_export, "Save conversation to JSON"),
    "/save": (_cmd_export, "Save conversation to JSON (alias)"),
    "/retry": (_cmd_retry, "Retry the last message (resend to agent)"),
    "/undo": (_cmd_undo, "Remove the last user/assistant exchange"),
    "/compact": (_cmd_compact, "Compact context (summarize history)"),
    "/rename": (_cmd_rename, "Rename the current session for easy resume"),
    "/resume": (_cmd_resume, "Resume by number, name, or id prefix"),
    "/goal": (_cmd_goal, "Set or manage a standing goal (auto-continues until done)"),
    "/subgoal": (_cmd_subgoal, "Add or manage acceptance criteria on the active goal"),
    "/btw": (_cmd_btw, "Quick aside answered in parallel \u2014 no tools, not persisted"),
    "/queue": (
        _cmd_queue,
        "Run as the NEXT turn after the current run finishes | list | edit <n> | insert <n> | remove <n> | clear",
    ),
    "/q": (_cmd_queue, "Run as the next turn after current run (alias)"),
    "/background": (_cmd_background, "Run NOW in a parallel independent agent (own session)"),
    "/bg": (_cmd_background, "Run now in a parallel independent agent (alias)"),
    "/stop": (_cmd_stop, "Kill all running background tasks"),
    "/steer": (_cmd_steer, "Course-correct the CURRENT run mid-task (injected between tool batches)"),
    "/checkpoint": (
        _cmd_checkpoint,
        "Durable file snapshots: list | create <label> <path...> | diff <id> | restore <id>",
    ),
    # Model & Config
    "/model": (_cmd_model, "View or switch model"),
    "/config": (_cmd_config, "Show config | set <field> <value> | env <KEY> <value> | path"),
    "/upgrade": (_cmd_upgrade, "Self-upgrade agentica via pip (check | --pre)"),
    "/cron": (_cmd_cron, "Scheduled jobs: list | add | edit | pause | resume | remove | runs | run | daemon"),
    "/cost": (_cmd_cost, "Show token usage and cost"),
    "/usage": (_cmd_cost, "Show token usage and cost (alias)"),
    "/debug": (_cmd_debug, "Show debug info"),
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
