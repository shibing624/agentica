# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared helpers used by multiple CLI slash-command handlers
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from agentica.cli.commands.context import CommandContext
from agentica.cli.runtime import (
    get_console,
    create_agent,
)
from agentica.subagent import get_subagent_configs
from agentica.skills import (
    get_skill_registry,
    load_skills,
)
from agentica.skills.skill_registry import reset_skill_registry




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
    from agentica.agent.history_filter import strip_tool_artifacts_from_memory

    strip_tool_artifacts_from_memory(agent.working_memory)



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
        background_process_registry=ctx.background_processes,
        peer_session=ctx.peer_session,
        worktree_binder=ctx.worktree_binder,
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
    """Import the subagent loader lazily to keep CLI startup lightweight."""
    import agentica.subagent_loader as loader  # noqa: PLC0415

    return loader



def _get_defined_agents_for_display() -> list:
    """Return effective file definitions with paths for ``/agents``."""
    loader = _get_subagent_loader()
    if loader is not None:
        return loader.list_defined_agents()
    return [
        {
            "id": name,
            "name": cfg.name,
            "description": cfg.description,
            "allowed_tools": cfg.allowed_tools,
            "denied_tools": cfg.denied_tools,
            "tool_call_limit": cfg.tool_call_limit,
            "model_tier": cfg.model_tier,
            "source": cfg.source,
            "path": cfg.path,
        }
        for name, cfg in get_subagent_configs().items()
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



def _update_task_tool_auxiliary_model(agent, auxiliary_model) -> None:
    """Repoint the BuiltinTaskTool's cheap-tier model on the live agent."""
    from agentica.tools.builtin_task_tool import BuiltinTaskTool

    if not agent or not agent.tools:
        return
    for tool in agent.tools:
        if isinstance(tool, BuiltinTaskTool):
            tool._auxiliary_model = auxiliary_model
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
