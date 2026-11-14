# -*- coding: utf-8 -*-
"""Build the tool set shared by every Agentica product surface.

CLI, Web, Desktop and IM are outlets of one agent harness. They use the same
bundled skills catalog and the same built-in tools; only the callbacks and
session-local state differ. This module owns that shared tool construction so a
surface cannot silently lose a capability that its skills still advertise.
"""
from __future__ import annotations

from typing import Any, Optional

from agentica.git.binding import WorktreeBinder
from agentica.tools.cron_tool import CronTool
from agentica.tools.self_manage_tool import CLI_RESTART_HINT, SelfManageTool
from agentica.tools.worktree_tool import WorktreeTool


def build_product_tools(
    *,
    surface: str,
    work_dir: str,
    owner: Optional[str] = None,
    session_id: Optional[str] = None,
    service: Optional[Any] = None,
    agent_config: Optional[dict] = None,
    cron_job_runner: Optional[Any] = None,
    extra_tools: Optional[list] = None,
    restart_hint: str = "",
    enter_work_dir: Optional[Any] = None,
) -> list:
    """Return the extra tools mounted on a product agent.

    ``surface`` selects only surface-specific wiring (CLI restart wording,
    gateway cron ownership, session-bound worktree config). It never decides
    *which* bundled capabilities exist: all surfaces get ``self_manage``,
    ``cronjob`` and ``worktree``.
    """
    if surface not in {"cli", "gateway", "im", "headless", "cron"}:
        raise ValueError(f"unknown product surface: {surface}")

    if agent_config is None:
        agent_config = {"work_dir": work_dir}
    elif "work_dir" not in agent_config:
        agent_config["work_dir"] = work_dir

    if service is not None and session_id and surface in {"gateway", "im"}:
        resolved = service.get_session_work_dir(session_id, owner)
        agent_config = dict(agent_config)
        agent_config["work_dir"] = resolved

    tools = [
        WorktreeTool(WorktreeBinder(agent_config=agent_config, enter_work_dir=enter_work_dir)),
        SelfManageTool(
            restart_hint=CLI_RESTART_HINT if surface == "cli" else (restart_hint or ""),
        ),
        CronTool(job_runner=cron_job_runner, owner=owner if surface != "cli" else None),
    ]
    tools.extend(extra_tools or [])
    return tools


__all__ = ["build_product_tools"]
