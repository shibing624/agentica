# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified 3-tier tool permission model.

Shared by the SDK (Agent/DeepAgent), the CLI, and the Gateway so all three
surfaces expose the exact same vocabulary and behavior. Tools stay in the
schema in every tier; the difference is whether the runner parks before
``fc.execute()`` (see ``agentica.agent.approvals``).

  - "ask"       : Ask for approval. Reads (including outside the work
                  directory), read-only ``execute``, network tools
                  (``web_search`` / ``fetch_url``), memory, ``task`` /
                  ``delegate``, skills, and other builtins run without
                  prompting. Parking: every ``write_file`` / ``apply_patch``
                  (even inside the work directory) and non-read-only
                  ``execute``.
  - "auto"      : Approve for me. Reads (including outside the work
                  directory), in-workspace writes, every ``execute``,
                  network tools, and every builtin / skill / third-party
                  tool (``self_manage``, ``cronjob``, ``get_skill_info``, …
                  regardless of ``action``) run without prompting. Parks
                  only for file *writes* outside the work directory or to
                  sensitive paths.
  - "allow-all" : Full Access. Never parks. Hard refusals for ``/etc``,
                  ``~/.ssh`` and similar write targets still apply; they
                  raise ``PermissionError`` rather than showing a card.

Callers should not construct their own mode strings — always compare against
``PERMISSION_MODES`` / use ``validate_permission_mode`` so a typo fails loud
instead of silently falling through to the most permissive behavior.
"""
from typing import List, Optional, Set

PERMISSION_MODES = ("ask", "auto", "allow-all")

# Historically the tools "ask" exposed by hiding the rest. The classifier
# no longer uses this set (tools stay visible); kept as a named list of
# tools that are safe to auto-allow inside the workspace.
READ_ONLY_TOOLS: Set[str] = frozenset({
    "read_file", "glob", "grep", "web_search", "fetch_url",
    "write_todos", "task",
    "list_agents",
    "search_memory",
})


def validate_permission_mode(mode: str) -> None:
    """Raise ValueError if `mode` is not one of PERMISSION_MODES."""
    if mode not in PERMISSION_MODES:
        raise ValueError(f"Invalid permission mode: {mode!r}. Must be one of {PERMISSION_MODES}.")


def read_only_whitelist(mode: str) -> Optional[List[str]]:
    """Query-level tool whitelist for `mode`. Always None: every tier exposes all tools."""
    validate_permission_mode(mode)
    return None


def sandbox_should_be_enabled(mode: str) -> bool:
    """Whether write operations should be path-restricted to work_dir for `mode`."""
    return mode != "allow-all"
