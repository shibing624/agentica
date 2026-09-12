# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI preferences that outlive one process.

``/reasoning``, ``/statusbar``, ``/debug``, ``/permissions`` and ``/tools add``
used to live only in the process that set them: the change was gone at the next
launch and, for ``/permissions``, even earlier than that — an agent rebuild
(``/resume``, ``/model``) re-reads ``agent_config`` rather than the live agent,
so the tier silently fell back to ``allow-all``. They are recorded in two
places, mirroring how the model profile is stored:

* ``project.json``'s ``cli`` block — what a **new** CLI in this work_dir starts
  with, next to ``active_profile``;
* ``<session_id>.meta.json``'s ``cli`` block — what **this** session was last
  running, authoritative on ``/resume`` so a second session in the same
  work_dir cannot drag the first one back to its own view settings.

Who wins at startup, highest first:

1. an explicit flag for a setting that has one (``--debug``, ``--permissions``);
2. the resumed session's sidecar;
3. this work_dir's ``project.json``;
4. the built-in default.

Deliberately *not* persisted: the extra tools loaded with ``/tools add``. They
join the tool schema of every request for the rest of the session, so the
deliberate way to have them is ``--tools`` on the command line.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agentica.agent.permissions import PERMISSION_MODES
from agentica.utils.log import logger

# Read/written as a group: the CLI settings that are not model config. Adding a
# key here is the only change needed for it to be persisted, read back, and (if
# it is a view toggle) pushed to the status bar / stream loop.
CLI_PREF_KEYS = ("show_reasoning", "statusbar_visible", "debug", "permissions", "extra_tools")

# The subset the TUI keeps live in ``tui_state`` for the current process.
_VIEW_PREF_KEYS = ("show_reasoning", "statusbar_visible", "debug")

# Keys whose value must be a plain bool / a known permission tier.
_BOOL_PREF_KEYS = ("show_reasoning", "statusbar_visible", "debug")

_DEFAULTS: Dict[str, Any] = {
    "show_reasoning": True,
    "statusbar_visible": True,
    "debug": False,
}


def TOOL_REGISTRY_NAMES() -> tuple:
    """Names accepted by ``--tools`` / ``/tools add`` (lazy: the registry
    imports every provider module, which must not happen at ``prefs`` import
    time)."""
    from agentica.cli.runtime import TOOL_REGISTRY

    return tuple(TOOL_REGISTRY)


def normalize_cli_prefs(raw: Any) -> Dict[str, Any]:
    """Keep only known preference keys with a valid value.

    A hand-edited ``project.json`` or a sidecar written by an older version is
    an external input: a junk value (``"yes"``, ``"strict"``, a nested dict)
    is dropped rather than carried into ``agent_config``, where it would
    surface as a permission mode no tier check accepts.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in CLI_PREF_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if key in _BOOL_PREF_KEYS and isinstance(value, bool):
            out[key] = value
        elif key == "permissions" and isinstance(value, str) and value in PERMISSION_MODES:
            out[key] = value
        elif key == "extra_tools":
            # ``None`` is the "forget the saved set" marker and must survive a
            # second pass, because every writer normalizes before persisting.
            if value is None:
                out[key] = None
            elif isinstance(value, list):
                # Registry names only: `/tools add-from` runs a user .py module
                # at load time, so its name must never be replayed at startup
                # from a file. `add` already refuses names outside
                # TOOL_REGISTRY, and this second filter keeps a hand-edited
                # project.json from smuggling an arbitrary module path back in.
                names = [n for n in value if isinstance(n, str) and n in TOOL_REGISTRY_NAMES()]
                if value and not names:
                    # Every name was unknown — a hand-edit, or a tool this
                    # version no longer ships. Drop the key rather than reading
                    # it as "clear everything", so a downgrade cannot silently
                    # forget the user's tools.
                    continue
                # ``[]`` is a decision ("no saved tools") and is carried as
                # ``None``, which both stores treat as "remove the key".
                # Without this, `/tools remove` of the last tool would leave
                # the name on disk and the next launch would load it back.
                out[key] = sorted(set(names)) or None
    return out


# ---------------------------------------------------------------------------
# project.json — what a new CLI in this work_dir starts with
# ---------------------------------------------------------------------------


def read_project_prefs(work_dir: Optional[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    """Read this work_dir's saved CLI preferences (``{}`` when never set)."""
    if not work_dir:
        return {}
    from agentica.project_store import get_project_cli_prefs, project_base_dir

    return normalize_cli_prefs(get_project_cli_prefs(project_base_dir(work_dir, user_id)))


def write_project_prefs(
    work_dir: Optional[str],
    updates: Dict[str, Any],
    user_id: Optional[str] = None,
) -> None:
    """Merge ``updates`` into this work_dir's ``project.json`` ``cli`` block.

    Never raises: a preference is a convenience, and a read-only or
    concurrently-written ``project.json`` must not turn ``/reasoning off``
    into a traceback.
    """
    updates = normalize_cli_prefs(updates)
    if not work_dir or not updates:
        return
    from agentica.project_store import (
        ensure_project_work_dir,
        project_base_dir,
        update_project_cli_prefs,
    )

    base = project_base_dir(work_dir, user_id)
    try:
        ensure_project_work_dir(base, work_dir)
        update_project_cli_prefs(base, updates)
    except OSError as exc:
        logger.debug(f"Could not persist CLI preferences in {base}: {exc}")


# ---------------------------------------------------------------------------
# Startup merge
# ---------------------------------------------------------------------------


def apply_cli_prefs(
    agent_config: Dict[str, Any],
    prefs: Dict[str, Any],
    *,
    replace: bool = False,
) -> None:
    """Merge saved preferences onto ``agent_config`` (in place).

    The merged view is kept on ``agent_config["_cli_prefs"]``, which is what
    the TUI seeds from and what survives an agent rebuild; only the settings
    with a startup flag are written through to the fields the rest of the CLI
    reads (``debug``, ``permissions``), and only when that flag was not given.
    ``show_reasoning`` / ``statusbar_visible`` have no field of their own —
    they live in ``tui_state`` (see :func:`sync_view_prefs_to_tui`).

    ``replace=True`` is for adopting a *different* source of truth — switching
    to a resumed session's sidecar. Merging there would keep whatever the
    previously selected session had set for the keys the new one never
    touched, which is a stale view, not a preference.
    """
    prefs = normalize_cli_prefs(prefs)
    if not prefs and not replace:
        return
    merged = {} if replace else dict(agent_config.get("_cli_prefs") or {})
    merged.update(prefs)
    # Remember what the project layer said, so a later session switch has a
    # fallback for the keys its own sidecar does not name.
    if not replace:
        agent_config["_project_cli_prefs"] = dict(merged)
    if merged:
        agent_config["_cli_prefs"] = merged
    else:
        agent_config.pop("_cli_prefs", None)
    if "debug" in prefs and not agent_config.get("_debug_explicit"):
        agent_config["debug"] = bool(prefs["debug"])
    if "permissions" in prefs and not agent_config.get("_permissions_explicit"):
        agent_config["permissions"] = prefs["permissions"]


def apply_session_cli_prefs(agent_config: Dict[str, Any], session_prefs: Dict[str, Any]) -> None:
    """Adopt a resumed session's preferences over this work_dir's.

    The session's sidecar beats ``project.json`` for the keys it names, and the
    project's value is still the fallback for the ones it does not — the same
    layering the model profile already uses. Building the result in one step
    (rather than layering onto the live ``_cli_prefs``) is what stops a second
    ``/resume`` from inheriting the first session's view settings.
    """
    project_layer = agent_config.get("_project_cli_prefs")
    merged = dict(project_layer) if isinstance(project_layer, dict) else {}
    merged.update(normalize_cli_prefs(session_prefs))
    apply_cli_prefs(agent_config, merged, replace=True)


def sync_view_prefs_to_tui(tui_state: Optional[Dict[str, Any]], agent_config: Dict[str, Any]) -> None:
    """Push the resolved view preferences into the live ``tui_state`` dict.

    Called with the empty dict at startup (which is why every key falls back to
    its default) and again after ``/resume``, because ``tui_state`` is created
    once and never rebuilt — the status bar and the reasoning stream read it
    directly.
    """
    if tui_state is None:
        return
    prefs = agent_config.get("_cli_prefs") or {}
    for key in _VIEW_PREF_KEYS:
        value = prefs.get(key, _DEFAULTS.get(key))
        if value is not None:
            tui_state[key] = bool(value)


def record_cli_prefs(
    agent_config: Dict[str, Any],
    agent: Any,
    updates: Dict[str, Any],
) -> None:
    """Persist a preference set by a slash command, in both scopes.

    The live value is written to ``agent_config["_cli_prefs"]`` first so that
    an agent rebuild later in this process keeps it even if the disk write
    fails.
    """
    updates = normalize_cli_prefs(updates)
    if not updates:
        return

    merged = dict(agent_config.get("_cli_prefs") or {})
    merged.update(updates)
    agent_config["_cli_prefs"] = merged

    user_id = agent_config.get("user_id")
    session_log = None
    if agent is not None:
        session_log = getattr(agent, "_session_log", None)
        if user_id is None:
            user_id = getattr(agent, "user_id", None)

    # Both come from objects a test (or a sibling caller) may have stubbed:
    # anything that is not the type the path builders accept would turn a
    # toggle into a TypeError, so it degrades to the process cwd / no user.
    if not isinstance(user_id, str):
        user_id = None
    work_dir = agent_config.get("work_dir") or getattr(agent, "work_dir", None)
    if not isinstance(work_dir, str) or not work_dir:
        work_dir = os.getcwd()
    write_project_prefs(work_dir, updates, user_id=user_id)

    if session_log is not None and hasattr(session_log, "set_cli_prefs"):
        try:
            session_log.set_cli_prefs(updates)
        except Exception as exc:  # noqa: BLE001 — sidecar write must not break the command
            logger.debug(f"Could not persist CLI preferences to session sidecar: {exc}")
