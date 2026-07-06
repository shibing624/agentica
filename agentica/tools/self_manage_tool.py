# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Self-management tool — lets the agent inspect and modify its own
runtime configuration and upgrade itself.

Single compressed action-oriented tool (mirrors cron_tool design) to avoid
schema/context bloat. The agent calls ``self_manage(action=...)`` to read config,
edit a config.yaml profile field, set/delete a .env variable, check for a new
release, or self-upgrade via pip.

Secrets are masked on read. Edits to config.yaml preserve comments; .env edits
preserve other lines and take effect in-process immediately.
"""
from typing import Optional

from agentica.tools.base import Tool
from agentica.tools.decorators import tool
from agentica.tools.helpers import tool_result, tool_error
from agentica.cli import self_manage as sm


# Whether the agent can literally restart itself to load a new version
# depends on the product surface. The terminal CLI is a foreground process
# the user is driving, so exiting and rerunning it is something *they* can
# do themselves. The gateway (web chat + Feishu/WeCom/WeChat/Telegram/
# Discord/DingTalk/QQ bot channels) is a long-running server process — the
# chat user has no way to restart it, only whoever manages the deployment
# does. Default to the surface-agnostic phrasing; the CLI opts into the more
# specific one via `SelfManageTool(restart_hint=CLI_RESTART_HINT)`.
DEFAULT_RESTART_HINT = (
    "Restarting is required to load a new version. This must be done by "
    "whoever manages this deployment (restarting the server process) — you "
    "cannot do this yourself from the conversation, and neither can the user "
    "if they're just chatting with you."
)
CLI_RESTART_HINT = "Restart the CLI to load the new version (exit and rerun `agentica`)."


def _build_description(restart_hint: str) -> str:
    return f"""Inspect and modify agentica's own runtime configuration, or upgrade itself.

Actions:
- action='show'                 -> return current config.yaml profiles + settings (cron scheduler
                                   enabled/interval, etc.) + .env vars (secrets masked)
- action='set_config'           -> edit a config.yaml profile field.
                                   Requires key + value. Optional profile (defaults to active).
                                   Editable keys: model_provider, model_name, base_url, api_key,
                                   max_tokens, temperature, reasoning_effort, top_p, context_window.
- action='set_env'              -> set a .env variable. Requires key + value.
                                   Pass value='-' to delete the variable.
- action='check_upgrade'        -> report current vs latest PyPI version (no install).
- action='upgrade'              -> pip install -U agentica (requires confirm=True). {restart_hint}
- action='install_skill'        -> install a skill from a git URL or local path. Requires value=<source>.
                                   Optional force=confirm-style via confirm=True to overwrite existing.

Use this to optimize your own setup, e.g. raise max_tokens, switch model, or add an API key.
Config file edits persist; model changes take effect on next agent rebuild/restart."""


_SELF_MANAGE_DESCRIPTION = _build_description(DEFAULT_RESTART_HINT)


def _do_self_manage(
    action: str,
    key: Optional[str] = None,
    value: Optional[str] = None,
    profile: Optional[str] = None,
    confirm: bool = False,
    restart_hint: str = DEFAULT_RESTART_HINT,
) -> str:
    """Pure business implementation of the self_manage tool.

    This is the single source of truth for every action. Both the module-level
    ``@tool``-decorated entry point and ``SelfManageTool.self_manage`` delegate
    here, so neither of them depends on the decorator's ``__call__`` transparency
    (which was a fragile implicit contract).
    """
    action = (action or "").strip().lower()

    if action == "show":
        return tool_result(
            config=sm.read_config_summary(),
            env=sm.read_dotenv(),
            config_path=sm.config_file_path(),
            env_path=sm.dotenv_path(),
            current_version=sm.get_current_version(),
        )

    if action == "set_config":
        if not key or value is None:
            return tool_error("set_config requires 'key' and 'value'.")
        try:
            updated = sm.set_profile_field(key, value, profile)
        except ValueError as e:
            return tool_error(str(e))
        return tool_result(
            success=True,
            profile=profile or "active",
            updated_field=key,
            new_value=sm.mask_secret(key, value),
            note="Saved to config.yaml. Takes effect on next agent rebuild/restart.",
            profile_snapshot=updated,
        )

    if action == "set_env":
        if not key:
            return tool_error("set_env requires 'key'.")
        try:
            if value == "-" or value is None:
                sm.set_dotenv_var(key, None)
                return tool_result(success=True, deleted=key, note="Removed from .env and process env.")
            sm.set_dotenv_var(key, value)
        except ValueError as e:
            return tool_error(str(e))
        return tool_result(
            success=True,
            key=key,
            value=sm.mask_secret(key, value),
            note="Saved to .env and applied to current process.",
        )

    if action == "install_skill":
        source = value or key
        if not source:
            return tool_error("install_skill requires 'value' = git URL or local path.")
        try:
            from agentica.skills import install_skills
            installed = install_skills(source, force=confirm)
        except Exception as e:
            return tool_error(f"Skill install failed: {e}")
        return tool_result(
            success=True,
            installed=[s.name for s in installed],
            note="Skill(s) installed at user level. Use list_skills to see them.",
        )

    if action == "check_upgrade":
        current = sm.get_current_version()
        latest = sm.get_latest_version()
        if latest is None:
            return tool_error("Could not reach PyPI to check latest version.",
                              current_version=current)
        return tool_result(
            current_version=current,
            latest_version=latest,
            upgrade_available=sm.is_upgrade_available(current, latest),
        )

    if action == "upgrade":
        current = sm.get_current_version()
        latest = sm.get_latest_version()
        if not sm.is_upgrade_available(current, latest):
            return tool_result(success=True, current_version=current,
                               note="Already up to date.")
        if not confirm:
            return tool_error(
                "Upgrade requires confirm=True (this installs a new version and "
                "needs a CLI restart afterwards).",
                current_version=current, latest_version=latest,
            )
        code, output = sm.run_pip_upgrade("agentica")
        return tool_result(
            success=(code == 0),
            exit_code=code,
            output=output.strip()[-2000:],  # surface real pip signal, capped
            note=restart_hint if code == 0 else "pip failed; see output.",
        )

    return tool_error(
        f"Unknown action '{action}'. Valid: show, set_config, set_env, "
        f"check_upgrade, upgrade, install_skill."
    )


@tool(name="self_manage", description=_SELF_MANAGE_DESCRIPTION, is_destructive=True)
def self_manage(
    action: str,
    key: Optional[str] = None,
    value: Optional[str] = None,
    profile: Optional[str] = None,
    confirm: bool = False,
) -> str:
    """Self-management tool handler. See decorator description for actions."""
    return _do_self_manage(action, key, value, profile, confirm)


class SelfManageTool(Tool):
    """Self-management tool for Agent integration.

    Usage:
        agent = Agent(tools=[SelfManageTool()])

    Args:
        restart_hint: surface-specific phrasing for how a new version is
            loaded after ``action='upgrade'``, baked into both the tool
            description and the upgrade result's ``note``. Defaults to
            ``DEFAULT_RESTART_HINT`` (safe for any non-CLI surface); the CLI
            passes ``CLI_RESTART_HINT`` since the user can restart it themselves.
    """

    def __init__(self, restart_hint: Optional[str] = None):
        self._restart_hint = restart_hint or DEFAULT_RESTART_HINT
        super().__init__(name="self_manage", description=_build_description(self._restart_hint))
        self.register(self.self_manage, is_destructive=True)

    # No docstring on the method: Tool.register() falls back to self.description
    # (set in __init__ from _build_description()) when function.__doc__ is None,
    # so the LLM-facing schema stays in sync with the surface-specific restart_hint
    # without duplicating the description text.
    def self_manage(
        self,
        action: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        profile: Optional[str] = None,
        confirm: bool = False,
    ) -> str:
        return _do_self_manage(action, key, value, profile, confirm, restart_hint=self._restart_hint)

    def __repr__(self) -> str:
        return "SelfManageTool()"
