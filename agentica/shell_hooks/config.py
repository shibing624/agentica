# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Configuration for the external hook egress.

Read once at install time, like the notify sink: an egress is either wired or it
is not, and a mid-run config flip would leave half a channel behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.global_config import get_setting
from agentica.utils.log import logger

#: The six events on this wire. Four are lifecycle notices; the two ``needs.*``
#: ones take the other path (a request with a reply) and do not go through the
#: fire-and-forget dispatch, but they are gated by the same ``events`` block so a
#: user has one place to switch things off.
SHELL_HOOK_EVENTS = (
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "needs.approval",
    "needs.input",
)


def _env(name: str) -> Optional[str]:
    """Read ``AGENTICA_HOOKS_<name>``, treating empty as unset."""
    value = os.getenv(f"AGENTICA_HOOKS_{name}")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str) -> Optional[bool]:
    value = _env(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on")


def _parse_command(raw: Any) -> List[str]:
    """The command as an argv list.

    A list is the only accepted shape. A string is refused rather than split:
    splitting would invent quoting rules the user did not write, and the wire
    format is a JSON document, so a shell is never needed to pass it. A user who
    wants a shell writes ``["/bin/sh", "-c", "..."]`` explicitly, which is also
    how they recover things we do not send (``$PPID``, the controlling tty).
    """
    if isinstance(raw, str):
        if raw.strip():
            logger.warning(
                "shell hooks: settings.hooks.command must be an argv list, not a "
                "string; ignoring it. Write [\"/abs/path/to/notifier\"] — a shell "
                "is not implied, and $HOME is not expanded here."
            )
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(part).strip() for part in raw if str(part).strip()]


@dataclass
class ShellHooksConfig:
    """Resolved hook egress configuration."""

    enabled: bool = False
    command: List[str] = field(default_factory=list)
    events: Dict[str, bool] = field(
        default_factory=lambda: {e: True for e in SHELL_HOOK_EVENTS}
    )

    def __post_init__(self) -> None:
        """Treat ``events`` as a partial override over the full default set.

        ``event_enabled`` reads an absent key as off, which is right for a
        resolved config. Applied to a hand-written ``{"run.started": False}``
        that rule would silently switch off the other five as well, so the
        mapping is completed against the defaults here, once.
        """
        merged = {e: True for e in SHELL_HOOK_EVENTS}
        merged.update(
            {k: bool(v) for k, v in (self.events or {}).items() if k in merged}
        )
        self.events = merged

    def event_enabled(self, event: str) -> bool:
        return bool(self.events.get(event, False))

    @property
    def effective(self) -> bool:
        """Wired only when it is switched on *and* there is something to run."""
        return bool(self.enabled and self.command)


def load_shell_hooks_config(config: Optional[Dict[str, Any]] = None) -> ShellHooksConfig:
    """Resolve ``settings.hooks`` plus env overrides. Env beats config.yaml."""
    try:
        block = get_setting("hooks", {}, config=config)
    except Exception as exc:  # a broken config must not break startup
        logger.debug(f"shell hooks: could not read settings.hooks: {exc}")
        block = {}
    if not isinstance(block, dict):
        block = {}

    cfg = ShellHooksConfig()
    if "enabled" in block:
        cfg.enabled = bool(block["enabled"])
    cfg.command = _parse_command(block.get("command"))
    events = block.get("events")
    if isinstance(events, dict):
        for name in SHELL_HOOK_EVENTS:
            if name in events:
                cfg.events[name] = bool(events[name])

    if _env_bool("ENABLED") is not None:
        cfg.enabled = bool(_env_bool("ENABLED"))
    command = _env("COMMAND")
    if command is not None:
        cfg.command = _parse_command(command.split())

    return cfg
