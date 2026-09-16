# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Configuration for the external hook egress.

Read once at install time, like the notify sink: an egress is either wired or it
is not, and a mid-run config flip would leave half a channel behind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.global_config import get_setting
from agentica.utils.log import logger

#: Events on this wire. ``needs.approval`` and ``needs.input`` take the reply
#: path; every other value is a fire-and-forget notice.
SHELL_HOOK_EVENTS = (
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "tool.started",
    "tool.completed",
    "session.started",
    "session.ended",
    "needs.approval",
    "needs.input",
    "needs.resolved",
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


def _parse_command(raw: Any, *, location: str) -> List[str]:
    """The command as an argv list.

    A list is the only accepted shape. A string is refused rather than split:
    splitting would invent quoting rules the user did not write, and the wire
    format is a JSON document, so a shell is never needed to pass it. A user who
    wants a shell writes ``["/bin/sh", "-c", "..."]`` explicitly.
    """
    if isinstance(raw, str):
        if raw.strip():
            logger.warning(
                f"shell hooks: {location}.command must be an argv list, not a "
                "string; ignoring it. Write [\"/abs/path/to/notifier\"] — a shell "
                "is not implied, and $HOME is not expanded here."
            )
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(part).strip() for part in raw if str(part).strip()]


@dataclass
class HookConsumer:
    """One named hook process and its event subscription."""

    name: str
    command: List[str] = field(default_factory=list)
    enabled: bool = True
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
        return bool(self.enabled and self.command and self.events.get(event, False))


@dataclass
class ShellHooksConfig:
    """Resolved multi-consumer hook egress configuration."""

    enabled: bool = False
    consumers: List[HookConsumer] = field(default_factory=list)

    @property
    def effective(self) -> bool:
        """Wired only when it is switched on *and* there is something to run."""
        return bool(self.enabled and any(c.enabled and c.command for c in self.consumers))

    def consumers_for(self, event: str) -> List[HookConsumer]:
        """Enabled consumers subscribed to ``event``, in config order."""
        if not self.enabled:
            return []
        return [consumer for consumer in self.consumers if consumer.event_enabled(event)]


def _parse_consumers(raw: Any, *, location: str) -> List[HookConsumer]:
    """Parse the only supported hook shape: a list of named consumers."""
    if not isinstance(raw, list):
        if raw is not None:
            logger.warning(f"shell hooks: {location} must be a list; ignoring it.")
        return []
    consumers: List[HookConsumer] = []
    names = set()
    for index, item in enumerate(raw):
        item_location = f"{location}[{index}]"
        if not isinstance(item, dict):
            logger.warning(f"shell hooks: {item_location} must be an object; ignoring it.")
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            logger.warning(f"shell hooks: {item_location}.name is required; ignoring it.")
            continue
        if name in names:
            logger.warning(f"shell hooks: duplicate consumer name {name!r}; ignoring it.")
            continue
        names.add(name)
        events = item.get("events")
        consumers.append(
            HookConsumer(
                name=name,
                command=_parse_command(item.get("command"), location=item_location),
                enabled=bool(item.get("enabled", True)),
                events=events if isinstance(events, dict) else {},
            )
        )
    return consumers


def load_shell_hooks_config(config: Optional[Dict[str, Any]] = None) -> ShellHooksConfig:
    """Resolve ``settings.hooks`` plus env overrides. Env beats config.yaml."""
    try:
        block = get_setting("hooks", {}, config=config)
    except Exception as exc:  # a broken config must not break startup
        logger.debug(f"shell hooks: could not read settings.hooks: {exc}")
        block = {}
    if not isinstance(block, dict):
        block = {}

    cfg = ShellHooksConfig(
        enabled=bool(block.get("enabled", False)),
        consumers=_parse_consumers(
            block.get("consumers"), location="settings.hooks.consumers"
        ),
    )

    env_enabled = _env_bool("ENABLED")
    if env_enabled is not None:
        cfg.enabled = env_enabled
    raw_consumers = _env("CONSUMERS")
    if raw_consumers is not None:
        try:
            parsed_consumers = json.loads(raw_consumers)
        except ValueError:
            logger.warning(
                "shell hooks: AGENTICA_HOOKS_CONSUMERS must be a JSON array; "
                "ignoring configured consumers."
            )
            parsed_consumers = None
        cfg.consumers = _parse_consumers(
            parsed_consumers, location="AGENTICA_HOOKS_CONSUMERS"
        )

    return cfg
