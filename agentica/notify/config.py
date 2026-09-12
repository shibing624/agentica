# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Configuration for the external notify sink.

Read once at install time (never per turn): a sink is either built or it is
not, and a config flip mid-run would leave a half-wired channel behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agentica.global_config import get_setting
from agentica.utils.log import logger

#: The four run lifecycle events, all non-blocking. ``needs.*`` is not in here:
#: those are two-way and gated by their own switches.
RUN_EVENTS = (
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelled",
)

DEFAULT_SOCKET = "~/Library/Application Support/VPet/notify.sock"
DEFAULT_TOKEN_FILE = "~/Library/Application Support/VPet/notify.token"
DEFAULT_TIMEOUT_SECONDS = 55

#: Non-blocking delivery is a fire-and-forget observation: it must never hold a
#: run up, so its timeout is a couple of seconds even though a decision may wait
#: a minute.
DELIVERY_TIMEOUT_SECONDS = 2.0

#: Bounded on purpose. A wedged desktop app must not let the queue grow until
#: agentica eats memory; the oldest event is dropped instead.
QUEUE_MAXSIZE = 256


def _env(name: str) -> Optional[str]:
    """Read ``AGENTICA_NOTIFY_<name>``, treating empty as unset."""
    value = os.getenv(f"AGENTICA_NOTIFY_{name}")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str) -> Optional[bool]:
    value = _env(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class NotifyConfig:
    """Resolved sink configuration. See the docs for each field's meaning."""

    enabled: bool = False
    socket: str = DEFAULT_SOCKET
    token: str = ""
    token_file: str = DEFAULT_TOKEN_FILE
    approve_from_desktop: bool = False
    events: Dict[str, bool] = field(default_factory=lambda: {e: True for e in RUN_EVENTS})
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def event_enabled(self, event: str) -> bool:
        return bool(self.events.get(event, False))

    @property
    def resolved_socket(self) -> str:
        return os.path.expanduser(self.socket)

    @property
    def resolved_token_file(self) -> str:
        return os.path.expanduser(self.token_file)

    def resolved_token(self) -> str:
        """Inline token, else the token file's contents, else empty.

        Read lazily so a token file written by the desktop app after agentica
        started still works on the next request.
        """
        if self.token:
            return self.token
        path = self.resolved_token_file
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""


def load_notify_config(config: Optional[Dict[str, Any]] = None) -> NotifyConfig:
    """Resolve the sink config from ``settings.notify`` plus env overrides.

    Environment beats config.yaml, matching the rest of the project.
    """
    try:
        block = get_setting("notify", {}, config=config)
    except Exception as exc:  # a broken config must not break startup
        logger.debug(f"notify sink: could not read settings.notify: {exc}")
        block = {}
    if not isinstance(block, dict):
        block = {}

    cfg = NotifyConfig()
    if "enabled" in block:
        cfg.enabled = bool(block["enabled"])
    cfg.socket = str(block.get("socket") or cfg.socket)
    cfg.token = str(block.get("token") or "")
    if block.get("token_file"):
        cfg.token_file = str(block["token_file"])
    cfg.approve_from_desktop = bool(block.get("approve_from_desktop", False))
    timeout = block.get("timeout_seconds")
    if isinstance(timeout, (int, float)) and timeout > 0:
        cfg.timeout_seconds = float(timeout)

    events = block.get("events")
    if isinstance(events, dict):
        # Per-event opt-out; an unlisted event keeps its default (on).
        for name in RUN_EVENTS:
            if name in events:
                cfg.events[name] = bool(events[name])

    if _env_bool("ENABLED") is not None:
        cfg.enabled = bool(_env_bool("ENABLED"))
    if _env_bool("APPROVE_FROM_DESKTOP") is not None:
        cfg.approve_from_desktop = bool(_env_bool("APPROVE_FROM_DESKTOP"))
    for key, attr in (("SOCKET", "socket"), ("TOKEN", "token"), ("TOKEN_FILE", "token_file")):
        value = _env(key)
        if value is not None:
            setattr(cfg, attr, value)
    timeout_env = _env("TIMEOUT_SECONDS")
    if timeout_env:
        try:
            parsed = float(timeout_env)
            if parsed > 0:
                cfg.timeout_seconds = parsed
        except ValueError:
            logger.debug(f"notify sink: ignoring non-numeric timeout_seconds={timeout_env!r}")

    return cfg
