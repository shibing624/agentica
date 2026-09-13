# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The hook egress — fire the user's command at lifecycle points.

Fire-and-forget for the four ``run.*`` events: spawn, write the document, and
return. The blocking ``needs.*`` path is ``requests.py``, because it needs a
reply and a race rather than a notification.

Installed once per process, like the sink, so a mid-run config flip cannot leave
a half-wired channel behind. ``enabled: false`` — or an enabled block with no
command — wires nothing at all: no thread, no process.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentica.shell_hooks.config import ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload
from agentica.utils.log import logger

#: Install-time decision: None means "this process runs no hook command".
_shell_hooks: Optional[ShellHooksConfig] = None


def install_hook_egress(
    config: Optional[ShellHooksConfig] = None,
) -> Optional[ShellHooksConfig]:
    """Wire the egress, or return None when there is nothing to run."""
    global _shell_hooks
    cfg = config if config is not None else load_shell_hooks_config()
    _shell_hooks = cfg if cfg.effective else None
    return _shell_hooks


def get_hook_egress() -> Optional[ShellHooksConfig]:
    """The installed config, or None when this process runs no hook command."""
    return _shell_hooks


def reset_hook_egress_for_tests() -> None:
    """Drop the installed egress so a test starts from a clean process state."""
    global _shell_hooks
    _shell_hooks = None


def hook_egress_dispatch(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
) -> None:
    """Send one lifecycle event to the user's command. Never raises, never blocks.

    Called alongside the notify sink rather than instead of it: a broken consumer
    and a broken sink must not be able to take each other down, and observation
    must never break a run.
    """
    cfg = _shell_hooks
    if cfg is None or not cfg.event_enabled(event):
        return
    try:
        body = dict(payload or {})
        doc = build_payload(
            event,
            session_id=session_id,
            work_dir=work_dir,
            run_id=_run_id(agent),
            prompt=body.get("prompt"),
            extra={k: v for k, v in body.items() if k != "prompt"},
        )
        HookProcess(cfg.command, doc).start()
    except Exception as exc:
        logger.debug(f"shell hooks: could not send {event}: {exc}")


def _run_id(agent: Any) -> Optional[str]:
    try:
        return getattr(getattr(agent, "run_context", None), "run_id", None)
    except Exception:
        return None
