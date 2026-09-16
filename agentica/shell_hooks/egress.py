# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The hook egress — fire subscribed consumer commands at lifecycle points.

Notices are fire-and-forget: spawn every subscriber, write the document, and
return. The blocking ``needs.*`` path is ``requests.py`` because it needs a
first-valid-reply race.

Installed once per process, like the sink, so a mid-run config flip cannot leave
a half-wired channel behind. ``enabled: false`` — or an enabled block with no
command — wires nothing at all: no thread, no process.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any, Dict, Optional, Set

from agentica.shell_hooks.config import ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload
from agentica.utils.log import logger

#: Install-time decision: None means "this process runs no hook consumers".
_shell_hooks: Optional[ShellHooksConfig] = None
_installed = False
_install_lock = threading.Lock()
NOTICE_PROCESS_TIMEOUT_SECONDS = 30.0
_live_notices: Set[HookProcess] = set()
_live_lock = threading.Lock()
_atexit_registered = False


def install_hook_egress(
    config: Optional[ShellHooksConfig] = None,
) -> Optional[ShellHooksConfig]:
    """Wire the egress, or return None when there is nothing to run."""
    global _installed, _shell_hooks
    cfg = config if config is not None else load_shell_hooks_config()
    with _install_lock:
        _shell_hooks = cfg if cfg.effective else None
        _installed = True
        return _shell_hooks


def ensure_hook_egress_installed() -> Optional[ShellHooksConfig]:
    """Load process configuration once, including outside the interactive CLI."""
    global _installed, _shell_hooks
    if _installed:
        return _shell_hooks
    with _install_lock:
        if not _installed:
            cfg = load_shell_hooks_config()
            _shell_hooks = cfg if cfg.effective else None
            _installed = True
        return _shell_hooks


def get_hook_egress() -> Optional[ShellHooksConfig]:
    """The installed config, or None when this process runs no hook consumers."""
    return _shell_hooks


def reset_hook_egress_for_tests() -> None:
    """Drop the installed egress so a test starts from a clean process state."""
    global _installed, _shell_hooks
    _shutdown_notice_processes()
    with _install_lock:
        _shell_hooks = None
        _installed = False


def hook_egress_dispatch(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    agent: Any = None,
) -> None:
    """Send one notice to every subscribed consumer. Never raises, never blocks.

    Called alongside the notify sink rather than instead of it: a broken consumer
    and a broken sink must not be able to take each other down, and observation
    must never break a run.
    """
    cfg = ensure_hook_egress_installed()
    if cfg is None:
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
        for consumer in cfg.consumers_for(event):
            process = HookProcess(consumer.command, doc)
            if process.start():
                with _live_lock:
                    _live_notices.add(process)
                    _register_shutdown()
                threading.Thread(
                    target=_finish_notice_process,
                    args=(process,),
                    name="agentica-hook-notice-cleanup",
                    daemon=True,
                ).start()
    except Exception as exc:
        logger.debug(f"shell hooks: could not send {event}: {exc}")


def _run_id(agent: Any) -> Optional[str]:
    if agent is None or agent.run_context is None:
        return None
    return agent.run_context.run_id


def _register_shutdown() -> None:
    """Register once. Caller holds ``_live_lock``."""
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_shutdown_notice_processes)
    _atexit_registered = True


def _shutdown_notice_processes() -> None:
    """Kill outstanding notice groups. Daemon cleanup does not survive exit."""
    with _live_lock:
        processes = list(_live_notices)
        _live_notices.clear()
    for process in processes:
        process.kill()


def _finish_notice_process(process: HookProcess) -> None:
    """Bound and reap a fire-and-forget consumer process group."""
    try:
        process.wait(timeout=NOTICE_PROCESS_TIMEOUT_SECONDS)
        process.kill()
    finally:
        with _live_lock:
            _live_notices.discard(process)
