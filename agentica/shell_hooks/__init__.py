# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External shell hooks — run named consumer commands at lifecycle
points and hand each one JSON on stdin.

This is the *executable* kind of hook. It is not ``agentica/hooks.py``
(``AgentHooks`` / ``RunHooks``), which are in-process Python observers and stay
as they are. See ``docs/rfcs/external-hook-egress.md``.
"""

from agentica.shell_hooks.config import (
    HookConsumer,
    ShellHooksConfig,
    load_shell_hooks_config,
)
from agentica.shell_hooks.egress import (
    ensure_hook_egress_installed,
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)
from agentica.shell_hooks.events import (
    emit_request_resolved,
    emit_session_ended,
    emit_session_started,
)
from agentica.shell_hooks.requests import (
    HookRequest,
    approval_payload,
    question_payload,
    start_hook_request,
)

__all__ = [
    "HookRequest",
    "HookConsumer",
    "ShellHooksConfig",
    "approval_payload",
    "get_hook_egress",
    "ensure_hook_egress_installed",
    "emit_request_resolved",
    "emit_session_ended",
    "emit_session_started",
    "hook_egress_dispatch",
    "install_hook_egress",
    "load_shell_hooks_config",
    "question_payload",
    "reset_hook_egress_for_tests",
    "start_hook_request",
]
