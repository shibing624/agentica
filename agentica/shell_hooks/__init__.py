# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External shell hooks — run the user's own command at lifecycle
points and hand it JSON on stdin, the way every other coding CLI does.

This is the *executable* kind of hook. It is not ``agentica/hooks.py``
(``AgentHooks`` / ``RunHooks``), which are in-process Python observers and stay
as they are. See ``docs/rfcs/external-hook-egress.md``.
"""

from agentica.shell_hooks.config import ShellHooksConfig, load_shell_hooks_config
from agentica.shell_hooks.egress import (
    get_hook_egress,
    hook_egress_dispatch,
    install_hook_egress,
    reset_hook_egress_for_tests,
)
from agentica.shell_hooks.requests import (
    HookRequest,
    approval_payload,
    question_payload,
    start_hook_request,
)

__all__ = [
    "HookRequest",
    "ShellHooksConfig",
    "approval_payload",
    "get_hook_egress",
    "hook_egress_dispatch",
    "install_hook_egress",
    "load_shell_hooks_config",
    "question_payload",
    "reset_hook_egress_for_tests",
    "start_hook_request",
]
