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

__all__ = ["ShellHooksConfig", "load_shell_hooks_config"]
