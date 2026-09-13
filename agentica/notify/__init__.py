# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External notify sink — tell a local app what a run is doing.

Talks to a desktop app over a local Unix socket, in one direction. **Observe
only**: the app is shown what a run is doing and is not asked anything back.
Replies — an approval, a question — travel through the user's own hook command
(``agentica/shell_hooks``), which is the one channel that can answer for the
person at the terminal. See ``sink.py`` for the contract; see
``docs/getting-started/notify-sink.md`` for the user-facing setup.
"""

from agentica.notify.config import NotifyConfig, load_notify_config
from agentica.notify.sink import (
    NotifySink,
    get_sink,
    goal_finished,
    install_sink,
    notify_sink_dispatch,
    reset_sink_for_tests,
    set_attach_endpoint,
    set_idle_provider,
)

__all__ = [
    "NotifyConfig",
    "NotifySink",
    "get_sink",
    "goal_finished",
    "install_sink",
    "load_notify_config",
    "notify_sink_dispatch",
    "reset_sink_for_tests",
    "set_attach_endpoint",
    "set_idle_provider",
]
