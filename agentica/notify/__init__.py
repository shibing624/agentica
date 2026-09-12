# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External notify sink — tell a local app what a run is doing, and
let the user answer from there instead of the terminal.

Talks to a desktop app over a local Unix socket. The app is an input surface, not
an authority: a ``y`` pressed there counts as the user's own answer for that
session and interaction, with the same effect as typing it. The app never
decides anything by itself and never auto-approves. See ``sink.py`` for the
contract; see ``docs/getting-started/notify-sink.md`` for the user-facing setup.
"""

from agentica.notify.config import NotifyConfig, load_notify_config
from agentica.notify.sink import (
    NotifySink,
    get_sink,
    goal_finished,
    install_sink,
    notify_sink_dispatch,
    reset_sink_for_tests,
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
    "set_idle_provider",
]
