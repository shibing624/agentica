# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: External notify sink — tell a local app what a run is doing.

A one-way observation channel plus (optionally) a two-way decision channel,
talking to a desktop app over a local Unix socket. See ``sink.py`` for the
contract and the degradation ladder; see ``docs/getting-started/notify-sink.md``
for the user-facing setup.
"""

from agentica.notify.config import NotifyConfig, load_notify_config
from agentica.notify.sink import (
    NotifySink,
    get_sink,
    install_sink,
    notify_sink_dispatch,
    reset_sink_for_tests,
)

__all__ = [
    "NotifyConfig",
    "NotifySink",
    "get_sink",
    "install_sink",
    "load_notify_config",
    "notify_sink_dispatch",
    "reset_sink_for_tests",
]
