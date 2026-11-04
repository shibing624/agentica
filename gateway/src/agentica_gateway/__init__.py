# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Agentica Gateway - FastAPI server + Web UI + IM bots + cron scheduler.

Ships as its own distribution, ``agentica-gateway``, released in lockstep with
``agentica`` (same version, exact pin)::

    pip install agentica-gateway              # server + Web UI + cron
    pip install "agentica-gateway[feishu]"    # + Feishu/Lark bot
    pip install "agentica-gateway[telegram]"  # + Telegram bot (etc.)

``pip install "agentica[gateway]"`` still works and pulls the same thing.
"""
try:
    import fastapi  # noqa: F401
except ImportError as e:
    raise ImportError(
        "agentica_gateway needs its dependencies. Install with:\n\n"
        "    pip install agentica-gateway              # FastAPI + Web UI + cron\n"
        "    pip install \"agentica-gateway[feishu]\"    # + Feishu/Lark bot\n"
    ) from e

from agentica.version import __version__
