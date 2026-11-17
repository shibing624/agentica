# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: A session's transcript and the facts read from it.

``log`` is the JSONL + sidecar store; ``rows`` says what one row means;
``trace`` / ``search`` / ``index`` are read models over the log; ``catalog`` is
the one session list every surface (Web, IM, CLI) shows. Cross-session memory
(MEMORY.md, recall, intents) lives in ``agentica.memory``.
"""
from agentica.session.log import SessionLog

__all__ = ["SessionLog"]
