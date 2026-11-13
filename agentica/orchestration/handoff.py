# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deprecated alias for ``agentica.subagents.handoff``.

The module moved because its only real consumer is ``SubagentRegistry.spawn``:
keeping it under ``orchestration`` made a leaf capability import its own
composition layer. This shim keeps the old import path working for a deprecation
window (same convention as ``agentica.gateway``); callers should import from
``agentica.subagents.handoff``. Will be removed in agentica 1.7.
"""
from __future__ import annotations

from agentica.subagents.handoff import HandoffContext, default_handoff_mapper

__all__ = ["HandoffContext", "default_handoff_mapper"]
