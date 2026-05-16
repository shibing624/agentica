# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Process-local frequency cap for boundary-triggered hooks.

`MemoryExtractHooks` and `ExperienceCaptureHooks` use an "idle gate" to
avoid firing the same expensive LLM call twice in a few seconds. The gate
is intentionally weak: it's a frequency cap, not a global mutex. Two
properties matter:

  1. It MUST be scoped by user_id. If user A just extracted and user B
     immediately follows, B's extraction cannot be suppressed by A's
     stamp — that would be cross-tenant data loss.

  2. It does NOT need to be persisted across processes or machines.
     "Strong" rate limits belong at the gateway / per-user quota layer,
     not in a JSON file racing with N async workers.

This module is therefore a tiny in-memory keyed-by-(user, slot) dict,
shared across all hook instances in the same process. No disk IO,
no cross-process race, no multi-tenant leak.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

_DEFAULT_USER = "default"

_stamps: Dict[Tuple[str, str], float] = {}
_lock = threading.Lock()


def _key(user_id: str | None, slot: str) -> Tuple[str, str]:
    return (user_id or _DEFAULT_USER, slot)


def should_skip(user_id: str | None, slot: str, min_seconds_between: int) -> bool:
    """Return True if (user_id, slot) ran less than ``min_seconds_between`` ago."""
    if min_seconds_between <= 0:
        return False
    with _lock:
        last = _stamps.get(_key(user_id, slot))
    if last is None:
        return False
    return (time.time() - last) < min_seconds_between


def stamp(user_id: str | None, slot: str) -> None:
    """Record that (user_id, slot) just ran."""
    with _lock:
        _stamps[_key(user_id, slot)] = time.time()


def reset() -> None:
    """Clear all stamps. Test helper; not meant for production paths."""
    with _lock:
        _stamps.clear()
