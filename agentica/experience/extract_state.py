# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cross-process state + frequency caps for boundary-triggered hooks.

Several hooks (memory extraction, correction judging, future curators) used
to fire an LLM call after every single turn. That blew up token cost and
made the user wait between prompts. We now batch these into boundary
triggers — every N turns or `on_pre_compact` — and additionally gate them
with a minimum-seconds-between-runs cap that persists across processes
(opening the CLI three times in a minute won't re-extract three times).

State lives at ``~/.agentica/extract_state.json``:

    {
      "memory_extract": {"last_at": "2026-05-17T02:45:04"},
      "correction_judge": {"last_at": "2026-05-17T02:44:50"}
    }

Single file, single source of truth. No per-session keys — the gate is
intentionally global so concurrent agents in the same user dir don't all
race to extract at the same time.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentica.config import AGENTICA_HOME
from agentica.utils.log import logger


def _state_path() -> Path:
    return Path(AGENTICA_HOME).expanduser() / "extract_state.json"


def _load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("extract state unreadable, resetting: %s", exc)
        return {}


def _save_state(state: dict) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("failed to persist extract state: %s", exc)


def _parse_iso(value) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def should_skip(slot: str, min_seconds_between: int) -> bool:
    """Return True if ``slot``'s last run was less than ``min_seconds_between`` ago.

    ``slot`` is a free-form string (e.g. ``"memory_extract"``,
    ``"correction_judge"``) — each consumer picks its own. A zero or negative
    cap disables the gate entirely.
    """
    if min_seconds_between <= 0:
        return False
    state = _load_state()
    entry = state.get(slot) or {}
    last_at = _parse_iso(entry.get("last_at"))
    if last_at is None:
        return False
    elapsed = datetime.now().timestamp() - last_at
    return elapsed < min_seconds_between


def stamp(slot: str) -> None:
    """Record that ``slot`` just ran. Best-effort; never raises."""
    state = _load_state()
    state[slot] = {"last_at": datetime.now().isoformat(timespec="seconds")}
    _save_state(state)
