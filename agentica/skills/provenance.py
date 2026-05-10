# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Append-only provenance log for generated Agentica skills.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROVENANCE_FILENAME = "provenance.jsonl"


def get_provenance_path(skill_dir: Path) -> Path:
    """Return the append-only provenance log path for one generated skill."""
    return Path(skill_dir) / PROVENANCE_FILENAME


def append_provenance_event(skill_dir: Path, event: Dict[str, Any]) -> Path:
    """Append one complete provenance event as a JSONL record.

    JSONL keeps each event independently parseable. Opening with ``O_APPEND``
    makes each single-line write append at the end of the file, which is safer
    for concurrent workers than rewriting one YAML/JSON document.
    """
    path = get_provenance_path(skill_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def read_provenance_events(skill_dir: Path) -> List[Dict[str, Any]]:
    """Read parseable provenance events from a generated skill directory."""
    path = get_provenance_path(skill_dir)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


__all__ = [
    "PROVENANCE_FILENAME",
    "get_provenance_path",
    "append_provenance_event",
    "read_provenance_events",
]
