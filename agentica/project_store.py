# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical project storage paths under ~/.agentica/projects/

Owns the ``<projects>/<user>/<sanitize(work_dir)>/`` layout used by sessions,
tool-result spill, and per-project metadata. One ``project.json`` in that
directory holds directory-level fields (``work_dir``, ``active_profile``,
``approvals``). Session-level sidecars (``<id>.meta.json``) stay next to
their ``.jsonl``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agentica.utils.log import logger

PROJECT_FILE = "project.json"


def _iso_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _projects_dir() -> str:
    """Live ``AGENTICA_PROJECTS_DIR`` (or ``$AGENTICA_HOME/projects``)."""
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    return os.getenv("AGENTICA_PROJECTS_DIR", os.path.join(home, "projects"))


def project_file_path(base_dir: Any) -> Path:
    return Path(base_dir) / PROJECT_FILE


def projects_root(user_id: Optional[str] = None) -> str:
    """Return ``~/.agentica/projects/<user>/`` — parent of every project dir."""
    from agentica.compression.tool_result_storage import safe_user_segment

    return os.path.join(_projects_dir(), safe_user_segment(user_id))


def project_base_dir(work_dir: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Return ``~/.agentica/projects/<user>/<sanitize(work_dir)>/``.

    Slug hashing uses the work_dir string as given (no ``realpath``), matching
    historical SessionLog / tool-result paths so ``project.json``, sessions,
    and spill files share one directory.

    Re-reads ``AGENTICA_HOME`` / ``AGENTICA_PROJECTS_DIR`` each call so tests
    can isolate via env without relying on the frozen
    ``agentica.config.AGENTICA_PROJECTS_DIR`` import-time constant.
    """
    from agentica.compression.tool_result_storage import sanitize_path

    return os.path.join(
        projects_root(user_id),
        sanitize_path(work_dir or os.getcwd()),
    )


def read_project_file(base_dir: Any) -> Dict[str, Any]:
    path = project_file_path(base_dir)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Could not read {path}: {e}")
        return {}


def write_project_file(base_dir: Any, data: Dict[str, Any]) -> None:
    """Atomically write ``project.json`` (mode 0o600)."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = project_file_path(base)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def ensure_project_work_dir(base_dir: Any, work_dir: str) -> None:
    """Record ``work_dir`` once; keep any existing ``active_profile``."""
    data = read_project_file(base_dir)
    recorded = data.get("work_dir")
    if isinstance(recorded, str) and recorded:
        return
    # Store the same string SessionLog / callers use as the project key
    # (typically an absolute path). expanduser only for readability when the
    # caller passed ``~/...``.
    data["work_dir"] = os.path.expanduser(work_dir) if work_dir.startswith("~") else work_dir
    data.setdefault("created_at", _iso_now())
    try:
        write_project_file(base_dir, data)
    except OSError as e:
        logger.debug(f"Could not write {project_file_path(base_dir)}: {e}")


def get_project_active_profile(base_dir: Any) -> Optional[str]:
    name = read_project_file(base_dir).get("active_profile")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None


def set_project_active_profile(base_dir: Any, name: str) -> None:
    data = read_project_file(base_dir)
    data["active_profile"] = name.strip()
    write_project_file(base_dir, data)


def clear_project_active_profile(base_dir: Any) -> bool:
    data = read_project_file(base_dir)
    if "active_profile" not in data:
        return False
    del data["active_profile"]
    write_project_file(base_dir, data)
    return True
