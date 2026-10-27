# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Point-in-time git state for the first turn of a session.

A new session starts blind to what is uncommitted, which branch it is on and
what was just committed, so "commit it" / "continue" begins with a round of
``git status`` calls. One snapshot, prefixed to the first user message (not
the system prompt, which must stay cache-stable), answers that up front. It
is labelled as a snapshot: the model re-checks before acting on it.
"""
import subprocess
from typing import List, Optional

from agentica.utils.log import logger

GIT_SNAPSHOT_OPEN = "<git_snapshot>"
GIT_SNAPSHOT_CLOSE = "</git_snapshot>"
STATUS_MAX_LINES = 20
LOG_LINES = 3
GIT_TIMEOUT_S = 2.0


def _git(args: List[str], cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        # External process boundary: no git, or a huge repo / network FS that
        # cannot answer in time. The snapshot is optional context.
        logger.debug(f"git snapshot skipped ({type(e).__name__}: {e})")
        return None
    return out.stdout.rstrip("\n") if out.returncode == 0 else None


def build_git_snapshot(work_dir: str) -> Optional[str]:
    """``<git_snapshot>`` for ``work_dir``, or None when it is not a git work tree."""
    if _git(["rev-parse", "--is-inside-work-tree"], work_dir) != "true":
        return None
    branch = _git(["branch", "--show-current"], work_dir) or "(detached HEAD)"
    status = _git(["status", "--short"], work_dir)
    log = _git(["log", "--oneline", f"-{LOG_LINES}"], work_dir)
    if status is None:
        return None
    lines = [
        GIT_SNAPSHOT_OPEN,
        "Git state when this session started; it may have changed since, so check before acting on it.",
        f"Branch: {branch}",
    ]
    changed = status.splitlines()
    if changed:
        lines.append(f"Uncommitted ({len(changed)}):")
        lines.extend(changed[:STATUS_MAX_LINES])
        if len(changed) > STATUS_MAX_LINES:
            lines.append(f"... {len(changed) - STATUS_MAX_LINES} more")
    else:
        lines.append("Working tree clean.")
    if log:
        lines.append("Recent commits:")
        lines.extend(log.splitlines())
    lines.append(GIT_SNAPSHOT_CLOSE)
    return "\n".join(lines)
