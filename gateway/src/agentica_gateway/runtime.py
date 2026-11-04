# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Where a running gateway publishes how to reach it.

A desktop shell (or a second terminal) has to answer two questions before it
can do anything: *is a gateway already up on this data root*, and *what port
and token does it answer on*. Neither is knowable from the outside once
``--port 0`` is in play, so the process that binds the socket writes the answer
down and removes it on the way out.

The file lives under ``$AGENTICA_CACHE_DIR`` because it is runtime state, not
configuration: it is meaningless after the process exits, and it holds a
credential, so it is written ``0600``.

One file per data root, not one per process. Two gateways on one root is
already ambiguous (they share sessions, cron jobs and the peers tree), so the
newest one wins the file and says so in the log. Removal is pid-guarded: a
gateway never deletes a record another process owns, which is what keeps a
desktop-spawned instance from unpublishing the terminal one on its way out.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger

RUNTIME_FILE = Path(AGENTICA_CACHE_DIR) / "gateway" / "runtime.json"


@dataclass
class GatewayRuntime:
    """How to reach one running gateway."""

    pid: int
    host: str
    port: int
    token: str
    version: str

    @property
    def url(self) -> str:
        """The address a client should open. Never the bind address.

        ``0.0.0.0`` is a way to listen, not a place to connect to — handing it
        to a browser or an HTTP client is the kind of thing that works on Linux
        and fails on Windows.
        """
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "token": self.token,
            "version": self.version,
            "url": self.url,
        }


def is_pid_alive(pid: int) -> bool:
    """Whether ``pid`` still exists. Signal 0 checks without delivering."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by somebody else — which for this file means the record
        # is stale in a way we must not act on either.
        return True
    return True


def publish(runtime: GatewayRuntime) -> Path:
    """Write the record, ``0600``, and warn if it displaces a live gateway."""
    existing = read()
    if existing is not None and existing.pid != runtime.pid and is_pid_alive(existing.pid):
        logger.warning(
            f"Another gateway (pid {existing.pid}) is already published on this "
            f"data root at {existing.url}; overwriting the record. A desktop "
            f"shell will now attach to this one."
        )

    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a reader never sees a half-written file, and create
    # the temp file 0600 from the start rather than fixing the mode afterwards
    # (the token would be world-readable for the width of that window).
    tmp = RUNTIME_FILE.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(runtime.to_dict(), fh, indent=2)
    os.replace(tmp, RUNTIME_FILE)
    return RUNTIME_FILE


def read() -> Optional[GatewayRuntime]:
    """Read the record, or None when there is none / it is unreadable.

    A truncated or hand-edited file reads as "no gateway": the caller's next
    move is to start one, which is also the right move when the file is junk.
    """
    try:
        raw = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return GatewayRuntime(
            pid=int(raw["pid"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw.get("token") or ""),
            version=str(raw.get("version") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def unpublish(pid: int) -> None:
    """Remove the record only if ``pid`` owns it."""
    current = read()
    if current is None or current.pid != pid:
        return
    try:
        RUNTIME_FILE.unlink()
    except OSError:
        pass
