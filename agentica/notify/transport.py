# -*- coding: utf-8 -*-
"""Shared process identity attached to every external egress payload."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

_attach_endpoint: Optional[Dict[str, str]] = None


def set_attach_endpoint(socket: Optional[str], peer_id: Optional[str]) -> None:
    """Publish this process's live attach endpoint."""
    global _attach_endpoint
    endpoint: Dict[str, str] = {}
    if socket:
        endpoint["attach_socket"] = str(socket)
    if peer_id:
        endpoint["peer_id"] = str(peer_id)
    _attach_endpoint = endpoint or None


def build_transport(work_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return stable process identity and the current terminal attach route."""
    transport: Dict[str, Any] = {"ppid": os.getppid()}
    cwd = work_dir or os.getcwd()
    if cwd:
        transport["cwd"] = str(cwd)
    tty = _tty_name()
    if tty:
        transport["tty"] = tty
    if _attach_endpoint:
        transport.update(_attach_endpoint)
    return transport


def reset_transport_for_tests() -> None:
    """Clear process-wide attach metadata between tests."""
    global _attach_endpoint
    _attach_endpoint = None


def _tty_name() -> Optional[str]:
    """Return the controlling terminal name when stdin has one."""
    try:
        return os.ttyname(sys.stdin.fileno())
    except (AttributeError, OSError, ValueError):
        return None
