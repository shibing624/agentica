# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Attach RPC client, re-exported from the core package.

The client used to live here, which made it unreachable for the callers that
need it most: the cron daemon runs in the CLI process, and ``agentica`` cannot
import ``agentica_gateway`` (a separate distribution, and an optional one). The
implementation now sits next to the server that speaks the same protocol
(``agentica.cli.attach``), and this module keeps the old import path working so
gateway code did not have to change all at once.
"""
from __future__ import annotations

from typing import Optional

from agentica.cli.attach import (
    AttachClient as _AttachRpcClient,
    prompt_live_session,
    read_attach_token,
)
from agentica.peers import PeerInfo


def _read_attach_token(peer: PeerInfo) -> Optional[str]:
    """Back-compat wrapper: the shared reader takes ``(socket, peer_id)``."""
    return read_attach_token(peer.attach_socket, peer.peer_id)


__all__ = ["_AttachRpcClient", "_read_attach_token", "prompt_live_session", "read_attach_token"]
