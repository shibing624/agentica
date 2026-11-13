# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Minimal synchronous JSON-RPC client for CLI AttachServer Unix sockets.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agentica.peers import PeerInfo


class _AttachRpcClient:
    """Minimal synchronous JSON-RPC client for a CLI session's AttachServer socket."""

    def __init__(self, sock_path: str, token: str, timeout: float = 10.0) -> None:
        self.sock_path = sock_path
        self.token = token
        self.timeout = timeout

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.sock_path)
            buf = b""
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"authToken": self.token, "clientCapabilities": {}},
            }
            sock.sendall(json.dumps(init_req).encode("utf-8") + b"\n")
            init_resp, buf = self._read_line(sock, buf)
            if "error" in init_resp:
                raise RuntimeError(init_resp["error"].get("message", "attach auth failed"))
            if method == "initialize":
                return init_resp.get("result") or {}

            req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": method,
                "params": params or {},
            }
            sock.sendall(json.dumps(req).encode("utf-8") + b"\n")
            resp, _ = self._read_line(sock, buf)
            if "error" in resp:
                raise RuntimeError(resp["error"].get("message", f"attach {method} failed"))
            return resp.get("result") or {}

    @staticmethod
    def _read_line(sock: socket.socket, buf: bytes) -> Tuple[Dict[str, Any], bytes]:
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise RuntimeError("attach socket closed unexpectedly")
            buf += chunk
        line, _, rest = buf.partition(b"\n")
        return json.loads(line.decode("utf-8")), rest


def _read_attach_token(peer: PeerInfo) -> Optional[str]:
    from agentica.cli.attach import token_path

    if not peer.attach_socket:
        return None
    sock_file = Path(peer.attach_socket)
    tok_file = sock_file.with_suffix(".token")
    if not tok_file.is_file():
        tok_file = token_path(peer.peer_id)
    try:
        return tok_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
