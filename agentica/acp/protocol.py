# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Protocol Handler

JSON-RPC 2.0 over newline-delimited JSON on stdio, as required by the Agent
Client Protocol. stdout carries protocol frames only; diagnostics go to
stderr (see ACPServer.run).
"""


import json
import sys
from typing import Any, Dict, Optional, TextIO, Union

from agentica.acp.types import ACPRequest, ACPResponse
from agentica.utils.log import logger


class ACPProtocolHandler:
    """Reads and writes one JSON-RPC message per line on the stdio transport."""

    def __init__(self, in_stream: Optional[TextIO] = None, out_stream: Optional[TextIO] = None):
        # Captured at construction: ACPServer.run() redirects sys.stdout so
        # agent output cannot corrupt the transport, while frames keep going
        # to the real stdout.
        self._in = in_stream if in_stream is not None else sys.stdin
        self._out = out_stream if out_stream is not None else sys.stdout

    def read_message(self) -> Optional[ACPRequest]:
        """Read one NDJSON request; None means EOF."""
        while True:
            line = self._in.readline()
            if not line:
                return None
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                logger.error(f"Invalid ACP frame: {error}")
                continue
            if "method" not in data:
                logger.warning("ACP frame without method ignored")
                continue
            return ACPRequest.from_dict(data)

    def _write(self, body: Dict[str, Any]) -> None:
        self._out.write(json.dumps(body, ensure_ascii=False) + "\n")
        self._out.flush()

    def write_message(self, response: ACPResponse) -> None:
        """Write a JSON-RPC response as one line."""
        self._write(response.to_dict())

    def send_error(self, request_id: Union[str, int, None], code: int, message: str, data: Any = None) -> None:
        """Send a JSON-RPC error response."""
        request_id = 0 if request_id is None else request_id
        self.write_message(ACPResponse.create_error(request_id, code, message, data))

    def send_success(self, request_id: Union[str, int], result: Dict[str, Any]) -> None:
        """Send a JSON-RPC success response."""
        self.write_message(ACPResponse.create_success(request_id, result))

    def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a one-way notification; notifications carry no id."""
        self._write({"jsonrpc": "2.0", "method": method, "params": params})
