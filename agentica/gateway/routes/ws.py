# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
WebSocket gateway: /ws endpoint + ConnectionManager.
"""
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agentica.utils.log import logger
from .. import deps
from ..config import settings

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections by client_id."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.debug(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)
        logger.debug(f"WebSocket disconnected: {client_id}")

    async def send_event(self, client_id: str, event: str, payload: dict) -> None:
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_json({"type": "event", "event": event, "payload": payload})

    async def broadcast(self, event: str, payload: dict) -> None:
        dead: list[str] = []
        for cid, ws in self.active_connections.items():
            try:
                await ws.send_json({"type": "event", "event": event, "payload": payload})
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    def count(self) -> int:
        return len(self.active_connections)


# Module-level singleton so other modules can broadcast events
ws_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket gateway endpoint.

    Handshake protocol:
        client → {"type":"req","method":"connect","params":{"auth":{"token":"..."},"client":{"id":"..."}}}
        server → {"type":"res","ok":true,"payload":{"type":"hello-ok","protocol":1,...}}
    """
    client_id = None
    try:
        await websocket.accept()
        data = await websocket.receive_json()

        if data.get("method") != "connect":
            await websocket.close(code=4000, reason="Must connect first")
            return

        params = data.get("params", {})
        client_id = params.get("client", {}).get("id", "unknown")
        ws_manager.active_connections[client_id] = websocket
        logger.debug(f"WebSocket connected: {client_id}")

        await websocket.send_json({
            "type": "res",
            "id": data.get("id"),
            "ok": True,
            "payload": {
                "type": "hello-ok",
                "protocol": 1,
                "policy": {"tickIntervalMs": 15000},
            },
        })

        while True:
            message = await websocket.receive_json()
            await _handle_message(websocket, client_id, message)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if client_id:
            ws_manager.disconnect(client_id)


async def _handle_message(ws: WebSocket, client_id: str, message: dict) -> None:
    """Dispatch an incoming WebSocket request message."""
    if message.get("type") != "req":
        return

    method = message.get("method")
    params = message.get("params", {})
    req_id = message.get("id", "")

    try:
        result: Dict[str, Any] = {}

        if method == "health":
            result = {"status": "ok", "connections": ws_manager.count()}

        elif method == "status":
            from agentica.cron.jobs import list_jobs as cron_list_jobs
            active_jobs = len(cron_list_jobs(include_disabled=False))
            result = {
                "channels": deps.channel_manager.get_status() if deps.channel_manager else {},
                "scheduler": {"active_jobs": active_jobs},
            }

        elif method == "agent":
            # Streaming agent via WebSocket (content-only; tool events not yet supported)
            text = params.get("message", "")
            session_id = params.get("sessionId", "default")
            user_id = params.get("userId", settings.default_user_id)

            async def on_content(delta: str):
                await ws_manager.send_event(client_id, "agent.content", {
                    "delta": delta,
                    "sessionId": session_id,
                    "userId": user_id,
                })

            if deps.agent_service:
                chat_result = await deps.agent_service.chat_stream(
                    message=text,
                    session_id=session_id,
                    user_id=user_id,
                    on_content=on_content,
                )
                result = {
                    "content": chat_result.content,
                    "toolCalls": chat_result.tool_calls,
                    "sessionId": chat_result.session_id,
                    "userId": chat_result.user_id,
                }
            else:
                result = {"error": "Agent service not ready"}

        elif method == "send":
            channel = params.get("channel")
            target = params.get("target")
            content = params.get("message")
            if deps.channel_manager:
                success = await deps.channel_manager.send(channel, target, content)
                result = {"status": "sent" if success else "failed"}
            else:
                result = {"status": "failed", "error": "Channel manager not ready"}

        else:
            raise ValueError(f"Unknown method: {method}")

        await ws.send_json({"type": "res", "id": req_id, "ok": True, "payload": result})

    except Exception as e:
        await ws.send_json({
            "type": "res",
            "id": req_id,
            "ok": False,
            "error": {"code": "ERROR", "message": str(e)},
        })
