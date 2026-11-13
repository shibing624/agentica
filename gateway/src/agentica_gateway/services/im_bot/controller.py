# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ImBotController core: state persistence, live CLI peer discovery, and inbound entry.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agentica.memory.session_log import SessionLog
from agentica.peers import PeerInfo, list_live_peers
from agentica.utils.log import logger

from ...card_page import CardRegistry
from ...channels.base import ChannelType, Message
from .attach import _read_attach_token
from .commands import ImChatState, ImCommand, parse_im_command
from .interactions import ImInteractionsMixin
from .menus import ImMenusMixin
from .turns import ImTurnsMixin


def _pkg():
    return sys.modules["agentica_gateway.services.im_bot"]


class ImBotController(ImMenusMixin, ImInteractionsMixin, ImTurnsMixin):
    """Session-bound IM controller for local Agentica workspaces and CLI terminals."""

    def __init__(
        self,
        *,
        channel_manager: Any,
        agent_service: Any,
        card_registry: Optional[CardRegistry] = None,
        state_file: Optional[Path] = None,
        user_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self._channel_manager = channel_manager
        self._agent_service = agent_service
        self._cards = card_registry if card_registry is not None else CardRegistry()
        self._user_id_provider = user_id_provider or (lambda: _pkg().settings.default_user_id)
        if state_file is not None:
            self._state_file = Path(state_file)
        else:
            from agentica.config import AGENTICA_HOME

            self._state_file = Path(AGENTICA_HOME) / "gateway" / "im_bot_state.json"
        self._states: Dict[str, ImChatState] = {}
        self._turn_tasks: Dict[str, asyncio.Task] = {}
        self._queued_turns: Dict[str, asyncio.Queue] = {}
        self._load_states()

    @property
    def card_registry(self) -> CardRegistry:
        return self._cards

    def _user_id(self) -> str:
        return self._user_id_provider() or _pkg().settings.default_user_id

    # ── Persistence ────────────────────────────────────────────────────

    def _load_states(self) -> None:
        if not self._state_file.is_file():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in raw.get("chats") or []:
            if not isinstance(item, dict):
                continue
            try:
                ch = ChannelType(item.get("channel", ""))
            except ValueError:
                continue
            cid = str(item.get("channel_id") or "")
            key = str(item.get("chat_key") or f"{ch.value}:{cid}")
            if not cid:
                continue
            self._states[key] = ImChatState(
                chat_key=key,
                channel=ch,
                channel_id=cid,
                work_dir=str(item.get("work_dir") or ""),
                session_id=str(item.get("session_id") or ""),
                verbose=bool(item.get("verbose", False)),
            )

    def _save_states(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"chats": [s.to_persisted() for s in self._states.values()]}
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._state_file)
        except OSError as exc:
            logger.warning(f"im_bot: failed to save state: {exc}")

    def state_for(self, message: Message) -> ImChatState:
        key = f"{message.channel.value}:{message.channel_id}"
        state = self._states.get(key)
        if state is None:
            state = ImChatState(
                chat_key=key,
                channel=message.channel,
                channel_id=message.channel_id,
            )
            self._states[key] = state
        if not state.work_dir or not Path(state.work_dir).is_dir():
            state.work_dir = self._default_work_dir()
        if not state.session_id:
            state.session_id = self._default_session_id(state.work_dir)
            self._bind_session(state, state.work_dir, state.session_id)
        return state

    def _default_work_dir(self) -> str:
        live = list_live_peers()
        for peer in live:
            if peer.cwd and Path(peer.cwd).is_dir():
                return peer.cwd
        base = str(_pkg().settings.base_dir)
        if base and Path(base).is_dir():
            return os.path.realpath(base)
        return os.getcwd()

    def _default_session_id(self, work_dir: str) -> str:
        for peer in list_live_peers():
            if peer.session_id and peer.cwd and os.path.realpath(peer.cwd) == os.path.realpath(work_dir):
                return peer.session_id
        sessions = SessionLog.list_sessions(work_dir=work_dir, user_id=self._user_id())
        for entry in sessions:
            sid = entry.get("session_id")
            if sid and not str(sid).startswith("scheduled_"):
                return str(sid)
        return str(uuid.uuid4())

    def _bind_session(self, state: ImChatState, work_dir: str, session_id: str) -> None:
        real_dir = os.path.realpath(os.path.expanduser(work_dir))
        state.work_dir = real_dir
        state.session_id = session_id
        uid = self._user_id()
        if self._agent_service is not None:
            self._agent_service.set_session_work_dir(session_id, real_dir, owner=uid)
        self._save_states()

    def find_live_cli_peer(self, session_id: str) -> Optional[PeerInfo]:
        """Return the live CLI peer serving ``session_id`` if its attach socket is up."""
        if not session_id:
            return None
        for peer in list_live_peers():
            if peer.session_id == session_id and peer.attach_socket:
                if Path(peer.attach_socket).exists() and _read_attach_token(peer):
                    return peer
        return None

    async def stop(self) -> None:
        for task in list(self._turn_tasks.values()):
            if not task.done():
                task.cancel()
        self._turn_tasks.clear()

    # ── Inbound Entry ──────────────────────────────────────────────────

    async def handle(self, message: Message) -> bool:
        """Process an inbound IM message."""
        state = self.state_for(message)
        media_refs = message.metadata.get("media") or []
        cmd = parse_im_command(message.content or "")

        if cmd.kind == "empty" and not media_refs:
            return True

        if cmd.kind == "number" and not media_refs:
            if state.has_active_pending():
                reply = await self._handle_number(state, cmd.number)
                await self._reply(state, reply)
                return True
            if cmd.number == 0:
                await self._reply(state, self._render_menu(state))
                return True
            cmd = ImCommand(kind="chat", arg=cmd.arg)

        if state.has_active_pending() and state.pending.kind == "interaction" and cmd.kind == "chat" and not media_refs:
            reply = await self._handle_interaction_text(state, cmd.arg)
            await self._reply(state, reply)
            return True

        if cmd.kind in ("menu", "help", "status", "ps", "stop_bg", "project", "new", "session", "model", "cancel", "verbose"):
            if cmd.kind not in ("cancel", "stop_bg"):
                state.clear_pending()
            reply = await self._dispatch_command(state, cmd, message)
            if reply:
                await self._reply(state, reply)
            return True

        state.clear_pending()
        await self._submit_turn(state, message, text_override=cmd.arg if cmd.kind == "chat" else None)
        return True
