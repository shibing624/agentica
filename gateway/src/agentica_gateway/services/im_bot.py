# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM Bot controller for remote-controlling local Agentica sessions.

Replaces the peer-mailbox bridge with a session-bound controller modeled after
OpenBitFun's ``remote_connect/bot/`` and agent-autoloop's ``card-page``:
- Each IM chat binds to a ``(work_dir, session_id)`` on this machine, sharing
  the exact same ``SessionLog`` JSONL as the local CLI and Web UI.
- Commands (``/menu``, ``/switch``, ``/new``, ``/resume``, ``/model``,
  ``/cancel``, ``/status``, ``/verbose``, ``/concise``, ``/help``) and numeric
  menu replies (``1..N``, ``0``) manage the bound workspace and session.
- Free-form text submits a turn to the bound session:
  * If that session is currently open in a local CLI terminal (with an
    ``AttachServer`` socket), the turn is injected directly into the live CLI
    process and its final answer is returned when the turn finishes.
  * Otherwise, ``AgentService`` runs the turn against the shared ``SessionLog``
    so resuming in the Mac terminal later picks up seamlessly.
- Pending approvals and ``ask_user_question`` prompts are delivered both as
  numbered IM options and as a zero-JS capability-URL card (``/c/<token>``).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentica.agent.approvals import ApprovalDecision, PendingApproval
from agentica.cli.approvals import (
    approval_decision_from_key,
    format_approval_prompt,
    visible_approval_decisions,
)
from agentica.config.profiles import get_active_profile_name, get_profiles
from agentica.memory.session_log import SessionLog, local_turn_stamp
from agentica.peers import PeerInfo, list_live_peers
from agentica.tools.ask_user_question_tool import AskUserQuestionTimeout
from agentica.utils.log import logger

from ..card_page import CardItem, CardRegistry, gateway_public_url
from ..channels.base import Channel, ChannelType, Message
from ..config import settings
from . import chat_images, live_turn
from .im_files import collect_auto_push_files, format_file_size

PENDING_TTL_SECS = 300.0
PENDING_INVALID_LIMIT = 3
SESSION_PAGE_SIZE = 8
CHUNK_CHARS = 1800
ACK_DELAY_SECS = 4.0

_NUMERIC_SUFFIX_CHARS = set(".。、,，:：;；)）]】")


@dataclass(frozen=True)
class ImCommand:
    """Parsed IM command or chat message."""

    kind: str
    arg: str = ""
    number: int = -1


def normalize_im_command_text(text: str) -> str:
    """Normalize full-width digits ``０..９`` to ASCII ``0..9``."""
    out: List[str] = []
    for ch in (text or "").strip():
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19:
            out.append(chr(code - 0xFF10 + ord("0")))
        else:
            out.append(ch)
    return "".join(out)


def strip_numeric_reply_suffix(text: str) -> str:
    s = (text or "").strip()
    while s and s[-1] in _NUMERIC_SUFFIX_CHARS:
        s = s[:-1].strip()
    return s


def parse_im_command(text: str) -> ImCommand:
    """Classify one inbound IM line into a control command, number, or chat turn."""
    normalized = normalize_im_command_text(text)
    trimmed = normalized.strip()
    if not trimmed:
        return ImCommand(kind="empty")

    head, _, rest = trimmed.partition(" ")
    lower_head = head.lower()
    rest = rest.strip()

    if lower_head in ("/start", "/menu", "/m") or trimmed == "菜单":
        return ImCommand(kind="menu")
    if lower_head in ("/help", "/h", "/?") or trimmed in ("帮助", "？", "?"):
        return ImCommand(kind="help")
    if lower_head in ("/status", "/s") or trimmed == "状态":
        return ImCommand(kind="status")
    if lower_head in ("/switch", "/workspace", "/switch_workspace") or trimmed in ("切换", "切换工作区"):
        return ImCommand(kind="switch", arg=rest)
    if lower_head in ("/new", "/n", "/new_session") or trimmed in ("新建", "新会话", "新建会话"):
        return ImCommand(kind="new", arg=rest)
    if lower_head in ("/resume", "/r", "/sessions", "/resume_session") or trimmed in (
        "恢复", "恢复会话", "会话列表",
    ):
        return ImCommand(kind="resume", arg=rest)
    if lower_head in ("/model", "/switch_model") or trimmed == "切换模型":
        return ImCommand(kind="model", arg=rest)
    if lower_head in ("/cancel", "/stop", "/cancel_task") or trimmed in ("取消", "停止"):
        return ImCommand(kind="cancel", arg=rest)
    if lower_head == "/verbose" or trimmed == "详细":
        return ImCommand(kind="verbose", arg="on")
    if lower_head == "/concise" or trimmed == "简洁":
        return ImCommand(kind="verbose", arg="off")
    if lower_head == "/goal":
        return ImCommand(kind="goal", arg=rest)

    num_token = strip_numeric_reply_suffix(trimmed)
    if num_token.isdigit():
        n = int(num_token)
        if 0 <= n <= 99:
            return ImCommand(kind="number", number=n, arg=trimmed)

    return ImCommand(kind="chat", arg=trimmed)


@dataclass
class PendingAction:
    """In-memory pending menu or blocking interaction for an IM chat."""

    kind: str  # "menu" | "workspace" | "session" | "model" | "interaction"
    options: List[Any] = field(default_factory=list)
    page: int = 0
    has_more: bool = False
    interaction_kind: str = ""  # "ask" | "approval"
    request_id: str = ""
    session_id: str = ""
    card_token: str = ""
    attach_peer: Optional[PeerInfo] = None
    resolve_fn: Optional[Callable[[str], bool]] = None


@dataclass
class ImChatState:
    """Per-chat controller state (work_dir + session_id persisted to disk)."""

    chat_key: str
    channel: ChannelType
    channel_id: str
    work_dir: str = ""
    session_id: str = ""
    verbose: bool = False
    pending: Optional[PendingAction] = None
    pending_expires_at: float = 0.0
    pending_invalid_count: int = 0

    def set_pending(self, action: PendingAction, ttl: float = PENDING_TTL_SECS) -> None:
        self.pending = action
        self.pending_expires_at = time.time() + ttl
        self.pending_invalid_count = 0

    def clear_pending(self) -> None:
        self.pending = None
        self.pending_expires_at = 0.0
        self.pending_invalid_count = 0

    def has_active_pending(self) -> bool:
        if self.pending is None:
            return False
        if time.time() > self.pending_expires_at:
            self.clear_pending()
            return False
        return True

    def to_persisted(self) -> Dict[str, Any]:
        return {
            "chat_key": self.chat_key,
            "channel": self.channel.value,
            "channel_id": self.channel_id,
            "work_dir": self.work_dir,
            "session_id": self.session_id,
            "verbose": self.verbose,
        }


def _short_dir(path: str) -> str:
    if not path:
        return "workspace"
    p = Path(path.rstrip("/"))
    return p.name or str(p)


def _fmt_elapsed(seconds: float) -> str:
    total = max(0, int(seconds + 0.5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m" if m else f"{h}h"
    if m:
        return f"{m}m{s}s" if s else f"{m}m"
    return f"{s}s"


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


HELP_TEXT = (
    "📱 Agentica 远控菜单命令：\n"
    "  /menu (菜单)    打开快捷控制菜单\n"
    "  /resume (恢复)  查看并切换会话（含正在运行的终端）\n"
    "  /new (新建)     在当前目录新建会话\n"
    "  /switch (切换)  切换本机项目工作区\n"
    "  /model          切换当前会话模型\n"
    "  /status (状态)  查看当前绑定会话与执行状态\n"
    "  /cancel (取消)  中止当前正在运行的任务\n"
    "  /verbose / /concise  切换详细/简洁汇报\n\n"
    "直接发送任何文字即为向当前绑定会话下发任务；出现编号菜单或审批时直接回复数字即可。"
)

MENU_COMMANDS = [
    ("/resume", "恢复 / 切换会话（含本机活跃终端）"),
    ("/new", "新建会话"),
    ("/switch", "切换项目工作区"),
    ("/model", "切换模型 Profile"),
    ("/status", "查看当前状态"),
    ("/cancel", "停止当前任务"),
]

_APPROVAL_LABELS = {
    "allow": "允许一次 (Allow)",
    "allow_prefix": "允许同类 (Allow similar)",
    "deny": "拒绝 (Deny)",
    "deny_prefix": "拒绝同类 (Deny similar)",
}

_SOURCE_LABELS = {
    "cli": "CLI",
    "wechat": "微信",
    "wecom": "企微",
    "feishu": "飞书",
    "telegram": "TG",
    "dingtalk": "钉钉",
    "qq": "QQ",
    "discord": "Discord",
    "slack": "Slack",
    "gateway": "Web",
}


class ImBotController:
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
        self._user_id_provider = user_id_provider or (lambda: settings.default_user_id)
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
        return self._user_id_provider() or settings.default_user_id

    # ── Persistence ────────────────────────────────────────────────────

    def _load_states(self) -> None:
        if not self._state_file.is_file():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
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
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                tmp.chmod(0o600)
            except OSError:
                pass
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
        base = str(settings.base_dir)
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
            log = self._agent_service.session_log_for(session_id, owner=uid)
            cur_source = log.get_source()
            if not cur_source or cur_source == "gateway":
                log.set_source(state.channel.value)
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

        if cmd.kind in ("menu", "help", "status", "switch", "new", "resume", "model", "cancel", "verbose"):
            if cmd.kind != "cancel":
                state.clear_pending()
            reply = await self._dispatch_command(state, cmd, message)
            if reply:
                await self._reply(state, reply)
            return True

        state.clear_pending()
        await self._submit_turn(state, message, text_override=cmd.arg if cmd.kind == "chat" else None)
        return True

    # ── Command Dispatch & Menus ───────────────────────────────────────

    async def _dispatch_command(self, state: ImChatState, cmd: ImCommand, message: Message) -> str:
        if cmd.kind == "menu":
            return self._render_menu(state)
        if cmd.kind == "help":
            return HELP_TEXT
        if cmd.kind == "status":
            return self._render_status(state)
        if cmd.kind == "verbose":
            state.verbose = cmd.arg == "on"
            self._save_states()
            mode_str = "详细模式（附带工具统计与思考摘要）" if state.verbose else "简洁模式（仅推送最终结论与产物文件）"
            return f"已切换为：{mode_str}"
        if cmd.kind == "switch":
            if cmd.arg and Path(os.path.expanduser(cmd.arg)).is_dir():
                target_dir = os.path.realpath(os.path.expanduser(cmd.arg))
                sid = self._default_session_id(target_dir)
                self._bind_session(state, target_dir, sid)
                return self._render_bound_banner(state, prefix="✅ 已切换工作区")
            return self._open_workspace_picker(state)
        if cmd.kind == "new":
            new_sid = str(uuid.uuid4())
            self._bind_session(state, state.work_dir, new_sid)
            if cmd.arg:
                await self._submit_turn(state, message, text_override=cmd.arg)
                return ""
            return self._render_bound_banner(state, prefix="🆕 已新建并绑定会话")
        if cmd.kind == "resume":
            return self._open_session_picker(state, page=0)
        if cmd.kind == "model":
            if cmd.arg:
                return await self._apply_model_choice(state, cmd.arg)
            return self._open_model_picker(state)
        if cmd.kind == "cancel":
            return await self._cancel_active_turn(state)
        return HELP_TEXT

    def _session_title(self, work_dir: str, session_id: str) -> str:
        uid = self._user_id()
        log = SessionLog(session_id=session_id, work_dir=work_dir, user_id=uid)
        name = log.get_name()
        if name:
            return name
        if log.path.is_file() and log.path.stat().st_size > 0:
            preview = SessionLog.session_preview(str(log.path), max_chars=48)
            first = chat_images.strip_markers(preview.get("first_user") or "").strip()
            if first:
                return " ".join(first.split())[:40]
        return f"会话 {session_id[:8]}"

    def _is_busy(self, session_id: str) -> bool:
        peer = self.find_live_cli_peer(session_id)
        if peer is not None and peer.busy:
            return True
        uid = self._user_id()
        if self._agent_service is not None and self._agent_service.is_session_active(session_id, owner=uid):
            return True
        return session_id in self._turn_tasks and not self._turn_tasks[session_id].done()

    def _render_bound_banner(self, state: ImChatState, *, prefix: str = "📌 当前控制目标") -> str:
        peer = self.find_live_cli_peer(state.session_id)
        title = self._session_title(state.work_dir, state.session_id)
        busy = self._is_busy(state.session_id)
        runtime_tag = (
            f"🟢 终端在线 ({peer.name})" if peer is not None else "💻 后台共享会话"
        )
        status_tag = "⏳ 运行中" if busy else "空闲"
        lines = [
            f"{prefix}：",
            f"  项目：{_short_dir(state.work_dir)} ({state.work_dir})",
            f"  会话：{title} [{state.session_id[:8]}]",
            f"  模式：{runtime_tag} · {status_tag}",
        ]
        if peer is not None and peer.task:
            lines.append(f"  当前任务：{peer.task}")
        return "\n".join(lines)

    def _render_menu(self, state: ImChatState) -> str:
        banner = self._render_bound_banner(state)
        lines = [banner, "", "回复数字选择操作："]
        cmds: List[str] = []
        for idx, (cmd_str, label) in enumerate(MENU_COMMANDS, 1):
            lines.append(f"  {idx}. {label} ({cmd_str})")
            cmds.append(cmd_str)
        lines.append("\n直接输入文字即可向当前会话派发任务。")
        state.set_pending(PendingAction(kind="menu", options=cmds))
        return "\n".join(lines)

    def _render_status(self, state: ImChatState) -> str:
        uid = self._user_id()
        banner = self._render_bound_banner(state, prefix="📊 当前会话状态")
        lines = [banner]
        if self._agent_service is not None:
            cfg = self._agent_service.session_model_config(state.session_id, owner=uid)
            prof = cfg.get("profile") or "default"
            lines.append(f"  模型：{cfg.get('model_provider')}/{cfg.get('model_name')} (profile: {prof})")
        log = SessionLog(session_id=state.session_id, work_dir=state.work_dir, user_id=uid)
        if log.path.is_file():
            lines.append(f"  日志：{format_file_size(log.path.stat().st_size)}")
        lines.append(f"  汇报：{'详细 (/verbose)' if state.verbose else '简洁 (/concise)'}")
        return "\n".join(lines)

    def _list_workspace_choices(self, current_dir: str) -> List[str]:
        seen = set()
        choices: List[str] = []

        def _add(p: Optional[str]) -> None:
            if not p:
                return
            real = os.path.realpath(os.path.expanduser(p))
            if real in seen or not Path(real).is_dir():
                return
            seen.add(real)
            choices.append(real)

        _add(current_dir)
        for peer in list_live_peers():
            _add(peer.cwd)
        _add(str(settings.base_dir))
        for proj in SessionLog.list_projects(user_id=self._user_id()):
            _add(proj.get("work_dir"))
        return choices[:15]

    def _open_workspace_picker(self, state: ImChatState) -> str:
        choices = self._list_workspace_choices(state.work_dir)
        if not choices:
            return "未找到可切换的项目工作区。"
        live_by_dir = {}
        for peer in list_live_peers():
            if peer.cwd:
                live_by_dir[os.path.realpath(peer.cwd)] = peer
        lines = ["📂 选择要控制的项目工作区（回复数字 1..N，0 返回菜单）："]
        for idx, path in enumerate(choices, 1):
            cur = " [当前]" if os.path.realpath(path) == os.path.realpath(state.work_dir) else ""
            peer = live_by_dir.get(os.path.realpath(path))
            live_tag = f" 🟢终端:{peer.name}" if peer else ""
            lines.append(f"  {idx}. {_short_dir(path)}{cur}{live_tag}\n     {path}")
        state.set_pending(PendingAction(kind="workspace", options=choices))
        return "\n".join(lines)

    def _collect_session_items(self, work_dir: str) -> List[Dict[str, Any]]:
        uid = self._user_id()
        live_map: Dict[str, PeerInfo] = {}
        items: List[Dict[str, Any]] = []
        seen_ids = set()

        for peer in list_live_peers():
            if peer.session_id:
                live_map[peer.session_id] = peer
                if os.path.realpath(peer.cwd or "") == os.path.realpath(work_dir):
                    seen_ids.add(peer.session_id)
                    items.append({
                        "session_id": peer.session_id,
                        "work_dir": peer.cwd or work_dir,
                        "title": peer.task or self._session_title(peer.cwd or work_dir, peer.session_id),
                        "stamp": "终端在线",
                        "source": "cli",
                        "live_peer": peer,
                    })

        for entry in SessionLog.list_sessions(work_dir=work_dir, user_id=uid):
            sid = str(entry.get("session_id") or "")
            if not sid or sid.startswith("scheduled_") or sid in seen_ids:
                continue
            if int(entry.get("size_bytes") or 0) <= 0 and sid not in live_map:
                continue
            seen_ids.add(sid)
            peer = live_map.get(sid)
            title = entry.get("name") or self._session_title(work_dir, sid)
            stamp = local_turn_stamp(entry.get("last_timestamp") or entry.get("mtime"))
            src = str(entry.get("source") or ("cli" if entry.get("cli") else ""))
            items.append({
                "session_id": sid,
                "work_dir": entry.get("work_dir") or work_dir,
                "title": title,
                "stamp": stamp,
                "source": src,
                "live_peer": peer,
            })

        # Also surface live CLI sessions in other directories so the user can attach to any open terminal
        for peer in list_live_peers():
            if peer.session_id and peer.session_id not in seen_ids:
                seen_ids.add(peer.session_id)
                items.append({
                    "session_id": peer.session_id,
                    "work_dir": peer.cwd or work_dir,
                    "title": f"[{_short_dir(peer.cwd)}] {peer.task or self._session_title(peer.cwd or work_dir, peer.session_id)}",
                    "stamp": "其他终端在线",
                    "source": "cli",
                    "live_peer": peer,
                })

        # Also include recent sessions from other projects on this machine so a closed CLI is always one tap away
        other_entries: List[Tuple[float, Dict[str, Any], str]] = []
        for proj in SessionLog.list_projects(user_id=uid):
            p_dir = proj.get("work_dir") or ""
            if not p_dir or os.path.realpath(p_dir) == os.path.realpath(work_dir):
                continue
            for entry in SessionLog.list_sessions(base_dir=proj.get("base_dir"), work_dir=p_dir, user_id=uid)[:3]:
                sid = str(entry.get("session_id") or "")
                if not sid or sid.startswith("scheduled_") or sid in seen_ids:
                    continue
                if int(entry.get("size_bytes") or 0) <= 0:
                    continue
                other_entries.append((float(entry.get("mtime") or 0.0), entry, p_dir))

        other_entries.sort(key=lambda x: x[0], reverse=True)
        for _, entry, p_dir in other_entries[:5]:
            sid = str(entry.get("session_id") or "")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            raw_title = entry.get("name") or self._session_title(p_dir, sid)
            stamp = local_turn_stamp(entry.get("last_timestamp") or entry.get("mtime"))
            src = str(entry.get("source") or ("cli" if entry.get("cli") else ""))
            items.append({
                "session_id": sid,
                "work_dir": p_dir,
                "title": f"[{_short_dir(p_dir)}] {raw_title}",
                "stamp": stamp,
                "source": src,
                "live_peer": live_map.get(sid),
            })
        return items

    def _open_session_picker(self, state: ImChatState, page: int = 0) -> str:
        all_items = self._collect_session_items(state.work_dir)
        if not all_items:
            return f"当前项目 [{_short_dir(state.work_dir)}] 暂无历史会话。发送 /new 可新建会话。"
        start = page * SESSION_PAGE_SIZE
        if start >= len(all_items):
            page = 0
            start = 0
        page_items = all_items[start : start + SESSION_PAGE_SIZE]
        has_more = (start + SESSION_PAGE_SIZE) < len(all_items)

        lines = [f"🗂 选择要挂载的会话（项目：{_short_dir(state.work_dir)}，第 {page + 1} 页）："]
        for idx, item in enumerate(page_items, 1):
            sid = item["session_id"]
            cur = " [当前]" if sid == state.session_id else ""
            src_label = _SOURCE_LABELS.get(str(item.get("source") or ""))
            src_tag = f" [{src_label}]" if src_label else ""
            peer: Optional[PeerInfo] = item.get("live_peer")
            if peer is not None:
                badge = "⏳终端运行中" if peer.busy else f"🟢终端:{peer.name}"
            elif self._is_busy(sid):
                badge = "⏳后台运行中"
            else:
                badge = item.get("stamp") or ""
            badge_str = f" ({badge})" if badge else ""
            lines.append(f"  {idx}. {item['title']}{src_tag}{cur}{badge_str} [{sid[:6]}]")
        if has_more:
            lines.append("  0. 下一页")
        state.set_pending(
            PendingAction(
                kind="session",
                options=page_items,
                page=page,
                has_more=has_more,
            )
        )
        return "\n".join(lines)

    def _open_model_picker(self, state: ImChatState) -> str:
        profiles = get_profiles()
        if not profiles:
            return "未在 config.yaml 中找到配置的模型 Profile。"
        uid = self._user_id()
        current_prof = (
            self._agent_service.session_profile_name(state.session_id, owner=uid)
            if self._agent_service
            else ""
        ) or get_active_profile_name()
        options: List[Tuple[str, str]] = []
        lines = ["🤖 选择当前会话使用的模型（回复数字 1..N）："]
        for idx, (name, prof) in enumerate(profiles.items(), 1):
            prov = prof.get("model_provider", "")
            mname = prof.get("model_name", "")
            label = f"{name} ({prov}/{mname})"
            cur = " [当前]" if name == current_prof else ""
            lines.append(f"  {idx}. {label}{cur}")
            options.append((name, label))
        state.set_pending(PendingAction(kind="model", options=options))
        return "\n".join(lines)

    async def _apply_model_choice(self, state: ImChatState, name: str) -> str:
        profiles = get_profiles()
        if name not in profiles:
            return f"未知模型 Profile：{name}"
        if self._agent_service is not None:
            cfg = await self._agent_service.set_session_profile(
                state.session_id, name, owner=self._user_id(),
            )
            return f"✅ 已切换模型为：{name} ({cfg.get('model_provider')}/{cfg.get('model_name')})"
        return f"✅ 已切换模型为：{name}"

    async def _handle_number(self, state: ImChatState, n: int) -> str:
        pending = state.pending
        if pending is None:
            return self._render_menu(state)

        if pending.kind == "interaction":
            return await self._handle_interaction_number(state, n)

        if n == 0:
            if pending.kind == "session" and pending.has_more:
                return self._open_session_picker(state, page=pending.page + 1)
            state.clear_pending()
            return self._render_menu(state)

        idx = n - 1
        if not (0 <= idx < len(pending.options)):
            return self._on_invalid_pending(state, max_n=len(pending.options))

        kind = pending.kind
        chosen = pending.options[idx]
        state.clear_pending()

        if kind == "menu":
            fake_msg = Message(
                channel=state.channel,
                channel_id=state.channel_id,
                sender_id="user",
                sender_name="user",
                content=str(chosen),
                message_id="menu-select",
            )
            return await self._dispatch_command(state, parse_im_command(str(chosen)), fake_msg)

        if kind == "workspace":
            target_dir = str(chosen)
            sid = self._default_session_id(target_dir)
            self._bind_session(state, target_dir, sid)
            return self._render_bound_banner(state, prefix="✅ 已切换工作区")

        if kind == "session":
            item = chosen
            self._bind_session(state, item["work_dir"], item["session_id"])
            return self._render_bound_banner(state, prefix="✅ 已挂载会话")

        if kind == "model":
            prof_name, _ = chosen
            return await self._apply_model_choice(state, prof_name)

        return self._render_menu(state)

    def _on_invalid_pending(self, state: ImChatState, max_n: int) -> str:
        state.pending_invalid_count += 1
        if state.pending_invalid_count >= PENDING_INVALID_LIMIT:
            state.clear_pending()
            return "多次输入无效，已退出当前菜单。发送 /menu 重新打开。"
        return f"请输入有效编号（1..{max_n}），或发 0 返回菜单。"

    # ── Interactions (Approval / AskUserQuestion + /c/<token>) ─────────

    def _card_url(self, card: CardItem) -> str:
        base = gateway_public_url(settings.host, settings.port)
        return f"{base}/c/{card.token}"

    async def _publish_interaction(
        self,
        state: ImChatState,
        *,
        interaction_kind: str,
        title: str,
        text: str,
        options: List[str],
        values: List[str],
        session_id: str,
        request_id: str,
        resolve_fn: Callable[[str], bool],
        attach_peer: Optional[PeerInfo] = None,
    ) -> CardItem:
        def _on_card_resolved(card: CardItem) -> None:
            val = card.selected_value
            resolve_fn(val)
            if state.pending is not None and state.pending.card_token == card.token:
                state.clear_pending()

        card = self._cards.create(
            kind="approval" if interaction_kind == "approval" else "ask",
            title=title,
            text=text,
            options=options,
            values=values,
            recommend=0,
            allow_custom_text=(interaction_kind != "approval"),
            session_id=session_id,
            request_id=request_id,
            on_resolve=_on_card_resolved,
        )

        action = PendingAction(
            kind="interaction",
            options=list(values),
            interaction_kind=interaction_kind,
            request_id=request_id,
            session_id=session_id,
            card_token=card.token,
            attach_peer=attach_peer,
            resolve_fn=resolve_fn,
        )
        state.set_pending(action)

        header = "⚠️ [需要审批]" if interaction_kind == "approval" else "❓ [需要你的确认]"
        lines = [f"{header} {title}"]
        if text:
            lines.append(text)
        if options:
            lines.append("")
            for idx, opt_label in enumerate(options, 1):
                lines.append(f"  {idx}. {opt_label}")
        lines.append(f"\n👉 一键处理：{self._card_url(card)}")
        if interaction_kind == "approval":
            lines.append("（可直接点链接，或在此回复数字 1..N / y 允许 / n 拒绝）")
        else:
            lines.append("（可直接点链接，或在此回复数字 1..N / 直接输入文字回答）")
        await self._reply(state, "\n".join(lines))
        return card

    async def _handle_interaction_number(self, state: ImChatState, n: int) -> str:
        pending = state.pending
        if pending is None or pending.kind != "interaction":
            return self._render_menu(state)
        idx = n - 1
        if not (0 <= idx < len(pending.options)):
            return self._on_invalid_pending(state, max_n=len(pending.options))
        chosen_val = str(pending.options[idx])
        card_token = pending.card_token
        resolve_fn = pending.resolve_fn
        state.clear_pending()

        if card_token:
            self._cards.answer(card_token, choice=idx, by="IM", fire_callback=False)
        ok = resolve_fn(chosen_val) if resolve_fn is not None else False
        card = self._cards.get_by_token(card_token) if card_token else None
        label = card.selected_label if card is not None else chosen_val
        return f"✓ 已选择：{label}" if ok else "该请求已处理或已过期。"

    async def _handle_interaction_text(self, state: ImChatState, text: str) -> str:
        pending = state.pending
        if pending is None or pending.kind != "interaction":
            return self._render_menu(state)
        if pending.interaction_kind == "approval":
            decision = approval_decision_from_key(text, pending.options)
            if decision is None:
                return f"请回复数字 1..{len(pending.options)} 或 y / n，或点击上方卡片链接处理。"
            idx = pending.options.index(decision) if decision in pending.options else 0
            return await self._handle_interaction_number(state, idx + 1)

        card_token = pending.card_token
        resolve_fn = pending.resolve_fn
        state.clear_pending()
        if card_token:
            self._cards.answer(card_token, text=text, by="IM", fire_callback=False)
        ok = resolve_fn(text) if resolve_fn is not None else False
        return f"✓ 已发送回答：{text}" if ok else "该提问已处理或已过期。"

    async def handle_hook_event(
        self,
        payload: Dict[str, Any],
        *,
        wait_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Handle a CLI shell_hook event (needs.approval / needs.input / request.resolved).

        Forwards the CLI's blocking prompt to the active IM chat with a ``/c/<token>``
        capability link and numbered options, and optionally awaits the user's answer.
        """
        event = str(payload.get("hook_event_name") or "")
        request_id = str(payload.get("request_id") or uuid.uuid4().hex[:10])
        session_id = str(payload.get("session_id") or "")

        if event == "request.resolved":
            card = self._cards.find_open(request_id=request_id)
            if card is not None:
                by = str(payload.get("decided_by") or "terminal")
                self._cards.expire(card.token, reason=f"已在 {by} 处理")
            for st in self._states.values():
                if st.pending is not None and st.pending.request_id == request_id:
                    st.clear_pending()
            return {"ok": True, "request_id": request_id}

        if event not in ("needs.approval", "needs.input"):
            return {"ok": False, "error": f"unsupported hook event: {event}"}

        # Pick the IM chat bound to this session, or the most recently active IM chat
        target_state: Optional[ImChatState] = None
        for st in self._states.values():
            if session_id and st.session_id == session_id:
                target_state = st
                break
        if target_state is None and self._states:
            target_state = list(self._states.values())[-1]

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def _resolve_hook(val: str) -> bool:
            if not fut.done():
                fut.set_result(val)
                return True
            return False

        if event == "needs.approval":
            raw_opts = [str(o) for o in (payload.get("options") or ["allow", "deny"])]
            vis_opts = list(visible_approval_decisions(raw_opts))
            labels = [_APPROVAL_LABELS.get(o, o) for o in vis_opts]
            tool_name = str(payload.get("tool_name") or "tool")
            question = str(payload.get("question") or f"允许执行 {tool_name}？")
            preview = str(payload.get("preview") or "")
            text = f"$ {preview}" if preview else ""
            ikind = "approval"
            values = vis_opts
        else:
            raw_opts = [str(o) for o in (payload.get("options") or [])]
            labels = raw_opts
            values = raw_opts
            question = str(payload.get("question") or payload.get("prompt") or "请输入你的回答")
            text = ""
            ikind = "ask"

        if target_state is not None:
            card = await self._publish_interaction(
                target_state,
                interaction_kind=ikind,
                title=question,
                text=text,
                options=labels,
                values=values,
                session_id=session_id,
                request_id=request_id,
                resolve_fn=_resolve_hook,
            )
        else:
            def _on_card(c: CardItem) -> None:
                _resolve_hook(c.selected_value)

            card = self._cards.create(
                kind="approval" if ikind == "approval" else "ask",
                title=question,
                text=text,
                options=labels,
                values=values,
                allow_custom_text=(ikind != "approval"),
                session_id=session_id,
                request_id=request_id,
                on_resolve=_on_card,
            )

        out: Dict[str, Any] = {
            "card_id": card.id,
            "token": card.token,
            "link": self._card_url(card),
            "request_id": request_id,
        }
        if wait_seconds > 0:
            try:
                chosen_val = await asyncio.wait_for(fut, timeout=wait_seconds)
                if event == "needs.approval":
                    out["decision"] = chosen_val
                else:
                    out["answer"] = chosen_val
            except asyncio.TimeoutError:
                out["pending"] = True
        return out

    # ── Cancel ─────────────────────────────────────────────────────────

    async def _cancel_active_turn(self, state: ImChatState) -> str:
        sid = state.session_id
        peer = self.find_live_cli_peer(sid)
        cancelled_any = False
        if peer is not None and peer.attach_socket:
            token = _read_attach_token(peer)
            if token:
                try:
                    client = _AttachRpcClient(peer.attach_socket, token, timeout=5.0)
                    await asyncio.to_thread(client.call, "session/cancel", {})
                    cancelled_any = True
                except Exception as exc:
                    logger.debug(f"im_bot: attach cancel failed: {exc}")

        uid = self._user_id()
        if self._agent_service is not None:
            res = await live_turn.cancel_and_wait(
                self._agent_service, session_id=sid, owner=uid, timeout=5.0,
            )
            if res.get("cancelled") or self._agent_service.cancel_session(sid, owner=uid):
                cancelled_any = True

        q = self._queued_turns.get(sid)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                    cancelled_any = True
                except asyncio.QueueEmpty:
                    break

        if state.has_active_pending() and state.pending.kind == "interaction":
            if state.pending.card_token:
                self._cards.expire(state.pending.card_token, "已取消")
            state.clear_pending()

        if cancelled_any:
            return f"🛑 已发送中止指令（会话 {sid[:8]}）。"
        return f"当前会话（{sid[:8]}）没有正在运行的任务。"

    # ── Turn Execution (Attach CLI or Gateway Local) ───────────────────

    async def _submit_turn(
        self,
        state: ImChatState,
        message: Message,
        *,
        text_override: Optional[str] = None,
    ) -> None:
        # Capture immutable target at submission time (OpenBitFun RemoteBotTarget rule)
        target_work_dir = state.work_dir
        target_session_id = state.session_id
        uid = self._user_id()

        # Fetch inbound media if any
        fetched_media = []
        wanted_media = message.metadata.get("media") or []
        if wanted_media and self._channel_manager is not None:
            ch = self._channel_manager.get_channel(message.channel)
            if ch is not None:
                fetched_media = await ch.fetch_media(message)

        text = (text_override if text_override is not None else (message.content or "")).strip()
        if not text and not fetched_media:
            if wanted_media:
                await self._reply(state, "没能下载这条图片/语音，请再发一次，或配一句文字。")
            return

        if not text and fetched_media:
            kind = fetched_media[0].kind
            text = {
                "image": "请看这条图片。",
                "voice": "请听这条语音。",
                "video": "请看这条视频。",
            }.get(kind, "请查看这条媒体。")

        peer = self.find_live_cli_peer(target_session_id) if not fetched_media else None
        if peer is not None:
            task = asyncio.create_task(
                self._run_attached_cli_turn(state, peer, target_work_dir, target_session_id, text)
            )
            self._turn_tasks[target_session_id] = task
            return

        # Gateway local execution: steer if already running on this session
        if (
            not fetched_media
            and self._agent_service is not None
            and self._agent_service.is_session_active(target_session_id, owner=uid)
        ):
            q = self._queued_turns.get(target_session_id)
            if (q is None or q.empty()) and self._agent_service.steer_session(
                target_session_id, text, owner=uid,
            ):
                logger.info(f"[im_bot] steered follow-up into running session {target_session_id}")
                return

        q = self._queued_turns.get(target_session_id)
        if q is None:
            q = asyncio.Queue()
            self._queued_turns[target_session_id] = q
        await q.put((state, target_work_dir, target_session_id, text, fetched_media))

        worker = self._turn_tasks.get(target_session_id)
        if worker is None or worker.done():
            self._turn_tasks[target_session_id] = asyncio.create_task(
                self._drain_session_queue(target_session_id, q)
            )

    async def _run_attached_cli_turn(
        self,
        state: ImChatState,
        peer: PeerInfo,
        work_dir: str,
        session_id: str,
        text: str,
    ) -> None:
        token = _read_attach_token(peer)
        if not token or not peer.attach_socket:
            # Fallback to local execution if socket disappeared
            q = self._queued_turns.setdefault(session_id, asyncio.Queue())
            await q.put((state, work_dir, session_id, text, []))
            await self._drain_session_queue(session_id, q)
            return

        if self._agent_service is not None:
            uid = self._user_id()
            self._agent_service.set_session_work_dir(session_id, work_dir, owner=uid)
            self._agent_service.session_log_for(session_id, owner=uid).set_source(
                state.channel.value
            )
            live_turn.bump_pulse(uid)

        started = time.monotonic()
        done_event = asyncio.Event()
        short = _short_dir(work_dir)

        async def _ack_if_slow() -> None:
            try:
                await asyncio.wait_for(done_event.wait(), timeout=ACK_DELAY_SECS)
            except asyncio.TimeoutError:
                await self._reply(
                    state,
                    f"⏳ 已发送到终端 [{peer.name}] 执行中，跑完自动汇报（发 /cancel 可中止）",
                )

        async def _watch_interactions() -> None:
            seen_req_ids = set()
            while not done_event.is_set():
                await asyncio.sleep(0.35)
                if done_event.is_set():
                    break
                try:
                    probe = _AttachRpcClient(peer.attach_socket, token, timeout=3.0)
                    snap = await asyncio.to_thread(probe.call, "session/load", {})
                except Exception:
                    continue
                pi = snap.get("pendingInteraction")
                if not isinstance(pi, dict):
                    continue
                req_id = str(pi.get("requestId") or "")
                if not req_id or req_id in seen_req_ids:
                    continue
                seen_req_ids.add(req_id)
                ikind = "approval" if pi.get("kind") == "approval" else "ask"
                raw_opts = [str(o) for o in (pi.get("options") or [])]
                if ikind == "approval":
                    labels = [_APPROVAL_LABELS.get(o, o) for o in raw_opts]
                    values = raw_opts
                    title = "终端请求工具执行权限"
                else:
                    labels = raw_opts
                    values = [str(i) for i in range(1, len(raw_opts) + 1)]
                    title = str(pi.get("prompt") or "终端正在等待你的输入")

                def _resolve_cli(ans: str, rid: str = req_id) -> bool:
                    try:
                        c = _AttachRpcClient(peer.attach_socket, token, timeout=5.0)
                        r = c.call("session/answer", {"answer": ans, "requestId": rid})
                        return bool(r.get("accepted"))
                    except Exception:
                        return False

                await self._publish_interaction(
                    state,
                    interaction_kind=ikind,
                    title=title,
                    text=str(pi.get("prompt") or "") if ikind == "approval" else "",
                    options=labels,
                    values=values,
                    session_id=session_id,
                    request_id=req_id,
                    resolve_fn=_resolve_cli,
                    attach_peer=peer,
                )

        ack_task = asyncio.create_task(_ack_if_slow())
        watch_task = asyncio.create_task(_watch_interactions())
        try:
            client = _AttachRpcClient(peer.attach_socket, token, timeout=620.0)
            result = await asyncio.to_thread(
                client.call,
                "session/prompt",
                {"sessionId": session_id, "prompt": text},
            )
        except Exception as exc:
            await self._reply(state, f"❌ 终端 [{peer.name}] 执行异常：{exc}")
            return
        finally:
            done_event.set()
            ack_task.cancel()
            watch_task.cancel()
            live_turn.bump_pulse(self._user_id())

        elapsed = time.monotonic() - started
        stop_reason = result.get("stopReason")
        answer = (result.get("agenticaAnswer") or "").strip()
        if stop_reason == "cancelled":
            await self._reply(state, f"🛑 终端 [{peer.name}] 任务已取消。")
            return
        if stop_reason == "agentica_pending" and not answer:
            await self._reply(
                state,
                f"⏳ 终端 [{peer.name}] 仍在后台继续执行（可发 /status 查看状态）。",
            )
            return
        if answer:
            header = f"[{short} · {_fmt_elapsed(elapsed)}]\n" if elapsed >= 10.0 or state.verbose else ""
            await self._reply(state, f"{header}{answer}")
            await self._push_deliverable_files(state, answer, work_dir)

    async def _drain_session_queue(self, session_id: str, q: asyncio.Queue) -> None:
        uid = self._user_id()
        carried = None
        while True:
            if carried is not None:
                batch = [carried]
                carried = None
            else:
                if q.empty():
                    return
                batch = [await q.get()]
            while not q.empty():
                batch.append(q.get_nowait())

            state, work_dir, sid, _, _ = batch[-1]
            texts = [item[3] for item in batch if item[3]]
            media = []
            for item in batch:
                media.extend(item[4])
            merged_text = "\n\n".join(texts)

            try:
                await self._run_gateway_turn(state, work_dir, sid, merged_text, media)
            finally:
                for _ in batch:
                    if _ is not carried:
                        try:
                            q.task_done()
                        except ValueError:
                            pass

            if self._agent_service is not None:
                late = self._agent_service.take_undelivered_steer(sid, owner=uid)
                if late:
                    carried = (state, work_dir, sid, "\n\n".join(late), [])
                    continue
            if q.empty():
                return

    async def _run_gateway_turn(
        self,
        state: ImChatState,
        work_dir: str,
        session_id: str,
        text: str,
        media: List[Any],
    ) -> None:
        if self._agent_service is None:
            return
        uid = self._user_id()
        self._agent_service.set_session_work_dir(session_id, work_dir, owner=uid)
        self._agent_service.session_log_for(session_id, owner=uid).set_source(
            state.channel.value
        )

        from ..routes import settings as settings_routes

        mode = settings_routes.approval_mode_pref(uid)
        if mode:
            self._agent_service.set_session_approval_mode(session_id, mode, owner=uid)

        loop = asyncio.get_running_loop()

        def _ask_callback(prompt: str, options: Optional[List[str]] = None) -> str:
            req_id = f"ask-{uuid.uuid4().hex[:8]}"
            fut: asyncio.Future = loop.create_future()
            opts = [str(o) for o in (options or [])]
            vals = opts if opts else []

            def _resolve(ans: str) -> bool:
                if not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, ans)
                    return True
                return False

            asyncio.run_coroutine_threadsafe(
                self._publish_interaction(
                    state,
                    interaction_kind="ask",
                    title=prompt,
                    text="",
                    options=opts,
                    values=vals,
                    session_id=session_id,
                    request_id=req_id,
                    resolve_fn=_resolve,
                ),
                loop,
            )
            try:
                future_concurrent = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(fut, timeout=PENDING_TTL_SECS), loop
                )
                return str(future_concurrent.result(timeout=PENDING_TTL_SECS + 2.0))
            except Exception:
                raise AskUserQuestionTimeout(PENDING_TTL_SECS)

        self._agent_service.set_session_ask_callback(session_id, _ask_callback, owner=uid)

        turn = live_turn.start(session_id, owner=uid, user_message=text)
        sub_q = turn.subscribe()
        done_event = asyncio.Event()
        started = time.monotonic()
        short = _short_dir(work_dir)

        def _on_content(delta: str) -> None:
            turn.publish({"event": "content", "data": delta})

        def _on_tool_call(name: str, args: dict, tool_call_id: str = "") -> None:
            turn.publish({
                "event": "tool_call",
                "data": {"name": name, "args": args, "tool_call_id": tool_call_id},
            })

        def _on_tool_result(name: str, res_str: str, extra: Optional[dict] = None) -> None:
            data = {"name": name, "result": res_str}
            if extra:
                data.update(extra)
            turn.publish({"event": "tool_result", "data": data})

        def _on_thinking(delta: str) -> None:
            turn.publish({"event": "thinking", "data": delta})

        async def _ack_if_slow() -> None:
            try:
                await asyncio.wait_for(done_event.wait(), timeout=ACK_DELAY_SECS)
            except asyncio.TimeoutError:
                await self._reply(
                    state,
                    f"⏳ 已在 [{short}] 启动执行，跑完自动通知你（发 /cancel 可中止）",
                )

        async def _watch_turn_events() -> None:
            while not done_event.is_set():
                try:
                    ev = await asyncio.wait_for(sub_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if ev is None:
                    break
                if ev.get("event") == "approval_request":
                    data = ev.get("data") or {}
                    tc_id = str(data.get("tool_call_id") or "")
                    if not tc_id:
                        continue
                    pending_obj = PendingApproval(
                        tool_call_id=tc_id,
                        name=str(data.get("name") or "tool"),
                        arguments=data.get("args") or {},
                        question=str(data.get("question") or ""),
                        preview=str(data.get("preview") or ""),
                        similar_label=str(data.get("similar_label") or ""),
                        options=tuple(data.get("options") or ("allow", "deny")),
                    )
                    vis_opts = list(visible_approval_decisions(pending_obj.options))
                    labels = [_APPROVAL_LABELS.get(o, o) for o in vis_opts]

                    def _resolve_approval(decision_str: str, tid: str = tc_id) -> bool:
                        d: ApprovalDecision = decision_str  # type: ignore[assignment]
                        return turn.approvals.decide(tid, d)

                    await self._publish_interaction(
                        state,
                        interaction_kind="approval",
                        title=pending_obj.question or f"允许执行 {pending_obj.name}？",
                        text=f"$ {pending_obj.preview}" if pending_obj.preview else "",
                        options=labels,
                        values=vis_opts,
                        session_id=session_id,
                        request_id=tc_id,
                        resolve_fn=_resolve_approval,
                    )

        ack_task = asyncio.create_task(_ack_if_slow())
        watch_task = asyncio.create_task(_watch_turn_events())
        try:
            if hasattr(self._agent_service, "chat_stream"):
                result = await self._agent_service.chat_stream(
                    message=text,
                    session_id=session_id,
                    user_id=uid,
                    owner=uid,
                    on_content=_on_content,
                    on_tool_call=_on_tool_call,
                    on_tool_result=_on_tool_result,
                    on_thinking=_on_thinking,
                    media=media or None,
                    include_usage=False,
                )
            else:
                result = await self._agent_service.chat(
                    message=text,
                    session_id=session_id,
                    user_id=uid,
                    owner=uid,
                    media=media or None,
                )
        except Exception as exc:
            turn.finish("failed")
            await self._reply(state, f"❌ 执行出错：{exc}")
            return
        finally:
            done_event.set()
            turn.unsubscribe(sub_q)
            ack_task.cancel()
            watch_task.cancel()

        elapsed = time.monotonic() - started
        reply = (result.content or "").strip()
        if result.media_notes:
            notes = "\n".join(result.media_notes)
            reply = f"{notes}\n\n{reply}" if reply else notes

        # Send the IM text reply FIRST, before finishing the LiveTurn or pushing files,
        # so the phone receives the answer at the exact moment generation completes.
        if reply:
            header = ""
            if state.verbose or elapsed >= 10.0 or (result.tool_calls and result.tool_calls >= 2):
                tools_part = f" · {result.tool_calls} tools" if result.tool_calls else ""
                header = f"[{short} · {_fmt_elapsed(elapsed)}{tools_part}]\n"
            await self._reply(state, f"{header}{reply}")

        turn.publish({
            "event": "done",
            "data": {
                "session_id": session_id,
                "tool_calls": result.tool_calls,
                "tools_used": result.tools_used,
                "response_time": round(elapsed, 2),
                "turn_usage": result.turn_usage,
            },
        })
        turn.finish("completed")

        if reply:
            await self._push_deliverable_files(state, result.content or "", work_dir)

    async def _push_deliverable_files(
        self,
        state: ImChatState,
        text: str,
        work_dir: str,
    ) -> None:
        if not text or self._channel_manager is None:
            return
        files = collect_auto_push_files(text, work_dir)
        for item in files:
            try:
                data = Path(item.path).read_bytes()
            except OSError:
                continue
            sent = await self._channel_manager.send_file(
                state.channel,
                state.channel_id,
                data=data,
                filename=item.name,
                mime_type=item.mime_type,
            )
            if sent:
                logger.info(
                    f"[im_bot] auto-pushed file {item.name} ({format_file_size(item.size)}) "
                    f"to {state.channel.value}:{state.channel_id}"
                )

    async def _reply(self, state: ImChatState, text: str) -> None:
        if not text or self._channel_manager is None:
            return
        for chunk in Channel.split_text(text, CHUNK_CHARS):
            await self._channel_manager.send(state.channel, state.channel_id, chunk)



