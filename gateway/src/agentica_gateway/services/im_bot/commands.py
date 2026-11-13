# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM command grammar, chat state dataclasses, and formatting helpers.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentica.peers import PeerInfo

from ...channels.base import ChannelType

PENDING_TTL_SECS = 300.0
PENDING_INVALID_LIMIT = 3
SESSION_PAGE_SIZE = 8
WORKSPACE_PAGE_SIZE = 8
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
    if lower_head in ("/status") or trimmed == "状态":
        return ImCommand(kind="status")
    if lower_head in ("/project", "/projects", "/p", "/workspace") or trimmed in (
        "项目", "切换项目", "切换工作区",
    ):
        return ImCommand(kind="project", arg=rest)
    if lower_head in ("/new", "/n", "/new_session") or trimmed in ("新建", "新会话", "新建会话"):
        return ImCommand(kind="new", arg=rest)
    if lower_head in ("/session", "/sessions", "/s") or trimmed in (
        "会话", "切换会话", "会话列表",
    ):
        return ImCommand(kind="session", arg=rest)
    if lower_head in ("/model", "/switch_model") or trimmed == "切换模型":
        return ImCommand(kind="model", arg=rest)
    if lower_head in ("/ps", "/jobs", "/bg") or trimmed in ("后台", "后台任务"):
        return ImCommand(kind="ps", arg=rest)
    if lower_head in ("/stop",) or (head == "停止" and rest):
        if rest:
            return ImCommand(kind="stop_bg", arg=rest)
        return ImCommand(kind="cancel", arg="")
    if lower_head in ("/cancel", "/cancel_task") or trimmed in ("取消", "停止"):
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

    kind: str  # "menu" | "project" | "session" | "session_with_new" | "model" | "interaction"
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
class QueuedTurn:
    """One line waiting for a session's IM worker."""

    state: "ImChatState"
    work_dir: str
    session_id: str
    text: str
    media: List[Any] = field(default_factory=list)
    # A finished background command's report: a turn of its own, logged as
    # automated, never merged with what the user typed.
    automated: bool = False
    # Other chats bound to the same session that also get the report's answer.
    also_notify: List["ImChatState"] = field(default_factory=list)
    from_queue: bool = True
    # A scheduled task's run: the job named on the turn's log entry, and where
    # the scheduler waits for the run's result.
    input_meta: Optional[Dict[str, Any]] = None
    done: Optional["asyncio.Future"] = None


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


def _workspace_labels(paths: List[str]) -> Dict[str, str]:
    """Label each path with its leaf, or ``<parent>/<leaf>`` when nested or colliding."""
    home = os.path.realpath(os.path.expanduser("~"))
    parts = {p: Path(p.rstrip("/")) for p in paths}
    leaf_counts: Dict[str, int] = {}
    for p in parts.values():
        leaf = (p.name or str(p)).lower()
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
    labels: Dict[str, str] = {}
    for raw, p in parts.items():
        leaf = p.name or str(p)
        under_another = any(
            raw != other and other != home and raw.startswith(other.rstrip("/") + os.sep)
            for other in paths
        )
        if (under_another or leaf_counts[leaf.lower()] > 1) and p.parent.name:
            labels[raw] = f"{p.parent.name}/{leaf}"
        else:
            labels[raw] = leaf
    return labels


def _fmt_elapsed(seconds: float) -> str:
    total = max(0, int(seconds + 0.5))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m" if m else f"{h}h"
    if m:
        return f"{m}m{s}s" if s else f"{m}m"
    return f"{s}s"


# The IM command set, rendered both as the user's /help and as the agent's
# instructions (model_factory) so the two cannot disagree.
# (command, Chinese alias, what it does, what it does — for the agent)
IM_COMMANDS: List[Tuple[str, str, str, str]] = [
    ("/menu", "菜单", "打开快捷控制菜单", "numbered control menu"),
    ("/session", "会话", "查看并切换当前项目的会话（含正在运行的终端）",
     "list this project's sessions and live CLI terminals, attach this chat to one"),
    ("/new", "新建", "在当前目录新建会话", "start a new session in the bound directory"),
    ("/project", "项目", "切换本机项目工作区（多会话时继续选择会话）",
     "bind this chat to another project directory and pick its session"),
    ("/model", "", "切换当前会话模型", "pick the config.yaml profile this session runs on"),
    ("/status", "状态", "查看当前目录、会话、模型、实时执行步骤与后台任务",
     "show the bound directory, session, model, live turn progress and background jobs"),
    ("/ps", "后台", "查看后台任务实时进度、最新日志并支持停止",
     "list running and recent background jobs, inspect live output tail, or stop a job"),
    ("/cancel", "取消", "中止当前正在运行的对话轮次（/stop <编号|all> 停止后台任务）",
     "stop the running turn (/stop <id|all> stops background jobs)"),
    ("/verbose", "详细", "汇报附带工具统计与思考摘要", "replies include tool stats and reasoning"),
    ("/concise", "简洁", "只汇报最终结论与产物文件", "replies carry only the final answer and files"),
    ("/help", "帮助", "显示本帮助", "this list"),
]

HELP_TEXT = (
    "📱 Agentica 远控菜单命令：\n"
    + "".join(
        f"  {cmd}{f' ({alias})' if alias else ''}  {desc}\n" for cmd, alias, desc, _ in IM_COMMANDS
    )
    + "\n直接发送任何文字即为向当前绑定会话下发任务；出现编号菜单或审批时直接回复数字即可。"
    + "\n也可以直接问助手「你是什么模型」「改下配置」「装个 skill」「检查更新」，它会用 self_manage 处理。"
)

MENU_COMMANDS = [
    ("/session", "切换当前项目的会话"),
    ("/new", "新建会话"),
    ("/project", "切换项目工作区"),
    ("/model", "切换模型 Profile"),
    ("/status", "查看当前状态与实时步骤"),
    ("/cancel", "停止当前任务"),
    ("/ps", "查看后台任务与实时日志"),
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
