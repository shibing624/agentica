# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IM Bot controller package for remote-controlling local Agentica sessions.

- Each IM chat binds to a ``(work_dir, session_id)`` on this machine, sharing
  the exact same ``SessionLog`` JSONL as the local CLI and Web UI.
- Commands (``/menu``, ``/project``, ``/session``, ``/new``, ``/model``,
  ``/cancel``, ``/status``, ``/ps``, ``/verbose``, ``/concise``, ``/help``)
  and numeric menu replies (``1..N``, ``0``) manage the bound project and session.
- Free-form text submits a turn to the bound session (live CLI attach when a
  terminal holds it, otherwise ``AgentService`` against the shared ``SessionLog``).
- Pending approvals and ``ask_user_question`` prompts are delivered both as
  numbered IM options and as a zero-JS capability-URL card (``/c/<token>``).
"""
from __future__ import annotations

from agentica.peers import send_message as send_peer_message

from ...config import settings
from ..agent_service import ChatResult, is_system_temp_path
from .attach import _AttachRpcClient, _read_attach_token
from .commands import (
    ACK_DELAY_SECS,
    CHUNK_CHARS,
    HELP_TEXT,
    IM_COMMANDS,
    MENU_COMMANDS,
    PENDING_INVALID_LIMIT,
    PENDING_TTL_SECS,
    SESSION_PAGE_SIZE,
    WORKSPACE_PAGE_SIZE,
    ImChatState,
    ImCommand,
    PendingAction,
    QueuedTurn,
    _APPROVAL_LABELS,
    _SOURCE_LABELS,
    _fmt_elapsed,
    _short_dir,
    _workspace_labels,
    normalize_im_command_text,
    parse_im_command,
    strip_numeric_reply_suffix,
)
from .controller import ImBotController

__all__ = [
    "ACK_DELAY_SECS",
    "CHUNK_CHARS",
    "ChatResult",
    "HELP_TEXT",
    "IM_COMMANDS",
    "ImBotController",
    "ImChatState",
    "ImCommand",
    "MENU_COMMANDS",
    "PENDING_INVALID_LIMIT",
    "PENDING_TTL_SECS",
    "PendingAction",
    "QueuedTurn",
    "SESSION_PAGE_SIZE",
    "WORKSPACE_PAGE_SIZE",
    "_APPROVAL_LABELS",
    "_AttachRpcClient",
    "_SOURCE_LABELS",
    "_fmt_elapsed",
    "_read_attach_token",
    "_short_dir",
    "_workspace_labels",
    "is_system_temp_path",
    "normalize_im_command_text",
    "parse_im_command",
    "send_peer_message",
    "settings",
    "strip_numeric_reply_suffix",
]
