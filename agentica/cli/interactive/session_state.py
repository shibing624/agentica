# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SessionState and input-request types for the interactive TUI
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

from agentica.agent.approvals import ApprovalRegistry, SessionGrants
from agentica.goals import GoalManager
from agentica.tools.background_processes import BackgroundProcessRegistry

# ==================== SessionState ====================


@dataclass
class _InputRequest:
    """A pending ask_user_question tool request awaiting a typed reply.

    Created by the ask_user_question_callback on the background agent thread; the main
    prompt_toolkit thread fulfils it by putting the user's line on ``result``.
    Putting the ``CANCELLED`` sentinel unblocks the agent thread so it can raise
    :class:`AgentCancelledError` — this is how Ctrl+C escapes a pending prompt.
    """

    CANCELLED: ClassVar[object] = object()

    prompt: str
    options: Optional[List[str]] = None
    result: "queue.Queue" = field(default_factory=lambda: queue.Queue(maxsize=1))
    resolved: bool = False
    # "ask" is ask_user_question (typed line + Enter). "approval" is the
    # Codex y/p/esc tool-approval prompt; the agent thread waits on the
    # ApprovalRegistry future, not on ``result``.
    kind: str = "ask"
    approval_id: Optional[str] = None
    approval_loop: Any = None
    approval_registry: Any = None
    approval_pending: Any = None

    def submit(self, answer: str) -> bool:
        """Deliver the user's answer exactly once.

        Returns ``True`` only when this call won the race to resolve the
        request. Late submissions after cancel/submit are ignored.
        """
        if self.resolved:
            return False
        try:
            self.result.put_nowait(answer)
            self.resolved = True
            return True
        except queue.Full:
            self.resolved = True
            return False

    def cancel(self) -> bool:
        """Wake up the blocked agent thread and tell it the user aborted."""
        if self.resolved:
            return False
        try:
            self.result.put_nowait(_InputRequest.CANCELLED)
            self.resolved = True
            return True
        except queue.Full:
            # Someone already answered — nothing to unblock.
            self.resolved = True
            return False


@dataclass
class SessionState:
    """All mutable session state in one place.

    Replaces the scattered single-element list containers
    (``[False]``, ``[0]``, ``[agent]``) with typed fields.
    """

    should_exit: bool = False
    agent_running: bool = False
    current_agent: Any = None
    image_counter: int = 0
    paste_counter: int = 0
    attached_images: List = field(default_factory=list)
    pasted_files: List = field(default_factory=list)
    last_ctrl_c: float = 0.0
    # Background tasks — owned by session, not module-global
    bg_tasks: Dict[str, dict] = field(default_factory=dict)
    bg_task_counter: int = 0
    background_processes: BackgroundProcessRegistry = field(default_factory=BackgroundProcessRegistry)
    # This terminal's end of the cross-session peer channel (agentica/peers.py).
    peer_session: Any = None
    # Standing-goal loop (see agentica/goals.py).
    goal_manager: Optional[GoalManager] = None
    goal_lock: threading.Lock = field(default_factory=threading.Lock)
    # Token + wall-clock baselines for per-turn budget accounting (S2).
    # A fresh CostTracker is created per Agent.run(), so each turn's
    # cost_tracker holds THIS turn's tokens only. extract_turn_signals
    # takes that as the delta and accumulates it onto this running baseline.
    goal_tokens_baseline: int = 0
    # Active ask_user_question tool request. When the agent (running in the background
    # process_loop thread) calls the ask_user_question tool, it parks on a result queue
    # and sets this field so the main prompt_toolkit thread routes the next typed
    # line into the queue instead of pending_queue. None when no request pending.
    # ``kind="approval"`` is the same slot used for Codex y/p/esc tool approval.
    input_request: Optional["_InputRequest"] = None
    # Session-scoped tool-approval memory. Prefix ("allow similar") grants
    # load from / save to this project's ``project.json``; allow-once stays
    # in process. Shared across /model rebuilds.
    approval_registry: ApprovalRegistry = field(default_factory=ApprovalRegistry)
    approval_grants: SessionGrants = field(default_factory=SessionGrants)
    approval_loop: Any = None
    # Serialises parked CLI cards. Created on the running loop.
    approval_prompt_lock: Any = None
    # Cron scheduler daemon thread (started when settings cron.enabled is true).
    cron_thread: Optional[threading.Thread] = None
    cron_stop_event: Optional[threading.Event] = None


__all__ = ['_InputRequest', 'SessionState']
