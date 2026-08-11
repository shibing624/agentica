# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CommandContext, PendingQueue, and concurrent-command constants
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.goals import is_goal_generated_prompt



@dataclass
class CommandContext:
    """Shared context passed to all command handlers.

    Replaces the scattered **kwargs parameter bags with a single,
    type-checkable object.
    """

    agent_config: dict
    current_agent: Any  # Agent instance
    extra_tools: Optional[List] = None
    extra_tool_names: Optional[List[str]] = None
    workspace: Any = None  # Optional[Workspace]
    skills_registry: Any = None
    tui_state: Optional[dict] = None
    pending_queue: Any = None  # PendingQueue
    agent_running: bool = False
    attached_images: Optional[list] = None
    image_counter: Optional[list] = None
    # Background tasks — instance-level, not module-global
    bg_tasks: Dict[str, dict] = field(default_factory=dict)
    bg_task_counter: int = 0
    background_processes: Any = None
    # Cross-session peer channel (agentica.peers.PeerSession). Shared with every
    # rebuilt agent so /resume, /model and friends keep the messaging tools and
    # this terminal's mailbox identity.
    peer_session: Any = None
    # Persistent goal loop (see agentica/goals.py). Same instance is shared
    # between the post-turn hook and /goal handlers, guarded by goal_lock.
    goal_manager: Any = None  # Optional[GoalManager]
    goal_lock: Any = None  # Optional[threading.Lock]
    # Callback the ask_user_question/confirm tools use to read via the TUI input box
    # instead of a blocking input(). Must be preserved across agent rebuilds
    # (/model, /newchat, /reload, …) or those paths reintroduce the deadlock.
    ask_user_question_callback: Any = None
    # TUI-owned callback for opening large, read-only content outside terminal
    # scrollback. Commands return compact inline output and send full history
    # through this callback when the interactive application is available.
    open_pager_callback: Any = None



# ==================== PendingQueue ====================


class PendingQueue:
    """Thread-safe observable queue with list/clear/remove support.

    Each enqueued item is paired with a wall-clock timestamp so the TUI
    queue bar can show when each pending message was submitted.
    """

    def __init__(self):
        self._deque = collections.deque()
        self._timestamps = collections.deque()
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._deque.append(item)
            self._timestamps.append(time.time())

    def get(self, timeout: float = 0.1):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._deque:
                    self._timestamps.popleft()
                    return self._deque.popleft()
            if time.monotonic() >= deadline:
                raise queue.Empty
            time.sleep(0.02)

    def peek_all(self) -> list:
        with self._lock:
            return list(self._deque)

    def peek_all_with_timestamps(self) -> list:
        """Return ``[(item, ts_epoch_seconds), ...]`` snapshot."""
        with self._lock:
            return list(zip(self._deque, self._timestamps))

    def qsize(self) -> int:
        with self._lock:
            return len(self._deque)

    def empty(self) -> bool:
        with self._lock:
            return len(self._deque) == 0

    def clear(self):
        with self._lock:
            self._deque.clear()
            self._timestamps.clear()

    def remove_index(self, idx: int) -> bool:
        with self._lock:
            if 0 <= idx < len(self._deque):
                del self._deque[idx]
                del self._timestamps[idx]
                return True
            return False

    def replace_index(self, idx: int, item) -> bool:
        """Replace the item at ``idx`` in place and refresh its timestamp.

        Returns ``False`` if ``idx`` is out of range. Refreshing the timestamp
        makes the TUI queue bar treat the edit as a re-submission so the
        "x seconds ago" label reflects the latest user intent.
        """
        with self._lock:
            if 0 <= idx < len(self._deque):
                self._deque[idx] = item
                self._timestamps[idx] = time.time()
                return True
            return False

    def insert_index(self, idx: int, item) -> bool:
        """Insert ``item`` at position ``idx`` (0-based).

        ``idx == len(queue)`` is allowed and equivalent to ``put`` (append).
        Returns ``False`` for any other out-of-range index so callers can
        report the error with the same shape as ``remove_index``.
        """
        with self._lock:
            if 0 <= idx <= len(self._deque):
                self._deque.insert(idx, item)
                self._timestamps.insert(idx, time.time())
                return True
            return False



# ==================== Concurrent commands ====================

# Commands that can execute while the agent is streaming (non-blocking).
# Readonly info commands + queue/bg management.
CONCURRENT_CMDS = frozenset(
    {
        "/bg",
        "/background",
        "/ps",
        "/stop",
        "/q",
        "/queue",
        "/steer",
        "/usage",
        "/config",
        "/debug",
        "/history",
        "/help",
        "/tools",
        "/skills",
        "/permissions",
        "/statusbar",
        "/sb",
        "/reasoning",
        "/status",
        "/agents",
        "/agent",
        # /goal and /subgoal: status/pause/clear/list subcommands are concurrent-safe.
        # Handlers reject "set new objective" when agent_running.
        "/goal",
        "/subgoal",
    }
)



# ==================== Helpers ====================


def queue_ahead_of_goal_continuation(pending_queue: PendingQueue, payload) -> None:
    """Enqueue ``payload``, but ahead of any prompt the goal loop queued itself.

    A continuation prompt is written by the goal loop, not typed by the user,
    so letting it go first would spend a whole turn before the agent ever sees
    the correction. ``payload`` is any queue payload (plain text, a tagged
    relay, or a text+images tuple).
    """
    for idx, item in enumerate(pending_queue.peek_all()):
        body = item[0] if isinstance(item, tuple) else item
        if isinstance(body, str) and is_goal_generated_prompt(body):
            if pending_queue.insert_index(idx, payload):
                return
            break
    pending_queue.put(payload)


IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".svg",
        ".ico",
    }
)
