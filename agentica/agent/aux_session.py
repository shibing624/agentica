# -*- coding: utf-8 -*-
"""Auxiliary-task sessions — small, isolated histories for side LLM calls.

Reasonix keeps planner and executor in separate sessions so each maintains its
own stable prompt-cache prefix (internal/agent/coordinator.go). Judge /
compression-style side calls here already never touch the main transcript, but
until now each call was also a fresh two-message request: no prefix to cache
beyond the system prompt.

``AuxSession`` gives one named side task its own bounded message list:

- the system prompt stays the first message and byte-stable;
- each call appends one (user, assistant) exchange — append-only within the
  bound, so the provider's automatic prefix cache keeps hitting the head;
- nothing here is ever aliased to the main conversation object;
- ``reset()`` when the task's identity changes (e.g. a new goal objective).

The bound is in exchanges, not messages: one commit = one user + one assistant.
Trimming drops the OLDEST pair first, which keeps the newest context and (on
repeat overflow) breaks the prefix at a single moving point rather than the
head — the same cache compromise the transcript eviction policy makes.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from agentica.model.message import Message

__all__ = ["AuxSession"]


@dataclass
class AuxSession:
    """A bounded, isolated history for one auxiliary LLM task."""

    purpose: str
    max_exchanges: int = 8
    _messages: List[Message] = field(default_factory=list)

    def context_messages(self) -> List[Message]:
        """Copy of the accumulated exchanges, oldest first."""
        return list(self._messages)

    def build_request(self, system_prompt: str, user_prompt: str) -> List[Message]:
        """system + accumulated exchanges + the new user turn, in wire order."""
        return [
            Message(role="system", content=system_prompt),
            *self.context_messages(),
            Message(role="user", content=user_prompt),
        ]

    def commit(self, user_text: str, assistant_text: str) -> None:
        """Record one exchange and trim the oldest pairs beyond the bound."""
        self._messages.append(Message(role="user", content=user_text))
        self._messages.append(Message(role="assistant", content=assistant_text))
        overflow = len(self._messages) - 2 * self.max_exchanges
        if overflow > 0:
            del self._messages[:overflow]

    def reset(self) -> None:
        """Drop all exchanges (task identity changed; prefix restarts)."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages) // 2
