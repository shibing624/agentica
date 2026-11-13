# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Dispatch identity ("ticket") shared by every execution substrate.

A *ticket* names one unit of work as it crosses a process or runtime boundary:
the turn that started it, the ``background/<id>.json`` record of the shell
command or delegated worker it started, the environment of that child process,
the cron job that fired it. Today each of those carries its own id
(``BackgroundProcess.id``, ``RunResponse.run_id``, ``session_id``, a subagent
id) and nothing ties them together, so "the same job" has no name that survives
a substrate change. This module gives it one.

It is a name, not a state machine. Nothing in the runtime branches on it, so an
agent that never mentions it behaves exactly as it did before. Two rules keep it
from becoming a fourth copy of something:

1. **Minted once, at the dispatch boundary**, then only inherited. The runner
   binds it at the start of a turn; a child run, a background command started by
   that turn, a delegated process and a cron run all *inherit* the value rather
   than minting a new one. Inheritance is what makes the record chain readable.
2. **Never a source of truth for anything.** Liveness, exit status and delivery
   are owned by the ``background/*.json`` record and the session log. A ticket
   groups those records; it does not summarise them.

**Absent means unmanaged.** Records written before this module existed have no
``ticket_id``, and every reader must treat a missing one as "not tracked"
rather than minting a replacement — a made-up id is worse than no id, because
it silently claims a relationship that does not exist.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional, Tuple
from uuid import uuid4

#: Environment variable a child process reads to inherit its parent's dispatch.
#: This is how `delegate` links a worker to the turn that started it: the OS
#: process and its environment are the only channel that survives exec.
TICKET_ENV = "AGENTICA_TICKET_ID"

_current: ContextVar[Optional[str]] = ContextVar("agentica_ticket_id", default=None)


def new_ticket_id() -> str:
    """Mint a fresh ticket id.

    Short and log-friendly on purpose: it appears in ``background/*.json``, in
    session sidecars and in a delegated worker's environment, so it has to be
    something a human can compare by eye across two files.
    """
    return f"tk_{uuid4().hex[:12]}"


def normalize_ticket_id(value: object) -> Optional[str]:
    """Return ``value`` as a ticket id, or ``None`` when it is not one.

    Readers use this instead of ``if rec["ticket_id"]`` so a malformed or
    non-string field degrades to "unmanaged" rather than crashing a view.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def current_ticket_id() -> Optional[str]:
    """The ticket this turn belongs to, or ``None`` when nothing is bound.

    The environment fallback is what makes a delegated child inherit: it is
    started with ``AGENTICA_TICKET_ID`` set, so its very first turn binds the
    parent's ticket without anyone passing a parameter through ``--query``.
    """
    bound = _current.get()
    if bound:
        return bound
    return normalize_ticket_id(os.environ.get(TICKET_ENV))


def set_ticket_id(ticket_id: Optional[str]) -> Token:
    """Bind ``ticket_id`` as current; returns the token ``reset_ticket_id`` needs."""
    return _current.set(normalize_ticket_id(ticket_id))


def reset_ticket_id(token: Token) -> None:
    """Undo :func:`set_ticket_id`. Safe to call when nothing was set."""
    try:
        _current.reset(token)
    except ValueError:
        # A different context (a thread, or a task that outlived its parent's
        # turn) cannot reset another context's token. Leaving the value bound
        # would NOT be harmless: the next dispatch in this context resolves
        # ``current_ticket_id()`` first and would *inherit* a ticket belonging to
        # work that has ended. Clearing this context's value is the only thing
        # available, and it keeps a finished dispatch from naming the next one.
        _current.set(None)


def resolve_ticket_id(ticket_id: Optional[str] = None) -> str:
    """The ticket this dispatch gets, without binding it.

    Order: an explicit id wins, then whatever this context already inherited
    (a subagent inside a parent turn, a delegated child reading its env), then a
    fresh one. Every dispatch ends up with a value, which is what lets a tool
    running inside the turn stamp it onto the record it writes.
    """
    return normalize_ticket_id(ticket_id) or current_ticket_id() or new_ticket_id()


def enter_ticket(ticket_id: Optional[str] = None) -> Tuple[str, Token]:
    """Resolve *and* bind, returning ``(ticket_id, token)`` in one step.

    The pair is the point. ``set_ticket_id(bind_ticket(...))`` looks equivalent
    and is not: ``bind_ticket`` already set the value, so the token it produces
    restores *that* value, and the ``reset`` silently keeps the ticket bound for
    every later turn in the process. Resolving first makes the token restore
    whatever was there before this dispatch.
    """
    resolved = resolve_ticket_id(ticket_id)
    return resolved, _current.set(resolved)


def bind_ticket(ticket_id: Optional[str] = None) -> str:
    """Bind the ticket for a dispatch and return it, when the caller has no token to keep.

    The runner uses :func:`enter_ticket` instead: it needs the token so the
    ticket cannot leak past the turn.
    """
    resolved = resolve_ticket_id(ticket_id)
    set_ticket_id(resolved)
    return resolved


@contextmanager
def ticket_scope(ticket_id: Optional[str]) -> Iterator[Optional[str]]:
    """Run a block with ``ticket_id`` bound, restoring the previous value.

    For work that leaves the turn's context on purpose — the ``/background``
    thread is the canonical case, because a ``contextvars`` value does not
    follow a ``threading.Thread``. Without this the thread's record would be
    unmanaged even though its parent turn had a ticket.
    """
    token = set_ticket_id(ticket_id)
    try:
        yield normalize_ticket_id(ticket_id)
    finally:
        reset_ticket_id(token)


__all__ = [
    "TICKET_ENV",
    "bind_ticket",
    "enter_ticket",
    "resolve_ticket_id",
    "current_ticket_id",
    "new_ticket_id",
    "normalize_ticket_id",
    "reset_ticket_id",
    "set_ticket_id",
    "ticket_scope",
]
