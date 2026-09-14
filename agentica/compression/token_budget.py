# -*- coding: utf-8 -*-
"""Model-visible context-window budget (Codex TokenBudget, post-#27438).

#27438 injected remaining tokens at 25/50/75% usage crossings. Later Codex
replaced that with a once-per-window remaining-token reminder plus a
full-window identity block on each new window. Fragments are separate
user messages, never folded into the frozen system prefix — a per-turn
oil-gauge in the system message would reprice the whole conversation.
"""
from typing import Optional

from agentica.model.message import Message


CONTEXT_WINDOW_OPEN = "<context_window>\n"
CONTEXT_WINDOW_CLOSE = "\n</context_window>"

# Idle /compact leaves this mark so the next user turn can fold the preamble
# into that request (avoids consecutive user roles on Bedrock / some gateways).
WINDOW_CONTINUATION_MARK = (
    "New context window started. "
    "Continue from session notes and search_session."
)

# Reminder fires once per window when remaining tokens drop to this share
# of the working window (Codex: reminder_threshold_tokens; we derive it).
REMINDER_REMAINING_RATIO = 0.25


def reminder_threshold(working_window: int) -> int:
    """Tokens remaining at which the once-per-window reminder fires."""
    if working_window <= 0:
        return 0
    return max(1, int(working_window * REMINDER_REMAINING_RATIO))


def tokens_remaining(context_tokens: int, working_window: int) -> int:
    if working_window <= 0:
        return 0
    return max(0, working_window - max(0, context_tokens))


def full_window_text(
    window_id: int,
    tokens_left: int,
    notes_path: Optional[str] = None,
) -> str:
    """Full-context metadata for a freshly opened window.

    Only advertises the recovery paths that exist. Without a session log there
    is no notes file to keep current and ``search_session`` has nothing to
    read (``context_tool`` answers "No session log on this agent"), so naming
    either one sends the model after a tool that cannot help. What remains is
    the carried digest, which is self-contained.

    A reset is stated only when there is something to recover from (Codex
    #29256 carries ``Previous context window id`` for the same reason). Every
    fragment used to say only ``Current context window N``, so after a cut the
    model could not tell a first window from a fifth and read the recovery
    advice below as boilerplate. ``window_id`` is per-process — it restarts at
    1 on resume, where the turns are *not* gone (they replay from the log), so
    staying quiet there is correct as well.
    """
    lines = [f"Current context window {window_id}."]
    if window_id > 1 and notes_path:
        lines.append(
            "This window follows a context reset: the earlier turns are not here."
        )
    lines.append(f"You have {tokens_left} tokens left in this context window.")
    if notes_path:
        lines.append("Recover prior facts with search_session.")
        lines.append(
            "Keep the session notes file current: goals, constraints, IDs, decisions."
        )
        lines.append(f"Session notes: {notes_path}")
    return CONTEXT_WINDOW_OPEN + "\n".join(lines) + CONTEXT_WINDOW_CLOSE


def remaining_text(tokens_left: int, notes_path: Optional[str] = None) -> str:
    """Threshold reminder — remaining tokens only."""
    lines = [f"You have {tokens_left} tokens left in this context window."]
    if notes_path:
        lines.append(
            f"Update session notes ({notes_path}) before the window rolls over."
        )
    return CONTEXT_WINDOW_OPEN + "\n".join(lines) + CONTEXT_WINDOW_CLOSE


def fallback_text(notes_path: Optional[str] = None) -> str:
    """Codex auto-compact fallback: write notes now, window is about to reset.

    Returns "" when there is no notes path — there is nothing to ask for, and
    the caller must not postpone a cut for an unactionable instruction.

    Deliberately does *not* borrow Codex's "do not continue the task" clause.
    The runner folds this into the current user turn and auto-compact keeps
    that turn, so the text outlives the cut and would land in the new window
    contradicting the fresh window's own instructions. Codex gets away with
    it because its fallback is a developer item the rollover consumes.
    """
    if not notes_path:
        return ""
    lines = [
        "This context window is about to reset.",
        "Write goals, constraints, IDs, paths, and decisions to the "
        "session notes file now — the next window starts without these turns.",
        f"Session notes: {notes_path}",
    ]
    return CONTEXT_WINDOW_OPEN + "\n".join(lines) + CONTEXT_WINDOW_CLOSE


def window_message(
    window_id: int,
    tokens_left: int,
    notes_path: Optional[str] = None,
) -> Message:
    return Message(
        role="user",
        content=full_window_text(window_id, tokens_left, notes_path),
    )


def remaining_message(tokens_left: int, notes_path: Optional[str] = None) -> Message:
    return Message(
        role="user",
        content=remaining_text(tokens_left, notes_path),
    )


def is_context_window_message(message: Message) -> bool:
    if message.role != "user":
        return False
    content = message.content
    if not isinstance(content, str):
        return False
    return content.startswith(CONTEXT_WINDOW_OPEN)


def is_pending_window_preamble(message: Message) -> bool:
    """True for the idle-/compact placeholder that must ride the next request."""
    if not is_context_window_message(message):
        return False
    return WINDOW_CONTINUATION_MARK in message.content


def fold_window_preamble(message: Message, preamble: str) -> None:
    """Put the new-window prompt on the next user turn, in place."""
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    message.content = f"{preamble}\n\n{content}" if content.strip() else preamble
