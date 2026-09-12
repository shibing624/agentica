# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Offer a pending question to the desktop app; fall back to the
terminal when it does not answer.

Smaller than the approval path in one way and subtler in another. It resolves to
a *string* rather than a typed decision (the ask callback is
``(prompt, options) -> str``), so there is no decision vocabulary to validate —
only "did we get a usable non-empty answer". And it blocks the caller for up to
the timeout, which is fine here because the ask callback already blocks: its
terminal implementation parks on a queue until the user types.

Falls back on every non-answer: disabled, not permitted, no socket, timeout,
HTTP error, unparseable body, empty string.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from agentica.notify.sink import get_sink
from agentica.utils.log import logger


def ask_via_desktop(
    prompt: str,
    options: Optional[List[str]] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Optional[str]:
    """Ask the desktop app. ``None`` means "use the terminal instead".

    Never raises: the caller's fallback is the terminal prompt, and an exception
    here would skip straight past it.
    """
    try:
        sink = get_sink()
        if sink is None:
            return None
        # ``kind`` tells the desktop app this is not an approval: the reply is
        # free text, not one of the approval enum values, and ``options`` may be
        # an arbitrary list rather than allow/deny. See the note in
        # ``approvals.py`` for why the two are kept apart on the wire.
        payload: dict = {"kind": "question", "question": str(prompt)}
        if options:
            payload["options"] = [str(o) for o in options]
        result = sink.await_decision(
            "needs.input",
            payload=payload,
            session_id=session_id,
            work_dir=work_dir,
        )
        if not result:
            return None
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
        return None
    except Exception as exc:
        logger.debug(f"notify sink: desktop question failed: {exc}")
        return None


def wrap_ask_callback(
    inner: Callable[..., str],
    *,
    session_id_getter: Optional[Callable[[], Optional[str]]] = None,
    work_dir_getter: Optional[Callable[[], Optional[str]]] = None,
) -> Callable[..., str]:
    """Wrap an ask callback so the desktop app gets first refusal.

    The terminal implementation is called unchanged when the desktop app does
    not answer, so the pre-existing behaviour is the fallback rather than
    something to be reimplemented here.
    """

    def bridging(prompt: str, options: Optional[List[str]] = None) -> str:
        session_id = session_id_getter() if session_id_getter else None
        work_dir = work_dir_getter() if work_dir_getter else None
        answer = ask_via_desktop(
            prompt, options, session_id=session_id, work_dir=work_dir
        )
        if answer is not None:
            # Logged because the terminal never saw this question: an answer
            # that appears from nowhere must still be traceable in the log.
            logger.info("[ask] answered from the desktop app")
            return answer
        return inner(prompt, options)

    return bridging
