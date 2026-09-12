# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Offer a parked approval to the desktop app, and hand its answer
back to the loop that is waiting for it.

The terminal prompt and the desktop app race on purpose: whoever answers first
wins, and the loser gets ``decide() -> False``, which is a normal outcome rather
than an error (the user may have hit y a moment before the button was clicked).

The blocking wait happens on its own daemon thread. It cannot run on the event
loop: the loop's thread is inside ``await waiter`` at that moment, so a blocking
call there would deadlock the very approval it is trying to resolve. Because it
is a different thread, the answer is published back with
``call_soon_threadsafe`` — resolving an asyncio future from another thread is
not safe.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from agentica.notify.sink import get_sink
from agentica.utils.log import logger


def _approval_payload(pending: Any) -> dict:
    """The metadata slice of a pending approval, per the contract.

    ``question`` / ``preview`` come from ``describe_approval`` and are already
    the human-facing summary, so they are the right thing to send. The raw
    ``arguments`` are deliberately NOT sent: the desktop app does not need the
    full command, and this channel stays metadata-only by design.
    """
    payload = {
        "approval_id": getattr(pending, "tool_call_id", "") or "",
        "kind": "permission",
        "tool": getattr(pending, "name", "") or "",
    }
    for field in ("question", "preview", "similar_label"):
        value = getattr(pending, field, None)
        if value:
            payload[field] = str(value)
    options = getattr(pending, "options", None)
    if options:
        payload["options"] = [str(o) for o in options]
    return payload


def publish_approval(
    pending: Any,
    registry: Any,
    loop: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> None:
    """Ask the desktop app about ``pending``. Returns immediately.

    Does nothing at all when the sink is absent, disabled, or not allowed to
    decide — in which case the terminal prompt is the only path, which is the
    pre-existing behaviour.
    """
    sink = get_sink()
    if sink is None or registry is None:
        return
    if not sink.config.approve_from_desktop:
        # Switch off = the app is still told a decision is pending (a /event
        # goes out below), but it can never answer.
        sink.emit_event(
            "needs.approval",
            session_id=session_id,
            work_dir=work_dir,
            payload=_approval_payload(pending),
        )
        return

    payload = _approval_payload(pending)
    tool_call_id = payload["approval_id"]
    if not tool_call_id:
        # Without a correlation id the answer could not be applied to anything.
        logger.debug("notify sink: pending approval has no tool_call_id; skipping")
        return

    def _wait_then_decide() -> None:
        try:
            result = sink.await_decision(
                "needs.approval",
                payload=payload,
                session_id=session_id,
                work_dir=work_dir,
            )
        except Exception as exc:
            # Belt and braces: await_decision already swallows everything, but
            # an observation channel must not be able to kill this thread with
            # an exception nobody is waiting to see.
            logger.debug(f"notify sink: approval wait failed: {exc}")
            return
        if not result:
            return  # no decision: the terminal prompt is still the answer path
        decision = result.get("decision")
        if not isinstance(decision, str):
            # An "answer" makes no sense for an approval; ignore rather than guess.
            return
        try:
            loop.call_soon_threadsafe(_apply, registry, tool_call_id, decision)
        except Exception as exc:
            # The loop can already be gone (a cancelled turn). Dropping the
            # decision is correct: nobody is waiting for it any more.
            logger.debug(f"notify sink: could not hand back approval decision: {exc}")

    def _apply(reg: Any, call_id: str, decision: str) -> None:
        # False means the id is unknown or was already decided — normally the
        # user answered in the terminal first. That is a race, not an error.
        applied = reg.decide(call_id, decision)
        if not applied:
            logger.debug(
                f"notify sink: approval {call_id} was already decided; "
                f"the desktop answer arrived second"
            )

    threading.Thread(
        target=_wait_then_decide,
        name="agentica-notify-approval",
        daemon=True,
    ).start()
