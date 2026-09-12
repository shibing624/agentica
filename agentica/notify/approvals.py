# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Offer a parked approval to the desktop app, and hand the user's
answer back to the loop that is waiting for it.

The desktop app is an **input surface**, not an authority. A ``y`` pressed there
is applied as the user's own answer, with exactly the effect of typing ``y`` in
the terminal for this session and this interaction. The app has no policy of its
own: it never auto-approves, and nothing is ever decided without the user having
said so somewhere.

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
        # ``kind`` distinguishes this from a question, and the two must not be
        # merged even though both are "a human decides something". They differ
        # in the *shape of the reply*: an approval comes back as one of four
        # fixed enum values (``registry.decide(id, "allow"|"deny"|...)``), while
        # a question comes back as an arbitrary string. The desktop app renders
        # them differently for that reason — fixed buttons vs N custom ones —
        # and without ``kind`` it would have to guess from the payload, which is
        # exactly the "infer meaning from body fields" pattern that splitting
        # ``/event`` from ``/await`` by path exists to avoid.
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
    timeout: Optional[float] = None,
) -> None:
    """Offer ``pending`` to the desktop app. Returns immediately.

    The desktop app is an *input surface*, not an authority: a ``y`` there is
    applied as the user's own answer, exactly as if they had typed it in the
    terminal. The app has no policy, never auto-approves, and is never asked
    whether it is "allowed" to answer — if the sink is installed, the user can
    answer from either place.

    Does nothing when no sink is installed, or when there is no registry to
    decide into (the non-interactive paths), which leaves the terminal prompt as
    the only path — the pre-existing behaviour.

    ``timeout`` is the caller's, not ours: it should be the same budget the
    terminal already gives the user, so a desktop answer is not held to a
    stricter clock than a typed one.
    """
    sink = get_sink()
    if sink is None or registry is None:
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
                timeout=timeout,
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
