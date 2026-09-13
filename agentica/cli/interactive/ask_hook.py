# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Offer a live ask_user_question prompt to the user's hook command.

The terminal's prompt slot is armed first and always checked first, so a typed
answer and a hook answer arriving at the same moment are settled by the slot's
own first-writer-wins rule — the same rule that already settles two typed lines.
The hook is given one short window per watchdog cycle rather than its own thread,
because a question has no registry future to park on: the answer is a string only
the prompt slot can deliver.

Why this replaced the old desktop path: that one called the sink *before* the
terminal prompt and blocked for the reply (``notify/questions.py``), so a desktop
that did not answer locked the terminal out entirely. Arming the terminal first
and polling the hook beside it is the same idea as the approval race, applied to
the one surface that has no future to wait on.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from agentica.utils.log import logger

#: How long each watchdog cycle lets the hook take before control returns to the
#: loop. Small on purpose: a typed answer arriving during this window is noticed
#: one tick late, and the tick is what keeps the prompt responsive.
HOOK_POLL_SECONDS = 0.15


class HookAsk:
    """A hook command offered the question the terminal is showing.

    ``poll`` is called from the terminal callback's watchdog loop, on that loop's
    own thread. It never blocks for longer than ``HOOK_POLL_SECONDS``, and it
    never decides anything: an unusable reply from the hook simply leaves the
    prompt open for the user to answer.
    """

    def __init__(self, request: Any):
        self._request = request

    @property
    def still_useful(self) -> bool:
        """True while a reply could still arrive from the hook."""
        return self._request.still_waiting

    def poll(self, req: Any) -> bool:
        """Give the hook one window; deliver its answer if it produced one.

        Returns True only when an answer reached ``req``'s slot. Free text is the
        whole reply for a question — there is no vocabulary to validate, and the
        only unusable outcome is an empty string, which ``parse_reply`` has
        already rejected by returning None.
        """
        try:
            reply = self._request.wait_for_reply(timeout=HOOK_POLL_SECONDS)
        except Exception as exc:
            # An observation channel must not raise into the prompt loop.
            logger.debug(f"shell hooks: question poll failed: {exc}")
            return False
        if not reply:
            return False
        answer = reply.get("answer")
        if not isinstance(answer, str):
            return False
        if req.submit(answer):
            logger.info("[ask] answered from the hook command")
        # False means the user typed first: their answer is already in the slot,
        # and this one is second.
        return True

    def stop(self) -> None:
        self._request.kill()


def start_hook_ask(
    prompt: str,
    options: Optional[List[str]] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Optional[HookAsk]:
    """Offer ``prompt`` to the user's hook command, or None when there is none."""
    try:
        from agentica.shell_hooks.requests import question_payload, start_hook_request

        request = start_hook_request(
            "needs.input",
            question_payload(
                prompt, options, session_id=session_id, work_dir=work_dir
            ),
        )
        return HookAsk(request) if request is not None else None
    except Exception as exc:
        logger.debug(f"shell hooks: could not offer the question: {exc}")
        return None
