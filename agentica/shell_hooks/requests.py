# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The blocking ``needs.*`` path — ask the user's command, and let
the terminal race it.

Two semantics are kept from the notify sink, and neither is negotiable:

1. **No wait cap invented here.** A deadline baked into this layer would mean a
   desktop answer got less time than a typed one. A command may set its own
   internal limit; that is the command's business and it is visible in the user's
   own config.
2. **The terminal and the hook race; whoever answers first wins.** The tool call
   is never parked on the hook process — the caller keeps its own parking (the
   registry future for an approval, the prompt slot for a question) and polls
   this while it waits. A reply that arrives second is a race, not an error.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentica.notify.wire import clip_text
from agentica.shell_hooks.egress import get_hook_egress
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload, parse_reply
from agentica.utils.log import logger


def approval_payload(
    pending: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """The metadata slice of a pending approval, on the hook wire.

    ``question`` / ``preview`` come from ``describe_approval`` and are already the
    human-facing summary, so they are the right thing to send. The raw
    ``arguments`` are deliberately NOT sent: a consumer does not need the full
    command, and this channel stays metadata-only.

    ``options`` is ``PendingApproval.options`` — the tuple the tool itself decided
    on (``_approval_options``). It is sent as-is rather than re-narrowed here: a
    consumer that renders exactly what it is given cannot show a button the
    terminal would reject.
    """
    extra: Dict[str, Any] = {"tool_name": getattr(pending, "name", "") or ""}
    tool_call_id = getattr(pending, "tool_call_id", "") or ""
    if tool_call_id:
        extra["tool_call_id"] = tool_call_id
    for field in ("question", "preview", "similar_label"):
        value = getattr(pending, field, None)
        if value:
            extra[field] = str(value)
    options = getattr(pending, "options", None)
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.approval",
        session_id=session_id,
        work_dir=work_dir,
        prompt=prompt,
        extra=extra,
    )


def question_payload(
    prompt: str,
    options: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """A pending question. No decision vocabulary: the reply is free text."""
    extra: Dict[str, Any] = {"question": clip_text(prompt) or ""}
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.input",
        session_id=session_id,
        work_dir=work_dir,
        prompt=clip_text(prompt),
        extra=extra,
    )


class HookRequest:
    """One in-flight ``needs.*`` request to the user's command.

    ``wait_for_reply`` is the whole interface a caller needs: it returns the reply
    when a usable one arrives, and ``None`` while there is nothing usable — which
    means "keep waiting, or answer in the terminal", never "the hook decided no".
    ``kill`` is what a caller does once the answer is no longer wanted.
    """

    def __init__(self, proc: HookProcess, event: str):
        self._proc = proc
        self._event = event

    @property
    def still_waiting(self) -> bool:
        """True while a reply could still arrive (the process is up)."""
        return self._proc.started and not self._proc.finished

    def wait_for_reply(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait up to ``timeout`` for a usable reply from the hook.

        ``timeout`` is the *caller's*, for its own polling convenience; it is not
        a cap on the user. It returns None on timeout and on every unusable reply,
        and the caller's terminal prompt remains the answer path.
        """
        try:
            self._proc.wait(timeout=timeout)
        except Exception as exc:
            logger.debug(f"shell hooks: waiting for {self._event} failed: {exc}")
            return None
        return parse_reply(self._proc.stdout, self._event)

    def kill(self) -> None:
        self._proc.kill()


def start_hook_request(event: str, payload: Dict[str, Any]) -> Optional[HookRequest]:
    """Spawn the user's command for a ``needs.*`` request, or None if there is none.

    Returns immediately. Nothing here blocks the run.
    """
    cfg = get_hook_egress()
    if cfg is None or not cfg.event_enabled(event):
        return None
    try:
        proc = HookProcess(cfg.command, payload)
        if not proc.start():
            return None
        return HookRequest(proc, event)
    except Exception as exc:
        logger.debug(f"shell hooks: could not start {event}: {exc}")
        return None
