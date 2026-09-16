# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The blocking ``needs.*`` path — ask every subscribed consumer,
and let the terminal race them.

Two semantics are kept from the notify sink, and neither is negotiable:

1. **No wait cap invented here.** A deadline baked into this layer would mean a
   desktop answer got less time than a typed one. A command may set its own
   internal limit; that is the command's business and it is visible in the user's
   own config.
2. **The terminal and every hook race; the first valid answer wins.** The tool call
   is never parked on a hook process — the caller keeps its own parking (the
   registry future for an approval, the prompt slot for a question) and polls
   this while it waits. A reply that arrives second is a race, not an error.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agentica.notify.wire import clip_text
from agentica.shell_hooks.egress import ensure_hook_egress_installed
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import build_payload, parse_reply
from agentica.utils.log import logger


def approval_payload(
    pending: Any,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    prompt: Optional[str] = None,
    request_id: Optional[str] = None,
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
    extra: Dict[str, Any] = {"tool_name": pending.name or ""}
    tool_call_id = pending.tool_call_id or ""
    if tool_call_id:
        extra["tool_call_id"] = tool_call_id
    for field, value in (
        ("question", pending.question),
        ("preview", pending.preview),
        ("similar_label", pending.similar_label),
    ):
        if value:
            extra[field] = str(value)
    options = pending.options
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.approval",
        session_id=session_id,
        work_dir=work_dir,
        request_id=request_id or str(uuid4()),
        prompt=prompt,
        extra=extra,
    )


def question_payload(
    prompt: str,
    options: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """A pending question. No decision vocabulary: the reply is free text."""
    extra: Dict[str, Any] = {"question": clip_text(prompt) or ""}
    if options:
        extra["options"] = [str(o) for o in options]
    return build_payload(
        "needs.input",
        session_id=session_id,
        work_dir=work_dir,
        request_id=request_id or str(uuid4()),
        prompt=clip_text(prompt),
        extra=extra,
    )


class HookRequest:
    """One in-flight ``needs.*`` request raced across all subscribed consumers.

    ``wait_for_reply`` is the whole interface a caller needs: it returns the reply
    when a usable one arrives, and ``None`` while there is nothing usable — which
    means "keep waiting, or answer in the terminal", never "the hook decided no".
    ``kill`` is what a caller does once the answer is no longer wanted.
    """

    def __init__(self, processes: List[HookProcess], event: str, request_id: str):
        self._processes = processes
        self._event = event
        self.request_id = request_id
        self._checked: set[int] = set()
        self._killed = False

    @property
    def still_waiting(self) -> bool:
        """True while at least one consumer could still produce a reply."""
        return not self._killed and any(
            index not in self._checked and process.started
            for index, process in enumerate(self._processes)
        )

    def wait_for_reply(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait up to ``timeout`` for a usable reply from the hook.

        ``timeout`` is the *caller's*, for its own polling convenience; it is not
        a cap on the user. It returns None on timeout and on every unusable reply,
        and the caller's terminal prompt remains the answer path.
        """
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0)
        while True:
            finished = [
                (index, process)
                for index, process in enumerate(self._processes)
                if index not in self._checked and process.finished
            ]
            finished.sort(key=lambda item: item[1].completed_at)
            for index, process in finished:
                self._checked.add(index)
                # The document is the decision. Exit status is not consulted:
                # a wrapper may print JSON and keep running, and a process that
                # already answered may then exit non-zero. Gating on returncode
                # raced with early JSON completion and made the same script
                # randomly allow or ignore.
                reply = parse_reply(
                    process.stdout,
                    self._event,
                    request_id=self.request_id,
                )
                # The complete reply is already buffered. End the process group
                # even when a wrapper kept stdout open or spawned descendants.
                process.kill()
                if reply is not None:
                    self._kill_except(index)
                    return reply
            if not self.still_waiting:
                return None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(0.01, remaining))
            else:
                time.sleep(0.01)

    def kill(self) -> None:
        self._killed = True
        for process in self._processes:
            process.kill()

    def _kill_except(self, winner: int) -> None:
        for index, process in enumerate(self._processes):
            if index != winner:
                process.kill()
                self._checked.add(index)


def start_hook_request(event: str, payload: Dict[str, Any]) -> Optional[HookRequest]:
    """Spawn subscribed consumers for a ``needs.*`` request, or return None.

    Returns immediately. Nothing here blocks the run.
    """
    cfg = ensure_hook_egress_installed()
    if cfg is None:
        return None
    try:
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            return None
        processes = []
        for consumer in cfg.consumers_for(event):
            process = HookProcess(consumer.command, payload)
            if process.start():
                processes.append(process)
        if not processes:
            return None
        return HookRequest(processes, event, request_id)
    except Exception as exc:
        logger.debug(f"shell hooks: could not start {event}: {exc}")
        return None
