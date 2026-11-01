# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The ``UserPromptSubmit`` hook: each turn's message may carry
context produced by the user's own commands.

The wire is Claude Code's, field for field, because that shape is what the
ecosystem already speaks (Codex and CodeBuddy copied it), so a hook script
written for any of them runs here unchanged:

* stdin — one JSON document: ``session_id``, ``transcript_path``, ``cwd``,
  ``hook_event_name`` and the **full** ``prompt``.
* stdout — ``hookSpecificOutput.additionalContext``, or plain text.
* exit code ``2`` — refuse the prompt; stderr is the reason.

This is not how agentica's own memory recall travels. Recall is built in
(``<memory_recall>`` / ``<session_recall>``) and runs in the same place, so a
consumer here adds context *beside* it and can never switch it off.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentica.notify.transport import build_transport
from agentica.guardrails.agent import InputGuardrailTripwireTriggered
from agentica.guardrails.core import GuardrailOutput
from agentica.shell_hooks.config import ShellHooksConfig
from agentica.shell_hooks.egress import ensure_hook_egress_installed
from agentica.shell_hooks.process import HookProcess
from agentica.shell_hooks.protocol import PROCESS_SESSION_ID, first_json_document

PROMPT_EVENT = "UserPromptSubmit"


class PromptBlocked(InputGuardrailTripwireTriggered):
    """A prompt a hook refused; ``reason`` is what to show the user.

    It is an input guardrail tripping, and is handled as one: the turn ends
    before any model call and is not recorded as a failed run.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(PROMPT_EVENT, GuardrailOutput.block(reason))

    def __str__(self) -> str:
        return f"{PROMPT_EVENT} hook blocked the prompt: {self.reason}"


def prompt_payload(
    query: str,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    transcript_path: Optional[str] = None,
) -> Dict[str, Any]:
    """The stdin document, in Claude Code's ``UserPromptSubmit`` shape.

    ``prompt`` is **not** clipped here, unlike the notice wire: a memory
    consumer searches with the user's actual words, and a truncated question
    recalls the wrong thing.
    """
    transport = build_transport(work_dir)
    doc: Dict[str, Any] = {
        "hook_event_name": PROMPT_EVENT,
        "session_id": str(session_id or PROCESS_SESSION_ID),
    }
    if transport["cwd"]:
        doc["cwd"] = str(transport["cwd"])
    if transcript_path:
        doc["transcript_path"] = str(transcript_path)
    doc["prompt"] = query
    doc["transport"] = transport
    return doc


def parse_prompt_reply(
    stdout: Optional[str],
    *,
    returncode: Optional[int],
    stderr: str = "",
) -> tuple[Optional[str], Optional[str]]:
    """``(context, refusal)`` from one consumer. Both are None for "nothing to say".

    An unreadable reply is no reply — never a refusal. The one asymmetry kept
    on purpose: a refusal spelled ``continue: false`` or ``decision: "block"``
    is honoured, because silently ignoring a script that meant to stop the turn
    is the failure this channel must not have.
    """
    if returncode == 2:
        return None, (stderr.strip() or "the prompt hook refused this message")
    if returncode != 0:
        # Any other failure is a non-blocking error, as in Claude Code: the
        # turn goes ahead and the hook's output is not trusted as context.
        return None, None
    body = first_json_document(stdout)
    if isinstance(body, dict):
        output = body.get("hookSpecificOutput")
        if isinstance(output, dict):
            context = output.get("additionalContext")
            if isinstance(context, str) and context.strip():
                return context.strip(), None
        if body.get("decision") == "block" or body.get("continue") is False:
            reason = body.get("reason")
            text = reason.strip() if isinstance(reason, str) else ""
            return None, (text or "the prompt hook refused this message")
        return None, None
    text = (stdout or "").strip()
    return (text, None) if text else (None, None)


def _remaining(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else max(deadline - time.monotonic(), 0)


def collect_prompt_context(
    query: str,
    *,
    session_id: Optional[str] = None,
    work_dir: Optional[str] = None,
    transcript_path: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[ShellHooksConfig] = None,
) -> Optional[str]:
    """Context every ``UserPromptSubmit`` consumer adds, joined in config order.

    Subscribers run in parallel. ``timeout`` is the caller's budget for this
    turn's recall: a consumer still running at the deadline is killed and
    contributes nothing, so a broken hook costs the turn a fixed few seconds
    instead of hanging it.
    """
    cfg = config if config is not None else ensure_hook_egress_installed()
    if cfg is None or not query.strip():
        return None
    consumers = cfg.consumers_for(PROMPT_EVENT)
    if not consumers:
        return None
    payload = prompt_payload(
        query,
        session_id=session_id,
        work_dir=work_dir,
        transcript_path=transcript_path,
    )
    deadline = time.monotonic() + timeout if timeout else None
    processes: List[HookProcess] = []
    try:
        for consumer in consumers:
            process = HookProcess(consumer.command, payload, capture_stderr=True)
            if process.start():
                processes.append(process)
        parts: List[str] = []
        for process in processes:
            # The reply is complete once stdout has one document or closes; the
            # exit code still has to be collected, because 2 is a refusal.
            if not process.wait(_remaining(deadline)):
                continue
            code = process.exit_code(_remaining(deadline))
            if code is None:
                continue
            context, refusal = parse_prompt_reply(
                process.stdout, returncode=code, stderr=process.stderr
            )
            if refusal is not None:
                raise PromptBlocked(refusal)
            if context:
                parts.append(context)
        return "\n\n".join(parts) or None
    finally:
        # Idempotent: a consumer that answered is reaped, a straggler is killed.
        for process in processes:
            process.kill()


__all__ = [
    "PROMPT_EVENT",
    "PromptBlocked",
    "collect_prompt_context",
    "parse_prompt_reply",
    "prompt_payload",
]
