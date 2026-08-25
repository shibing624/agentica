# -*- coding: utf-8 -*-
"""Interactive CLI tool-approval prompts (Codex-style y / p / esc / x).

The runner parks on ``Agent.approve`` before ``fc.execute()``. Interactive
sessions resolve that wait through ``_InputRequest`` + prompt_toolkit keys
(y / p / esc / x). ``--print`` / non-interactive sessions pass
``get_registry=None`` so anything that would park is denied immediately.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from agentica.agent.approvals import (
    EXECUTE_TOOLS,
    FILE_TOOLS,
    NETWORK_TOOLS,
    ApprovalDecision,
    PendingApproval,
    SessionGrants,
    apply_path_grant_on_agent,
    command_class_display,
    make_approve,
)

_DECISION_ORDER: Tuple[ApprovalDecision, ...] = (
    "allow",
    "allow_prefix",
    "deny",
    "deny_prefix",
)

_LETTER_TO_DECISION: Dict[str, ApprovalDecision] = {
    "y": "allow",
    "yes": "allow",
    "p": "allow_prefix",
    "n": "deny",
    "no": "deny",
    "esc": "deny",
    "escape": "deny",
    "x": "deny_prefix",
}


def visible_approval_decisions(
    options: Optional[Sequence[str]] = None,
) -> Tuple[ApprovalDecision, ...]:
    """Stable allow → allow-similar → deny → deny-similar, omitting hidden ones."""
    offered = set(options) if options is not None else set(_DECISION_ORDER)
    return tuple(d for d in _DECISION_ORDER if d in offered)


def approval_decision_from_key(
    key: str,
    options: Optional[Sequence[str]] = None,
) -> Optional[ApprovalDecision]:
    """Map y / p / esc / x / 1..N to a decision among the visible options."""
    k = (key or "").strip().lower()
    visible = visible_approval_decisions(options)
    if k.isdigit():
        idx = int(k) - 1
        if 0 <= idx < len(visible):
            return visible[idx]
        return None
    decision = _LETTER_TO_DECISION.get(k)
    if decision is None or decision not in visible:
        return None
    return decision


def is_approval_request(req: Any) -> bool:
    return req is not None and req.kind == "approval"


def format_approval_prompt(pending: PendingApproval) -> str:
    """Codex CLI copy for the TUI prompt widget."""
    name = pending.name
    preview = (pending.preview or "").strip() or _preview_from_arguments(name, pending.arguments)
    if name in EXECUTE_TOOLS:
        question = "Would you like to run the following command?"
        body = f"$ {preview}" if preview else "$"
        label = pending.similar_label or command_class_display(preview)
        allow_similar = (
            f"Yes, and don't ask again for `{label}` commands (p)" if label else (
                "Yes, and don't ask again for this class of command (p)"
            )
        )
        deny_similar = (
            f"No, and don't ask again for `{label}` commands (x)" if label else (
                "No, and don't ask again for this class of command (x)"
            )
        )
    elif name in FILE_TOOLS:
        question = "Would you like to allow this file operation?"
        body = preview
        allow_similar = "Yes, and don't ask again for this class of path (p)"
        deny_similar = "No, and don't ask again for this class of path (x)"
    elif name in NETWORK_TOOLS:
        question = "Would you like to allow this network request?"
        body = preview
        allow_similar = "Yes, and don't ask again for this class of network tool (p)"
        deny_similar = "No, and don't ask again for this class of network tool (x)"
    else:
        question = f"Would you like to allow this {name} call?"
        body = preview
        allow_similar = f"Yes, and don't ask again for this {name} tool (p)"
        deny_similar = f"No, and don't ask again for this {name} tool (x)"
    labels = {
        "allow": "Yes, proceed (y)",
        "allow_prefix": allow_similar,
        "deny": "No, deny (esc/n)",
        "deny_prefix": deny_similar,
    }
    lines = [question, "Environment: local", body, ""]
    for i, decision in enumerate(visible_approval_decisions(pending.options), 1):
        lines.append(f"{i}. {labels[decision]}")
    return "\n".join(lines)


def format_approval_record(pending: PendingApproval, decision: ApprovalDecision) -> str:
    """Transcript remnant after the TUI card is dismissed.

    The card lives in the prompt_toolkit layout, not scrollback. Hiding it
    would drop the command the user just approved unless we reprint it.
    """
    preview = (pending.preview or "").strip() or _preview_from_arguments(
        pending.name, pending.arguments
    )
    if pending.name in EXECUTE_TOOLS:
        body = f"$ {preview}" if preview else "$"
    else:
        body = preview or pending.name
    if decision == "allow":
        verdict = "✓ allowed"
    elif decision == "allow_prefix":
        verdict = "✓ allowed similar"
    elif decision == "deny_prefix":
        verdict = "✗ denied similar"
    else:
        verdict = "✗ denied"
    return f"{body}\n{verdict}"


def _print_approval_record(pending: Optional[PendingApproval], decision: ApprovalDecision) -> None:
    if pending is None:
        return
    from agentica.cli.interactive.console_io import _cprint

    _cprint(format_approval_record(pending, decision))


def submit_approval_decision(req: Any, decision: ApprovalDecision) -> bool:
    """Resolve one parked approval from the TUI thread.

    ``ApprovalRegistry`` futures belong to the agent asyncio loop, so decide
    is posted with ``call_soon_threadsafe``. Returns False if this request
    was already resolved.
    """
    if not is_approval_request(req) or req.resolved:
        return False
    registry = req.approval_registry
    loop = req.approval_loop
    tool_call_id = req.approval_id or ""
    if registry is None or not tool_call_id:
        return req.submit(decision)
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(registry.decide, tool_call_id, decision)
    else:
        registry.decide(tool_call_id, decision)
    return req.submit(decision)


def interrupt_approvals(state: Any) -> None:
    """Ctrl+C / turn-end: deny every pending wait and close the prompt."""
    from agentica.cli.interactive.console_io import _ask_active, _ask_state_lock

    req = state.input_request
    pending = req.approval_pending if is_approval_request(req) else None
    if is_approval_request(req):
        req.cancel()
        if state.input_request is req:
            state.input_request = None
    with _ask_state_lock:
        _ask_active[0] = False
    _print_approval_record(pending, "deny")
    registry = state.approval_registry
    loop = state.approval_loop
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(registry.deny_all)
    else:
        registry.deny_all()


def complete_approval(
    state: Any, decision: ApprovalDecision, *, app: Any = None
) -> bool:
    """Apply a keypress decision and hide the prompt."""
    from agentica.cli.interactive.console_io import _ask_active, _ask_state_lock

    req = state.input_request
    if req is None or req.kind != "approval":
        return False
    pending = req.approval_pending
    if state.input_request is req:
        state.input_request = None
    with _ask_state_lock:
        _ask_active[0] = False
    _print_approval_record(pending, decision)
    ok = submit_approval_decision(req, decision)
    if app is not None:
        app.invalidate()
    return ok


def build_noninteractive_approve(agent: Any) -> Callable:
    """``--print`` / SDK-style: park would hang, so the manual path is deny."""
    grants = SessionGrants()
    return make_approve(
        get_mode=lambda: agent.tool_config.permission_mode,
        get_grants=lambda: grants,
        get_registry=lambda: None,
        get_work_dir=lambda: agent.work_dir,
        publish=lambda pending: None,
        apply_path_grant=lambda path, prefix: apply_path_grant_on_agent(
            agent, path, prefix=prefix
        ),
        get_user_id=lambda: agent.user_id,
    )


def build_interactive_approve(state: Any, ui_holder: dict) -> Callable:
    """Park on ``_InputRequest``; y/p/esc on the prompt_toolkit thread decide."""
    from agentica.cli.interactive.console_io import _ask_active, _ask_state_lock
    from agentica.cli.interactive.session_state import _InputRequest

    def _agent():
        return state.current_agent

    def publish(pending: PendingApproval) -> None:
        loop = asyncio.get_running_loop()
        state.approval_loop = loop
        req = _InputRequest(
            prompt=format_approval_prompt(pending),
            kind="approval",
            options=list(pending.options),
            approval_id=pending.tool_call_id,
            approval_loop=loop,
            approval_registry=state.approval_registry,
            approval_pending=pending,
        )
        with _ask_state_lock:
            state.input_request = req
            _ask_active[0] = True
        app = ui_holder.get("app")
        if app is not None:
            app.invalidate()

    inner = make_approve(
        get_mode=lambda: _agent().tool_config.permission_mode if _agent() else "allow-all",
        get_grants=lambda: state.approval_grants,
        get_registry=lambda: state.approval_registry,
        get_work_dir=lambda: _agent().work_dir if _agent() else None,
        publish=publish,
        apply_path_grant=lambda path, prefix: apply_path_grant_on_agent(
            _agent(), path, prefix=prefix
        ),
        get_user_id=lambda: _agent().user_id if _agent() is not None else None,
    )

    async def approve(fc) -> ApprovalDecision:
        agent = _agent()
        if agent is not None and agent._cancelled:
            return "deny"
        loop = asyncio.get_running_loop()
        if state.approval_prompt_lock is None or state.approval_loop is not loop:
            state.approval_prompt_lock = asyncio.Lock()
        state.approval_loop = loop
        async with state.approval_prompt_lock:
            agent = _agent()
            if agent is not None and agent._cancelled:
                return "deny"
            try:
                return await inner(fc)
            finally:
                req = state.input_request
                if is_approval_request(req) and req.approval_id == (fc.call_id or ""):
                    with _ask_state_lock:
                        if state.input_request is req:
                            state.input_request = None
                        _ask_active[0] = False
                    app = ui_holder.get("app")
                    if app is not None:
                        app.invalidate()

    return approve


def _preview_from_arguments(name: str, arguments: Dict[str, Any]) -> str:
    if name in EXECUTE_TOOLS:
        return str(arguments.get("command") or "")
    if name in ("read_file", "write_file"):
        return str(arguments.get("file_path") or "")
    if name == "fetch_url":
        return str(arguments.get("url") or "")
    if name == "web_search":
        return str(arguments.get("queries") or "")
    return str(arguments or "")
