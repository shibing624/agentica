# -*- coding: utf-8 -*-
"""Interactive CLI tool-approval prompts (Codex three-option UX).

The runner parks on ``Agent.approve`` before ``fc.execute()``. Interactive
sessions resolve that wait through ``_InputRequest`` + prompt_toolkit keys
(y / p / esc). ``--print`` / non-interactive sessions pass
``get_registry=None`` so anything that would park is denied immediately.
"""
from __future__ import annotations

import asyncio
import shlex
from typing import Any, Callable, Dict, Optional

from agentica.agent.approvals import (
    EXECUTE_TOOLS,
    FILE_TOOLS,
    NETWORK_TOOLS,
    ApprovalDecision,
    PendingApproval,
    SessionGrants,
    apply_path_grant_on_agent,
    make_approve,
)
from agentica.tools.safety import split_compound_command

_KEY_TO_DECISION: Dict[str, ApprovalDecision] = {
    "y": "allow",
    "yes": "allow",
    "1": "allow",
    "p": "allow_prefix",
    "2": "allow_prefix",
    "n": "deny",
    "no": "deny",
    "3": "deny",
    "esc": "deny",
    "escape": "deny",
}


def approval_decision_from_key(key: str) -> Optional[ApprovalDecision]:
    """Map a Codex-style key (y / p / esc / 1 / 2 / 3) to a decision."""
    return _KEY_TO_DECISION.get((key or "").strip().lower())


def is_approval_request(req: Any) -> bool:
    return req is not None and req.kind == "approval"


def format_approval_prompt(pending: PendingApproval) -> str:
    """Codex CLI three-option copy for the TUI prompt widget."""
    name = pending.name
    preview = (pending.preview or "").strip() or _preview_from_arguments(name, pending.arguments)
    if name in EXECUTE_TOOLS:
        question = "Would you like to run the following command?"
        body = f"$ {preview}" if preview else "$"
        option2 = (
            "Yes, and don't ask again for commands that start with "
            f"`{_command_prefix_display(preview)}` (p)"
        )
    elif name in FILE_TOOLS:
        question = "Would you like to allow this file operation?"
        body = preview
        option2 = "Yes, and don't ask again for this class of path (p)"
    elif name in NETWORK_TOOLS:
        question = "Would you like to allow this network request?"
        body = preview
        option2 = "Yes, and don't ask again for this class of network tool (p)"
    else:
        question = f"Would you like to allow this {name} call?"
        body = preview
        option2 = f"Yes, and don't ask again for this {name} tool (p)"
    return (
        f"{question}\n"
        f"Environment: local\n"
        f"{body}\n"
        f"\n"
        f"1. Yes, proceed (y)\n"
        f"2. {option2}\n"
        f"3. No, and tell the agent what to do differently (esc)"
    )


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

    registry = state.approval_registry
    loop = state.approval_loop
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(registry.deny_all)
    else:
        registry.deny_all()
    req = state.input_request
    if is_approval_request(req):
        req.cancel()
        if state.input_request is req:
            state.input_request = None
    with _ask_state_lock:
        _ask_active[0] = False


def complete_approval(
    state: Any, decision: ApprovalDecision, *, app: Any = None
) -> bool:
    """Apply a keypress decision and hide the prompt."""
    from agentica.cli.interactive.console_io import _ask_active, _ask_state_lock

    req = state.input_request
    if req is None or req.kind != "approval":
        return False
    ok = submit_approval_decision(req, decision)
    if state.input_request is req:
        state.input_request = None
    with _ask_state_lock:
        _ask_active[0] = False
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
    )


def build_interactive_approve(state: Any, ui_holder: dict) -> Callable:
    """Park on ``_InputRequest``; y/p/esc on the prompt_toolkit thread decide."""
    from agentica.cli.interactive.console_io import _ask_active, _ask_state_lock
    from agentica.cli.interactive.session_state import _InputRequest

    lock_box: Dict[str, Any] = {"lock": None, "loop": None}

    def _agent():
        return state.current_agent

    def publish(pending: PendingApproval) -> None:
        loop = asyncio.get_running_loop()
        state.approval_loop = loop
        req = _InputRequest(
            prompt=format_approval_prompt(pending),
            kind="approval",
            approval_id=pending.tool_call_id,
            approval_loop=loop,
            approval_registry=state.approval_registry,
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
    )

    async def approve(fc) -> ApprovalDecision:
        agent = _agent()
        if agent is not None and agent._cancelled:
            return "deny"
        loop = asyncio.get_running_loop()
        if lock_box["loop"] is not loop:
            lock_box["lock"] = asyncio.Lock()
            lock_box["loop"] = loop
        async with lock_box["lock"]:
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


def _command_prefix_display(command: str) -> str:
    segments = split_compound_command(command) or [command]
    if not segments:
        return "..."
    try:
        tokens = shlex.split(segments[0])
    except ValueError:
        tokens = segments[0].split()
    if not tokens:
        return "..."
    if len(tokens) > 2:
        return f"{tokens[0]} {tokens[1]} ..."
    return " ".join(tokens)
