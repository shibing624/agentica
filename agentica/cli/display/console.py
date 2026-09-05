# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Console helpers, color scheme, and agent execution error formatting
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from rich.text import Text

from agentica.utils.log import logger

# Rich console color scheme (unified - no separate ANSI codes)
COLORS = {
    "user": "bright_cyan",
    "agent": "bright_green",
    "thinking": "yellow",
    "tool": "cyan",
    "error": "red",
}


_INTERNAL_REPEAT_FAILURE_NOTICE_RE = re.compile(
    r"\n?\[Notice: This exact call has failed \d+ times this run with the same error\. "
    r"Consider a different approach\.\]\s*"
)


def _strip_internal_tool_notices(text: str) -> str:
    """Remove model-facing retry nudges from the user-facing transcript."""
    return _INTERNAL_REPEAT_FAILURE_NOTICE_RE.sub("", text).rstrip()


def _is_diagnostic_execute_result(text: str) -> bool:
    return "(Note: Diagnostics found)" in text

_PASTE_PATH_RE = re.compile(r"@\S*[\\/]pastes[\\/]paste_\S+\.txt")


# Tool command/output blocks truncated in the CLI display during the current
# run. Remembered so the user can expand them on demand: Ctrl+O opens EVERY
# folded block in one pager (CC-style "expand all"). User input and write-tool
# diffs are always shown in full, so they are never stashed here. Cleared at
# the start of each run.
_truncated_blocks: List[Dict[str, str]] = []


def _public_truncated(block: Dict[str, str]) -> Dict[str, str]:
    return {"title": block.get("title", ""), "content": block.get("content", "")}


def remember_truncated(
    title: str,
    content: str,
    *,
    key: Optional[str] = None,
    only_replace: bool = False,
) -> None:
    """Stash a truncated block for on-demand expansion (Ctrl+O opens all).

    ``key`` replaces an earlier block from the same call (a folded execute
    command is upgraded to command + full output when the result arrives).
    ``only_replace`` updates that block and otherwise does nothing — used
    when the result itself was short enough to show inline.
    """
    if not content:
        return
    block: Dict[str, str] = {"title": title, "content": content}
    if key:
        block["key"] = key
        for i, existing in enumerate(_truncated_blocks):
            if existing.get("key") == key:
                _truncated_blocks[i] = block
                return
        if only_replace:
            return
    _truncated_blocks.append(block)


def get_last_truncated() -> Dict[str, str]:
    """Return a copy of the most recent truncated block (or empty)."""
    if not _truncated_blocks:
        return {"title": "", "content": ""}
    return _public_truncated(_truncated_blocks[-1])


def get_truncated_blocks() -> List[Dict[str, str]]:
    """Return all truncated blocks accumulated this run (newest last)."""
    return [_public_truncated(b) for b in _truncated_blocks]


def clear_truncated_blocks() -> None:
    """Drop all remembered truncated blocks (called at run start)."""
    _truncated_blocks.clear()

def _parse_provider_error_payload(message: str) -> Dict[str, Any]:
    """Extract common provider error fields from SDK exception text."""
    details: Dict[str, Any] = {"raw": message}

    def find_first_key(value: Any, target: str) -> Optional[Any]:
        if isinstance(value, dict):
            if target in value:
                return value[target]
            for item in value.values():
                found = find_first_key(item, target)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find_first_key(item, target)
                if found is not None:
                    return found
        return None

    status_match = re.search(r"Error code:\s*(\d+)", message, re.IGNORECASE)
    if status_match:
        details["status"] = status_match.group(1)

    payload_match = re.search(r"Error code:\s*\d+\s*-\s*(.+)\s*$", message, re.DOTALL | re.IGNORECASE)
    if not payload_match:
        return details

    payload_text = payload_match.group(1).strip()
    if not payload_text.startswith("{"):
        return details
    try:
        payload = ast.literal_eval(payload_text)
    except (ValueError, SyntaxError):
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return details
    if not isinstance(payload, dict):
        return details

    raw_error = payload.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    provider_message = error.get("message") if isinstance(error.get("message"), str) else find_first_key(payload, "message")
    if isinstance(provider_message, str):
        details["message"] = provider_message
    code = error.get("code") if error.get("code") is not None else find_first_key(payload, "code")
    if code is not None:
        details["code"] = str(code)
    error_type = error.get("type") if isinstance(error.get("type"), str) else find_first_key(payload, "type")
    if isinstance(error_type, str):
        details["type"] = error_type
    span_id = find_first_key(payload, "spanId")
    if isinstance(span_id, str):
        details["span_id"] = span_id
    return details


def _decode_error_window(error: json.JSONDecodeError, span: int = 30) -> str:
    """Return the text on either side of where JSON parsing gave up.

    Kept to one terminal line: it exists so the shape of the garbage is
    recognisable at a glance (two events glued together, an HTML error page,
    a truncated object). The whole chunk is one Ctrl+O away.
    """
    doc = error.doc
    if not doc:
        return ""
    start = max(0, error.pos - span)
    end = min(len(doc), error.pos + span)
    window = " ".join(doc[start:end].split())
    return f"{'…' if start else ''}{window}{'…' if end < len(doc) else ''}"


def _format_agent_execution_error(error: BaseException) -> Dict[str, Any]:
    """Build a concise CLI-facing error view while retaining raw details.

    ``raw`` is what Ctrl+O expands and what goes to the log, so it has to be
    something the short on-screen line does not already say. For a decode
    error ``str(error)`` is just "Extra data: line 1 column 309" — expanding
    that shows the user the same sentence twice. The chunk the endpoint
    actually sent is in ``.doc``, and nothing else keeps a copy of it.
    """
    raw = str(error)
    details = _parse_provider_error_payload(raw)
    low = raw.lower()
    status = details.get("status")
    provider_message = details.get("message")

    is_rate_limited = (
        status == "429"
        or "rate_limit" in low
        or "rate limit" in low
        or "限流" in raw
        or "tpm" in low
    )
    is_transient = is_rate_limited or any(
        hint in low
        for hint in ("connection", "timeout", "502", "503", "504", "gateway", "remote disconnected")
    )

    is_context_length = any(
        hint in low
        for hint in (
            "context_length_exceeded",
            "maximum context length",
            "maximum context",
            "prompt_too_long",
            "too many tokens",
        )
    )

    if is_rate_limited:
        summary = f"LLM rate limited ({status})" if status else "LLM rate limited"
        detail = provider_message or raw
        hint = "Type /retry after a short wait, or switch model/profile."
    elif is_context_length:
        # Oversized single queries (and irreducible prompt_too_long) must show
        # the provider's limit text, not a generic "execution failed".
        summary = "Input exceeds model context window"
        detail = provider_message or raw
        hint = "Shorten the message, /compact earlier history, or switch to a larger-context model."
    elif isinstance(error, json.JSONDecodeError):
        # A gateway that packs two SSE events onto one ``data:`` line surfaces
        # only as "Extra data: line 1 column N". Name the cause so the user
        # doesn't go looking for it in their prompt or config, and show the
        # text around the break — that is what says *which* garbage arrived.
        summary = "Malformed stream from the model endpoint"
        detail = f"The endpoint sent an unparsable SSE chunk: {raw}"
        window = _decode_error_window(error)
        if window:
            detail = f"{detail}\n  near the break: {window}"
        hint = "Type /retry to resend the last message."
    elif is_transient:
        summary = f"Transient LLM/API error ({status})" if status else "Transient LLM/API error"
        detail = provider_message or raw
        hint = "Type /retry to resend the last message."
    else:
        summary = f"Agent execution failed ({status})" if status else "Agent execution failed"
        detail = provider_message or raw
        hint = None

    if len(detail) > 500:
        detail = detail[:497] + "..."

    diagnostics = []
    for key, label in (
        ("code", "code"),
        ("type", "type"),
        ("span_id", "spanId"),
    ):
        value = details.get(key)
        if value:
            diagnostics.append(f"{label}={value}")

    if isinstance(error, json.JSONDecodeError) and error.doc:
        raw = (
            f"{raw}\n\nOffending chunk sent by the endpoint "
            f"({len(error.doc)} chars, parsing stopped at char {error.pos}):\n{error.doc}"
        )

    return {
        "summary": summary,
        "detail": detail,
        "diagnostics": " ".join(diagnostics),
        "hint": hint,
        "raw": raw,
    }


def display_agent_execution_error(console_instance, error: BaseException) -> Dict[str, Any]:
    """Render a structured agent error, log it, and retain raw for Ctrl+O.

    The Ctrl+O copy lives in a buffer that the next user turn clears, so it is
    gone the moment the user says anything else — which is exactly when a
    malformed-stream error tends to be looked at. The file log is the durable
    copy, and it is the only one an unattended run leaves behind at all.
    """
    view = _format_agent_execution_error(error)
    if view["raw"]:
        remember_truncated("Agent error · raw", view["raw"])
        # One log record per error, so the raw text is greppable as a unit.
        logger.error("%s: %s", view["summary"], " ".join(view["raw"].split()))

    headline = Text("● Error: ", style="bold red")
    headline.append(view["summary"], style="bold red")
    console_instance.print()
    console_instance.print(headline)
    if view["detail"]:
        console_instance.print(Text(f"  {view['detail']}", style="red"))
    if view["diagnostics"]:
        console_instance.print(Text(f"  {view['diagnostics']}", style="dim"))
    if view["hint"]:
        console_instance.print(Text(f"  {view['hint']}", style="dim"))
    console_instance.print(Text("  Ctrl+O shows raw provider error.", style="dim"))
    return view


