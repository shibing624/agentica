# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Console helpers, color scheme, and agent execution error formatting
"""

import ast
import json
import re
from typing import Any, ContextManager, Dict, List, Optional, Tuple, cast

from rich.text import Text

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


def remember_truncated(title: str, content: str) -> None:
    """Stash a truncated block for on-demand expansion (Ctrl+O opens all)."""
    if not content:
        return
    _truncated_blocks.append({"title": title, "content": content})


def get_last_truncated() -> Dict[str, str]:
    """Return a copy of the most recent truncated block (or empty)."""
    if not _truncated_blocks:
        return {"title": "", "content": ""}
    return dict(_truncated_blocks[-1])


def get_truncated_blocks() -> List[Dict[str, str]]:
    """Return all truncated blocks accumulated this run (newest last)."""
    return [dict(b) for b in _truncated_blocks]


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


def _format_agent_execution_error(error: BaseException) -> Dict[str, Any]:
    """Build a concise CLI-facing error view while retaining raw details."""
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
        # doesn't go looking for it in their prompt or config.
        summary = "Malformed stream from the model endpoint"
        detail = f"The endpoint sent an unparsable SSE chunk: {raw}"
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

    return {
        "summary": summary,
        "detail": detail,
        "diagnostics": " ".join(diagnostics),
        "hint": hint,
        "raw": raw,
    }


def display_agent_execution_error(console_instance, error: BaseException) -> Dict[str, Any]:
    """Render a structured agent error and retain raw details for Ctrl+O."""
    view = _format_agent_execution_error(error)
    if view["raw"]:
        remember_truncated("Agent error · raw", view["raw"])

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


class _GutteredConsole:
    """Console proxy that prepends every printed line with a gutter marker.

    Wraps an existing console-like object (either a raw ``rich.console.Console``
    or the CLI's ``ChatConsole`` adapter defined in ``interactive.py``) and
    rewrites ``.print()`` output so every visible line is prefixed with a
    colored gutter character (e.g. ``▏ `` for the assistant turn). All other
    attributes are forwarded to the underlying console.

    Rendering strategy — picked per-call in this order:

    1. If the underlying console exposes ``render_ansi(*args, **kwargs)``
       (that's the CLI's ``ChatConsole``), use it — it already gives us a
       fully-rendered ANSI string without touching stdout.
    2. Otherwise, if it exposes rich's ``.capture()`` context manager, use
       that — same idea, one indirection more.
    3. Otherwise, fall back to a plain pass-through ``.print(*args, **kwargs)``
       without any gutter. This lets bare mocks and non-rich consoles still
       work (they just lose the gutter decoration) rather than blowing up.

    Emission: the assembled gutter-prefixed ANSI text is handed back to
    the underlying console via ``_emit_ansi``, which prefers ``ChatConsole``'s
    line-oriented ``_cprint`` (integrates with prompt_toolkit's patch_stdout)
    and otherwise writes directly to ``console.file``.
    """

    def __init__(self, console, gutter_char: str = "▏", style: str = "#CD7F32"):
        self._console = console
        self._gutter_char = gutter_char
        self._style = style

    # ------------------------------------------------------------------ helpers

    def _render_ansi(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        """Render args to ANSI text.

        Returns ``(text, already_printed)``:
        - ``(str, False)`` — got real ANSI, caller should apply gutter and emit
        - ``(None, True)`` — no ANSI available, but a side-effect ``.print()``
          call already happened during the capture attempt (bare MagicMock);
          caller must NOT print again to avoid double output
        - ``(None, False)`` — neither ANSI nor side-effect print occurred;
          caller should degrade to a plain pass-through ``.print()``

        Prefers ``ChatConsole.render_ansi`` (side-effect free). Falls back to
        ``rich.Console.capture()`` (side-effect free on real consoles; on
        MagicMock the ``.capture()`` context still records ``.print()`` calls
        as its underlying implementation, which we treat as ``already_printed``).
        """
        render = getattr(self._console, "render_ansi", None)
        if callable(render):
            try:
                result = render(*args, **kwargs)
            except Exception:
                result = None
            if isinstance(result, str):
                return result, False
        capture = getattr(self._console, "capture", None)
        if callable(capture):
            try:
                with cast(ContextManager[Any], capture()) as cap:
                    self._console.print(*args, **kwargs)
                got = cap.get()
            except Exception:
                # capture blew up but ``self._console.print`` may or may not
                # have fired; be conservative and assume it did to avoid
                # duplicating output.
                return None, True
            if isinstance(got, str):
                return got, False
            # capture succeeded structurally but returned non-str (MagicMock).
            # The inner ``self._console.print`` call HAS already been recorded,
            # so the payload effectively landed. Signal already_printed.
            return None, True
        return None, False

    def _emit_ansi(self, ansi_text: str) -> None:
        """Send fully-assembled ANSI text back to the underlying console.

        Prefers ``ChatConsole._cprint`` (line-oriented, integrates with
        prompt_toolkit's patch_stdout); falls back to raw file writes.
        """
        if not ansi_text:
            return
        # ChatConsole exposes ``print`` that expects Rich markup, not raw ANSI.
        # We can't reuse it — it would double-render. Instead we go straight
        # to the module-level ``_cprint`` which is line-oriented and writes
        # ANSI verbatim (that's what ChatConsole itself uses internally).
        from agentica.cli.interactive.console_io import _cprint

        if hasattr(self._console, "render_ansi"):
            # Line-mode: strip a single trailing newline so we don't emit an
            # extra empty line, then feed each line through _cprint.
            text = ansi_text[:-1] if ansi_text.endswith("\n") else ansi_text
            for line in text.split("\n"):
                _cprint(line)
            return

        # Rich Console path: write directly to its file.
        file = getattr(self._console, "file", None)
        if file is not None:
            file.write(ansi_text)
            try:
                file.flush()
            except Exception:
                pass
            return

        # Last resort: pass-through print (no gutter).
        self._console.print(ansi_text, end="", markup=False, highlight=False)

    # ------------------------------------------------------------------ public

    @property
    def gutter_prefix_ansi(self) -> str:
        """The ``▏ `` prefix pre-rendered as ANSI, cached per instance.

        Uses whichever ANSI-rendering path the underlying console supports.
        If no path yields ANSI (bare mock, exotic console), returns a plain
        uncolored ``▏ `` — the gutter still exists structurally, just untinted.
        """
        cached = getattr(self, "_prefix_cache", None)
        if cached is not None:
            return cached
        text, _ = self._render_ansi(
            f"[{self._style}]{self._gutter_char}[/{self._style}] ",
            end="",
        )
        rendered = text if text is not None else f"{self._gutter_char} "
        self._prefix_cache = rendered
        return rendered

    def print(self, *args, **kwargs):
        text, already_printed = self._render_ansi(*args, **kwargs)
        if text is None:
            if not already_printed:
                # Neither ANSI nor a side-effect print — pass through so at
                # least the payload lands somewhere.
                self._console.print(*args, **kwargs)
            # If already_printed, the underlying console already saw a
            # (gutter-less) ``.print`` call during our capture attempt.
            # We can't retroactively add the gutter, so we accept the
            # ungutter'd output rather than duplicating it.
            return
        if not text:
            return
        # Rich normally ends with a newline; strip trailing newline once so we
        # don't emit an empty gutter line at the very bottom.
        trailing_newline = text.endswith("\n")
        if trailing_newline:
            text = text[:-1]
        prefix = self.gutter_prefix_ansi
        lines = text.split("\n")
        # Prepend the gutter to every physical line. Empty lines still get a
        # gutter so the visual bar stays continuous even in blank spacing.
        rebuilt = "\n".join(prefix + ln for ln in lines)
        if trailing_newline:
            rebuilt += "\n"
        self._emit_ansi(rebuilt)

    def __getattr__(self, name):
        # Forward width, size, is_terminal, options, capture, rule, status,
        # etc. — anything StreamDisplayManager might touch.
        return getattr(self._console, name)
