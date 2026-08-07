# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI display utilities - colors, formatting, stream display manager
"""
import ast
import difflib
import json
import os
import re
import textwrap
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agentica.cli.runtime import BUILTIN_TOOLS, TOOL_ICONS, get_console
from agentica.global_config import get_setting
from agentica.model.usage import Usage
from agentica.tools.patch_tool import parse_patch_envelope
from agentica.version import __version__

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


def print_header(model_provider: str, model_name: str, work_dir: Optional[str] = None,
                 extra_tools: Optional[List[str]] = None, shell_mode: bool = False):
    """Print the application header with version and model information"""
    box_width = min(get_console().width, 80)
    get_console().print("=" * box_width, style="bright_cyan")
    get_console().print(f"  Agentica CLI v{__version__} - Interactive AI Assistant")
    get_console().print(f"  Model: [bright_green]{model_provider}/{model_name}[/bright_green]")

    # Working directory
    cwd = work_dir or os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]
    if len(cwd) > 50:
        cwd = "..." + cwd[-47:]
    get_console().print(f"  Working Directory: {cwd}")

    # Built-in tools (always shown)
    get_console().print(f"  Built-in Tools: [white]{', '.join(BUILTIN_TOOLS)}[/white]")

    # Extra tools info
    if extra_tools:
        tools_str = ", ".join(extra_tools)
        if len(tools_str) > 55:
            tools_str = tools_str[:52] + "..."
        get_console().print(f"  Extra Tools: [bright_green]{tools_str}[/bright_green]")

    # Log file location (helps users find logs when debugging)
    from agentica.config import AGENTICA_LOG_FILE, AGENTICA_LOG_LEVEL
    if AGENTICA_LOG_FILE:
        log_path = AGENTICA_LOG_FILE
        if log_path.startswith(home):
            log_path = "~" + log_path[len(home):]
        get_console().print(f"  Log File ({AGENTICA_LOG_LEVEL}): [white]{log_path}[/white]")

    get_console().print("=" * box_width, style="bright_cyan")
    get_console().print()
    # Keyboard shortcuts
    get_console().print("  [bright_green]Enter[/bright_green]       Submit your message")
    get_console().print("  [bright_green]Ctrl+X[/bright_green]      Toggle Agent/Shell mode")
    get_console().print("  [bright_green]Ctrl+J[/bright_green]      Insert newline (Alt+Enter also works)")
    get_console().print("  [bright_green]Ctrl+D[/bright_green]      Exit and show resume command")
    get_console().print("  [bright_green]Ctrl+C[/bright_green]      Interrupt current operation (press twice to exit)")
    get_console().print("  [bright_green]Ctrl+V[/bright_green]      Paste image from clipboard (or just paste directly)")
    get_console().print("  [bright_green]Ctrl+O[/bright_green]      Expand truncated tool commands and output in pager (Ctrl+O or Esc to return)")
    get_console().print("  [bright_green]Alt+P[/bright_green]       Pause/resume live output while browsing terminal history")
    get_console().print()
    # Input features
    get_console().print("  [bright_green]@filename[/bright_green]   Type @ to auto-complete files (images auto-attach)")
    get_console().print("  [bright_green]/command[/bright_green]    Type / to see available commands (try /help)")
    get_console().print()


def format_session_summary(
    *, elapsed_seconds: float, usage: Usage, session_id: str | None
) -> Text:
    """Build the summary printed before ``/new`` starts a fresh chat."""
    elapsed = max(0, int(elapsed_seconds))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"

    text = Text()
    text.append(f"Worked for {duration} ", style="dim")
    text.append("─" * 42, style="dim")
    text.append("\n\nToken usage: ", style="dim")
    text.append(f"total={usage.total_tokens:,} input={usage.input_tokens:,}")
    cached_tokens = usage.input_tokens_details.cache_read_tokens
    if cached_tokens <= 0:
        cached_tokens = usage.input_tokens_details.cached_tokens
    if cached_tokens > 0:
        text.append(f" (+ {cached_tokens:,} cached)", style="dim")
    text.append(f" output={usage.output_tokens:,}")
    reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    if reasoning_tokens > 0:
        text.append(f" (reasoning {reasoning_tokens:,})", style="dim")
    if session_id:
        text.append("\nTo continue this session, run ", style="dim")
        text.append(f"agentica resume {session_id}", style="bold")
    return text


def parse_file_mentions(text: str) -> Tuple[str, List[Path]]:
    """Parse @file mentions and return text with mentioned files.
    
    Uses lookbehind to avoid matching email addresses.
    """
    pattern = r"(?:^|(?<=\s))@([\w./-]+)"
    mentioned_files = []
    
    for match in re.finditer(pattern, text):
        file_path_str = match.group(1)
        file_path = Path(file_path_str).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if file_path.exists() and file_path.is_file():
            mentioned_files.append(file_path)
    
    # Remove @ mentions from text for cleaner display
    processed_text = re.sub(pattern, r'\1', text)
    return processed_text, mentioned_files


def inject_file_contents(prompt_text: str, mentioned_files: List[Path]) -> str:
    """Inject file contents into the prompt."""
    if not mentioned_files:
        return prompt_text
    
    context_parts = [prompt_text, "\n\n## Referenced Files\n"]
    for file_path in mentioned_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            # Limit file content to reasonable size
            if len(content) > 20000:
                content = content[:20000] + "\n... (file truncated)"
            context_parts.append(
                f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
            )
        except Exception as e:
            context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")
    
    return "\n".join(context_parts)


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

    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
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

    if is_rate_limited:
        summary = f"LLM rate limited ({status})" if status else "LLM rate limited"
        detail = provider_message or raw
        hint = "Type /retry after a short wait, or switch model/profile."
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
                with capture() as cap:
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
        try:
            from agentica.cli.interactive import _cprint  # local import to avoid cycles
        except Exception:  # pragma: no cover — during shutdown / partial import
            _cprint = None

        if _cprint is not None and hasattr(self._console, "render_ansi"):
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


def _format_attachment_size(path: Path) -> str:
    """Return a compact file size for an attachment label."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def display_user_message(
    text: str,
    *,
    pasted_blocks: int = 0,
    pasted_lines: int = 0,
    images: Optional[List[Path]] = None,
) -> None:
    """Display the full user message and image attachments in one input panel.

    Long messages are no longer folded behind Ctrl+O — the complete text is
    always rendered inline so the user sees exactly what they sent.
    """
    # A new user turn starts here: drop the previous turn's folded blocks so
    # Ctrl+O expands the CURRENT turn's tool results, not stale history.
    clear_truncated_blocks()
    cleaned = _PASTE_PATH_RE.sub("", text).strip()
    if not cleaned and pasted_blocks:
        cleaned = f"[Pasted text: {pasted_lines} lines]"

    # Always render the complete message (no Ctrl+O fold); only colorize
    # @file mentions.
    pattern = r"(@[\w./-]+)"
    parts = re.split(pattern, cleaned)
    rich_text = Text()
    for part in parts:
        if part.startswith("@"):
            rich_text.append(part, style="bold magenta")
        else:
            rich_text.append(part, style=f"bold {COLORS['user']}")

    if pasted_blocks:
        suffix = "s" if pasted_blocks > 1 else ""
        rich_text.append(
            f" ({pasted_blocks} pasted block{suffix}, {pasted_lines} lines total)",
            style="dim",
        )

    for index, image in enumerate(images or [], start=1):
        if rich_text.plain:
            rich_text.append("\n")
        rich_text.append(
            f"📎 Image #{index} attached: {image.name} ({_format_attachment_size(image)})",
            style="dim",
        )

    # Echo historical user queries on a subtle full-width background so they are
    # easy to find while scanning a long conversation. No trailing blank line here:
    # the response section (start_tool_section / _start_response) adds its spacing.
    # A separate content column keeps wrapped and explicit continuation lines aligned.
    history = Table.grid(padding=(0, 1))
    history.add_column(no_wrap=True)
    history.add_column(ratio=1)
    history.add_row(Text("❯", style="bold bright_yellow"), rich_text)
    console = get_console()
    console.print()
    console.print(Padding(history, (0, 1), style="on rgb(35,35,35)"))


def get_file_completions(document_text: str) -> List[str]:
    """Get file completions for @ mentions."""
    import glob as glob_module

    # Find the @ mention being typed
    match = re.search(r"@([\w./-]*)$", document_text)
    if not match:
        return []
    
    partial = match.group(1)
    
    if partial:
        # Search for files matching the partial path (current dir only, not recursive)
        search_pattern = f"{partial}*"
        matches = glob_module.glob(search_pattern, recursive=False)
        # Also search one level of subdirectories (limited depth)
        if os.sep not in partial and "/" not in partial:
            for d in os.listdir("."):
                if os.path.isdir(d) and not d.startswith("."):
                    sub_matches = glob_module.glob(os.path.join(d, f"{partial}*"))
                    matches.extend(sub_matches[:5])
    else:
        # Show files in current directory
        matches = glob_module.glob("*")
    
    # Filter to only files (not directories) and limit results
    completions = []
    seen = set()
    for m in matches[:20]:
        if m in seen:
            continue
        seen.add(m)
        if os.path.isfile(m):
            completions.append(m)
        elif os.path.isdir(m):
            completions.append(m + "/")
    
    return completions


def show_help(skills_registry=None):
    """Display categorized help information."""
    categories = {
        "Session": {
            "/new":             "Start a new chat session",
            "/clear, /reset":   "Clear screen and reset conversation",
            "/rename <name>":   "Name current session for easy resume",
            "/resume [target]": "Resume by number, name, or id prefix",
            "/history":         "Show conversation history or full tool details",
            "/save, /export":   "Save conversation to JSON (no system prompts)",
            "/retry":           "Retry the last message (resend to agent)",
            "/undo":            "Remove the last user/assistant exchange",
            "/compact":         "Compact context (summarize history)",
            "/btw <question>":  "Ephemeral side question (no tools, not saved)",
            "/queue":           "Queue: <prompt> | list | clear | remove <n>",
            "/steer <text>":    "Guide the running agent mid-task (no interrupt)",
            "/checkpoint":      "Durable file snapshots: list | create | diff | restore",
            "/background":      "Run prompt in background (/bg alias)",
            "/ps":              "List background agents and terminals",
            "/stop":            "Stop background agents and terminals",
        },
        "Configure": {
            "/model [p/m]":     "Show or switch model",
            "/config":          "Show current configuration",
            "/usage":           "Token usage, cost, and what fills the context",
            "/debug":           "Show debug info (model, history count)",
            "/reasoning":       "Toggle reasoning display: on | off",
            "/statusbar, /sb":  "Toggle the status bar",
        },
        "Tools & Skills": {
            "/tools":           "Manage tools: add | remove | info | search",
            "/skills":          "Manage skills: search | browse | install | remove | inspect | tap",
        },
        "Automation": {
            "/cron":            "Scheduled jobs: list | add \"<prompt>\" <schedule> | edit | pause | resume | remove | runs | run | daemon on/off",
        },
        "Permissions": {
            "/permissions":     "View or set mode (ask/auto/allow-all)",
        },
        "Other": {
            "/help":            "Show this help message",
            "/exit, /quit":     "Exit and show the resume command",
        },
    }

    get_console().print()
    get_console().print("  [bold]Available Commands[/bold]")
    get_console().print()

    for category, commands in categories.items():
        get_console().print(f"  [bold]-- {category} --[/bold]")
        for cmd, desc in commands.items():
            get_console().print(f"    [bright_green]{cmd:<18}[/bright_green] [dim]{desc}[/dim]")
        get_console().print()

    # Skill auto-commands
    if skills_registry and len(skills_registry) > 0:
        skill_cmds = skills_registry.auto_commands()
        if skill_cmds:
            get_console().print("  [bold]-- Skill Commands --[/bold]")
            for slug, skill in skill_cmds.items():
                desc = skill.description[:50] if skill.description else ""
                get_console().print(f"    [bright_green]{slug:<18}[/bright_green] [dim]{desc}[/dim]")
            get_console().print()

    get_console().print("  [bold]Keyboard Shortcuts[/bold]")
    get_console().print()
    shortcuts = {
        "Enter":             "Submit your message",
        "Ctrl+X":            "Toggle Agent/Shell mode ($ = shell, > = agent)",
        "Ctrl+J, Alt+Enter": "Insert newline for multi-line input",
        "Ctrl+D":            "Exit and show resume command",
        "Ctrl+C":            "Interrupt current operation; press twice to exit",
        "Tab, Right Arrow":  "Accept completion / auto-suggestion",
        "Ctrl+V":            "Paste image from clipboard",
        "Ctrl+O":            "Expand truncated tool commands and output in pager",
    }
    for key, desc in shortcuts.items():
        get_console().print(f"    [bright_green]{key:<20}[/bright_green] [dim]{desc}[/dim]")
    get_console().print()

    get_console().print("  [bold]Input Features[/bold]")
    get_console().print()
    get_console().print("    [bright_green]@filename[/bright_green]           Reference a file - content injected into prompt")
    get_console().print("    [bright_green]/command[/bright_green]            Type / to see slash commands with auto-complete")
    get_console().print()
    get_console().print("  [dim]Tip: type your message and press Enter to chat![/dim]")
    get_console().print()


def _extract_filename(file_path: str) -> str:
    """Extract filename from a file path."""
    return Path(file_path).name


def _format_line_range(offset: int, limit: int) -> str:
    """Format line range as L{start}-{end}."""
    start = offset + 1 if offset else 1
    end = start + (limit or 500) - 1
    return f"L{start}-{end}"


def _shorten_path(file_path: str) -> str:
    """Shorten a file path for display: prefer relative path, keep outside paths intact."""
    if not file_path or file_path == ".":
        return "."
    p = Path(file_path)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _shorten_paths_in_command(command: str) -> str:
    """Shorten absolute paths embedded in a shell command."""
    cwd = str(Path.cwd())
    if cwd in command:
        command = command.replace(cwd + "/", "").replace(cwd, ".")
    return command


def _wrap_command_lines(command: str, width: int) -> List[str]:
    """Wrap a shell command for display while preserving explicit newlines.

    Long indivisible tokens (URLs, IDs, paths) stay intact instead of being
    split into misleading fragments. The full command is retained separately
    for Ctrl+O whenever the inline preview is folded.
    """
    wrapped: List[str] = []
    logical_lines = command.splitlines() or [""]
    for line in logical_lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(
            line,
            width=max(1, width),
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""])
    return wrapped


def _display_execute_command(console_instance, command: str) -> None:
    """Render an execute command as one header plus two continuation rows."""
    raw_command = str(command or "")
    display_command = _shorten_paths_in_command(raw_command)
    icon = TOOL_ICONS.get("execute", TOOL_ICONS["default"])
    header = f" {icon} execute "
    continuation = "   │ "
    width = max(1, int(getattr(console_instance, "width", 80) or 80) - len(header))
    command_lines = _wrap_command_lines(display_command, width)
    visible_lines = command_lines[:3]
    omitted = len(command_lines) - len(visible_lines)

    for index, line in enumerate(visible_lines):
        rendered = Text()
        if index == 0:
            rendered.append(f" {icon} ")
            rendered.append("execute", style="bold magenta")
            rendered.append(" ")
        else:
            rendered.append(continuation, style="dim")
        rendered.append(line, style="dim")
        console_instance.print(rendered)

    if omitted > 0:
        hint = Text()
        hint.append(continuation, style="dim")
        hint.append(f"… +{omitted} lines (Ctrl+O to expand)", style="dim italic")
        console_instance.print(hint)
        remember_truncated("Command · execute", raw_command)


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool call for user-friendly display."""
    # File reading tools - show filename and line range
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        filename = _shorten_path(file_path)
        offset = tool_args.get("offset", 0)
        limit = tool_args.get("limit", 500)
        line_range = _format_line_range(offset, limit)
        return f"{filename} ({line_range})"
    
    # File writing tools - show filename only
    if tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)
    
    # File editing tools - show filename only
    if tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        return _extract_filename(file_path)

    if tool_name == "apply_patch":
        patch = str(tool_args.get("patch", ""))
        count = len(re.findall(r"^\*\*\* (?:Add|Update|Delete) File: ", patch, re.MULTILINE))
        return f"{count} {'file' if count == 1 else 'files'}" if count else ""
    
    # Execute command - shorten absolute paths in command
    if tool_name == "execute":
        command = tool_args.get("command", "")
        return _shorten_paths_in_command(command)
    
    # Todo tools - list the todo items (show ALL todos, no truncation)
    if tool_name == "write_todos":
        todos = tool_args.get("todos", [])
        if isinstance(todos, list) and todos:
            todo_lines = []
            for todo in todos:
                if isinstance(todo, dict):
                    content = todo.get("content", "")
                    status = todo.get("status", "pending")
                    status_icon = "✓" if status == "completed" else "○" if status == "pending" else "◐"
                    todo_lines.append(f"{status_icon} {content}")
                else:
                    todo_lines.append(f"○ {str(todo)}")
            return "\n    ".join(todo_lines)
        return f"{len(todos)} items"
    
    # Web search - show search queries
    if tool_name == "web_search":
        queries = tool_args.get("queries", "")
        if isinstance(queries, list):
            return ", ".join(str(q)[:40] for q in queries[:3])
        return str(queries)[:80]
    
    # Fetch URL - show the URL
    if tool_name == "fetch_url":
        url = tool_args.get("url", "")
        if len(url) > 60:
            return url[:57] + "..."
        return url
    
    # ls/glob/grep - show shortened path/pattern
    if tool_name == "ls":
        directory = tool_args.get("directory", ".")
        return _shorten_path(directory)

    if tool_name == "glob":
        pattern = tool_args.get("pattern", "*")
        path = tool_args.get("path", ".")
        return f"{pattern} in {_shorten_path(path)}"

    if tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", ".")
        include = tool_args.get("include", "")
        display = f"'{pattern[:40]}' in {_shorten_path(path)}"
        if include:
            display += f" ({include})"
        return display
    
    # Task tool - show description
    if tool_name == "task":
        description = tool_args.get("description", "")
        if len(description) > 80:
            return description[:77] + "..."
        return description
    
    # Default format for other tools
    brief_args = []
    for key, value in tool_args.items():
        if isinstance(value, str):
            if len(value) > 40:
                value = value[:37] + "..."
            brief_args.append(f"{key}={value!r}")
        elif isinstance(value, (int, float, bool)):
            brief_args.append(f"{key}={value}")
        elif isinstance(value, list):
            brief_args.append(f"{key}=[{len(value)} items]")
        elif isinstance(value, dict):
            brief_args.append(f"{key}={{...}}")
    
    args_str = ", ".join(brief_args[:3])
    if len(brief_args) > 3:
        args_str += ", ..."
    
    return args_str if args_str else ""


def _display_tool_impl(console_instance, tool_name: str, tool_args: dict,
                       tool_count: int = 0) -> None:
    """Shared implementation for displaying a tool call."""
    icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
    display_str = format_tool_display(tool_name, tool_args)

    # Add blank line between tools for readability
    if tool_count > 1:
        console_instance.print()

    if tool_name == "execute":
        _display_execute_command(console_instance, tool_args.get("command", ""))
    # Special handling for write_todos - multi-line display.
    # Note: in this repo "task" is the dedicated subagent-spawn tool, so we
    # avoid using "tasks" as the label here to prevent confusion.
    elif tool_name == "write_todos" and "\n" in display_str:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta]:")
        console_instance.print(f"    {display_str}", style="dim")
    elif display_str:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta] [dim]{display_str}[/dim]")
    else:
        console_instance.print(f" {icon} [bold magenta]{tool_name}[/bold magenta]")


class StreamDisplayManager:
    """Manages CLI output display state for streaming responses.

    Assistant output uses a left gutter (▏) instead of a box.
    Thinking/reasoning uses a left gutter (▎) in magenta.
    """

    # Subagent rendering verbosity. Three explicit modes (see PR notes):
    #   "all"     — default. Show ``tool_started`` only (one line per call,
    #               consecutive same-tool dedup, ``[N]`` prefix when multiple
    #               subagents are concurrently active). Final response shown.
    #   "verbose" — also show ``tool_completed`` with elapsed time.
    #   "off"     — silent during execution; only the final response summary
    #               at ``subagent.end`` is shown.
    SUBAGENT_VERBOSITIES = ("all", "verbose", "off")

    def __init__(self, console_instance, subagent_verbosity: str = "all",
                 work_dir: Optional[Path] = None):
        self.console = console_instance
        configured_work_dir = (work_dir or Path.cwd()).expanduser().absolute()
        self._work_dir_input = configured_work_dir
        self._work_dir = configured_work_dir.resolve()
        self._raw_console = console_instance
        # Post-redesign: assistant/thinking output is emitted as plain text
        # (no left-side gutter bar). Only the user query keeps a ``▎`` prefix,
        # which acts as the sole visual delimiter between the human question
        # and the AI response body. The ``_assistant_console`` alias remains
        # so the rest of ``StreamDisplayManager`` doesn't have to branch, but
        # it now points at the raw console — the ``_GutteredConsole`` proxy
        # is only constructed on-the-fly inside ``display_user_message``.
        self._assistant_console = console_instance
        self._term_width = min(console_instance.width or 80, 120)
        if subagent_verbosity not in self.SUBAGENT_VERBOSITIES:
            subagent_verbosity = "all"
        self._subagent_verbosity = subagent_verbosity
        self._cli_markdown_mode = str(get_setting("cli_markdown", "auto") or "auto").lower()
        if self._cli_markdown_mode not in {"off", "auto", "on"}:
            self._cli_markdown_mode = "auto"
        self._turn_started_at = None
        # Populated by ``finalize()`` from its kwargs, then read back by
        # ``_build_turn_summary()``. Kept as instance state (rather than
        # threaded as params through the render helper) because the summary
        # is built inside ``finalize`` after the last render step and it's
        # cleaner than yet another positional-args tuple.
        self._summary_turn_no: int | None = None
        self._summary_delta_tokens: int | None = None
        self._summary_delta_cost_usd: float | None = None
        self._thinking_buffer = ""
        self._thinking_console = None
        self.reset()

    def reset(self):
        """Reset state for a new response."""
        self.in_thinking = False
        self.thinking_shown = False
        self.tool_count = 0
        self.in_tool_section = False
        self.response_started = False
        self.has_content_output = False
        self._response_buffer = []
        # Buffer for the CURRENT response segment — the run of assistant text
        # since the last boundary (tool call / thinking start) or turn start.
        # Flushed as plain text by ``_flush_segment_as_plain_text`` at the next
        # boundary (so it lands in the LLM's native order), or rendered as
        # Markdown by ``finalize`` when it turns out to be the final answer.
        self._segment_text = ""
        self._turn_started_at = None
        # Populated by ``finalize()`` from its kwargs, then read back by
        # ``_build_turn_summary()``. Kept as instance state (rather than
        # threaded as params through the render helper) because the summary
        # is built inside ``finalize`` after the last render step and it's
        # cleaner than yet another positional-args tuple.
        self._summary_turn_no: int | None = None
        self._summary_delta_tokens: int | None = None
        self._summary_delta_cost_usd: float | None = None
        self._thinking_buffer = ""
        self._thinking_console = None
        # Set of "task" tool_call_ids (or just a counter) for which we have
        # already streamed live subagent steps; the after-completion summary
        # in display_tool_result() should be suppressed in that case.
        self._subagent_live_shown = 0
        # Per-run state for batch prefixes + consecutive-tool dedup.
        # ``_subagent_index`` maps run_id → 1-based slot. Slots are reclaimed
        # at ``subagent.end`` so a long-lived parent can run many batches
        # without leaking. ``_subagent_last_tool`` stores the previous
        # (tool_name, info) per run_id to suppress consecutive duplicates
        # (e.g. an agent retrying the same read_file call).
        self._subagent_index: "OrderedDict[str, int]" = OrderedDict()
        self._subagent_last_tool: Dict[str, tuple] = {}
        self._next_subagent_slot: int = 0
        # File content captured at write-tool START time (before mutation) so
        # completion can render one real old→new file diff. Keyed by tool call
        # ID when available, otherwise by the raw file_path argument.
        self._write_old: Dict[str, Optional[str]] = {}
        # apply_patch can mutate several files atomically, so retain every
        # target's action and pre-call content for one combined real diff.
        self._patch_old: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}
        # tool_call_id of the tool block currently at the bottom of the
        # transcript — i.e. the call line (or merged block) printed last. A
        # ``⎿`` result body only reads as "belongs to the line above" while
        # this still matches its own id; see ``_print_result_anchor``.
        self._open_block_id: Optional[str] = None
        # Truncated blocks are cleared at the start of each user turn in
        # display_user_message(), NOT here. Clearing here would wipe the
        # user's just-remembered long query (display_user_message runs before
        # the manager is created), which used to make Ctrl+O show only tool
        # results and never the folded query.

    # No more box methods; gutter is used instead.

    def _flush_segment_as_plain_text(self):
        """Emit the current buffered segment as plain text and reset it.

        Called at every boundary that produces its own live output —
        ``start_thinking`` and ``start_tool_section``. Reaching such a
        boundary proves the preceding assistant text was a *mid-turn
        preamble* (not the final answer), so we print it now, right before
        the thinking / tool output, preserving the LLM's native emission
        order. The final segment, in contrast, stays buffered until
        ``finalize`` where it is rendered as Markdown in one shot.
        """
        if not self._segment_text:
            return
        # Split on newlines and print through the gutter console so lines
        # line up with the rest of the assistant output. Trailing partial
        # line (no ``\n``) still gets printed as its own line.
        for line in self._segment_text.split("\n"):
            self._assistant_console.print(line, highlight=False, markup=False)
        self._segment_text = ""

    def start_thinking(self):
        """Start thinking section.

        Post-redesign: no left-side gutter — the thinking segment is just
        raw text on the raw console, distinguished from the assistant's
        answer by an ``italic dim magenta`` style applied at print time.
        """
        if not self.thinking_shown:
            # Any assistant text buffered before this thinking block is a
            # mid-turn preamble (the model spoke, then thought). Flush it as
            # plain text FIRST so it appears above the thinking, matching the
            # LLM's native ``text → thinking`` emission order instead of being
            # held back until the next tool call.
            self._flush_segment_as_plain_text()
            self._raw_console.print()
            self.thinking_shown = True
            self.in_thinking = True

    def stream_thinking(self, content: str):
        """Stream thinking content with line-buffered output."""
        self._thinking_buffer += content
        while "\n" in self._thinking_buffer:
            line, self._thinking_buffer = self._thinking_buffer.split("\n", 1)
            self._raw_console.print(line, style="italic dim magenta", highlight=False, markup=False)

    def end_thinking(self):
        """End thinking section."""
        if self.in_thinking:
            # Flush thinking buffer
            if self._thinking_buffer:
                self._raw_console.print(self._thinking_buffer, style="italic dim magenta", highlight=False, markup=False)
                self._thinking_buffer = ""
            self.in_thinking = False
            self.response_started = False

    def start_tool_section(self):
        """Start tool section.

        Tool calls are part of the assistant's response, so they must render
        UNDER the ``╭─ Response ─╮`` header exactly like the text does. When a
        turn leads with a tool call and no preamble text (very common — the
        model often goes straight to a tool), the box hasn't been opened yet,
        so open it here. Without this the *first* tool call prints bare above
        the box while later tool calls (which follow some text) sit inside it —
        the "偶发第一个 tool call 跑到 Response 前面" inconsistency.
        """
        if not self.in_tool_section:
            if self.in_thinking:
                self.end_thinking()
            # Any assistant text buffered so far is now known to be a
            # mid-turn preamble (the model paused to call a tool). Flush
            # it to the screen as plain text so the user can read what the
            # model said before the tool call runs.
            self._flush_segment_as_plain_text()
            if not self.response_started:
                self.console.print()
                if self._turn_started_at is None:
                    self._turn_started_at = time.monotonic()
                self.response_started = True
            elif self.has_content_output:
                self.console.print()
            self.in_tool_section = True

    def display_tool(self, tool_name: str, tool_args: dict,
                     tool_call_id: Optional[str] = None):
        """Display a single tool call.

        Every tool call counts toward the per-turn total reported in the closing
        separator (``… · N tools``), INCLUDING read-only / write-diff tools whose
        call line is deferred to completion. Otherwise a turn that only ran
        ``read_file`` / ``grep`` / ``edit_file`` etc. would show "0 tools" and
        confuse the user — the visible tool calls and the reported count must
        agree.

        Read-only tools (``_DEFERRED_TOOLS``) skip the start-time call line and
        collapse into a single completion line that folds in elapsed time, e.g.
        ``  🔎 grep 'pat' in path - 5 lines (13ms)``. The live spinner still
        announces the running tool, so deferring the print costs no feedback.
        """
        # Count every tool call up front, before any deferred-print early
        # return, so the turn summary's "N tools" matches what the user saw.
        self.tool_count += 1
        if tool_name in self._DEFERRED_TOOLS:
            return
        if tool_name == "apply_patch":
            self._capture_patch_before_call(tool_args, tool_call_id)
            return
        if tool_name in self._WRITE_DIFF_TOOLS:
            self._capture_file_before_call(tool_name, tool_args, tool_call_id)
            return
        self.start_tool_section()
        _display_tool_impl(self._assistant_console, tool_name, tool_args, self.tool_count)
        self._open_block_id = tool_call_id

    def _resolve_diff_path(self, raw_path: str) -> Path:
        """Resolve display-only paths against the file tool's work directory."""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self._work_dir / path
        return path

    def _display_path(self, raw_path: str) -> str:
        """Return a normalized path relative to the configured work directory."""
        if not raw_path:
            return ""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self._work_dir_input / path
        lexical_path = Path(os.path.abspath(path))
        for work_root in (self._work_dir_input, self._work_dir):
            try:
                return lexical_path.relative_to(work_root).as_posix()
            except ValueError:
                continue
        return lexical_path.name

    def _shorten_workdir_text(self, content: str) -> str:
        """Replace absolute work-dir paths in tool output with relative paths."""
        roots = {str(self._work_dir_input), str(self._work_dir)}
        shortened = content
        for root in sorted(roots, key=len, reverse=True):
            if shortened == root:
                shortened = "."
            else:
                shortened = shortened.replace(root + os.sep, "")
        return shortened

    @staticmethod
    def _read_diff_path(path: Path) -> Optional[str]:
        """Read a small text file for display-only diffing."""
        try:
            if path.is_file() and path.stat().st_size < 512_000:
                return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
        return None

    def _capture_file_before_call(self, tool_name: str, tool_args: dict,
                                  tool_call_id: Optional[str] = None) -> None:
        """Capture a write target before execution for a real result diff."""
        fp = tool_args.get("file_path")
        if not fp:
            return
        key = tool_call_id or str(fp)
        path = self._resolve_diff_path(str(fp))
        self._write_old[key] = self._read_diff_path(path) if path.exists() else ""

    def _capture_patch_before_call(self, tool_args: dict,
                                   tool_call_id: Optional[str] = None) -> None:
        """Capture every apply_patch target before its atomic mutation."""
        patch = tool_args.get("patch")
        if not isinstance(patch, str):
            return
        try:
            operations = parse_patch_envelope(patch)
        except ValueError:
            return
        key = tool_call_id or patch
        self._patch_old[key] = [
            (
                operation.path,
                operation.action,
                None if operation.action == "add" else self._read_diff_path(
                    self._resolve_diff_path(operation.path)
                ),
            )
            for operation in operations
        ]

    @staticmethod
    def _fmt_elapsed(elapsed: Optional[float]) -> str:
        """Format elapsed seconds with ms-precision under 1s.

        Every tool call has a non-zero cost (subprocess spawn, file I/O,
        even pure-python work), so we always surface a number when one is
        provided — fast tools just report ``<1ms`` instead of being hidden.

        - None or negative   → ''            (no measurement available)
        - < 1ms              → ' (<1ms)'
        - < 1s               → ' (Nms)'      e.g. ' (5ms)', ' (120ms)'
        - < 10s              → ' (N.NNs)'    e.g. ' (1.23s)'
        - >= 10s             → ' (N.Ns)'     e.g. ' (12.3s)'
        """
        if elapsed is None or elapsed < 0:
            return ""
        if elapsed < 0.001:
            return " (<1ms)"
        if elapsed < 1.0:
            return f" ({int(round(elapsed * 1000))}ms)"
        if elapsed < 10.0:
            return f" ({elapsed:.2f}s)"
        return f" ({elapsed:.1f}s)"

    @staticmethod
    def _result_count_summary(tool_name: str, result_content: str) -> str:
        """One-word count summary for a deferred read-only tool's result."""
        if not result_content:
            return "no matches" if tool_name == "grep" else ""
        # read_file appends a "[Showing lines A-B of N total lines]" footer
        # only when truncated; the span encodes the exact content returned,
        # so count from that rather than the wrapped string (which would
        # inflate the number with the footer line). Anchor to the FINAL line
        # and the "[Showing lines" prefix — a bare "N-M" search can
        # false-match identical text in file content.
        if tool_name == "read_file":
            lines = str(result_content).rstrip().splitlines()
            if lines and re.match(r"\[File metadata: .+\]$", lines[-1]):
                lines = lines[:-1]
                while lines and lines[-1] == "":
                    lines = lines[:-1]
            tail = lines[-1] if lines else ""
            m = re.match(r"\[Showing lines (\d+)-(\d+) of \d+ total lines\]$", tail)
            if m:
                # Clamp reads past EOF (range start > end) to 0.
                n = max(0, int(m.group(2)) - int(m.group(1)) + 1)
                return f"{n} lines"
            # Empty file: the result is a system-reminder, not content lines.
            if result_content.startswith("<system-reminder>"):
                return "0 lines"
            n = len(lines)
            return f"{n} lines" if n else ""
        n = len(str(result_content).splitlines())
        if n == 0:
            return "no matches" if tool_name == "grep" else ""
        if tool_name == "grep":
            return f"{n} lines"
        if tool_name == "ls":
            return f"{n} items"
        if tool_name == "glob":
            return f"{n} files"
        if tool_name == "web_search":
            return f"{n} results"
        return f"{n} lines"

    def _display_deferred_merged(self, tool_name: str, tool_args: dict,
                                 result_content: str, is_error: bool,
                                 elapsed_str: str) -> None:
        """Print the single merged line for a deferred read-only tool.

        Format: ``  {icon} {name} {params} - {count} {elapsed}``
        (errors surface a truncated message instead of the count).
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        params = format_tool_display(tool_name, tool_args)
        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if params:
            line += f" [dim]{params}[/dim]"
        if is_error:
            err = str(result_content).replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(f"Tool error · {tool_name}", str(result_content))
            line += f" [red]- error: {err}{elapsed_str}[/red]"
        else:
            summary = self._result_count_summary(tool_name, result_content)
            if summary:
                line += f" [dim]- {summary}{elapsed_str}[/dim]"
            else:
                line += f" [dim]{elapsed_str}[/dim]"
        self._assistant_console.print(line)

    def _display_edit_merged(self, tool_name: str, tool_args: dict,
                             result_content: str, is_error: bool,
                             elapsed_str: str,
                             tool_call_id: Optional[str] = None) -> None:
        """One summary line + the FULL unified diff for edit tools.

        ``  ✎ edit_file config.py - Edited 1 file (+1 -1) (120ms)`` followed
        by one complete real-file diff. Errors surface a truncated message.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        raw_path = str(tool_args.get("file_path", ""))
        display_path = self._display_path(raw_path)

        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if display_path:
            line += f" [dim]{display_path}[/dim]"
        key = tool_call_id or tool_args.get("file_path", "")
        old_content = self._write_old.pop(key, None)
        if is_error:
            err = self._shorten_workdir_text(str(result_content)).replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(
                    f"Tool error · {tool_name}",
                    self._shorten_workdir_text(str(result_content)),
                )
            line += f" [red]- error: {err}{elapsed_str}[/red]"
            self._assistant_console.print(line)
            return

        new_content = self._read_diff_target(tool_args)
        diff_text = self._build_file_diff(old_content, new_content, display_path)
        added, removed = self._diff_line_counts(diff_text)
        line += f" [dim]- Edited 1 file (+{added} -{removed}){elapsed_str}[/dim]"
        self._assistant_console.print(line)
        if not diff_text:
            return
        self._assistant_console.print(Syntax(diff_text + "\n", "diff", theme="monokai",
                                  line_numbers=False))

    def _display_patch_summary(self, result_content: str, is_error: bool,
                               elapsed_str: str, tool_args: dict,
                               tool_call_id: Optional[str] = None) -> None:
        """Render apply_patch as one summary plus its real multi-file diff."""
        icon = TOOL_ICONS.get("apply_patch", TOOL_ICONS["default"])
        line = f"  {icon} [bold magenta]apply_patch[/bold magenta]"
        content = self._shorten_workdir_text(str(result_content).strip())
        key = tool_call_id or tool_args.get("patch", "")
        old_files = self._patch_old.pop(key, [])
        if is_error:
            self._assistant_console.print(line + f" [red]- error{elapsed_str}[/red]")
            error_lines = content.splitlines() or ["Unknown patch error"]
            max_lines = 8
            truncated = len(error_lines) > max_lines or any(len(item) > 120 for item in error_lines)
            for index, error_line in enumerate(error_lines[:max_lines]):
                if len(error_line) > 120:
                    error_line = error_line[:117] + "..."
                prefix = "    ⎿ " if index == 0 else "      "
                self._assistant_console.print(
                    f"{prefix}{error_line}",
                    style="dim red",
                    highlight=False,
                    markup=False,
                )
            remaining = len(error_lines) - max_lines
            if truncated:
                detail = f"{remaining} more lines" if remaining > 0 else "full error"
                self._assistant_console.print(
                    f"      ... ({detail} · Ctrl+O to expand)", style="dim italic"
                )
                remember_truncated("Tool error · apply_patch", content)
            return

        summary, _, details = content.partition("\n")
        summary = re.sub(r"^Successfully applied patch to ", "Edited ", summary)
        self._assistant_console.print(line + f" [dim]- {summary}{elapsed_str}[/dim]")
        diffs = []
        for raw_path, action, old_content in old_files:
            new_content = "" if action == "delete" else self._read_diff_path(
                self._resolve_diff_path(raw_path)
            )
            if action == "add":
                old_content = ""
            diff_text = self._build_file_diff(
                old_content,
                new_content,
                self._display_path(raw_path),
            )
            if diff_text:
                diffs.append(diff_text)
        if diffs:
            self._assistant_console.print(
                Syntax("\n".join(diffs) + "\n", "diff", theme="monokai", line_numbers=False)
            )
            return

        detail_lines = details.splitlines()
        file_lines = []
        while detail_lines and re.match(r"^[MAD] .+ \(\+\d+ -\d+\)$", detail_lines[0]):
            file_lines.append(detail_lines.pop(0))
        for index, file_line in enumerate(file_lines):
            branch = "└" if index == len(file_lines) - 1 else "├"
            self._assistant_console.print(
                f"    {branch} {file_line}", style="dim", highlight=False, markup=False
            )
        if detail_lines:
            self._assistant_console.print(
                "\n".join(detail_lines), style="dim", highlight=False, markup=False
            )

    def _read_diff_target(self, tool_args: dict) -> Optional[str]:
        """Read a write tool's post-call file content for display-only diffing."""
        fp = tool_args.get("file_path")
        if not fp:
            return None
        return self._read_diff_path(self._resolve_diff_path(str(fp)))

    @staticmethod
    def _build_file_diff(old_content: Optional[str], new_content: Optional[str],
                         filename: str) -> str:
        """Build one git-style unified diff from real pre/post file contents."""
        if old_content is None or new_content is None:
            return ""
        unified_lines = list(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=filename,
            tofile=filename,
            n=2,
            lineterm="",
        ))
        if not unified_lines:
            return ""
        hunks = "\n".join(unified_lines[2:]).rstrip("\n")
        return f"diff -- {filename}\n{hunks}"

    @staticmethod
    def _diff_line_counts(diff_text: str) -> tuple[int, int]:
        """Count added/removed content lines, excluding unified-diff headers."""
        lines = diff_text.splitlines()
        added = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
        removed = sum(line.startswith("-") and not line.startswith("---") for line in lines)
        return added, removed

    def _display_write_merged(self, tool_name: str, tool_args: dict,
                              result_content: str, is_error: bool,
                              elapsed_str: str,
                              tool_call_id: Optional[str] = None) -> None:
        """One summary line + the FULL old→new diff for write_file.

        Keeps the existing created/updated line-count summary and renders the
        same git-style real-file diff as edit tools. For a brand-new file the
        old side is empty. Successful diffs are never folded.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        raw_path = str(tool_args.get("file_path", ""))
        display_path = self._display_path(raw_path)
        new_content = str(tool_args.get("content", ""))
        result_str = str(result_content)
        key = tool_call_id or tool_args.get("file_path", "")
        old_content = self._write_old.pop(key, None)

        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if display_path:
            line += f" [dim]{display_path}[/dim]"
        if is_error:
            shortened_result = self._shorten_workdir_text(result_str)
            err = shortened_result.replace("\n", " ").strip()
            if len(err) > 80:
                err = err[:77] + "..."
                remember_truncated(f"Tool error · {tool_name}", shortened_result)
            line += f" [red]- error: {err}{elapsed_str}[/red]"
            self._assistant_console.print(line)
            return

        diff_text = self._build_file_diff(old_content, new_content, display_path)
        n_lines = len(new_content.splitlines())
        verb = "created" if "Created" in result_str else "updated"
        unit = "line" if n_lines == 1 else "lines"
        line += f" [dim]- ✓ {verb} {n_lines} {unit}{elapsed_str}[/dim]"
        self._assistant_console.print(line)
        if not diff_text:
            return
        # Render the FULL diff so the user always sees the complete file change,
        # never folded behind Ctrl+O.
        self._assistant_console.print(Syntax(diff_text + "\n", "diff", theme="monokai",
                                  line_numbers=False))

    # Read-only tools whose call line is deferred to completion so the call
    # line and elapsed time collapse into ONE line, e.g.
    # ``  🔎 grep 'pat' in path - 5 lines (13ms)``. No separate result footer.
    _DEFERRED_TOOLS = frozenset({"glob", "grep", "ls", "read_file", "web_search", "fetch_url"})

    # Single-file write tools: call line is deferred to completion and rendered
    # as one summary line plus the real pre/post unified diff. apply_patch is
    # handled separately from the executor's multi-file result summary.
    _WRITE_DIFF_TOOLS = frozenset({"edit_file", "write_file"})

    # Tools whose success result is pure noise on success. The call line itself
    # already tells the user what happened; errors are still surfaced.
    _SUPPRESS_RESULT_TOOLS = frozenset({"write_todos"})

    # Max result lines shown inline before folding (per-tool overrides below).
    _DEFAULT_MAX_RESULT_LINES = 4
    # execute: show up to this many lines inline; beyond that, a head+tail
    # window with the middle collapsed into a single dim hint line so long
    # command output stays scannable without flooding the transcript.
    _EXECUTE_MAX_INLINE_LINES = 20
    # execute head/tail window — the tail carries the command's final status /
    # output, which is usually what the user needs at a glance.
    _EXECUTE_HEAD_LINES = 10
    _EXECUTE_TAIL_LINES = 10
    _EXECUTE_DIAGNOSTIC_HEAD_LINES = 6
    _EXECUTE_DIAGNOSTIC_TAIL_LINES = 4

    def _print_result_anchor(self, tool_name: str, tool_args: dict) -> None:
        """Re-state the call a detached ``⎿`` result body belongs to.

        Tools render with two different strategies: ``_DEFERRED_TOOLS`` /
        ``_WRITE_DIFF_TOOLS`` emit ONE self-labelled block at completion, while
        every other tool prints its call line at START time (so a long
        ``execute`` is visible while it runs) and its result body at completion.
        The runner emits a whole parallel batch's completions together, so any
        earlier-called deferred tool's block lands *between* such a call line
        and its own body — and ``⎿``, which means "continues the line above",
        then points at the wrong call. One dim line naming the real call makes
        the block unambiguous without giving up the live call line.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        params = format_tool_display(tool_name, tool_args).replace("\n", " ")
        if len(params) > 60:
            params = params[:57] + "..."
        line = f"    ↳ {icon} {tool_name}"
        if params:
            line += f" {params}"
        self._assistant_console.print(line, style="dim")

    def display_tool_result(self, tool_name: str, result_content: str,
                            is_error: bool = False, elapsed: float = None,
                            tool_args: Optional[dict] = None,
                            tool_call_id: Optional[str] = None):
        """Display tool execution result.

        For ``_DEFERRED_TOOLS`` the call line was suppressed at start time, so
        here we emit the merged single line ``icon name params - count (elapsed)``.

        ``tool_call_id`` identifies which call this result belongs to; when the
        transcript's last tool block is a *different* call, a bare ``⎿`` body
        would be misread, so it gets an anchor line first. Callers that render
        one tool at a time can omit it.
        """
        elapsed_str = self._fmt_elapsed(elapsed)
        detached = self._open_block_id != tool_call_id
        # This tool's block is the transcript's last one from here on, whichever
        # branch below renders it.
        self._open_block_id = tool_call_id

        if tool_name in self._DEFERRED_TOOLS:
            self.start_tool_section()
            self._display_deferred_merged(
                tool_name, tool_args or {}, result_content, is_error, elapsed_str
            )
            return

        if tool_name == "apply_patch":
            self.start_tool_section()
            self._display_patch_summary(
                result_content, is_error, elapsed_str, tool_args or {}, tool_call_id
            )
            return

        if tool_name in self._WRITE_DIFF_TOOLS:
            self.start_tool_section()
            if tool_name == "write_file":
                self._display_write_merged(
                    tool_name, tool_args or {}, result_content, is_error,
                    elapsed_str, tool_call_id
                )
            else:
                self._display_edit_merged(
                    tool_name, tool_args or {}, result_content, is_error,
                    elapsed_str, tool_call_id
                )
            return

        # Suppress noisy success results; the call line is enough.
        if tool_name in self._SUPPRESS_RESULT_TOOLS and not is_error:
            return

        if not result_content:
            if detached:
                self._print_result_anchor(tool_name, tool_args or {})
            self._assistant_console.print(f"    [dim]⎿ done{elapsed_str}[/dim]")
            return

        if tool_name == "task":
            # Its footer already reads ``⎿ task done (...)`` and the subagent's
            # own steps are nested under it, so it never needs an anchor.
            self._display_task_result(result_content, is_error)
            return

        if detached:
            self._print_result_anchor(tool_name, tool_args or {})

        result_str = _strip_internal_tool_notices(str(result_content))

        if tool_name in ("grep", "glob", "execute", "ls", "read_file"):
            cwd = str(Path.cwd())
            if cwd in result_str:
                result_str = result_str.replace(cwd + "/", "").replace(cwd, ".")

        lines = result_str.splitlines()

        # execute: head/tail window with the middle hidden — the tail carries
        # the command's final status/output, which is usually what the user
        # needs to see at a glance.
        if tool_name == "execute":
            is_diagnostics = _is_diagnostic_execute_result(result_str)
            self._display_head_tail(
                lines,
                self._EXECUTE_DIAGNOSTIC_HEAD_LINES if is_diagnostics else self._EXECUTE_HEAD_LINES,
                self._EXECUTE_DIAGNOSTIC_TAIL_LINES if is_diagnostics else self._EXECUTE_TAIL_LINES,
                prefix="    ⎿ ", cont_prefix="      ",
                style="dim red" if is_error else "dim yellow" if is_diagnostics else "dim",
                error_prefix="    ⎿ ⚠ " if (is_error or is_diagnostics) else None,
                truncated_title=f"Tool output · {tool_name}",
                full_content=result_str,
                elapsed_str=elapsed_str,
            )
            return

        max_lines = self._DEFAULT_MAX_RESULT_LINES
        max_line_width = 120

        style = "dim red" if is_error else "dim"
        prefix = "    ⎿ " if not is_error else "    ⎿ ⚠ "
        cont_prefix = "      "

        display_lines = lines[:max_lines]
        for i, line in enumerate(display_lines):
            if len(line) > max_line_width:
                line = line[:max_line_width - 3] + "..."
            p = prefix if i == 0 else cont_prefix
            self._assistant_console.print(f"{p}{line}", style=style)

        remaining = len(lines) - max_lines
        if remaining > 0:
            self._assistant_console.print(
                f"{cont_prefix}... ({remaining} more lines · Ctrl+O to expand)", style="dim italic"
            )
            remember_truncated(f"Tool output · {tool_name}", result_str)
        if elapsed_str:
            self._assistant_console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")

    def _display_head_tail(self, lines: List[str], head: int, tail: int,
                           *, prefix: str, cont_prefix: str, style: str,
                           error_prefix: Optional[str] = None,
                           truncated_title: str, full_content: str,
                           elapsed_str: str = "",
                           max_line_width: int = 120) -> None:
        """Render a head/tail window with the middle hidden.

        Shows the first ``head`` lines and the last ``tail`` lines; anything in
        between is collapsed into a single dim ``(N hidden lines · Ctrl+O to expand)``
        hint line. The full content is still remembered for on-demand Ctrl+O
        expansion. Used for execute output where the tail carries the command's
        final status.
        """
        first_prefix = error_prefix or prefix
        n = len(lines)
        if n <= head + tail:
            show = lines
            hidden = 0
        else:
            show = lines[:head] + lines[-tail:]
            hidden = n - head - tail

        for i, line in enumerate(show):
            if len(line) > max_line_width:
                line = line[:max_line_width - 3] + "..."
            p = first_prefix if i == 0 else cont_prefix
            self._assistant_console.print(f"{p}{line}", style=style)

        if hidden > 0:
            # Single dim hint line between head and tail (no blank/.../blank
            # separator). The full content is still stashed for Ctrl+O expansion.
            self._assistant_console.print(
                f"{cont_prefix}({hidden} hidden lines · Ctrl+O to expand)",
                style="dim italic",
            )
            remember_truncated(truncated_title, full_content)
        if elapsed_str:
            self._assistant_console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")
    
    def _display_task_result(self, result_content: str, is_error: bool = False):
        """Display subagent task result.

        When live subagent events were already streamed via ``handle_event``,
        skip the per-tool summary (avoid duplication) and just print a brief
        execution-summary footer.
        """
        try:
            data = json.loads(result_content)
        except (ValueError, TypeError):
            self._assistant_console.print(f"    ⎿ {str(result_content)[:120]}", style="dim")
            return

        success = data.get("success", False)
        tool_summary = data.get("tool_calls_summary", [])
        exec_time = data.get("execution_time")
        tool_count = data.get("tool_count", len(tool_summary))

        if not success:
            error_msg = data.get("error", "Unknown error")
            self._assistant_console.print(f"    ⎿ ⚠ {error_msg[:120]}", style="dim red")
            if self._subagent_live_shown > 0:
                self._subagent_live_shown -= 1
            return

        # Live events already rendered the tool calls + final response — only
        # print a one-line summary footer to avoid duplicating output.
        if self._subagent_live_shown > 0:
            self._subagent_live_shown -= 1
            summary_parts = []
            if tool_count > 0:
                summary_parts.append(f"{tool_count} tool uses")
            if exec_time is not None:
                summary_parts.append(f"{exec_time:.1f}s")
            if summary_parts:
                self._assistant_console.print(
                    f"    [dim italic]⎿ task done ({', '.join(summary_parts)})[/dim italic]"
                )
            return

        # Fallback (no live callback registered): render the recap.
        max_shown = 8
        for i, tc in enumerate(tool_summary[:max_shown]):
            name = tc.get("name", "")
            info = tc.get("info", "")
            if len(info) > 90:
                info = info[:87] + "..."
            if i == 0:
                self._assistant_console.print(f"    ⎿ ", end="", style="dim")
            else:
                self._assistant_console.print(f"      ", end="")
            self._assistant_console.print(f"{name}", end="", style="dim bold")
            if info:
                self._assistant_console.print(f" {info}", style="dim")
            else:
                self._assistant_console.print(style="dim")

        if len(tool_summary) > max_shown:
            remaining = len(tool_summary) - max_shown
            self._assistant_console.print(f"      ... and {remaining} more tool calls", style="dim italic")

        summary_parts = []
        if tool_count > 0:
            summary_parts.append(f"{tool_count} tool uses")
        if exec_time is not None:
            summary_parts.append(f"cost: {exec_time:.1f}s")
        if summary_parts:
            summary_str = ", ".join(summary_parts)
            self._assistant_console.print(f"    [dim italic]Execution Summary: {summary_str}[/dim italic]")

    # ------------------------------------------------------------------
    # Live event rendering (subagent progress + compaction)
    # ------------------------------------------------------------------

    # Indent prefix for subagent inner steps; visually nests them under the
    # parent ``task`` tool call line.
    _SUB_INDENT = "    └─ "
    _SUB_CONT_INDENT = "       "

    def handle_event(self, event: dict) -> None:
        """Dispatch a runtime event from the agent (subagent / compression).

        Called synchronously by Runner / BuiltinTaskTool from the asyncio
        thread. While these events fire, the parent run is awaiting tool
        execution or starting a new turn, so the main thread is not mutating
        display state — direct console output is safe.
        """
        et = event.get("type", "")
        if et.startswith("subagent."):
            self._handle_subagent_event(et, event)
        elif et.startswith("compact."):
            self._handle_compact_event(et, event)

    def _subagent_prefix(self, run_id: Optional[str]) -> str:
        """Return ``[N] `` if multiple subagents are concurrently active.

        Single-subagent runs render with no numeric prefix to avoid noise.
        """
        if not run_id or len(self._subagent_index) < 2:
            return ""
        slot = self._subagent_index.get(run_id)
        return f"[dim]\\[{slot}][/dim] " if slot is not None else ""

    def _handle_subagent_event(self, et: str, event: dict) -> None:
        verbosity = self._subagent_verbosity
        run_id = event.get("run_id")

        if et == "subagent.start":
            self._subagent_live_shown += 1
            self._next_subagent_slot += 1
            if run_id:
                self._subagent_index[run_id] = self._next_subagent_slot
                self._subagent_last_tool.pop(run_id, None)

            if verbosity == "off":
                return

            agent_name = event.get("agent_name", "Subagent")
            task = event.get("task", "")
            preview = task.replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            max_turns = event.get("max_turns")
            tool_call_limit = event.get("tool_call_limit")
            budget = ""
            if max_turns is not None and tool_call_limit is not None:
                budget = f" [turns≤{max_turns}, calls≤{tool_call_limit}]"
            elif max_turns is not None:
                budget = f" [turns≤{max_turns}]"
            # Show which model is actually running the subagent: whether a task
            # went to the cheap auxiliary model or the main one is the single
            # most useful thing to know when judging its output.
            model_id = event.get("model_id")
            model_note = f" [dim]({model_id})[/dim]" if model_id else ""
            self._assistant_console.print(
                f"{self._SUB_INDENT}{self._subagent_prefix(run_id)}"
                f"[dim cyan]⮕ {agent_name}[/dim cyan]{model_note} "
                f"[dim italic]{preview}[/dim italic][dim]{budget}[/dim]"
            )

        elif et == "subagent.tool_started":
            if verbosity == "off":
                return
            tool_name = event.get("tool_name", "")
            info = event.get("info", "") or ""
            # Consecutive same-(tool, args) dedup: an agent that retries the
            # exact same call (or a stuck loop) shouldn't produce N identical
            # CLI lines. Only suppress when the previous tool from THIS run
            # had the same key — different runs / interleaved tools still
            # render normally.
            key = (tool_name, info)
            if run_id and self._subagent_last_tool.get(run_id) == key:
                return
            if run_id:
                self._subagent_last_tool[run_id] = key
            if len(info) > 100:
                info = info[:97] + "..."
            line = (
                f"{self._SUB_INDENT}{self._subagent_prefix(run_id)}"
                f"[dim magenta]{tool_name}[/dim magenta]"
            )
            if info:
                line += f" [dim]{info}[/dim]"
            self._assistant_console.print(line)

        elif et == "subagent.tool_completed":
            # Default ``all`` mode is tool-first: completion is hidden because
            # the started line already told the user "agent is doing X".
            # Verbose mode adds completion + elapsed for debugging.
            if verbosity != "verbose":
                # Always surface errors though — silent failures are worse
                # than slightly noisier output.
                if not event.get("is_error"):
                    return
            tool_name = event.get("tool_name", "")
            info = event.get("info", "") or ""
            if len(info) > 100:
                info = info[:97] + "..."
            is_error = event.get("is_error", False)
            elapsed_str = self._fmt_elapsed(event.get("elapsed"))
            prefix = self._subagent_prefix(run_id)
            if is_error:
                self._assistant_console.print(
                    f"{self._SUB_INDENT}{prefix}"
                    f"[dim red]⚠ {tool_name}[/dim red] [dim]{info}[/dim]"
                )
                return
            line = (
                f"{self._SUB_INDENT}{prefix}"
                f"[dim green]✓ {tool_name}[/dim green]"
            )
            if info:
                line += f" [dim]{info}[/dim]"
            if elapsed_str:
                line += f"[dim]{elapsed_str}[/dim]"
            self._assistant_console.print(line)

        elif et == "subagent.end":
            # Reclaim the slot before rendering so the prefix on the final
            # line reflects the active count after this subagent exits.
            if run_id:
                self._subagent_index.pop(run_id, None)
                self._subagent_last_tool.pop(run_id, None)

            response = event.get("response", "") or ""
            if response:
                preview = response.replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                # Final response is shown in every mode (including ``off``):
                # it's the actual answer the parent agent will consume.
                self._assistant_console.print(
                    f"{self._SUB_INDENT}[dim cyan]⤷[/dim cyan] "
                    f"[dim italic]{preview}[/dim italic]"
                )

    def _handle_compact_event(self, et: str, event: dict) -> None:
        # Micro-compact is an expected per-turn maintenance pass and fires too
        # frequently to be useful in the CLI. Keep it silent; surface only the
        # heavier compaction stages that change conversation structure.
        if et == "compact.micro":
            return

        is_main_agent = event.get("is_main_agent") is True
        prefix = "  " if is_main_agent else "    "
        if et == "compact.rule_based":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            self._assistant_console.print(
                f"{prefix}[dim yellow]🗜 compact [/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.2f}s)[/dim]"
            )
        elif et == "compact.auto":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            if is_main_agent:
                count = event.get("compaction_count")
                if isinstance(count, int) and count >= 2:
                    self._assistant_console.print(
                        f"{prefix}[bold yellow]⚠ Context has been auto-compacted {count} times.[/bold yellow]"
                    )
                    self._assistant_console.print(
                        f"{prefix}[yellow]Accuracy may degrade as summaries accumulate; "
                        "consider /new for a focused fresh session.[/yellow]"
                    )
                else:
                    self._assistant_console.print(
                        f"{prefix}[bold yellow]⚠ Context was automatically compacted "
                        "near the model limit.[/bold yellow]"
                    )
                    self._assistant_console.print(
                        f"{prefix}[yellow]Long sessions may become less accurate; "
                        "use /new for a focused fresh session.[/yellow]"
                    )
                return
            self._assistant_console.print(
                f"{prefix}[dim yellow]🗜 compact (auto / LLM-summarised)[/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.1f}s)[/dim]"
            )
        elif et == "compact.reactive":
            before = event.get("before", 0)
            after = event.get("after", 0)
            elapsed = event.get("elapsed", 0.0)
            if is_main_agent:
                self._assistant_console.print(
                    f"{prefix}[bold yellow]⚠ Context exceeded the model limit and was "
                    "compacted before retrying.[/bold yellow]"
                )
                self._assistant_console.print(
                    f"{prefix}[yellow]Long sessions may become less accurate; "
                    "use /new for a focused fresh session.[/yellow]"
                )
                return
            self._assistant_console.print(
                f"{prefix}[dim yellow]🗜 compact (reactive · prompt_too_long)[/dim yellow] "
                f"[dim]{before} → {after} msgs ({elapsed:.1f}s)[/dim]"
            )
    
    def end_tool_section(self):
        """End tool section."""
        if self.in_tool_section:
            self.in_tool_section = False
            self.response_started = False
    
    def start_response(self):
        """Start the assistant response section.

        With the gutter design there is no visible ``open`` — the visual
        signal is the left-side ``▏`` bar that appears on each printed line
        via ``self._assistant_console``. This method only manages state:
        prints a blank spacer line and records the turn start timestamp so
        ``finalize()`` can report elapsed time.
        """
        if not self.response_started:
            if self.in_thinking:
                self.end_thinking()
            if self.in_tool_section:
                self.end_tool_section()
            self.console.print()
            if self._turn_started_at is None:
                self._turn_started_at = time.monotonic()
            self.response_started = True

    def stream_response(self, content: str):
        """Buffer response content silently; render lazily on segment boundary.

        Assistant text is NOT printed token-by-token as it streams. Instead
        each chunk accumulates into ``_segment_text`` and is flushed at one of
        two well-defined moments:

        * On the next boundary that produces its own live output —
          ``start_thinking`` or ``start_tool_section``. Reaching such a
          boundary proves the segment was a *mid-turn preamble*, so it is
          printed as plain text right before the thinking / tool output. This
          keeps the LLM's native emission order (e.g. ``text → thinking →
          tool``) intact, whichever way it interleaves.
        * On ``finalize`` — no further boundary arrived, so the segment is the
          final answer. It is rendered as Markdown in one shot.

        The spinner ("⠋ answering… 3s") stays on the status bar the whole
        time, so the user still gets a heartbeat that the model is producing
        tokens. The tradeoff: the final answer isn't a live typewriter, but it
        also never flash-erases-and-redraws — and preambles land in order.
        """
        self.start_response()
        self._response_buffer.append(content)
        self._segment_text += content
        self.has_content_output = True

    def _should_render_markdown(self, text: str) -> bool:
        if self._cli_markdown_mode == "on":
            return True
        if self._cli_markdown_mode == "off":
            return False
        return _has_markdown(text)

    def _build_turn_summary(self) -> str:
        """Compose the compact summary shown in the closing separator.

        Format (Plan A):

            #N · HH:MM:SS · Xs · +Tk · +$C · N tools

        Field order and meaning:

        * ``#N``       — 1-based turn number within the session. Only shown
                         when caller supplied ``turn_no`` to :meth:`finalize`;
                         acts as a scrollback anchor ("see turn #7").
        * ``HH:MM:SS`` — Wall-clock time the turn ended. Always shown.
        * ``Xs``       — Net execution time (LLM + tools), NOT the wall-clock
                         seen in the status bar's ``⏱``. The two differ by CLI
                         overhead (render / disk / callbacks) — that gap is
                         real and intentional; see docs.
        * ``+Tk``      — Delta tokens produced by this turn (input+output).
                         ``K`` suffix when ≥ 1_000, else raw. Omitted when
                         caller didn't pass ``delta_tokens``.
        * ``+$C``      — Delta USD cost for this turn. Omitted when caller
                         didn't pass ``delta_cost_usd`` or the cost is 0
                         (avoid noise for free/local models).
        * ``N tools``  — Tool calls executed this turn. Omitted when 0.

        The design principle: the closing separator carries **per-turn API
        consumption** (immutable history), while the status bar carries the
        **current session context occupancy** plus session cost/time. Prompt
        tokens resent within a tool loop belong only to the former.
        """
        parts: list[str] = []
        if self._summary_turn_no is not None:
            parts.append(f"#{self._summary_turn_no}")
        parts.append(time.strftime("%H:%M:%S", time.localtime()))
        if self._turn_started_at is not None:
            elapsed = time.monotonic() - self._turn_started_at
            parts.append(f"{elapsed:.1f}s")
        if self._summary_delta_tokens is not None and self._summary_delta_tokens > 0:
            dt = self._summary_delta_tokens
            if dt >= 1000:
                parts.append(f"+{dt / 1000:.1f}K")
            else:
                parts.append(f"+{dt}")
        if self._summary_delta_cost_usd is not None and self._summary_delta_cost_usd > 0:
            parts.append(f"+${self._summary_delta_cost_usd:.2f}")
        if self.tool_count > 0:
            parts.append(f"{self.tool_count} tool{'s' if self.tool_count != 1 else ''}")
        return " " + " · ".join(parts) + " "

    def finalize(
        self,
        turn_no: int | None = None,
        delta_tokens: int | None = None,
        delta_cost_usd: float | None = None,
    ):
        """Finalize the assistant turn: flush buffers and draw the closing rule.

        With the gutter design there is no visible box to close. Instead we
        render a compact dim separator at the bottom whose body carries a
        summary of *what this turn cost* (Plan A layout):

            ──── #7 · 20:29:29 · 6.4s · +3.2K · +$0.08 · 4 tools ────

        The separator is drawn on the *raw* console (no gutter), so it acts
        as a hard boundary between this turn and the next user query.

        Args:
            turn_no: 1-based turn number within the session. Rendered as
                ``#N`` at the head of the summary. When ``None`` the field
                is omitted — useful for tests or synthetic turns that have
                no meaningful ordinal.
            delta_tokens: Tokens consumed by this turn — every prompt token
                (cached ones included) plus output, summed over the turn's API
                calls. A tool loop sends the prompt once per call, so this can
                run well above the context size shown in the status bar; that
                one is the context the next main request will carry.
                Rendered as ``+Tk`` (``+3.2K`` when ≥ 1000, ``+42`` else).
                Omitted when ``None`` or ``<= 0``.
            delta_cost_usd: USD cost incurred by this turn. Rendered as
                ``+$C``. Omitted when ``None`` or ``<= 0`` so free/local
                models don't show a noisy ``+$0.00``.

        The caller (interactive loop) is expected to source ``delta_*``
        from the per-run ``cost_tracker`` (which is itself per-``run()``
        scoped in agentica, so its totals ARE the per-turn deltas — no
        snapshotting needed here).
        """
        self._summary_turn_no = turn_no
        self._summary_delta_tokens = delta_tokens
        self._summary_delta_cost_usd = delta_cost_usd

        if self.in_thinking:
            self.end_thinking()
        if self.in_tool_section:
            self.end_tool_section()

        if self.has_content_output:
            # ``full_text`` (whole turn) drives the render-mode decision so
            # a code fence anywhere in the turn still upgrades the tail to
            # Markdown. The actual render target is the final segment —
            # earlier segments were already emitted as plain text at their
            # trailing thinking/tool boundary.
            full_text = "".join(self._response_buffer)
            tail_text = self._segment_text
            if tail_text:
                if self._should_render_markdown(full_text):
                    # Constrain markdown width to ``console.width - 4`` so
                    # code blocks and headings don't hug the terminal edge
                    # — leaves a small right-side gutter of breathing
                    # room. Without this cap, code fences and heading
                    # underlines stretch to the far right and look loud.
                    md_width = max(20, (self._raw_console.width or 80) - 4)
                    self._assistant_console.print(Markdown(tail_text), width=md_width)
                else:
                    # No markdown features detected — print as plain text,
                    # preserving line breaks exactly as the model sent them.
                    for line in tail_text.split("\n"):
                        self._assistant_console.print(line, highlight=False, markup=False)
                self._segment_text = ""

        # Draw the closing separator only if this turn actually produced
        # output (tool calls, streamed text, or thinking). A no-op turn
        # shouldn't leave a stray separator floating on the screen.
        #
        # The separator is a fixed-width string ``──── SUMMARY ────`` rather
        # than a ``rich.rule.Rule`` because Rule spans the full console width
        # at render time. If the user resizes the terminal afterwards, old
        # Rules already in scrollback keep their original width and look off.
        # A fixed short separator is width-agnostic — it looks the same at
        # 80 cols and at 200 cols, before and after resize.
        if self.has_content_output or self.tool_count > 0:
            summary = self._build_turn_summary().strip()
            edge = "─" * 4
            self.console.print(f"[dim]{edge} {summary} {edge}[/dim]")


def display_tool_call(tool_name: str, tool_args: dict) -> None:
    """Display a tool call with icon and colored tool name."""
    _display_tool_impl(get_console(), tool_name, tool_args)


def _has_markdown(text: str) -> bool:
    """Detect if text contains Markdown formatting worth rendering."""
    if not text:
        return False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith(("- ", "* ", "+ ", "> ")):
            return True
        if any(stripped.startswith(f"{i}. ") for i in range(1, 10)):
            return True
        if "```" in line or "**" in line or "`" in line:
            return True
    return "\n|" in text and "\n|-" in text


def render_markdown_response(console_instance, text: str) -> None:
    """Render a complete response as rich Markdown if it contains formatting."""
    if _has_markdown(text):
        console_instance.print(Markdown(text))
    else:
        console_instance.print(text, style=COLORS["agent"])


def display_diff(console_instance, file_path: str, old_content: str, new_content: str) -> None:
    """Display unified diff between old and new file content."""
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
        lineterm="",
    ))
    if diff_lines:
        diff_text = "\n".join(diff_lines)
        console_instance.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))


def _format_tokens_short(n: int) -> str:
    """Format token count with K/M suffix for compact display."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def context_pct_style(pct: float) -> str:
    """Return Rich style name based on context usage percentage."""
    if pct >= 95:
        return "bold red"
    if pct >= 80:
        return "red"
    if pct >= 50:
        return "yellow"
    return "green"


def build_context_bar(pct: float, width: int = 10) -> str:
    """Build a visual context usage bar like [████░░░░░░]."""
    safe = max(0.0, min(100.0, pct))
    filled = round((safe / 100) * width)
    return f"[{'█' * filled}{'░' * max(0, width - filled)}]"


def display_token_stats(
    console_instance,
    cost_tracker,
    *,
    context_window: int = 128000,
    context_tokens: int = 0,
    tool_use_count: int = 0,
    elapsed_seconds: float = 0.0,
) -> None:
    """Display compact per-response stats footer with color-graded context.

    Format example::

        ctx 50.0% (64K / 128K) [████░░░░░░] · 2 tools · 5.32s · $0.0034
    """
    if cost_tracker is None:
        return

    used_pct = (
        context_tokens / context_window * 100 if context_window > 0 else 0.0
    )
    pct_style = context_pct_style(used_pct)
    bar = build_context_bar(used_pct)

    parts = [
        f"[{pct_style}]ctx {used_pct:.1f}%[/{pct_style}] "
        f"({_format_tokens_short(context_tokens)} / "
        f"{_format_tokens_short(context_window)}) "
        f"[{pct_style}]{bar}[/{pct_style}]"
    ]

    if tool_use_count > 0:
        label = "tool" if tool_use_count == 1 else "tools"
        parts.append(f"[dim]{tool_use_count} {label}[/dim]")

    if elapsed_seconds > 0:
        parts.append(f"[dim]{elapsed_seconds:.2f}s[/dim]")

    # Prompt-cache hits / writes (Anthropic-style, e.g. Venus proxying Claude).
    cache_read = cost_tracker.total_cache_read_tokens
    cache_write = cost_tracker.total_cache_write_tokens
    if cache_read or cache_write:
        seg = []
        if cache_read:
            seg.append(f"{_format_tokens_short(cache_read)} cache_read")
        if cache_write:
            seg.append(f"{_format_tokens_short(cache_write)} cache_write")
        parts.append(f"[dim]{' · '.join(seg)}[/dim]")

    cost = cost_tracker.total_cost_usd
    cost_str = f"${cost:.4f}" if cost < 0.01 else f"${cost:.2f}"
    parts.append(f"[dim]{cost_str}[/dim]")

    console_instance.print(f"{'  ·  '.join(parts)}")


# ---------------------------------------------------------------------------
# Persistent TUI status bar (prompt_toolkit fragments)
# ---------------------------------------------------------------------------

def _ctx_fg_style(pct: float) -> str:
    """Return a prompt_toolkit style class for context usage percentage."""
    if pct >= 95:
        return "class:sb-critical"
    if pct >= 80:
        return "class:sb-bad"
    if pct >= 50:
        return "class:sb-warn"
    return "class:sb-good"


def format_duration_compact(seconds: float) -> str:
    """Format seconds into compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _format_status_work_dir(work_dir: str) -> str:
    """Return a home-relative absolute path for the persistent status bar."""
    if not work_dir:
        return ""
    path = os.path.abspath(os.path.expanduser(work_dir))
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _compact_status_work_dir(work_dir: str) -> str:
    """Keep the project name visible when the full path does not fit."""
    formatted = _format_status_work_dir(work_dir)
    if not formatted or formatted == "~":
        return formatted
    return Path(formatted).name


def build_status_bar_fragments(
    *,
    model_name: str = "",
    model_provider: str = "",
    profile_name: str = "",
    thinking_mode: str = "",
    work_dir: str = "",
    git_branch: str = "",
    context_tokens: int = 0,
    context_window: int = 0,
    cost_usd: float = 0.0,
    active_seconds: float = 0.0,
    last_turn_seconds: float = 0.0,
    spinner_text: str = "",
    terminal_width: int = 80,
    agent_running: bool = False,
    background_terminal_count: int = 0,
):
    """Build prompt_toolkit formatted-text fragments for the persistent status bar.

    Time display uses *agent active time* (sum of all LLM + tool
    execution durations) rather than session wall-clock, plus the
    most recent turn's latency.

    Adapts to terminal width by trying progressively smaller layouts. A wide
    terminal shows model/effort, project path, Git branch/profile, context,
    cost, and timing. The narrowest layout retains model/effort and turn time.

    The model label is rendered as ``provider/model`` when a provider is
    supplied (e.g. ``openai/gpt-4o``). The active Agentica profile name is
    shown first; it is independent from the Git branch.

    When ``agent_running`` is ``True``:
      - ``spinner_text`` (typically a single spinner glyph like ``⠋``) is
        prepended as the leftmost fragment, giving users a heartbeat
        signal that the agent is working.
      - Every ``class:sb*`` class name is swapped for its ``-active``
        variant, which the CLI style sheet paints with a slightly darker
        ``bg:#0f0f1a`` background. This visual downshift makes it clear
        the bar is in "working" state without hiding any of the (still
        updating) numeric fields — users often want to watch tokens and
        cost tick during long turns.
    """
    base = model_name.split("/")[-1] if "/" in model_name else model_name
    if model_provider:
        label = f"{model_provider}/{base}"
    else:
        label = base
    if len(label) > 26:
        label = label[:23] + "..."
    pct = (context_tokens / context_window * 100) if context_window > 0 else 0.0
    pct_label = f"{pct:.0f}%"
    fg = _ctx_fg_style(pct)
    cost_str = f"${cost_usd:.4f}" if cost_usd < 0.01 else f"${cost_usd:.2f}"

    turn_str = f"⏱ {last_turn_seconds:.1f}s" if last_turn_seconds > 0 else ""
    total_str = f"Σ {format_duration_compact(active_seconds)}" if active_seconds > 0 else ""
    bg_full = ""
    bg_short = ""
    if background_terminal_count > 0:
        noun = "terminal" if background_terminal_count == 1 else "terminals"
        bg_full = (
            f"{background_terminal_count} background {noun} running"
            " · /ps to view · /stop to close"
        )
        bg_short = f"{background_terminal_count} bg · /ps · /stop"

    full_work_dir = _format_status_work_dir(work_dir)
    compact_work_dir = _compact_status_work_dir(work_dir)
    ctx_used = _format_tokens_short(context_tokens) if context_tokens else "0"
    ctx_total = _format_tokens_short(context_window) if context_window else "?"

    def compose(
        *,
        project: str = "",
        branch: str = "",
        profile: str = "",
        context_detail: bool = True,
        show_context: bool = True,
        show_cost: bool = True,
        background_detail: bool = True,
    ):
        frags = [("class:sb", " ▸ ")]
        if profile:
            frags.append(("class:sb-dim", f"{profile} "))
        frags.append(("class:sb-strong", label))
        if thinking_mode:
            frags.append(("class:sb-dim", f" {thinking_mode}"))
        if project:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", project),
            ])
        if branch:
            separator = " · " if project else " │ "
            frags.extend([
                ("class:sb-dim", separator),
                ("class:sb", branch),
            ])
        if show_context:
            frags.append(("class:sb-dim", " │ "))
            if context_detail:
                frags.append(("class:sb", f"{ctx_used}/{ctx_total} "))
            frags.append((fg, pct_label))
        if show_cost:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", cost_str),
            ])
        if turn_str:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", turn_str),
            ])
        if total_str:
            frags.append(("class:sb-dim", "  "))
            frags.append(("class:sb-dim", total_str))
        bg_text = bg_full if background_detail else bg_short
        if bg_text:
            frags.extend([
                ("class:sb-dim", " │ "),
                ("class:sb", bg_text),
            ])
        frags.append(("class:sb", " "))
        return frags

    candidates = [
        compose(project=full_work_dir, branch=git_branch, profile=profile_name),
        compose(
            project=full_work_dir, branch=git_branch, profile=profile_name,
            show_cost=False,
        ),
        compose(
            project=compact_work_dir, branch=git_branch, profile=profile_name,
            show_cost=False,
        ),
        compose(
            project=compact_work_dir, branch=git_branch, profile=profile_name,
            context_detail=False, show_cost=False, background_detail=False,
        ),
        compose(
            profile=profile_name, context_detail=False, show_cost=False,
            background_detail=False,
        ),
        compose(profile=profile_name, show_context=False, show_cost=False, background_detail=False),
        compose(show_context=False, show_cost=False, background_detail=False),
    ]
    if terminal_width < 52:
        candidates.insert(
            0,
            compose(show_context=False, show_cost=False, background_detail=False),
        )
    spinner_width = len(spinner_text) + 2 if agent_running and spinner_text else 0
    available_width = max(1, terminal_width - spinner_width)
    frags = next(
        (candidate for candidate in candidates if sum(len(text) for _, text in candidate) <= available_width),
        candidates[-1],
    )

    # ── Agent-running visual downshift ─────────────────────────────────
    # Two things happen when the agent is actively producing output:
    #   1. Prepend spinner_text as the leftmost fragment (heartbeat).
    #   2. Rewrite every ``class:sb*`` fragment to ``class:sb*-active`` so
    #      the CLI style sheet paints them on ``bg:#0f0f1a`` (one shade
    #      darker than the idle ``#1a1a2e``). This is intentionally subtle
    #      — the bar stays legible and the numeric fields keep updating.
    if agent_running:
        if spinner_text:
            # Use the base class name; the rewrite pass below tacks on ``-active``.
            frags.insert(0, ("class:sb-spin", f" {spinner_text} "))
        frags = [
            (
                cls + "-active"
                if cls.startswith("class:sb") and not cls.endswith("-active")
                else cls,
                text,
            )
            for (cls, text) in frags
        ]

    return frags
