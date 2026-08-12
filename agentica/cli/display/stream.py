# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: StreamDisplayManager: live tool/response rendering during a run
"""

import difflib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.markdown import Markdown
from rich.syntax import Syntax

from agentica.cli.runtime import TOOL_ICONS
from agentica.global_config import get_setting
from agentica.tools.patch_tool import parse_patch_envelope

from .console import (
    _is_diagnostic_execute_result,
    _strip_internal_tool_notices,
    remember_truncated,
)
from .messages import _has_markdown
from .tool_format import _display_tool_impl, format_tool_display


def _is_background_execute(tool_args: Optional[dict], result_str: str = "") -> bool:
    """True when this execute call/result is a detached background start."""
    bg = (tool_args or {}).get("background")
    if bg is True or bg == "true" or bg == 1 or bg == "1":
        return True
    return str(result_str).startswith("Started background command")


def _parse_ask_user_exchange(result_str: str) -> Optional[Tuple[str, str]]:
    """Pull ``(prompt, response)`` out of an ask_user_question result.

    Returns None for anything that isn't that payload — a select-mode error
    dict, or a tool of the same name from elsewhere — so the caller can fall
    back to the generic result renderer instead of showing a blank exchange.
    """
    try:
        payload = json.loads(result_str)
    except ValueError:
        return None
    if not isinstance(payload, dict) or "response" not in payload:
        return None
    return (
        str(payload.get("prompt", "")),
        str(payload.get("response", "")),
        payload.get("raw_input"),
    )


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
        the sporadic first-tool-before-response inconsistency.
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
        collapse into a single completion line, e.g.
        ``  🔎 grep 'pat' in path - 5 lines`` (slow calls also fold in the
        elapsed time). The live spinner still announces the running tool, so
        deferring the print costs no feedback.
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
        """Format elapsed seconds; fast calls render nothing.

        Only SLOW tool calls surface a duration — a fast ``grep`` /
        ``read_file`` reporting ``(13ms)`` is pure noise. The 1s cutoff keeps
        quick commands clean while long-running execute tasks (foreground or
        background) still show their cost.

        - None / negative → ''            (no measurement available)
        - < 1s            → ''            (fast — not worth reporting)
        - < 10s           → ' (N.NNs)'    e.g. ' (1.23s)'
        - >= 10s          → ' (N.Ns)'     e.g. ' (12.3s)'
        """
        if elapsed is None or elapsed < 1.0:
            return ""
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
                # Keep the TAIL — the final exception line is what matters.
                err = "..." + err[-77:]
                remember_truncated(f"Tool error · {tool_name}", str(result_content))
            line += f" [yellow]- error: {err}{elapsed_str}[/yellow]"
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
                             tool_call_id: Optional[str] = None,
                             tool_display_meta: Optional[dict] = None) -> None:
        """One summary line + the FULL unified diff for edit tools.

        ``  ✎ edit_file config.py - Edited 1 file (+1 -1)`` followed by one
        complete real-file diff. Errors surface a truncated message.
        """
        icon = TOOL_ICONS.get(tool_name, TOOL_ICONS["default"])
        raw_path = str(tool_args.get("file_path", ""))
        display_path = self._display_path(raw_path)

        line = f"  {icon} [bold magenta]{tool_name}[/bold magenta]"
        if display_path:
            line += f" [dim]{display_path}[/dim]"
        key = tool_call_id or tool_args.get("file_path", "")
        captured_old_content = self._write_old.pop(key, None)
        if is_error:
            err = self._shorten_workdir_text(str(result_content)).replace("\n", " ").strip()
            if len(err) > 80:
                err = "..." + err[-77:]
                remember_truncated(
                    f"Tool error · {tool_name}",
                    self._shorten_workdir_text(str(result_content)),
                )
            line += f" [yellow]- error: {err}{elapsed_str}[/yellow]"
            self._assistant_console.print(line)
            return

        changes = (tool_display_meta or {}).get("files") or []
        change = changes[0] if changes else None
        old_content = change.get("before") if change else captured_old_content
        new_content = change.get("after") if change else self._read_diff_target(tool_args)
        if change:
            display_path = self._display_path(str(change.get("path") or raw_path))
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
                               tool_call_id: Optional[str] = None,
                               tool_display_meta: Optional[dict] = None) -> None:
        """Render apply_patch as one summary plus its real multi-file diff."""
        icon = TOOL_ICONS.get("apply_patch", TOOL_ICONS["default"])
        line = f"  {icon} [bold magenta]apply_patch[/bold magenta]"
        content = self._shorten_workdir_text(str(result_content).strip())
        key = tool_call_id or tool_args.get("patch", "")
        old_files = self._patch_old.pop(key, [])
        if is_error:
            self._assistant_console.print(line + f" [yellow]- error{elapsed_str}[/yellow]")
            error_lines = content.splitlines() or ["Unknown patch error"]
            max_lines = 8
            truncated = len(error_lines) > max_lines or any(len(item) > 120 for item in error_lines)
            # Keep the TAIL of the error — the cause lands at the end.
            hidden = max(0, len(error_lines) - max_lines)
            for index, error_line in enumerate(error_lines[-max_lines:]):
                if len(error_line) > 120:
                    error_line = "..." + error_line[-117:]
                prefix = "    ⎿ " if index == 0 else "      "
                self._assistant_console.print(
                    f"{prefix}{error_line}",
                    style="dim yellow",
                    highlight=False,
                    markup=False,
                )
            if truncated:
                detail = f"{hidden} earlier lines hidden" if hidden > 0 else "full error"
                self._assistant_console.print(
                    f"      ... ({detail} · Ctrl+O to expand)", style="dim italic"
                )
                remember_truncated("Tool error · apply_patch", content)
            return

        summary, _, details = content.partition("\n")
        summary = re.sub(r"^Successfully applied patch to ", "Edited ", summary)
        self._assistant_console.print(line + f" [dim]- {summary}{elapsed_str}[/dim]")
        changes = (tool_display_meta or {}).get("files") or []
        if changes:
            files = [
                (change.get("path", ""), change.get("action", "update"),
                 change.get("before"), change.get("after"))
                for change in changes
            ]
        else:
            files = [
                (raw_path, action, old_content,
                 "" if action == "delete" else self._read_diff_path(self._resolve_diff_path(raw_path)))
                for raw_path, action, old_content in old_files
            ]
        diffs = []
        for raw_path, action, old_content, new_content in files:
            if action == "add" and old_content is None:
                old_content = ""
            if action == "delete" and new_content is None:
                new_content = ""
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
                              tool_call_id: Optional[str] = None,
                              tool_display_meta: Optional[dict] = None) -> None:
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
                err = "..." + err[-77:]
                remember_truncated(f"Tool error · {tool_name}", shortened_result)
            line += f" [yellow]- error: {err}{elapsed_str}[/yellow]"
            self._assistant_console.print(line)
            return

        changes = (tool_display_meta or {}).get("files") or []
        if changes:
            change = changes[0]
            old_content = change.get("before")
            new_content = change.get("after")
            display_path = self._display_path(str(change.get("path") or raw_path))
        if old_content is None and changes and changes[0].get("action") == "add":
            old_content = ""
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
    # line and result summary collapse into ONE line, e.g.
    # ``  🔎 grep 'pat' in path - 5 lines``. No separate result footer.
    _DEFERRED_TOOLS = frozenset({"glob", "grep", "ls", "read_file", "web_search", "fetch_url"})

    # Single-file write tools: call line is deferred to completion and rendered
    # as one summary line plus the real pre/post unified diff. apply_patch is
    # handled separately from the executor's multi-file result summary.
    _WRITE_DIFF_TOOLS = frozenset({"edit_file", "write_file"})

    # Tools whose success result is pure noise on success. The call line itself
    # already tells the user what happened; errors are still surfaced.
    _SUPPRESS_RESULT_TOOLS = frozenset({"write_todos"})

    # Peer discovery / delivery, wait, and delegate: short multi-line status
    # whose Log:/Command:/path lines must stay intact. Never fold behind Ctrl+O
    # or the default 120-char ellipsis. ``task`` is NOT here — its result is
    # JSON with a live-stream dedup path (``_display_task_result``); call-side
    # briefs already share ``_format_handoff_display`` with delegate.
    _FULL_RESULT_TOOLS = frozenset({"list_agents", "send_message", "wait", "delegate"})

    # Human-in-the-loop. Their result is the only durable record of the
    # exchange — the question widget lives in the prompt_toolkit layout and is
    # gone the moment the user answers — so it replays both sides in full.
    _ASK_USER_TOOLS = frozenset({"ask_user_question"})

    # Max result lines shown inline before folding (per-tool overrides below).
    _DEFAULT_MAX_RESULT_LINES = 4
    # execute: show up to this many lines inline; beyond that, a tail-only
    # window (codex-style) with the head collapsed into a single dim hint
    # line — the tail carries the command's final status / exception, which
    # is what the user needs at a glance.
    _EXECUTE_MAX_INLINE_LINES = 10
    _EXECUTE_TAIL_LINES = 6
    _EXECUTE_ERROR_TAIL_LINES = 12
    _EXECUTE_DIAGNOSTIC_TAIL_LINES = 8

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
        params = format_tool_display(tool_name, tool_args)
        # task / delegate briefs must stay intact on the anchor line too —
        # collapsing them to 60 chars re-introduces the same omission the
        # start-time call line was fixed to avoid.
        if tool_name not in ("task", "delegate", "send_message"):
            params = params.replace("\n", " ")
            if len(params) > 60:
                params = params[:57] + "..."
        line = f"    ↳ {icon} {tool_name}"
        if params:
            if "\n" in params:
                self._assistant_console.print(line, style="dim")
                for part in params.splitlines():
                    self._assistant_console.print(f"      {part}", style="dim")
                return
            line += f" {params}"
        self._assistant_console.print(line, style="dim")

    def display_tool_result(self, tool_name: str, result_content: str,
                            is_error: bool = False, elapsed: Optional[float] = None,
                            tool_args: Optional[dict] = None,
                            tool_call_id: Optional[str] = None,
                            tool_display_meta: Optional[dict] = None):
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
                result_content, is_error, elapsed_str, tool_args or {}, tool_call_id,
                tool_display_meta,
            )
            return

        if tool_name in self._WRITE_DIFF_TOOLS:
            self.start_tool_section()
            if tool_name == "write_file":
                self._display_write_merged(
                    tool_name, tool_args or {}, result_content, is_error,
                    elapsed_str, tool_call_id, tool_display_meta
                )
            else:
                self._display_edit_merged(
                    tool_name, tool_args or {}, result_content, is_error,
                    elapsed_str, tool_call_id, tool_display_meta
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

        # execute: foreground uses a tail-only window; background start text is
        # status + Log: path (same class as wait/delegate) and goes FULL.
        if tool_name == "execute":
            if _is_background_execute(tool_args, result_str):
                self._display_full_result_lines(lines, is_error=is_error, elapsed_str=elapsed_str)
                return
            is_diagnostics = _is_diagnostic_execute_result(result_str)
            if is_error:
                tail = self._EXECUTE_ERROR_TAIL_LINES
            elif is_diagnostics:
                tail = self._EXECUTE_DIAGNOSTIC_TAIL_LINES
            else:
                tail = self._EXECUTE_TAIL_LINES
            self._display_tail_window(
                lines, tail,
                inline=self._EXECUTE_MAX_INLINE_LINES,
                prefix="    ⎿ ", cont_prefix="      ",
                style="dim yellow" if (is_error or is_diagnostics) else "dim",
                error_prefix="    ⎿ ⚠ " if (is_error or is_diagnostics) else None,
                truncated_title=f"Tool output · {tool_name}",
                full_content=result_str,
                elapsed_str=elapsed_str,
            )
            return

        if tool_name in self._FULL_RESULT_TOOLS:
            self._display_full_result_lines(lines, is_error=is_error, elapsed_str=elapsed_str)
            return

        if tool_name in self._ASK_USER_TOOLS and not is_error:
            exchange = _parse_ask_user_exchange(result_str)
            if exchange is not None:
                self._display_ask_user_exchange(*exchange, elapsed_str=elapsed_str)
                return

        if is_error:
            # Error details live at the END of the output — fold the head and
            # keep the tail, same codex-style window execute uses.
            self._display_tail_window(
                lines, self._DEFAULT_MAX_RESULT_LINES,
                prefix="    ⎿ ", cont_prefix="      ",
                style="dim yellow", error_prefix="    ⎿ ⚠ ",
                truncated_title=f"Tool output · {tool_name}",
                full_content=result_str,
                elapsed_str=elapsed_str,
            )
            return

        style = "dim"
        prefix = "    ⎿ "
        cont_prefix = "      "

        max_lines = self._DEFAULT_MAX_RESULT_LINES
        max_line_width = 120

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

    def _display_ask_user_exchange(
        self, prompt: str, response: str, raw_input: Optional[str] = None,
        *, elapsed_str: str
    ) -> None:
        """Replay a human-in-the-loop question and the user's answer, unclipped.

        Both sides go into the transcript so scrolling back weeks later still
        shows what the agent asked and what the user chose to do about it.
        ``raw_input`` is the user's full typed reply when it was resolved to an
        option (e.g. "3, because workers=10 is ok" → option 3); showing it
        preserves the rationale that would otherwise vanish.
        """
        cont_prefix = "      "
        # Both sides are free text — a model-written question or a typed answer
        # containing "[bold]" is content, not styling, and rich would eat it.
        for label, body in (("Q", prompt), ("A", response)):
            body_lines = body.splitlines() or [""]
            head = "    ⎿ " if label == "Q" else cont_prefix
            self._assistant_console.print(
                f"{head}{label}: {body_lines[0]}", style="dim", highlight=False, markup=False
            )
            for line in body_lines[1:]:
                self._assistant_console.print(
                    f"{cont_prefix}   {line}", style="dim", highlight=False, markup=False
                )
        if raw_input and raw_input != response:
            raw_lines = raw_input.splitlines() or [""]
            self._assistant_console.print(
                f"{cont_prefix}   (your input: {raw_lines[0]})",
                style="dim italic", highlight=False, markup=False,
            )
            for line in raw_lines[1:]:
                self._assistant_console.print(
                    f"{cont_prefix}   {line}", style="dim italic", highlight=False, markup=False
                )
        if elapsed_str:
            self._assistant_console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")

    def _display_full_result_lines(
        self, lines: List[str], *, is_error: bool, elapsed_str: str
    ) -> None:
        """Print every result line without width or line-count truncation."""
        style = "dim yellow" if is_error else "dim"
        prefix = "    ⎿ " if not is_error else "    ⎿ ⚠ "
        cont_prefix = "      "
        for i, line in enumerate(lines):
            p = prefix if i == 0 else cont_prefix
            self._assistant_console.print(f"{p}{line}", style=style)
        if elapsed_str:
            self._assistant_console.print(f"{cont_prefix}{elapsed_str.lstrip()}", style="dim")

    def _display_tail_window(self, lines: List[str], tail: int,
                             *, inline: int = 0,
                             prefix: str, cont_prefix: str, style: str,
                             error_prefix: Optional[str] = None,
                             truncated_title: str, full_content: str,
                             elapsed_str: str = "",
                             max_line_width: int = 120) -> None:
        """Render a tail-only window with the head folded (codex-style).

        Output up to ``inline`` lines is shown in full. Longer output keeps
        only the last ``tail`` lines — the tail carries the command's final
        status / exception, which is what the user needs at a glance — and
        folds the head into one leading dim ``… +N lines (Ctrl+O to expand)``
        hint. The full content is still remembered for Ctrl+O expansion.
        """
        n = len(lines)
        hidden = 0 if n <= inline else max(0, n - tail)
        show = lines[hidden:]

        if hidden > 0:
            self._assistant_console.print(
                f"{error_prefix or prefix}… +{hidden} lines (Ctrl+O to expand)",
                style="dim italic",
            )
            remember_truncated(truncated_title, full_content)
            first_prefix = cont_prefix
        else:
            first_prefix = error_prefix or prefix

        for i, line in enumerate(show):
            if len(line) > max_line_width:
                line = line[:max_line_width - 3] + "..."
            p = first_prefix if i == 0 else cont_prefix
            self._assistant_console.print(f"{p}{line}", style=style)
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
            self._assistant_console.print(f"    ⎿ ⚠ {error_msg[:120]}", style="dim yellow")
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
            # Full task text — truncating here hides the brief the parent wrote.
            task = str(event.get("task", "") or "").strip()
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
            prefix = (
                f"{self._SUB_INDENT}{self._subagent_prefix(run_id)}"
                f"[dim cyan]⮕ {agent_name}[/dim cyan]{model_note}"
            )
            if not task:
                self._assistant_console.print(f"{prefix}[dim]{budget}[/dim]")
            elif "\n" in task:
                self._assistant_console.print(f"{prefix}[dim]{budget}[/dim]")
                for line in task.splitlines():
                    self._assistant_console.print(
                        f"{self._SUB_INDENT}  [dim italic]{line}[/dim italic]"
                    )
            else:
                self._assistant_console.print(
                    f"{prefix} [dim italic]{task}[/dim italic][dim]{budget}[/dim]"
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
        # Eviction only replaces old tool-result bodies and leaves the call
        # visible in the transcript. Keep it silent; surface only the heavier
        # compaction stages that change conversation structure.
        if et == "compact.evict":
            return

        is_main_agent = event.get("is_main_agent") is True
        prefix = "  " if is_main_agent else "    "
        if et == "compact.auto":
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
