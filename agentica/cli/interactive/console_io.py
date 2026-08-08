# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Console IO helpers: cprint, pager, ChatConsole, OSC8 stripping
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from io import StringIO
from typing import List, Optional, Tuple

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console as RichConsole

from agentica.cli.display import format_session_summary, resumable_session_id
from agentica.cli.runtime import get_console

from .session_state import SessionState

def _print_interactive_exit_summary(state: SessionState, tui_state: dict) -> None:
    """Print the final session summary when the interactive TUI exits."""
    agent = state.current_agent
    if agent is None:
        return

    get_console().print(
        format_session_summary(
            elapsed_seconds=time.monotonic() - tui_state["session_started_at"],
            usage=agent.model.usage,
            session_id=resumable_session_id(agent),
        )
    )


# ==================== Output bridge for patch_stdout ====================


# Serializes the background streaming thread's terminal writes against a
# foreground pager (Ctrl+O). ``run_in_terminal`` only suspends prompt_toolkit's
# own renderer, not the agent thread's ``print_formatted_text`` calls, so while
# ``less`` owns the TTY those writes would interleave with the pager screen
# (blank gaps / spliced-in diff text / cursor desync). Holding this lock for the
# whole pager session parks the streaming thread's writes until the pager
# closes; the agent's LLM stream keeps draining into its unbounded queue
# meanwhile, so nothing is lost or timed out — buffered chunks render in order
# afterwards. RLock (not Lock) because the no-pager fallback in
# ``_open_in_pager`` re-enters ``_cprint`` on the same thread that holds it.
_tty_write_lock = threading.RLock()


# While a ask_user_question prompt is armed, the agent worker thread is parked
# on a queue waiting for the user's line. Background terminal writes (_cprint →
# print_formatted_text under patch_stdout) are serialized onto the main
# prompt_toolkit event loop via run_in_terminal; a burst of them while the user
# is typing an answer can starve the loop so Enter / Ctrl+C / Ctrl+D never get
# processed (the CLI appears frozen). We drop background writes while an ask is
# pending — the agent itself is blocked so there is normally nothing to print,
# and anything else (/btw, cron) would only corrupt the answer prompt anyway.
_ask_active = [False]
_ask_state_lock = threading.RLock()
_output_pause_lock = threading.RLock()
_output_paused = False
_paused_output: List[str] = []


# Rich emits OSC 8 terminal hyperlinks for Markdown links. prompt_toolkit's
# ANSI parser does not understand OSC sequences and renders their payload as
# visible text (for example ``8;id=...;/path/to/file.py:42...8;;``). Strip only
# the unsupported hyperlink wrapper; the styled link label remains intact.
_OSC8_PATTERN = re.compile(
    r"(?:\x1b\]|\x9d)8;[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)"
)


def _strip_unsupported_osc8(text: str) -> str:
    """Remove OSC 8 wrappers before prompt_toolkit parses Rich ANSI output."""
    return _OSC8_PATTERN.sub("", text)


def _print_prompt_toolkit_ansi(text: str) -> None:
    """Render ANSI supported by prompt_toolkit without leaking OSC payloads."""
    print_formatted_text(ANSI(_strip_unsupported_osc8(text)))


def _install_sigquit_escape(handler):
    """Install a temporary SIGQUIT handler when the platform supports it."""
    if os.name == "nt":
        return None
    try:
        previous_handler = signal.getsignal(signal.SIGQUIT)
        signal.signal(signal.SIGQUIT, handler)
    except (AttributeError, ValueError, OSError):
        return None
    return signal.SIGQUIT, previous_handler


def _restore_sigquit_escape(installation) -> None:
    """Restore the SIGQUIT handler saved by _install_sigquit_escape."""
    if installation is None:
        return
    signal_number, previous_handler = installation
    try:
        signal.signal(signal_number, previous_handler)
    except (ValueError, OSError):
        pass


def _cprint(text: str):
    """Print ANSI text through prompt_toolkit's renderer.

    Inside ``patch_stdout()`` context, raw ``print()`` ANSI escapes are
    swallowed.  Routing through ``print_formatted_text(ANSI(...))`` lets
    prompt_toolkit parse the escapes and render colors correctly.
    """
    if _ask_active[0]:
        return
    with _output_pause_lock:
        if _output_paused:
            _paused_output.append(text)
            return
        with _tty_write_lock:
            _print_prompt_toolkit_ansi(text)


def _toggle_output_pause() -> Tuple[bool, int]:
    """Pause transcript rendering, or flush output accumulated while paused."""
    global _output_paused
    with _output_pause_lock:
        _output_paused = not _output_paused
        if _output_paused:
            return True, 0

        buffered = list(_paused_output)
        _paused_output.clear()
        with _tty_write_lock:
            for line in buffered:
                _print_prompt_toolkit_ansi(line)
        return False, len(buffered)


def _clear_output_pause() -> None:
    """Discard buffered transcript output when an interactive session ends."""
    global _output_paused
    with _output_pause_lock:
        _output_paused = False
        _paused_output.clear()


def _less_supports_lesskey(pager: str) -> bool:
    """True if this ``less`` accepts ``--lesskey-content`` (needed to bind Ctrl+O
    to quit). Cached per-process. Probed by invoking the real option rather than
    parsing ``--help`` text — less's own help misprints the option as
    ``--lesskey-context``, so a substring check on help would always be False.
    """
    global _LESS_LESSKEY_OK
    if _LESS_LESSKEY_OK is None:
        try:
            r = subprocess.run(
                [pager, "--lesskey-content=\n#command\n^O quit\n", os.devnull],
                capture_output=True, text=True, timeout=3,
            )
            # Unsupported builds print "There is no lesskey-content=... option".
            _LESS_LESSKEY_OK = "no lesskey-content" not in (r.stderr or "").lower()
        except Exception:
            _LESS_LESSKEY_OK = False
    return _LESS_LESSKEY_OK


_LESS_LESSKEY_OK: Optional[bool] = None


def _compile_lesskey(bindings: str):
    """Compile a lesskey ``bindings`` source (e.g. ``\\n#command\\n^O quit\\n``)
    to a temp file via the ``lesskey`` binary and return its path, or ``None``
    when no compiler is available. Used to inject key bindings on old ``less``
    builds that lack ``--lesskey-content``.
    """
    import tempfile

    lesskey_bin = shutil.which("lesskey")
    if not lesskey_bin:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".lesskey", delete=False, encoding="utf-8") as src:
        src.write(bindings)
        src_path = src.name
    compiled_path = src_path + ".bin"
    try:
        r = subprocess.run([lesskey_bin, "-o", compiled_path, src_path],
                           capture_output=True, text=True, timeout=3)
        if r.returncode != 0 or not os.path.exists(compiled_path):
            return None
        return compiled_path
    except Exception:
        return None
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _open_in_pager(title: str, content: str) -> None:
    """Open ``content`` in a pager so the user can view the full truncated
    block(s), then press ``Ctrl+O`` or ``Esc`` to return — the terminal is
    restored, giving CC-style expand/hide semantics without flooding the
    inline transcript.

    Key binding strategy (in order of preference):
    1. ``less --lesskey-content``: bind ``Ctrl+O`` to quit and the up/down
       arrow sequences to ``back-line``/``forw-line``. Esc is NOT bound (a lone
       ``^[`` leaf would shadow the arrow keys, which also start with ESC —
       see the comment above ``lesskey_src``), so quitting is Ctrl+O or the
       built-in ``q``. The bottom prompt line (``-P``) shows a one-line cheat
       sheet (scroll / search / jump / return) so the user knows how to drive
       less without leaving the CLI.
    2. Old ``less`` without ``--lesskey-content`` but with the ``lesskey``
       compiler: compile the same lesskey source and feed it via ``LESSKEY``.
    3. No ``lesskey`` compiler: plain ``less`` — only the built-in ``q`` quits.
    """
    import tempfile

    pager = shutil.which("less") or shutil.which("more")
    # lesskey source binding Ctrl+O (^O) to quit, plus the up/down arrow
    # sequences to back-line/forw-line so they scroll line-by-line.
    #
    # We deliberately do NOT bind `^[ quit` (Esc to quit): in lesskey, the
    # lone `^[` leaf node matches as soon as less reads the ESC byte and never
    # waits for the rest of an arrow sequence (which also starts with ESC, e.g.
    # ESC[B / ESCOB). Verified on less 668: with `^[ quit` present, pressing
    # Down quits every time — the `^[[B forw-line` binding is shadowed and
    # never reached. Esc-to-quit and arrow-scrolling are mutually exclusive,
    # so we keep the arrows (the user's intuitive expectation) and rely on
    # Ctrl+O / q to quit. Both cursor-key modes are covered (^[ [ X normal
    # mode, ^[ O X application mode).
    lesskey_src = (
        "\n#command\n"
        "^O quit\n"
        "^[[A back-line\n"
        "^[[B forw-line\n"
        "^[OA back-line\n"
        "^[OB forw-line\n"
    )
    lesskey_ok = pager is not None and _less_supports_lesskey(pager) and "less" in (pager or "")

    if lesskey_ok:
        return_hint = "Ctrl+O/q to return"
        prompt_line = "↑↓/d/u scroll · / search · g/G first/last · Ctrl+O/q return"
    elif pager is not None and "less" in pager and _compile_lesskey(lesskey_src):
        return_hint = "Ctrl+O/q to return"
        prompt_line = "↑↓/d/u scroll · / search · g/G first/last · Ctrl+O/q return"
    else:
        return_hint = "q to return"
        prompt_line = "↑↓/d/u scroll · / search · g/G first/last · q return"

    header = (
        f"=== {title} · {len(content.splitlines())} lines "
        f"({return_hint}) ===\n\n"
    )
    full = header + content
    if not pager:
        con = get_console()
        con.print()
        con.print(f"[dim]{title}[/dim]")
        con.print(full)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(full)
        path = f.name
    compiled_lesskey = None
    try:
        if lesskey_ok:
            subprocess.run(
                [pager, "-R", "-f", f"--lesskey-content={lesskey_src}", "-P", prompt_line, path],
            )
        elif "less" in pager:
            compiled_lesskey = _compile_lesskey(lesskey_src)
            if compiled_lesskey:
                env = dict(os.environ, LESSKEY=compiled_lesskey)
                subprocess.run(
                    [pager, "-R", "-f", "-P", prompt_line, path], env=env,
                )
            else:
                subprocess.run([pager, "-R", "-f", "-P", prompt_line, path])
        else:
            subprocess.run([pager, "-P", prompt_line, path])
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        if compiled_lesskey:
            try:
                os.unlink(compiled_lesskey)
            except OSError:
                pass


class ChatConsole:
    """Rich Console adapter for prompt_toolkit's patch_stdout context.

    Renders Rich markup to an in-memory buffer, then outputs line-by-line
    through ``_cprint`` so colors/formatting work correctly while the
    input area stays pinned at the bottom.
    """

    def __init__(self):
        self._buffer = StringIO()
        self._inner = RichConsole(
            file=self._buffer,
            force_terminal=True,
            color_system="truecolor",
            highlight=False,
        )

    def render_ansi(self, *args, **kwargs) -> str:
        """Render Rich markup to an ANSI string without emitting it.

        Lets callers batch multiple lines into a single ``run_in_terminal``
        cycle (see ``_cli_ask_user_question_callback._show_prompt``).
        """
        self._buffer.seek(0)
        self._buffer.truncate()
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
        self._inner.print(*args, **kwargs)
        return self._buffer.getvalue()

    def print(self, *args, **kwargs):
        output = self.render_ansi(*args, **kwargs)
        for line in output.rstrip("\n").split("\n"):
            _cprint(line)

    @property
    def width(self):
        return shutil.get_terminal_size((80, 24)).columns


def _print_boxed_result(label: str, question: str, result_text: str, color: str = "cyan"):
    """Print a question + answer inside a colored box.

    Args:
        label:       Box title (e.g. "BTW", "Background #1")
        question:    Original user question (displayed truncated)
        result_text: Full answer text (displayed in full, never truncated)
        color:       Rich color for box frame
    """
    con = get_console()
    tw = 80
    try:
        tw = min(shutil.get_terminal_size((80, 24)).columns, 100)
    except Exception:
        pass
    fill = max(0, tw - len(label) - 5)
    con.print()
    con.print(f"[{color}]╭─ {label} {'─' * fill}╮[/{color}]")
    # Question — truncate display for readability
    q_display = question[: tw - 8] + ("..." if len(question) > tw - 8 else "")
    con.print(f"  [dim]Q: {q_display}[/dim]")
    # Answer — show in full, never truncate
    if result_text:
        for line in result_text.splitlines():
            con.print(f"  {line}")
    else:
        con.print("  (no output)")
    con.print(f"[{color}]╰{'─' * (tw - 2)}╯[/{color}]")


__all__ = ['_print_interactive_exit_summary', '_tty_write_lock', '_ask_active', '_ask_state_lock', '_output_pause_lock', '_output_paused', '_paused_output', '_OSC8_PATTERN', '_strip_unsupported_osc8', '_print_prompt_toolkit_ansi', '_install_sigquit_escape', '_restore_sigquit_escape', '_cprint', '_toggle_output_pause', '_clear_output_pause', '_less_supports_lesskey', '_LESS_LESSKEY_OK', '_compile_lesskey', '_open_in_pager', 'ChatConsole', '_print_boxed_result']
