# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI interactive mode - main interaction loop
"""

import json
import os
import queue
import re
import shutil
import subprocess
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from io import StringIO

from prompt_toolkit import print_formatted_text
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea
from rich.console import Console as RichConsole

from agentica.cli.config import (
    get_console,
    set_active_console,
    history_file,
    configure_tools,
    create_agent,
    _generate_session_id,
)
from agentica.cli.display import (
    StreamDisplayManager,
    build_status_bar_fragments,
    print_header,
    parse_file_mentions,
    inject_file_contents,
    display_user_message,
    get_file_completions,
    get_truncated_blocks,
)
from agentica.cli.permissions import PermissionManager
from agentica.run_display import RunDisplayEventKind, classify_run_response
from agentica.run_response import AgentCancelledError
from agentica.utils.log import logger, suppress_console_logging
from agentica.workspace import Workspace
from agentica.skills import load_skills, get_skill_registry
from agentica.global_config import get_active_profile_name

from agentica.cli.commands import (
    CommandContext,
    COMMAND_REGISTRY,
    COMMAND_HANDLERS,
    CONCURRENT_CMDS,
    PendingQueue,
    IMAGE_EXTENSIONS,
    _run_async_safe,
    _detach_goal_tool,
)
from agentica.goals import CONTINUATION_PROMPT_PREFIX, GoalManager


# ==================== SessionState ====================


@dataclass
class SessionState:
    """All mutable session state in one place.

    Replaces the scattered single-element list containers
    (``[False]``, ``[0]``, ``[agent]``) with typed fields.
    """

    shell_mode: bool = False
    should_exit: bool = False
    agent_running: bool = False
    current_agent: Any = None
    image_counter: int = 0
    session_tokens: int = 0
    paste_counter: int = 0
    attached_images: List = field(default_factory=list)
    pasted_files: List = field(default_factory=list)
    last_ctrl_c: float = 0.0
    # Background tasks — owned by session, not module-global
    bg_tasks: Dict[str, dict] = field(default_factory=dict)
    bg_task_counter: int = 0
    # Standing-goal loop (see agentica/goals.py).
    goal_manager: Optional[GoalManager] = None
    goal_lock: threading.Lock = field(default_factory=threading.Lock)
    # Token + wall-clock baselines for per-turn budget accounting (S2).
    # We snapshot cumulative CostTracker totals BEFORE each turn and diff
    # AFTER so the goal loop sees the delta, not the whole run.
    goal_tokens_baseline: int = 0


# ==================== Output bridge for patch_stdout ====================


def _cprint(text: str):
    """Print ANSI text through prompt_toolkit's renderer.

    Inside ``patch_stdout()`` context, raw ``print()`` ANSI escapes are
    swallowed.  Routing through ``print_formatted_text(ANSI(...))`` lets
    prompt_toolkit parse the escapes and render colors correctly.
    """
    print_formatted_text(ANSI(text))


def _less_supports_lesskey(pager: str) -> bool:
    """True if this ``less`` accepts ``--lesskey-content`` (needed to bind Ctrl+o
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
    block(s), then press ``Ctrl+o`` or ``Esc`` to return — the terminal is
    restored, giving CC-style expand/hide semantics without flooding the
    inline transcript.

    Key binding strategy (in order of preference):
    1. ``less --lesskey-content``: bind ``Ctrl+o`` AND ``Esc`` to quit, and bind
       the up/down arrow sequences to ``back-line``/``forw-line`` so they are
       not shadowed by the Esc binding (arrow keys start with ESC). Hint:
       ``Ctrl+o or Esc to return`` (Ctrl+o mirrors the expand key; Esc is the
       conventional CC close key).
    2. Old ``less`` without ``--lesskey-content`` but with the ``lesskey``
       compiler: compile the same lesskey source (quit keys + arrow scrolls)
       and feed it via the ``LESSKEY`` env var. Same hint.
    3. No ``lesskey`` compiler: plain ``less`` — only the built-in ``q`` quits.
       Hint: ``q to return``.
    """
    import tempfile

    pager = shutil.which("less") or shutil.which("more")
    # lesskey source binding Ctrl+o (^O) and Esc (^[) to quit, PLUS the
    # up/down arrow sequences explicitly bound to back-line/forw-line. Without
    # the arrow bindings, ^[ quit would shadow the arrow keys (which start with
    # ESC, e.g. ESC[A / ESCOA) — a lone ^[ leaf matches immediately and less
    # never reads the rest of the sequence, so pressing Up would quit. Binding
    # the full arrow sequences makes less do longest-prefix matching: ESC[A →
    # back-line (scroll up), a lone ^[ (Esc held with no following byte within
    # less's key timeout) → quit. Both cursor-key modes are covered (^[ [ A
    # normal mode, ^[ O A application mode).
    lesskey_src = (
        "\n#command\n"
        "^O quit\n"
        "^[ quit\n"
        "^[[A back-line\n"
        "^[[B forw-line\n"
        "^[OA back-line\n"
        "^[OB forw-line\n"
    )
    lesskey_ok = pager is not None and _less_supports_lesskey(pager) and "less" in (pager or "")

    if lesskey_ok:
        return_hint = "Ctrl+o or Esc to return"
    elif pager is not None and "less" in pager and _compile_lesskey(lesskey_src):
        return_hint = "Ctrl+o or Esc to return"
    else:
        return_hint = "q to return"

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
                [pager, "-R", f"--lesskey-content={lesskey_src}", "-P", "Ctrl+o or Esc to return", path],
            )
        elif "less" in pager:
            compiled_lesskey = _compile_lesskey(lesskey_src)
            if compiled_lesskey:
                env = dict(os.environ, LESSKEY=compiled_lesskey)
                subprocess.run(
                    [pager, "-R", "-P", "Ctrl+o or Esc to return", path], env=env,
                )
            else:
                subprocess.run([pager, "-R", "-P", "q to return", path])
        else:
            subprocess.run([pager, "-P", "q to return", path])
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

    def print(self, *args, **kwargs):
        self._buffer.seek(0)
        self._buffer.truncate()
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
        self._inner.print(*args, **kwargs)
        output = self._buffer.getvalue()
        for line in output.rstrip("\n").split("\n"):
            _cprint(line)

    @property
    def width(self):
        return shutil.get_terminal_size((80, 24)).columns


# ==================== Image Attachment Helpers ====================


def _split_path_input(raw: str) -> tuple:
    """Split a leading file path token from trailing free-form text."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == "\\" and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                token = raw[1:pos]
                remainder = raw[pos + 1 :].strip()
                return token, remainder
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == "\\" and pos + 1 < len(raw) and raw[pos + 1] == " ":
            pos += 2
        elif ch == " ":
            break
        else:
            pos += 1

    token = raw[:pos].replace("\\ ", " ")
    remainder = raw[pos:].strip()
    return token, remainder


def _resolve_attachment_path(raw_path: str) -> Optional[Path]:
    """Resolve a user-supplied local attachment path."""
    token = str(raw_path or "").strip()
    if not token:
        return None
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()
    if not token:
        return None

    expanded = os.path.expandvars(os.path.expanduser(token))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def _detect_file_drop(user_input: str) -> Optional[dict]:
    """Detect if user_input starts with a real local file path."""
    if not isinstance(user_input, str):
        return None
    stripped = user_input.strip()
    if not stripped:
        return None

    starts_like_path = (
        stripped.startswith("/")
        or stripped.startswith("~")
        or stripped.startswith("./")
        or stripped.startswith("../")
        or stripped.startswith('"/')
        or stripped.startswith('"~')
        or stripped.startswith("'/")
        or stripped.startswith("'~")
    )
    if not starts_like_path:
        return None

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None:
        return None

    return {
        "path": drop_path,
        "is_image": drop_path.suffix.lower() in IMAGE_EXTENSIONS,
        "remainder": remainder,
    }


def _try_attach_clipboard_image(attached_images: list, image_counter: list) -> bool:
    """Check clipboard for an image and attach it if found."""
    from agentica.cli.clipboard import save_clipboard_image
    from agentica.config import AGENTICA_HOME

    img_dir = Path(AGENTICA_HOME) / "images"
    image_counter[0] += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"clip_{ts}_{image_counter[0]}.png"

    if save_clipboard_image(img_path):
        attached_images.append(img_path)
        return True
    image_counter[0] -= 1
    return False


# ==================== Shell command ====================


def _handle_shell_command(user_input: str, work_dir: Optional[str] = None) -> None:
    """Execute a shell command directly."""
    con = get_console()
    con.print(f"[dim]$ {user_input}[/dim]")
    try:
        result = subprocess.run(
            user_input,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir or os.getcwd(),
        )
        if result.stdout:
            con.print(result.stdout, end="")
        if result.stderr:
            con.print(result.stderr, style="red", end="")
        if result.returncode != 0:
            con.print(f"[dim]Exit code: {result.returncode}[/dim]")
    except Exception as e:
        con.print(f"[red]Error: {e}[/red]")
    con.print()


# ==================== BTW concurrent handler ====================


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


def _run_btw_concurrent(agent, question: str, tui_state: dict):
    """Run a BTW side question in a background thread.

    Uses a fresh agent with NO tools but WITH a snapshot of the main agent's
    conversation history, so it can answer side questions in context.
    """
    try:
        from agentica import Agent
        from agentica.memory.models import AgentRun
        from agentica.model.message import Message
        from agentica.run_response import RunResponse

        # Snapshot conversation context from the main agent (same as /bg)
        context_snapshot = []
        if agent and agent.working_memory and agent.working_memory.messages:
            for msg in agent.working_memory.messages:
                if msg.role in ("user", "assistant") and msg.content:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_snapshot.append(Message(role=msg.role, content=content))
            context_snapshot = context_snapshot[-10:]

        btw_agent = Agent(
            model=agent.model,
            tools=[],
            instructions="You are a helpful assistant answering a quick side question. "
            "You have NO tools, NO skills, NO file access. "
            "Answer concisely based on your knowledge and conversation context.",
            session_id=_generate_session_id(),
            debug=False,
            add_history_to_context=True,
        )

        # Inject context snapshot so the BTW agent can see prior conversation
        if context_snapshot:
            synthetic_run = AgentRun(
                response=RunResponse(messages=context_snapshot),
            )
            btw_agent.working_memory.runs.append(synthetic_run)

        response = btw_agent.run_sync(question)
        result_text = response.content if response else "(no answer)"
    except Exception as e:
        result_text = f"Error: {e}"

    _print_boxed_result("BTW", question, result_text, color="cyan")


# ==================== Image OCR fallback ====================

# Per-image and total limits for OCR text injection
_OCR_PER_IMAGE_CHARS = 50_000
_OCR_TOTAL_CHARS = 200_000
_OCR_TIMEOUT_SECS = 30


def _ocr_single_image(image_path: str) -> str:
    """OCR a single image, returning extracted text (truncated to limit)."""
    try:
        from imgocr import ImgOcr

        ocr = ImgOcr()
        result = ocr.ocr(image_path)
        text = " ".join(item["text"] for item in result if "text" in item)
        if len(text) > _OCR_PER_IMAGE_CHARS:
            text = text[:_OCR_PER_IMAGE_CHARS] + f"\n... (truncated, {len(text)} chars total)"
        return text
    except ImportError:
        return ""
    except Exception:
        return ""


def _ocr_images_parallel(image_paths: list) -> str:
    """OCR multiple images in parallel with timeout. Returns combined text."""
    from concurrent.futures import ThreadPoolExecutor

    results = []
    total_len = 0
    with ThreadPoolExecutor(max_workers=min(len(image_paths), 4)) as pool:
        futures = {pool.submit(_ocr_single_image, p): p for p in image_paths}
        for future in futures:
            path = futures[future]
            name = Path(path).name
            try:
                text = future.result(timeout=_OCR_TIMEOUT_SECS)
            except Exception:
                text = ""

            if not text:
                continue

            if total_len + len(text) > _OCR_TOTAL_CHARS:
                remaining = _OCR_TOTAL_CHARS - total_len
                if remaining > 100:
                    text = text[:remaining] + "\n..."
                else:
                    break

            if len(image_paths) > 1:
                results.append(f"[{name}]\n{text}")
            else:
                results.append(text)
            total_len += len(text)

    return "\n\n".join(results)


# ==================== Goal loop hook ====================


def _maybe_continue_goal(
    state: "SessionState",
    pending_queue: PendingQueue,
    tui_state: dict,
) -> None:
    """After each agent turn, decide whether to enqueue a continuation prompt.

    Invariants:
    - Real user input ALWAYS preempts the goal loop. If any non-continuation,
      non-internal item is already queued, we defer.
    - A cancelled agent (Ctrl+C) pauses the goal instead of evaluating —
      otherwise the judge sees a half-finished response, judges "not done",
      and the user's cancel immediately gets re-queued.
    - Empty response: skip (nothing to judge).
    - GoalManager.evaluate_after_turn() is async; we bridge with _run_async_safe.
    - token_delta is read from CostTracker totals diffed against the
      pre-turn baseline; elapsed comes from tui_state["last_turn_seconds"].
    """
    mgr = state.goal_manager
    if mgr is None or not mgr.is_active():
        return

    agent = state.current_agent
    if agent is None:
        return

    if agent._cancelled:
        mgr.pause(reason="user-interrupted")
        _cprint("  ⊙ Goal paused (user interrupted).")
        return

    # User real input takes priority.
    for item, _ts in pending_queue.peek_all_with_timestamps():
        if isinstance(item, tuple):
            text = str(item[0]) if item[0] != "__BTW__" else ""
        else:
            text = str(item)
        if not text or text.startswith("__"):
            continue
        if text.startswith(CONTINUATION_PROMPT_PREFIX):
            continue
        return  # real user message waiting — let it run first

    final_text = ""
    if agent.run_response is not None:
        final_text = agent.run_response.content or ""
    if not final_text.strip():
        return

    # Compute per-turn token delta from the CostTracker (cumulative).
    token_delta = 0
    if agent.run_response is not None and agent.run_response.cost_tracker is not None:
        ct = agent.run_response.cost_tracker
        total_now = ct.total_input_tokens + ct.total_output_tokens
        token_delta = max(0, total_now - state.goal_tokens_baseline)
        state.goal_tokens_baseline = total_now

    elapsed_sec = float(tui_state.get("last_turn_seconds", 0.0) or 0.0)

    # Tool-call summary for the judge + the tool-stuck counter.
    tool_pairs: List[Tuple[str, bool]] = []
    if agent.run_response is not None:
        for tc in agent.run_response.tool_calls:
            if tc.tool_name:
                tool_pairs.append((tc.tool_name, bool(tc.is_error)))

    with state.goal_lock:
        try:
            decision = _run_async_safe(
                mgr.evaluate_after_turn(
                    final_text,
                    token_delta=token_delta,
                    elapsed_sec=elapsed_sec,
                    tool_calls=tool_pairs or None,
                )
            )
        except Exception as exc:
            _cprint(f"  [goal] evaluator failed: {exc}")
            return

    if decision.message:
        _cprint(f"  {decision.message}")

    # If the loop ended (complete / paused / budget_limited), detach the
    # tool — otherwise it lingers on a goal that no longer auto-continues.
    if decision.status in ("complete", "paused", "budget_limited"):
        _detach_goal_tool(agent)

    if decision.should_continue and decision.continuation_prompt:
        pending_queue.put(decision.continuation_prompt)


# ==================== Stream response ====================


def _process_stream_response(
    current_agent,
    final_input: str,
    session_tokens: list,
    tui_state: dict,
    *,
    images: Optional[list] = None,
    permission_manager: Optional[PermissionManager] = None,
) -> None:
    """Process the agent's streaming response and display it."""
    con = get_console()

    def _set_spinner(text: str = ""):
        tui_state["spinner_text"] = text
        if text:
            tui_state["_spinner_base"] = text
            tui_state["_tool_start"] = time.monotonic()
        else:
            tui_state["_spinner_base"] = ""

    _THINKING_FRAMES = ["✦ thinking...", "✧ thinking..."]
    tui_state["_thinking"] = True
    tui_state["spinner_text"] = _THINKING_FRAMES[0]
    request_start = perf_counter()

    try:
        from agentica.run_config import RunConfig
        from agentica.run_context import RunSource

        run_config = RunConfig(stream_intermediate_steps=True, source=RunSource.cli)

        # Permission integration: restrict tools based on permission mode
        if permission_manager and permission_manager.mode == "strict":
            from agentica.cli.permissions import READ_ONLY_TOOLS

            run_config.enabled_tools = list(READ_ONLY_TOOLS)

        run_kwargs = {"config": run_config}

        # When images are attached, run LLM vision API and OCR in parallel.
        # Both results are merged into the prompt for the final agent call.
        # If one fails, the other provides a fallback.
        if images:
            image_paths = [str(p) for p in images]
            _set_spinner("analyzing images")

            ocr_future = None
            vision_result = None
            vision_error = None

            from concurrent.futures import ThreadPoolExecutor

            def _run_vision():
                """Call LLM API with images (blocking, runs in thread)."""
                kwargs = dict(run_kwargs)
                kwargs["images"] = image_paths
                chunks = []
                for chunk in current_agent.run_stream_sync(final_input, **kwargs):
                    if chunk and chunk.content:
                        chunks.append(chunk.content)
                return "".join(chunks)

            with ThreadPoolExecutor(max_workers=2) as pool:
                # Start OCR and vision API in parallel
                ocr_future = pool.submit(_ocr_images_parallel, image_paths)
                vision_future = pool.submit(_run_vision)

                # Collect results (both may succeed, one may fail)
                try:
                    vision_result = vision_future.result(timeout=120)
                except Exception as e:
                    vision_error = str(e)

                try:
                    ocr_text = ocr_future.result(timeout=_OCR_TIMEOUT_SECS + 10)
                except Exception:
                    ocr_text = ""

            # Build combined image context — keep it concise, focus on user's query
            extra_parts = []
            if vision_result:
                extra_parts.append(f"[Image description]\n{vision_result}")
            if ocr_text and ocr_text.strip():
                extra_parts.append(f"[Text extracted from image]\n{ocr_text}")

            if extra_parts:
                final_input += "\n\n" + "\n\n".join(extra_parts)

            # Clean up: remove messages from the vision trial run (if any were added)
            wm = current_agent.working_memory
            if vision_result or vision_error:
                # The vision run added messages to working memory; remove them
                # so the combined re-run starts clean
                if wm.runs:
                    wm.runs.pop()
                while wm.messages and wm.messages[-1].role in ("assistant", "tool"):
                    wm.messages.pop()
                if wm.messages and wm.messages[-1].role == "user":
                    wm.messages.pop()

            _set_spinner("")

        # Subagent verbosity follows the global ``--debug`` flag (carried
        # via ``tui_state`` since this helper has no direct access to the
        # CLI args): developers debugging a flow want completion + elapsed
        # for every child tool; end users get the tool-first single-line
        # view by default.
        subagent_verbosity = "verbose" if tui_state.get("debug") else "all"
        display = StreamDisplayManager(con, subagent_verbosity=subagent_verbosity)
        # Register live-event callback so the subagent's tool calls and
        # compression events render in real time (instead of being a black
        # box until the parent tool result arrives).
        current_agent._event_callback = display.handle_event

        response_stream = current_agent.run_stream_sync(final_input, **run_kwargs)

        shown_tool_count = 0

        for chunk in response_stream:
            if current_agent._cancelled:
                raise AgentCancelledError("Agent run cancelled by user")

            if chunk is None:
                continue

            display_event = classify_run_response(chunk)

            if display_event.kind == RunDisplayEventKind.METADATA_SKIP:
                continue
            if display_event.kind == RunDisplayEventKind.TELEMETRY_ONLY:
                continue

            if display_event.kind == RunDisplayEventKind.TOOL_STARTED:
                if chunk.tools and len(chunk.tools) > shown_tool_count:
                    for tool_info in chunk.tools[shown_tool_count:]:
                        tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                        tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                        if isinstance(tool_args, str):
                            try:
                                tool_args = json.loads(tool_args)
                            except ValueError:
                                tool_args = {"args": tool_args}

                        display.display_tool(tool_name, tool_args)
                        tui_state["_thinking"] = False
                        _set_spinner(f"🔧 {tool_name}")
                    shown_tool_count = len(chunk.tools)
                continue

            if display_event.kind == RunDisplayEventKind.TOOL_COMPLETED:
                _set_spinner("")
                tui_state["_thinking"] = True
                if chunk.tools:
                    for tool_info in reversed(chunk.tools):
                        if "content" in tool_info:
                            tool_name = tool_info.get("tool_name") or tool_info.get("name", "unknown")
                            result_content = tool_info.get("content", "")
                            is_error = tool_info.get("tool_call_error", False)
                            elapsed = (tool_info.get("metrics") or {}).get("time")
                            tool_args = tool_info.get("tool_args") or tool_info.get("arguments") or {}
                            display.display_tool_result(
                                tool_name,
                                str(result_content) if result_content else "",
                                is_error=is_error,
                                elapsed=elapsed,
                                tool_args=tool_args,
                            )
                            break
                continue

            has_content = chunk.content and isinstance(chunk.content, str)
            has_reasoning = chunk.reasoning_content

            if not has_content and not has_reasoning:
                continue

            if has_reasoning and not has_content:
                if tui_state.get("show_reasoning", True):
                    _set_spinner("")
                    tui_state["_thinking"] = False
                    display.start_thinking()
                    display.stream_thinking(chunk.reasoning_content)
                continue

            if has_content:
                _set_spinner("")
                tui_state["_thinking"] = False
                display.stream_response(chunk.content)

        display.finalize()
        _set_spinner("")
        tui_state["_thinking"] = False

        # Surface loop-break reasons (death spiral / max turns / cost budget).
        # These no longer ride inside the response content, so render a notice
        # here to keep the CLI user informed about a truncated run.
        if current_agent.run_response.break_reason:
            con.print(
                f"\n[yellow]⚠ Run stopped early "
                f"({current_agent.run_response.break_reason}): "
                f"{current_agent.run_response.break_message}[/yellow]"
            )

        elapsed = perf_counter() - request_start
        tui_state["last_turn_seconds"] = elapsed
        tui_state["active_seconds"] = tui_state.get("active_seconds", 0.0) + elapsed

        cost_tracker = current_agent.run_response.cost_tracker
        if cost_tracker and cost_tracker.turns > 0:
            context_tokens = cost_tracker.last_input_tokens
            context_window = current_agent.model.context_window if current_agent.model else 128000
            tui_state["context_tokens"] = context_tokens
            tui_state["context_window"] = context_window
            tui_state["cost_usd"] = tui_state.get("cost_usd", 0.0) + cost_tracker.total_cost_usd
            tui_state["total_api_calls"] = tui_state.get("total_api_calls", 0) + cost_tracker.turns

        if not display.has_content_output and display.tool_count == 0 and not display.thinking_shown:
            _set_spinner("")
            con.print("[info]Agent returned no content.[/info]")

    except KeyboardInterrupt:
        current_agent.cancel()
        _set_spinner("")
        deadline = time.monotonic() + 3.0
        while current_agent._running and time.monotonic() < deadline:
            time.sleep(0.05)
        current_agent._running = False
        current_agent._cancelled = False
        con.print("\n[yellow]⚡ Agent cancelled.[/yellow] [dim][用户中断了回答][/dim]")
    except AgentCancelledError:
        _set_spinner("")
        current_agent._running = False
        current_agent._cancelled = False
        con.print("\n[yellow]⚡ Agent cancelled.[/yellow] [dim][用户中断了回答][/dim]")
    except Exception as e:
        _set_spinner("")
        msg = str(e)
        con.print(f"\n[bold red]Error during agent execution: {msg}[/bold red]")
        # Transient connection / gateway failures are usually worth a retry.
        low = msg.lower()
        if any(h in low for h in ("connection", "timeout", "502", "503", "504", "gateway", "remote disconnected")):
            con.print("[dim]  Transient network error — type /retry to resend the last message.[/dim]")
    finally:
        # Clear the live-event callback so it doesn't outlive this run.
        current_agent._event_callback = None


# ==================== TUI setup ====================


def _setup_tui(
    state: SessionState,
    skills_registry,
    tui_state: dict,
    pending_queue: PendingQueue,
    image_counter_ref: list,
    dispatch_cmd=None,
):
    """Build the prompt_toolkit Application with fixed-bottom input area."""

    _image_counter_ref = image_counter_ref

    class AgenticaCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/"):
                parts = text.split(None, 1)
                if len(parts) >= 2:
                    cmd = parts[0].lower()
                    if skills_registry:
                        skill_cmds = skills_registry.auto_commands()
                        skill = skill_cmds.get(cmd)
                        if skill and skill.argument_hint:
                            yield Completion(
                                skill.argument_hint,
                                start_position=0,
                                display=skill.argument_hint,
                                display_meta="argument",
                            )
                    return
                q = text.lower()
                for cmd_name, (_, desc) in COMMAND_REGISTRY.items():
                    if cmd_name.startswith(q):
                        yield Completion(cmd_name, start_position=-len(text), display=cmd_name, display_meta=desc)
                if skills_registry:
                    for slug, skill in skills_registry.auto_commands().items():
                        if slug.startswith(q) and slug not in COMMAND_REGISTRY:
                            desc = skill.description[:50] if skill.description else ""
                            yield Completion(
                                slug, start_position=-len(text), display=f"{slug} ({skill.name})", display_meta=desc
                            )
                return
            m = re.search(r"@([\w./-]*)$", text)
            if m:
                partial = m.group(1)
                for comp in get_file_completions(text):
                    yield Completion(comp, start_position=-len(partial), display=comp)

    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")
    def _newline2(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-d")
    def _exit_app(event):
        state.should_exit = True
        event.app.exit()

    @kb.add("c-c")
    def _handle_ctrl_c(event):
        now = time.time()
        if state.agent_running:
            if now - state.last_ctrl_c < 2.0:
                _cprint("\n⚡ Force exiting...")
                state.should_exit = True
                event.app.exit()
                return
            state.last_ctrl_c = now
            _cprint("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
            state.current_agent.cancel()
            # Pause any active standing goal so the post-turn hook doesn't
            # auto-requeue a continuation right after the user cancelled.
            if state.goal_manager is not None and state.goal_manager.is_active():
                state.goal_manager.pause(reason="user-interrupted")
        elif event.app.current_buffer.text:
            event.app.current_buffer.reset()
            event.app.invalidate()
        elif state.attached_images:
            state.attached_images.clear()
            event.app.invalidate()
        else:
            if now - state.last_ctrl_c < 2.0:
                state.should_exit = True
                event.app.exit()
            else:
                state.last_ctrl_c = now
                tui_state["spinner_text"] = "Press Ctrl+C again to exit (or Ctrl+D)"
                event.app.invalidate()

    @kb.add("c-x")
    def _toggle_shell(event):
        pending_queue.put("__TOGGLE_SHELL_MODE__")
        event.app.current_buffer.reset()

    @kb.add("enter")
    def _handle_enter(event):
        text = event.app.current_buffer.text.strip()
        has_images = bool(state.attached_images)
        if not text and not has_images:
            return

        images = list(state.attached_images)
        state.attached_images.clear()
        payload = (text, images) if images else text

        # Concurrent command dispatch — runs immediately even when agent is busy
        if state.agent_running and text.startswith("/") and dispatch_cmd:
            first_word = text.split()[0].lower()
            if first_word in CONCURRENT_CMDS:
                cmd_parts = text.split(maxsplit=1)
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                threading.Thread(
                    target=dispatch_cmd,
                    args=(first_word, cmd_args),
                    daemon=True,
                ).start()
                event.app.current_buffer.reset(append_to_history=True)
                event.app.invalidate()
                return

        # BTW side question — dispatch concurrently even when agent is busy
        if text.startswith("/btw ") and dispatch_cmd:
            cmd_args = text[5:].strip()
            if cmd_args:
                threading.Thread(
                    target=dispatch_cmd,
                    args=("/btw", cmd_args),
                    daemon=True,
                ).start()
                event.app.current_buffer.reset(append_to_history=True)
                event.app.invalidate()
                return

        pending_queue.put(payload)

        # The queue bar above the input box renders queued items live (with
        # per-item timestamps), so we deliberately do NOT print a notice
        # into the chat stream — that would interleave with the running
        # AI response box.

        event.app.current_buffer.reset(append_to_history=True)
        event.app.invalidate()

    def _accept_or_complete(event):
        buf = event.current_buffer
        if buf.complete_state:
            completion = buf.complete_state.current_completion
            if completion is None:
                buf.go_to_completion(0)
                completion = buf.complete_state and buf.complete_state.current_completion
            if completion:
                buf.apply_completion(completion)
        elif buf.suggestion and buf.suggestion.text:
            buf.insert_text(buf.suggestion.text)
        else:
            buf.start_completion()

    @kb.add("tab", eager=True)
    def _handle_tab(event):
        _accept_or_complete(event)

    @kb.add("right", eager=True)
    def _handle_right(event):
        buf = event.current_buffer
        if buf.complete_state or (buf.suggestion and buf.suggestion.text):
            _accept_or_complete(event)
        else:
            buf.cursor_right()

    @kb.add("up")
    def _history_up(event):
        event.app.current_buffer.auto_up(count=event.arg)

    @kb.add("down")
    def _history_down(event):
        event.app.current_buffer.auto_down(count=event.arg)

    @kb.add(Keys.BracketedPaste, eager=True)
    def _handle_paste(event):
        pasted = (event.data or "").replace("\r\n", "\n").replace("\r", "\n")
        if _try_attach_clipboard_image(state.attached_images, _image_counter_ref):
            n = len(state.attached_images)
            img = state.attached_images[-1]
            size_kb = img.stat().st_size // 1024 if img.exists() else 0
            _cprint(f"  📎 Image #{n} attached: {img.name} ({size_kb}KB)")
            event.app.invalidate()
        if pasted:
            line_count = pasted.count("\n")
            buf = event.current_buffer
            if line_count >= 5 and not buf.text.strip().startswith("/"):
                from agentica.config import AGENTICA_HOME

                paste_dir = Path(AGENTICA_HOME) / "pastes"
                paste_dir.mkdir(parents=True, exist_ok=True)
                state.paste_counter += 1
                ts = datetime.now().strftime("%H%M%S")
                paste_file = paste_dir / f"paste_{state.paste_counter}_{ts}.txt"
                paste_file.write_text(pasted, encoding="utf-8")
                state.pasted_files.append((paste_file, line_count + 1))
                placeholder = f"[Pasted text #{state.paste_counter}: {line_count + 1} lines -> {paste_file}]"
                prefix = ""
                if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != "\n":
                    prefix = "\n"
                buf.insert_text(prefix + placeholder)
            else:
                buf.insert_text(pasted)

    @kb.add("c-v")
    def _handle_ctrl_v(event):
        if _try_attach_clipboard_image(state.attached_images, _image_counter_ref):
            img = state.attached_images[-1]
            size_kb = img.stat().st_size // 1024 if img.exists() else 0
            _cprint(f"  📎 Image #{len(state.attached_images)} attached: {img.name} ({size_kb}KB)")
            event.app.invalidate()

    @kb.add("escape", "v")
    def _handle_alt_v(event):
        if _try_attach_clipboard_image(state.attached_images, _image_counter_ref):
            img = state.attached_images[-1]
            size_kb = img.stat().st_size // 1024 if img.exists() else 0
            _cprint(f"  📎 Image #{len(state.attached_images)} attached: {img.name} ({size_kb}KB)")
            event.app.invalidate()

    @kb.add("c-o")
    def _expand_last_truncated(event):
        """Expand truncated blocks in a pager (CC-style expand/hide).

        Opens EVERY block folded during the current run (user input, tool
        output, edit/write diffs) in one pager so the user can scroll through
        all of it — not just the most recent one. Press ``Ctrl+o`` or ``Esc``
        to return; the terminal is restored, giving expand/hide semantics
        without flooding the inline transcript.
        """
        blocks = get_truncated_blocks()
        if not blocks:
            return
        # Single block → keep its own title; many → one combined view.
        if len(blocks) == 1:
            title = blocks[0].get("title", "Content")
            content = blocks[0].get("content", "")
        else:
            title = f"Expanded blocks ({len(blocks)})"
            parts = []
            for b in blocks:
                bt = b.get("title", "Content")
                bc = b.get("content", "")
                parts.append(f"=== {bt} · {len(bc.splitlines())} lines ===\n{bc}")
            content = "\n\n".join(parts)
        run_in_terminal(lambda: _open_in_pager(title, content))
        event.app.invalidate()

    class _PlaceholderProcessor(Processor):
        def __init__(self, get_text):
            self._get_text = get_text

        def apply_transformation(self, ti):
            if not ti.document.text and ti.lineno == 0:
                text = self._get_text()
                if text:
                    return Transformation(fragments=ti.fragments + [("class:placeholder", text)])
            return Transformation(fragments=ti.fragments)

    def _get_placeholder():
        if state.agent_running:
            return "type + Enter to queue, Ctrl+C to cancel"
        return ""

    def _get_prompt():
        if state.agent_running:
            return [("class:prompt-working", "~ ")]
        if state.shell_mode:
            return [("class:shell-prompt", "$ ")]
        return [("class:prompt", "❯ ")]

    def _get_status_bar():
        tw = shutil.get_terminal_size().columns
        return build_status_bar_fragments(
            model_name=tui_state.get("model_name", ""),
            model_provider=tui_state.get("model_provider", ""),
            profile_name=tui_state.get("profile_name", ""),
            context_tokens=tui_state.get("context_tokens", 0),
            context_window=tui_state.get("context_window", 128000),
            cost_usd=tui_state.get("cost_usd", 0.0),
            active_seconds=tui_state.get("active_seconds", 0.0),
            last_turn_seconds=tui_state.get("last_turn_seconds", 0.0),
            spinner_text=tui_state.get("spinner_text", ""),
            terminal_width=tw,
        )

    history_dir = os.path.dirname(history_file)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    _MAX_INPUT_ROWS = 12

    def _get_input_height() -> Dimension:
        widget = input_area
        if widget is None:
            return Dimension(min=1, max=_MAX_INPUT_ROWS, preferred=1)
        # Count *visual* rows, not just logical lines. With wrap_lines=True a
        # single long line wraps onto multiple terminal rows; counting only
        # explicit '\n' (document.line_count) would keep the box one row tall
        # and hide the wrapped text. We estimate wrapped rows from the usable
        # text width (terminal width minus the 2-char prompt like "❯ ").
        try:
            term_width = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            term_width = 80
        usable_width = max(1, term_width - 2)
        total_rows = 0
        for line in widget.buffer.document.lines:
            # A line of N chars occupies ceil(N / usable_width) rows
            # (empty line still occupies 1 row).
            total_rows += max(1, -(-len(line) // usable_width))
        total_rows = max(1, total_rows)
        preferred = min(_MAX_INPUT_ROWS, total_rows)
        return Dimension(min=1, max=_MAX_INPUT_ROWS, preferred=preferred)

    input_area = TextArea(
        height=_get_input_height,
        prompt=_get_prompt,
        style="class:input-area",
        multiline=True,
        wrap_lines=True,
        history=FileHistory(history_file),
        completer=AgenticaCompleter(),
        complete_while_typing=True,
        auto_suggest=AutoSuggestFromHistory(),
    )
    input_area.control.input_processors.append(_PlaceholderProcessor(_get_placeholder))

    from prompt_toolkit.layout.containers import ConditionalContainer, FloatContainer, Float

    status_bar = ConditionalContainer(
        Window(content=FormattedTextControl(_get_status_bar), height=1, wrap_lines=False),
        filter=Condition(lambda: tui_state.get("statusbar_visible", True)),
    )

    def _get_spinner_fragments():
        text = tui_state.get("spinner_text", "")
        if not text:
            return []
        return [("class:spinner", f"  {text}")]

    spinner_widget = ConditionalContainer(
        Window(content=FormattedTextControl(_get_spinner_fragments), height=1, wrap_lines=False),
        filter=Condition(lambda: bool(tui_state.get("spinner_text", ""))),
    )

    def _get_queue_bar():
        pairs = pending_queue.peek_all_with_timestamps()
        if not pairs:
            return []
        display_pairs = []
        for item, ts in pairs:
            if isinstance(item, str) and (item.startswith("/") or item.startswith("__")):
                continue
            if isinstance(item, tuple) and item[0] == "__BTW__":
                continue
            display_pairs.append((item, ts))
        if not display_pairs:
            return []
        frags = [("class:queue-label", f"  Queued ({len(display_pairs)}): ")]
        for i, (item, ts) in enumerate(display_pairs[:3]):
            if isinstance(item, tuple):
                text = str(item[0])
            else:
                text = str(item)
            preview = text[:40] + ("..." if len(text) > 40 else "")
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
            if i > 0:
                frags.append(("class:queue-dim", "  |  "))
            frags.append(("class:queue-time", f"({ts_str}) "))
            frags.append(("class:queue-dim", preview))
        if len(display_pairs) > 3:
            frags.append(("class:queue-dim", f"  ... +{len(display_pairs) - 3} more"))
        return frags

    queue_bar = ConditionalContainer(
        Window(content=FormattedTextControl(_get_queue_bar), height=1, wrap_lines=False),
        filter=Condition(lambda: not pending_queue.empty()),
    )

    input_rule = Window(char="─", height=1, style="class:input-rule")

    body = HSplit([spinner_widget, queue_bar, input_rule, input_area, status_bar])
    layout = Layout(
        FloatContainer(
            content=body,
            floats=[Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=12))],
        )
    )

    style = PTStyle.from_dict(
        {
            "input-area": "#FFF8DC",
            "placeholder": "#555555 italic",
            "prompt": "ansicyan bold",
            "prompt-working": "#888888 italic",
            "shell-prompt": "ansigreen bold",
            "hint": "#555555 italic",
            "input-rule": "#CD7F32",
            "queue-label": "#FFD700 bold",
            "queue-dim": "#8B8682 italic",
            "queue-time": "#8FBC8F",
            "spinner": "#FFD700 italic",
            "sb": "bg:#1a1a2e #C0C0C0",
            "sb-strong": "bg:#1a1a2e #FFD700 bold",
            "sb-dim": "bg:#1a1a2e #8B8682",
            "sb-good": "bg:#1a1a2e #8FBC8F bold",
            "sb-warn": "bg:#1a1a2e #FFD700 bold",
            "sb-bad": "bg:#1a1a2e #FF8C00 bold",
            "sb-critical": "bg:#1a1a2e #FF6B6B bold",
            "sb-spin": "bg:#1a1a2e #FFD700 italic",
            "completion-menu": "bg:#1a1a2e #FFF8DC",
            "completion-menu.completion": "bg:#1a1a2e #FFF8DC",
            "completion-menu.completion.current": "bg:#333355 #FFD700",
        }
    )

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )

    return app


# ==================== Main entry ====================


def run_interactive(
    agent_config: dict,
    extra_tool_names: Optional[List[str]] = None,
    workspace: Optional[Workspace] = None,
    skills_registry=None,
):
    """Run the interactive CLI with fixed-bottom input area TUI."""

    if not agent_config.get("debug"):
        suppress_console_logging()

    perm_mode = agent_config.get("permissions", "auto")
    permission_manager = PermissionManager(mode=perm_mode)

    extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None
    current_agent = create_agent(agent_config, extra_tools, workspace, skills_registry)

    # Load custom subagent definitions (.agentica/agents/*.md) before the TUI
    # starts. load_all_agents is fail-soft internally; the outer guard keeps a
    # broken agent file from ever blocking the CLI.
    try:
        from agentica.subagent_loader import load_all_agents

        load_all_agents()
    except Exception as e:
        logger.warning(f"Failed to load custom subagents at startup: {e}")

    con = get_console()

    # Print header BEFORE entering TUI
    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        work_dir=agent_config.get("work_dir"),
        extra_tools=extra_tool_names,
        shell_mode=False,
    )

    if workspace and workspace.exists():
        con.print(f"  Workspace: [green]{workspace.path}[/green]")

    # Always scan installed skills for auto-commands
    if skills_registry is None or len(skills_registry) == 0:
        load_skills()
        scanned = get_skill_registry()
        if len(scanned) > 0:
            skills_registry = scanned

    if skills_registry and len(skills_registry) > 0:
        skill_cmds = skills_registry.auto_commands()
        if skill_cmds:
            cmds_str = ", ".join(skill_cmds.keys())
            con.print(f"  Skills: [cyan]{len(skills_registry)} loaded[/cyan] (commands: {cmds_str})")
    if perm_mode != "auto":
        con.print(f"  Permissions: [yellow]{perm_mode}[/yellow]")
    con.print()

    # Session state
    state = SessionState(current_agent=current_agent)

    tui_state = {
        "model_name": agent_config.get("model_name", ""),
        "model_provider": agent_config.get("model_provider", ""),
        "profile_name": get_active_profile_name(),
        "context_tokens": 0,
        "context_window": current_agent.model.context_window if current_agent.model else 128000,
        "cost_usd": 0.0,
        "active_seconds": 0.0,
        "last_turn_seconds": 0.0,
        "spinner_text": "",
        "show_reasoning": True,
        "statusbar_visible": True,
        "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_api_calls": 0,
        "debug": bool(agent_config.get("debug")),
    }

    pending_queue = PendingQueue()
    # Keep a mutable list wrapper for image_counter (needed by _try_attach_clipboard_image)
    _image_counter_ref = [0]
    session_tokens = [0]

    def _build_ctx() -> CommandContext:
        """Build a CommandContext from current session state."""
        return CommandContext(
            agent_config=agent_config,
            current_agent=state.current_agent,
            extra_tools=extra_tools,
            extra_tool_names=extra_tool_names,
            workspace=workspace,
            skills_registry=skills_registry,
            permission_manager=permission_manager,
            shell_mode=state.shell_mode,
            tui_state=tui_state,
            pending_queue=pending_queue,
            agent_running=state.agent_running,
            attached_images=state.attached_images,
            image_counter=_image_counter_ref,
            bg_tasks=state.bg_tasks,
            bg_task_counter=state.bg_task_counter,
            goal_manager=state.goal_manager,
            goal_lock=state.goal_lock,
        )

    def _dispatch_concurrent_cmd(cmd: str, cmd_args: str):
        """Dispatch a command — called from _handle_enter for concurrent execution."""
        # Special handling for /btw — run concurrently
        if cmd == "/btw":
            question = cmd_args.strip()
            if question and state.current_agent:
                _run_btw_concurrent(state.current_agent, question, tui_state)
            return

        handler = COMMAND_HANDLERS.get(cmd)
        if handler:
            ctx = _build_ctx()
            result = handler(ctx, cmd_args)
            if isinstance(result, dict):
                _apply_command_result(result)

    def _apply_command_result(result: dict):
        """Apply side effects from command handler results."""
        nonlocal skills_registry, extra_tool_names
        if "current_agent" in result:
            state.current_agent = result["current_agent"]
            tui_state["model_name"] = agent_config.get("model_name", "")
            tui_state["context_window"] = (
                state.current_agent.model.context_window if state.current_agent.model else 128000
            )
            session_tokens[0] = 0
            tui_state["context_tokens"] = 0
            tui_state["cost_usd"] = 0.0
        if result.get("model_switched"):
            tui_state["model_name"] = agent_config.get("model_name", "")
            tui_state["context_window"] = (
                state.current_agent.model.context_window if state.current_agent.model else 128000
            )
        if "skills_registry" in result:
            skills_registry = result["skills_registry"]
        if "extra_tool_names" in result:
            extra_tool_names = result["extra_tool_names"]
        if "goal_manager" in result:
            state.goal_manager = result["goal_manager"]
            # Reset per-turn token baseline whenever the manager changes
            # (new session, cleared goal, resumed session). Avoids carrying
            # the previous session's cumulative counts into a fresh goal.
            state.goal_tokens_baseline = 0

    app = _setup_tui(
        state,
        skills_registry,
        tui_state,
        pending_queue,
        image_counter_ref=_image_counter_ref,
        dispatch_cmd=_dispatch_concurrent_cmd,
    )

    # Activate ChatConsole for TUI — all get_console() calls now return this
    chat_console = ChatConsole()
    set_active_console(chat_console)

    # ── Background thread: process input queue and run agent ──
    def process_loop():
        nonlocal skills_registry
        while not state.should_exit:
            try:
                payload = pending_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if payload is None:
                continue

            if payload == "__CANCEL__":
                continue

            if payload == "__TOGGLE_SHELL_MODE__":
                state.shell_mode = not state.shell_mode
                mode_str = (
                    "Shell Mode ON - Commands execute directly"
                    if state.shell_mode
                    else "Agent Mode ON - AI processes your input"
                )
                _cprint(f"\n{mode_str}")
                app.invalidate()
                continue

            # If agent is currently running, re-queue
            if state.agent_running:
                pending_queue.put(payload)
                time.sleep(0.1)
                continue

            # Unpack payload
            submit_images = []
            is_btw = False
            if isinstance(payload, tuple):
                if payload[0] == "__BTW__":
                    is_btw = True
                    user_input = payload[1]
                else:
                    user_input, submit_images = payload
            else:
                user_input = str(payload)

            user_input = user_input.strip()
            if not user_input and not submit_images:
                continue

            if not user_input and submit_images:
                user_input = "What do you see in this image?"

            # Detect file drops
            dropped = _detect_file_drop(user_input)
            if dropped:
                if dropped["is_image"]:
                    submit_images.append(dropped["path"])
                    _cprint(f"  📎 Attached image: {dropped['path'].name}")
                    user_input = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"
                else:
                    user_input = f"@{dropped['path']} {dropped['remainder']}".strip()

            # Shell mode
            if state.shell_mode:
                if user_input.startswith("/") and user_input.split()[0].lower() in {
                    "/exit",
                    "/quit",
                    "/help",
                    "/model",
                    "/debug",
                    "/clear",
                    "/reset",
                }:
                    pass
                else:
                    _handle_shell_command(user_input, agent_config.get("work_dir"))
                    continue

            # Slash commands
            first_word = user_input.split()[0].lower() if user_input else ""
            skill_cmds = skills_registry.auto_commands() if skills_registry else {}
            is_command = first_word in COMMAND_HANDLERS or first_word in skill_cmds
            if is_command:
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                handler = COMMAND_HANDLERS.get(cmd)
                if handler:
                    ctx = _build_ctx()
                    try:
                        result = handler(ctx, cmd_args)
                    except Exception as e:
                        con.print(f"  [red]Command error: {e}[/red]")
                        app.invalidate()
                        continue
                    if result == "EXIT":
                        state.should_exit = True
                        if app.is_running:
                            app.exit()
                        break
                    if isinstance(result, dict):
                        _apply_command_result(result)
                    # Sync bg_task_counter back from ctx
                    state.bg_task_counter = ctx.bg_task_counter
                    app.invalidate()
                    continue
                else:
                    # Skill auto-command dispatch
                    matched_skill = skill_cmds.get(cmd)
                    if matched_skill:
                        skill_prompt = matched_skill.get_prompt()
                        user_instruction = cmd_args.strip()
                        skill_msg = f"# {matched_skill.name} Skill\n\n{skill_prompt}"
                        if user_instruction:
                            skill_msg += f"\n\n## User Request\n{user_instruction}"
                        _cprint(f"  Skill activated: {matched_skill.name}")
                        user_input = skill_msg

            # Expand paste references
            _paste_ref_re = re.compile(r"\[Pasted text #\d+: \d+ lines -> (.+?)\]")
            paste_refs = list(_paste_ref_re.finditer(user_input))
            n_pasted_blocks = len(paste_refs) + len(state.pasted_files)
            n_pasted_lines = sum(n for _, n in state.pasted_files) if state.pasted_files else 0
            if paste_refs:

                def _expand_ref(m):
                    p = Path(m.group(1))
                    if p.exists():
                        return p.read_text(encoding="utf-8")
                    return m.group(0)

                expanded = _paste_ref_re.sub(_expand_ref, user_input)
                n_pasted_lines += expanded.count("\n") + 1
                user_input = expanded
            state.pasted_files.clear()

            prompt_text, mentioned_files = parse_file_mentions(user_input)
            final_input = inject_file_contents(prompt_text, mentioned_files)

            if submit_images:
                for img in submit_images:
                    size_kb = img.stat().st_size // 1024 if img.exists() else 0
                    _cprint(f"  📎 {img.name} ({size_kb}KB) -> {img}")

            if not is_btw:
                display_user_message(
                    user_input,
                    pasted_blocks=n_pasted_blocks,
                    pasted_lines=n_pasted_lines,
                )

            turn_images = submit_images if submit_images else None

            # BTW: ephemeral side question (when agent is NOT running, via queue)
            if is_btw:
                _run_btw_concurrent(state.current_agent, final_input, tui_state)
                continue

            # Run agent
            state.agent_running = True
            app.invalidate()
            _process_stream_response(
                state.current_agent,
                final_input,
                session_tokens,
                tui_state,
                images=turn_images,
                permission_manager=permission_manager,
            )
            state.agent_running = False
            tui_state["spinner_text"] = ""

            # Standing-goal hook: decide whether to enqueue a continuation
            # for the next turn. Honors user-priority and cancel semantics.
            _maybe_continue_goal(state, pending_queue, tui_state)

            app.invalidate()

    process_thread = threading.Thread(target=process_loop, daemon=True)
    process_thread.start()

    # ── Spinner refresh thread ──
    _SPIN_ICONS = ["✦", "✧", "✦", "✧", "⊹", "✦", "✧", "⊹"]
    _TOOL_DOTS = ["", ".", "..", "..."]
    _frame_idx = [0]

    def spinner_loop():
        while not state.should_exit:
            if state.agent_running and app.is_running:
                if tui_state.get("_thinking"):
                    _frame_idx[0] = (_frame_idx[0] + 1) % len(_SPIN_ICONS)
                    icon = _SPIN_ICONS[_frame_idx[0]]
                    tui_state["spinner_text"] = f"{icon} thinking..."
                elif tui_state.get("_spinner_base", ""):
                    # Tool running — animate dots + show elapsed time
                    base = tui_state["_spinner_base"]
                    _frame_idx[0] = (_frame_idx[0] + 1) % len(_TOOL_DOTS)
                    dots = _TOOL_DOTS[_frame_idx[0]]
                    elapsed = time.monotonic() - tui_state.get("_tool_start", time.monotonic())
                    tui_state["spinner_text"] = f"{base}{dots} ({elapsed:.0f}s)"
                app.invalidate()
                time.sleep(0.4)
            else:
                time.sleep(0.3)

    spinner_thread = threading.Thread(target=spinner_loop, daemon=True)
    spinner_thread.start()

    # ── Run the TUI ──
    try:
        with patch_stdout():
            app.run()
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        state.should_exit = True
        set_active_console(None)

    get_console().print("\nThank you for using Agentica CLI. Goodbye!", style="bold green")
