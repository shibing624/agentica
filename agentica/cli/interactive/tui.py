# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: prompt_toolkit TUI setup, keybindings, and resize handling
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea

from agentica.cli.commands.context import CONCURRENT_CMDS, PendingQueue
from agentica.cli.commands.registry import COMMAND_REGISTRY
from agentica.cli.display import (
    build_status_bar_fragments,
    display_user_message,
    get_file_completions,
    get_truncated_blocks,
)
from agentica.cli.runtime import get_console, history_file
from agentica.utils.log import logger

from .attachments import (
    _deduplicate_image_attachments,
    _try_attach_clipboard_image,
    queue_item_preview,
)
from .console_io import (
    _ask_active,
    _cprint,
    _open_in_pager,
    _toggle_output_pause,
    _tty_write_lock,
)
from .session_state import SessionState

# ==================== TUI setup ====================


def _steer_or_queue(state: SessionState, pending_queue: PendingQueue, text: str, payload) -> bool:
    """Route plain text typed mid-run: steer the live run, queue on refusal.

    Returns True when the text was accepted as steering. A False from
    ``steer()`` means the run ended between the UI's ``agent_running`` check
    and the call (the TOCTOU gap ``Agent.steer`` documents) — the text falls
    back to the queue rather than being dropped.
    """
    agent = state.current_agent
    if agent is not None and agent.steer(text):
        return True
    pending_queue.put(payload)
    return False


def _queue_next_turn(state: SessionState, pending_queue: PendingQueue, text: str):
    """Queue the current input as the next turn and consume attachments."""
    images = _deduplicate_image_attachments(list(state.attached_images))
    state.attached_images.clear()
    payload = (text, images) if images else text
    pending_queue.put(payload)
    return payload, images


class _CleanResizeApplication(Application):
    """Application that collapses its bottom frame during terminal resize.

    In non-full-screen mode prompt_toolkit erases its bottom frame (spinner /
    hint / queue / rule / input box / status bar) using a *relative* cursor
    offset (``renderer._cursor_pos``). On resize the terminal reflows the whole
    visible area (including the scrollback transcript printed via patch_stdout)
    at the new width, so that recorded offset becomes stale: the default
    ``_on_resize`` moves the cursor up by the old row count and erases the
    wrong region, leaving ghost copies of the frame stacked in the scrollback.
    A full ``erase_screen`` does not help here because ESC[2J only clears the
    visible viewport, while ghost rows have already been reflowed *into* the
    scrollback where no escape sequence can reach them.

    Strategy: during a resize burst we hide the tall parts of the bottom frame
    (input prompt, queue bar and status bar) via ``tui_state["_resize_collapsed"]``
    and pin the input area to a single row, so that at most one row can be
    reflowed into scrollback per SIGWINCH. A short debounce timer restores the
    full frame once the user stops resizing.

    The transcript stays in the terminal's native scrollback, so scrolling up
    still shows prior output.
    """

    #: How long to keep the frame collapsed after the last SIGWINCH. Tuned so
    #: that continuous drag-resize keeps everything collapsed, while a single
    #: resize restores the spinner/rule within a quarter second.
    _RESIZE_DEBOUNCE_SEC: float = 0.25
    tui_state: dict | None = None

    def _on_resize(self) -> None:
        tui_state = self.tui_state
        if tui_state is not None:
            tui_state["_resize_collapsed"] = True
            # Cancel any pending restore timer and re-arm it. Repeated SIGWINCH
            # events (e.g. while the user is dragging the terminal edge) keep
            # bumping the deadline forward, so we only restore once the burst
            # has clearly ended.
            prev_handle = tui_state.pop("_resize_restore_handle", None)
            if prev_handle is not None:
                try:
                    prev_handle.cancel()
                except Exception:
                    pass
            try:
                loop = asyncio.get_event_loop()
                tui_state["_resize_restore_handle"] = loop.call_later(
                    self._RESIZE_DEBOUNCE_SEC,
                    self._restore_after_resize,
                )
            except Exception:
                # No running loop yet — restore immediately so we don't leave
                # the frame permanently collapsed.
                tui_state["_resize_collapsed"] = False

        # Still do the normal erase+redraw so the (now 1-row) frame lands in
        # a sane place. renderer.erase resets _cursor_pos and _last_screen,
        # forcing a fresh non-diff redraw at the new dimensions.
        renderer = self.renderer
        renderer.erase(leave_alternate_screen=False)
        self._request_absolute_cursor_position()
        self._redraw()

    def _restore_after_resize(self) -> None:
        tui_state = self.tui_state
        if tui_state is None:
            return
        tui_state["_resize_collapsed"] = False
        tui_state.pop("_resize_restore_handle", None)
        # Force a clean, non-diff repaint. A plain invalidate() would diff the
        # restored (full) frame against the collapsed (1-row) screen and redraw
        # the full frame below where the collapsed row landed — pushing the
        # collapsed row plus the old frame into scrollback as extra ghosts.
        # Erase with an absolute-cursor resync first, then redraw, so the
        # restored frame paints in exactly one correct position.
        try:
            renderer = self.renderer
            renderer.erase(leave_alternate_screen=False)
            self._request_absolute_cursor_position()
            self._redraw()
        except Exception as e:
            # Terminal I/O boundary: a failed repaint only leaves a cosmetic bad
            # frame (no data at risk), but log it so the cause is diagnosable.
            logger.debug(f"resize restore repaint failed: {e}")


# Trailing hint line under an armed ask_user_question prompt. Ctrl+\ (SIGQUIT →
# hard exit, installed in app.py) is advertised *only* here: prompt_toolkit's
# raw_mode clears ISIG, so the key is an inert 0x1c byte while the event loop is
# healthy. It becomes a real signal only inside run_in_terminal's cooked_mode
# window — which is where a loop starved by background writes gets stuck while
# an answer is pending. So this prompt is the one place the hint is true.
_ASK_KEY_HINT = "    Enter to answer · Ctrl+C to cancel · Ctrl+\\ if frozen"


def _ask_prompt_lines(req) -> list[str]:
    """Lines rendered above the input box while an ask_user_question is armed.

    Both the fragment builder and the widget height derive from this, so the
    reserved height can't drift from what is actually drawn.
    """
    lines = [f"  ? {req.prompt}"]
    if req.options:
        lines.extend(f"    {i}. {opt}" for i, opt in enumerate(req.options, 1))
    lines.append(_ASK_KEY_HINT)
    return lines


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
                _cprint("\n⚡ Force exiting... session summary will be shown below.")
                state.should_exit = True
                event.app.exit()
                return
            state.last_ctrl_c = now
            _ask_active[0] = False
            _cprint("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
            # If the agent is currently blocked in a ask_user_question tool call, the
            # asyncio task.cancel() route alone won't help: the tool runs on a
            # worker thread waiting on a queue.Queue.get(), and Python threads
            # can't be interrupted from asyncio. We must wake that thread by
            # putting a sentinel on the queue; the tool callback raises
            # AgentCancelledError, which the agent runtime unwinds cleanly.
            pending_req = state.input_request
            if pending_req is not None:
                cancelled = pending_req.cancel()
                logger.info(f"[ask] ctrl-c cancel: cancelled={cancelled}")
                if state.input_request is pending_req:
                    state.input_request = None
                tui_state["spinner_text"] = (
                    "Cancelled pending answer" if cancelled else "Answer prompt already closed"
                )
                # Clear whatever the user was typing into the answer field so
                # the next prompt starts fresh.
                try:
                    event.app.current_buffer.reset()
                except Exception:
                    pass
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
                tui_state["spinner_text"] = "Press Ctrl+C again to exit; summary appears below"
                event.app.invalidate()

    @kb.add("escape", "p")
    def _toggle_transcript_pause(event):
        paused, buffered_lines = _toggle_output_pause()
        tui_state["output_paused"] = paused
        if paused:
            tui_state["spinner_text"] = "Output paused · browse history · Alt+P to resume"
        else:
            tui_state["spinner_text"] = (
                f"Output resumed · flushed {buffered_lines} line(s)"
                if buffered_lines
                else "Output resumed"
            )
        event.app.invalidate()

    @kb.add("enter")
    def _handle_enter(event):
        raw_text = event.app.current_buffer.text
        text = raw_text.strip()
        has_images = bool(state.attached_images)

        # If the agent is waiting on a ask_user_question tool request, route this line
        # straight to the request's result queue (unblocking the agent thread)
        # instead of treating it as a new turn. Empty input is allowed here so
        # the user can accept a default/blank answer.
        if state.input_request is not None:
            req = state.input_request
            submitted = req.submit(raw_text)
            logger.info(
                f"[ask] enter submit: submitted={submitted} text={raw_text[:60]!r}"
            )
            if state.input_request is req:
                state.input_request = None
            if submitted:
                tui_state["spinner_text"] = "Submitted answer"
            else:
                tui_state["spinner_text"] = "Answer prompt already closed"
            event.app.current_buffer.reset(append_to_history=True)
            event.app.invalidate()
            return

        if not text and not has_images:
            return

        # The user is at the keyboard, so no agent-to-agent exchange running
        # underneath is unattended any more. Reopen the peer channel before the
        # line is dispatched: refusing "message the other session" because two
        # agents had been chatting earlier makes the cap look broken.
        if state.peer_session is not None:
            state.peer_session.note_user_turn()

        images = _deduplicate_image_attachments(list(state.attached_images))
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

        # Mid-run default: plain text STEERS the current run. Most follow-ups
        # typed while the agent works are corrections or extra context ("not
        # that file", "the error is actually 503") — queueing them until the
        # run ends means the agent finishes on stale assumptions and the next
        # turn is rework. Steering lands at the next tool-batch boundary
        # instead. Boundaries:
        # - slash input keeps its meaning: skill auto-commands and non-
        #   concurrent commands stay queued as next-turn prompts (concurrent
        #   commands and /btw already returned above)
        # - image attachments queue — the steer channel is text-only
        # - steer() refused (run ended in the TOCTOU gap) falls back to the
        #   queue; steer accepted but never drained (typed during the final
        #   inference) is promoted to a queued turn when the run ends. The
        #   message is never lost either way.
        # Gate on the post-dedup ``images`` (the payload ground truth), not the
        # earlier ``has_images`` snapshot: if another image source is ever added
        # between the two, an accepted steer would silently drop it.
        if state.agent_running and not images and not text.startswith("/"):
            if _steer_or_queue(state, pending_queue, text, payload):
                # Honest copy: acceptance only means "buffered for the next
                # inference boundary" — if the run finishes first, the text is
                # promoted to a queued turn and app.py says so explicitly then.
                display_user_message(text)
                get_console().print("  Guidance added to the current task. Tab queues the next turn.")
            event.app.current_buffer.reset(append_to_history=True)
            event.app.invalidate()
            return

        # Idle agent (or queued fallback): the message runs as a new turn and
        # is shown live in the bottom ``Queued (N):`` bar. The queue bar
        # already renders queued items with timestamps, so we deliberately do
        # NOT also print a notice into the chat stream — that would interleave
        # with the running AI response box.
        pending_queue.put(payload)

        event.app.current_buffer.reset(append_to_history=True)
        event.app.invalidate()

    @kb.add("tab", eager=True)
    def _handle_tab(event):
        buf = event.current_buffer
        if buf.complete_state or (buf.suggestion and buf.suggestion.text):
            _accept_or_complete(event)
            return
        raw_text = buf.text
        text = raw_text.strip()
        if not state.agent_running or state.input_request is not None or not text:
            _accept_or_complete(event)
            return
        _payload, images = _queue_next_turn(state, pending_queue, text)
        display_user_message(text, images=images)
        get_console().print("  Queued as the next turn. Enter steers the current task.")
        buf.reset(append_to_history=True)
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
            event.app.invalidate()
        if pasted:
            line_count = pasted.count("\n")
            buf = event.current_buffer
            if line_count >= 5 and not buf.text.strip().startswith("/"):
                from agentica.cli.runtime import CACHE_DIR

                paste_dir = Path(CACHE_DIR) / "pastes"
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
            event.app.invalidate()

    @kb.add("escape", "v")
    def _handle_alt_v(event):
        if _try_attach_clipboard_image(state.attached_images, _image_counter_ref):
            event.app.invalidate()

    @kb.add("c-o")
    def _expand_last_truncated(event):
        """Expand truncated blocks in a pager (CC-style expand/hide).

        Opens EVERY block folded during the current run (user input, tool
        output, edit/write diffs) in one pager so the user can scroll through
        all of it — not just the most recent one. Press ``Ctrl+O`` or ``Esc``
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

        def _paged():
            # Hold the TTY write lock for the whole pager session so the
            # background streaming thread can't print over `less`.
            with _tty_write_lock:
                _open_in_pager(title, content)

        run_in_terminal(_paged)
        event.app.invalidate()

    class _PlaceholderProcessor(Processor):
        def __init__(self, get_text):
            self._get_text = get_text

        def apply_transformation(self, transformation_input):
            if not transformation_input.document.text and transformation_input.lineno == 0:
                text = self._get_text()
                if text:
                    return Transformation(
                        fragments=transformation_input.fragments + [("class:placeholder", text)]
                    )
            return Transformation(fragments=transformation_input.fragments)

    def _get_placeholder():
        if state.input_request is not None:
            return "Type your answer, then Enter · Ctrl+C to abort"
        if state.agent_running:
            return "Enter to steer · Tab to queue next · Ctrl+C to cancel"
        return "Enter to send · Ctrl+J newline · / commands · @ files"

    def _get_prompt():
        if state.agent_running:
            return [("class:prompt-working", "~ ")]
        return [("class:prompt", "❯ ")]

    def _get_status_bar():
        tw = shutil.get_terminal_size().columns
        spinner = tui_state.get("spinner_text", "")
        # spinner_text is set exclusively while the agent is producing output
        # (streaming, tool execution, thinking). Use its presence as the
        # ground-truth signal for "agent is working right now" — avoids a
        # separate flag we'd have to keep in sync.
        return build_status_bar_fragments(
            model_name=tui_state.get("model_name", ""),
            model_provider=tui_state.get("model_provider", ""),
            profile_name=tui_state.get("profile_name", ""),
            thinking_mode=tui_state.get("thinking_mode", ""),
            work_dir=tui_state.get("work_dir", ""),
            git_branch=tui_state.get("git_branch", ""),
            context_tokens=tui_state.get("context_tokens", 0),
            context_window=tui_state.get("context_window", 128000),
            cost_usd=tui_state.get("cost_usd", 0.0),
            active_seconds=tui_state.get("active_seconds", 0.0),
            last_turn_seconds=tui_state.get("last_turn_seconds", 0.0),
            spinner_text=spinner,
            terminal_width=tw,
            agent_running=bool(spinner),
            background_terminal_count=state.background_processes.running_count(),
            goal_tokens_used=tui_state.get("goal_tokens_used"),
            goal_token_budget=tui_state.get("goal_token_budget"),
        )

    history_dir = os.path.dirname(history_file)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    _MAX_INPUT_ROWS = 8

    def _get_input_height() -> Dimension:
        widget = input_area
        if widget is None:
            return Dimension(min=1, max=1, preferred=1)
        # During a resize burst collapse the input box to exactly one row so the
        # whole bottom frame shrinks to a single line. This caps how much of the
        # frame the terminal can reflow into scrollback as ghost rows (see
        # _CleanResizeApplication). _resize_collapsed is set by _on_resize and
        # cleared by _restore_after_resize.
        if tui_state.get("_resize_collapsed"):
            return Dimension(min=1, max=1, preferred=1)
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
        needed = min(_MAX_INPUT_ROWS, total_rows)
        # IMPORTANT: pin min == max == preferred == needed (a fixed size, not
        # a range). prompt_toolkit's HSplit dimension solver treats `max` as
        # "how far this child may grow to soak up spare terminal rows", not
        # just a content cap: once every child hits its `preferred` size, a
        # second pass keeps growing children (up to their `max`) to fill the
        # rest of the reported terminal height. With `max=_MAX_INPUT_ROWS`
        # that inflated the input box to a full 8 rows of blank padding any
        # time the terminal had spare height, then the renderer had to jump
        # the cursor back up to the real (much higher) row — desyncing the
        # redraw and making already-typed lines appear to vanish/duplicate.
        # A fixed dimension (min=max=preferred) gives the box exactly the
        # rows its content needs, every render, with no room for the solver
        # to expand it. Growth beyond `_MAX_INPUT_ROWS` is handled by the
        # TextArea's own internal scrolling, which keeps the cursor line
        # visible.
        return Dimension(min=needed, max=needed, preferred=needed)

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
    input_processors = input_area.control.input_processors
    if input_processors is not None:
        input_processors.append(_PlaceholderProcessor(_get_placeholder))

    from prompt_toolkit.layout.containers import ConditionalContainer, FloatContainer, Float

    status_bar = ConditionalContainer(
        Window(content=FormattedTextControl(_get_status_bar), height=1, wrap_lines=False),
        filter=Condition(
            lambda: tui_state.get("statusbar_visible", True)
            and not tui_state.get("_resize_collapsed")
        ),
    )

    def _get_spinner_fragments():
        if tui_state.get("output_paused"):
            return [
                (
                    "class:spinner",
                    "  Output paused · browse history · Alt+P to resume",
                )
            ]
        text = tui_state.get("spinner_text", "")
        if not text:
            return []
        return [("class:spinner", f"  {text}")]

    # ── ask_user_question prompt widget ──
    # When the agent parks on a ask_user_question/confirm tool it sets
    # state.input_request. We render the question here, as part of the layout
    # on the main prompt_toolkit thread, instead of having the background
    # agent thread call print_formatted_text(). In a non-full-screen app that
    # background print triggers run_in_terminal (CPR + full redraw), which
    # races the spinner's invalidate() and desyncs the input cursor so the
    # box stops echoing keystrokes. Rendering inline in the layout removes
    # that race entirely.
    def _get_input_prompt_fragments():
        req = state.input_request
        if req is None:
            return []
        *question, hint = _ask_prompt_lines(req)
        return [
            ("class:input-prompt", "\n".join(question)),
            ("class:hint", f"\n{hint}"),
        ]

    def _get_input_prompt_height() -> int:
        req = state.input_request
        if req is None:
            return 0
        return sum(1 + line.count("\n") for line in _ask_prompt_lines(req))

    input_prompt_widget = ConditionalContainer(
        Window(
            content=FormattedTextControl(_get_input_prompt_fragments),
            height=_get_input_prompt_height,
            wrap_lines=True,
        ),
        filter=Condition(
            lambda: state.input_request is not None
            and not tui_state.get("_resize_collapsed")
        ),
    )

    def _get_queue_bar():
        pairs = pending_queue.peek_all_with_timestamps()
        if not pairs:
            return []
        frags = [("class:queue-label", f"  Queued ({len(pairs)}): ")]
        for i, (item, ts) in enumerate(pairs[:3]):
            text = queue_item_preview(item)
            preview = text[:40] + ("..." if len(text) > 40 else "")
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
            if i > 0:
                frags.append(("class:queue-dim", "  |  "))
            frags.append(("class:queue-time", f"({ts_str}) "))
            frags.append(("class:queue-dim", preview))
        if len(pairs) > 3:
            frags.append(("class:queue-dim", f"  ... +{len(pairs) - 3} more"))
        return frags

    queue_bar = ConditionalContainer(
        Window(content=FormattedTextControl(_get_queue_bar), height=1, wrap_lines=False),
        filter=Condition(
            lambda: not pending_queue.empty()
            and not tui_state.get("_resize_collapsed")
        ),
    )

    # NOTE: no ``input_rule`` and no standalone ``spinner_widget`` here.
    # The gutter design (assistant ``▏`` bar + closing ``Rule`` in the
    # transcript) already provides a hard boundary between the assistant
    # turn and the input line, so an extra horizontal rule above the input
    # would just stack redundant separators. The spinner text is folded into
    # the leftmost segment of ``status_bar`` (see ``build_status_bar_fragments``)
    # so we never occupy a full extra row for it.
    body = HSplit([input_prompt_widget, queue_bar, input_area, status_bar])
    layout = Layout(
        FloatContainer(
            content=body,
            floats=[Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=12))],
        )
    )

    style = PTStyle.from_dict(
        {
            "input-area": "bg:#20203a #F8F8F2 bold",
            "placeholder": "#555555 italic",
            "prompt": "#FFD700 bold",
            "prompt-working": "#888888 italic",
            "hint": "#555555 italic",
            "queue-label": "#FFD700 bold",
            "queue-dim": "#8B8682 italic",
            "queue-time": "#8FBC8F",
            "spinner": "#FFD700 italic",
            "input-prompt": "#FFD700 bold",
            "sb": "bg:#1a1a2e #C0C0C0",
            "sb-strong": "bg:#1a1a2e #FFD700 bold",
            "sb-dim": "bg:#1a1a2e #8B8682",
            "sb-good": "bg:#1a1a2e #8FBC8F bold",
            "sb-warn": "bg:#1a1a2e #FFD700 bold",
            "sb-bad": "bg:#1a1a2e #FF8C00 bold",
            "sb-critical": "bg:#1a1a2e #FF6B6B bold",
            "sb-spin": "bg:#1a1a2e #FFD700 italic",
            # Agent-running variants: same foreground palette, one shade
            # darker background (#0f0f1a instead of #1a1a2e) so the bar
            # visually "cools down" while work is in progress. Numeric fields
            # keep updating — this is intentional, users often watch tokens
            # and cost tick during long turns.
            "sb-active": "bg:#0f0f1a #C0C0C0",
            "sb-strong-active": "bg:#0f0f1a #FFD700 bold",
            "sb-dim-active": "bg:#0f0f1a #8B8682",
            "sb-good-active": "bg:#0f0f1a #8FBC8F bold",
            "sb-warn-active": "bg:#0f0f1a #FFD700 bold",
            "sb-bad-active": "bg:#0f0f1a #FF8C00 bold",
            "sb-critical-active": "bg:#0f0f1a #FF6B6B bold",
            "sb-spin-active": "bg:#0f0f1a #FFD700 italic",
            "completion-menu": "bg:#1a1a2e #FFF8DC",
            "completion-menu.completion": "bg:#1a1a2e #FFF8DC",
            "completion-menu.completion.current": "bg:#333355 #FFD700",
        }
    )

    app = _CleanResizeApplication(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    # Attached so _on_resize can flip the collapse flag and schedule the
    # debounce timer without needing to override __init__.
    app.tui_state = tui_state

    return app


__all__ = ['_CleanResizeApplication', '_ASK_KEY_HINT', '_ask_prompt_lines', '_setup_tui']
