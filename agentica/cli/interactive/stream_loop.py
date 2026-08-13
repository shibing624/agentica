# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Stream response processing, spinner, and live status refresh
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from time import perf_counter
from typing import Optional

from agentica.cli.context_usage import measure_context
from agentica.cli.rewind import extract_rewrite_paths, get_turn_checkpointer
from agentica.cli.display import (
    StreamDisplayManager,
    display_agent_execution_error,
    format_session_summary,
    resumable_session_id,
)
from agentica.cli.runtime import get_console
from agentica.cli.usage_display import ProviderUsageSummary
from agentica.run_display import RunDisplayEventKind, classify_run_response
from agentica.run_response import AgentCancelledError
from agentica.utils.async_utils import run_sync
from agentica.utils.log import logger

from .attachments import _ocr_images_parallel
from .session_state import _ToolResultSequencer

# ==================== BTW concurrent handler ====================


# Braille spinner frames — cycled continuously while the agent is alive so
# the user can tell a live process (spinner turning) from a hung one
# (spinner frozen) at a glance.
_BRAILLE_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Shown in the spinner line while the agent is parked on a ask_user_question/confirm
# tool. Steady (no animation) so the spinner thread can stop invalidate() churn
# that would otherwise fight the input renderer.
_WAITING_FOR_INPUT_TEXT = "⏸  waiting for your answer…"


def _render_spinner_text(frame_idx: int, phase: str, base: str, elapsed: float) -> str:
    """Render one spinner line: ``⠋ <phase label> (Ns)``.

    phase: ``thinking`` | ``reasoning`` | ``tool`` | ``answering`` |
           ``compacting`` | ``idle``
    base:  tool label (e.g. ``🔧 grep``) for the ``tool`` phase
    """
    if phase == "idle":
        return ""
    icon = _BRAILLE_SPINNER[frame_idx % len(_BRAILLE_SPINNER)]
    if phase == "tool" and base:
        return f"{icon} {base} ({elapsed:.0f}s)"
    if phase == "compacting":
        return f"{icon} 🗜 compacting context… ({elapsed:.0f}s)"
    if phase == "answering":
        return f"{icon} answering… ({elapsed:.0f}s)"
    if phase == "reasoning":
        return f"{icon} reasoning… ({elapsed:.0f}s)"
    return f"{icon} thinking… ({elapsed:.0f}s)"


# ==================== Stream response ====================


def _refresh_live_status(
    tui_state: dict,
    agent,
    request_start: float,
    cost_baseline: float,
    active_baseline: float,
    calls_baseline: int,
    goal_tokens_baseline: int,
) -> None:
    """Write live cost and timing fields from the current run's cost tracker.

    Idempotent — uses ``baseline + current-run-delta`` math instead of ``+=``,
    so calling it repeatedly mid-turn (from the spinner loop, ~8x/sec) and
    once more at turn end never double-counts. The cost_tracker is populated
    live by the model layer on every LLM API call (``model/base.py`` calls
    ``cost_tracker.record(...)`` as each call returns), so reading it at any
    point gives the running totals for the in-flight turn.

    Context occupancy is updated separately from Runner ``context.usage``
    events. Cost accounting cannot represent it because a turn may contain
    retries, tool loops, and auxiliary calls.

    ``goal_tokens_used`` follows the same baseline+delta rule so a standing
    goal's spend ticks during the turn; ``_maybe_continue_goal`` overwrites it
    with the manager's charged total once the turn is accounted for.
    """
    elapsed = perf_counter() - request_start
    tui_state["last_turn_seconds"] = elapsed
    tui_state["active_seconds"] = active_baseline + elapsed
    run_response = agent.run_response if agent is not None else None
    ct = run_response.cost_tracker if run_response is not None else None
    if ct is None:
        return
    if ct.turns > 0:
        tui_state["cost_usd"] = cost_baseline + ct.total_cost_usd
        tui_state["total_api_calls"] = calls_baseline + ct.turns
    if tui_state.get("goal_token_budget") is not None:
        turn_tokens = max(0, ct.total_input_tokens + ct.total_output_tokens)
        tui_state["goal_tokens_used"] = goal_tokens_baseline + turn_tokens


def _make_compact_phase_handler(set_phase, tui_state: dict):
    """Bridge ``compact.start`` / ``compact.end`` onto the spinner phase.

    Auto-compact blocks the turn on an LLM summarisation that routinely runs
    10-20s; without a phase change the spinner keeps saying "thinking" and the
    turn looks hung.

    The interrupted phase is restored afterwards rather than assumed to be
    "thinking": subagents share the parent's event callback, so a subagent
    compaction happens while the parent sits in its ``task`` tool phase, which
    would otherwise lose both its label and its clock. For the same reason the
    nesting is counted, not flagged — with several subagents summarising at once
    the first one to finish must not clear a notice the others still need.
    """
    depth = [0]
    interrupted = [("thinking", "", 0.0)]

    def handle(event: dict) -> None:
        et = event.get("type", "")
        if et == "compact.start":
            if depth[0] == 0:
                interrupted[0] = (
                    tui_state.get("_phase", "thinking"),
                    tui_state.get("_spinner_base", ""),
                    tui_state.get("_phase_start") or time.monotonic(),
                )
                set_phase("compacting")
            depth[0] += 1
        elif et == "compact.end":
            depth[0] = max(depth[0] - 1, 0)
            if depth[0] == 0:
                phase, base, started = interrupted[0]
                set_phase(phase, base)
                # Keep the interrupted phase's original clock: a tool that has
                # been running 40s must not come back reading 0s.
                tui_state["_phase_start"] = started

    return handle


def _record_main_auto_compaction(event: dict, tui_state: dict) -> None:
    """Count successful full auto-compactions visible in the active CLI session."""
    if event.get("type") != "compact.auto" or event.get("is_main_agent") is not True:
        return
    tui_state["compaction_count"] += 1
    event["compaction_count"] = tui_state["compaction_count"]


def _record_main_context_usage(event: dict, tui_state: dict) -> None:
    """Apply one main-agent request's actual context shape to the status bar."""
    if event.get("type") != "context.usage" or event.get("is_main_agent") is not True:
        return
    tui_state["context_tokens"] = event["context_tokens"]
    # Cache observability riding the same event: the previous request's hit
    # ratio for the bar, and the prefix-break index for debug diagnosis.
    tui_state["cache_hit_ratio"] = event.get("cache_hit_ratio")
    break_index = event.get("prefix_break_index")
    if break_index is not None:
        logger.debug(
            "cache prefix broke at message %d this request (cold tail from there on)",
            break_index,
        )
    if event["context_window"] > 0:
        tui_state["context_window"] = event["context_window"]


def _read_git_branch(work_dir: str) -> str:
    """Return the current branch for ``work_dir``, or empty outside Git."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _status_thinking_mode(agent, agent_config: dict) -> str:
    """Return the concise effective thinking label shown in the status bar."""
    effort = agent_config.get("reasoning") or agent_config.get("reasoning_effort")
    if effort:
        return str(effort)
    if agent is None or agent.model is None:
        return ""
    return agent.model.describe_thinking_mode()


def _seed_context_tokens(agent, tui_state: dict) -> None:
    """Show the context the session already carries before any API call lands.

    The system prompt and tool definitions already occupy the window before the
    first API call, and a resumed session brings its history on top of them.
    Measure the next main request locally; Runner events replace this estimate
    with the exact in-flight message shape while a turn is running.

    Shares ``measure_context`` with ``/usage`` so the headline number and the
    breakdown behind it can never disagree.

    This is context state, not an accounting record. Provider usage remains the
    source for footer token consumption and cost.
    """
    if agent is None or agent.model is None:
        return
    try:
        tui_state["context_tokens"] = run_sync(measure_context(agent)).total
    except Exception as e:
        # A status-bar estimate is never worth aborting startup or a command
        # over; leave whatever the bar was showing. Warn rather than whisper:
        # this path failing means prompt assembly is broken, which the next real
        # turn will hit too.
        logger.warning(f"Could not seed status-bar context tokens: {e}")


def _process_stream_response(
    current_agent,
    final_input: str,
    tui_state: dict,
    *,
    images: Optional[list] = None,
    work_dir: Optional[str] = None,
) -> None:
    """Process the agent's streaming response and display it."""
    con = get_console()

    def _set_phase(phase: str, base: str = ""):
        """Set the spinner phase and reset the per-phase elapsed timer.

        phases:
          thinking   — waiting for the first token / between tool calls
          reasoning  — streaming reasoning content
          tool       — running a tool (pass its label as ``base``)
          answering  — streaming the final response
          compacting — summarising the context to free room
          idle       — clear the spinner (run ended / cancelled / errored)

        The spinner thread renders a continuously spinning braille glyph +
        the phase label + elapsed seconds, so the user can always tell a live
        process (spinner turning) from a hung one (spinner frozen).
        """
        if phase == "idle":
            tui_state["spinner_text"] = ""
            tui_state["_spinner_base"] = ""
            tui_state["_phase"] = "idle"
            tui_state["_thinking"] = False
            return
        tui_state["_phase"] = phase
        tui_state["_phase_start"] = time.monotonic()
        tui_state["_spinner_base"] = base
        tui_state["_thinking"] = (phase == "thinking")

    _set_phase("thinking")
    tui_state["spinner_text"] = "⠋ thinking…"
    request_start = perf_counter()
    # Snapshot pre-turn baselines so live status refreshes (spinner loop +
    # turn-end reconcile) can express the running total as
    # ``baseline + current-run-delta`` — idempotent, no double-counting.
    # Stashed into tui_state because the spinner thread has no other way to
    # reach these locals.
    _cost_baseline = tui_state.get("cost_usd", 0.0)
    _active_baseline = tui_state.get("active_seconds", 0.0)
    _calls_baseline = tui_state.get("total_api_calls", 0)
    _goal_tokens_baseline = tui_state.get("goal_tokens_used", 0)
    _usage_entry_baseline = len(current_agent.model.usage.request_usage_entries)
    tui_state["_turn_request_start"] = request_start
    tui_state["_turn_cost_baseline"] = _cost_baseline
    tui_state["_turn_active_baseline"] = _active_baseline
    tui_state["_turn_calls_baseline"] = _calls_baseline
    tui_state["_turn_goal_tokens_baseline"] = _goal_tokens_baseline
    tui_state["_turn_usage_entry_baseline"] = _usage_entry_baseline

    turn_checkpointer = None
    try:
        from agentica.run_config import RunConfig
        from agentica.run_context import RunSource

        run_config = RunConfig(stream_intermediate_steps=True, source=RunSource.cli)

        # ── Per-turn rewind checkpoint ─────────────────────────────
        # Open a turn checkpoint BEFORE the run: msg_index = the conversation
        # length right now, so a later /rewind truncates back to here. The
        # file-tool snapshot happens below on each TOOL_STARTED event, before
        # the tool actually writes (Phase 1 yields started events, Phase 2
        # executes — see Model._run_function_calls_impl).
        turn_checkpointer = get_turn_checkpointer(
            tui_state, current_agent.session_id or "default"
        )
        turn_no = tui_state.get("turn_no", 0) + 1
        wm = current_agent.working_memory
        msg_index = len(wm.messages) if wm is not None else 0
        turn_checkpointer.begin_turn(turn_no, prompt=final_input, msg_index=msg_index)

        # Permission enforcement lives on the Agent itself now (tool_config.
        # permission_mode + sandbox_config, see agentica.agent.permissions) —
        # set once at build time by create_agent() and switchable at runtime
        # via current_agent.set_permission_mode() (/permissions command).
        # No per-run RunConfig override needed here.

        image_paths_for_model: list[str] | None = None

        # Preserve local paths for later file operations. Vision-capable base
        # models receive the original image directly; text-only models receive
        # external OCR observations instead.
        if images:
            image_paths = [str(Path(p).resolve()) for p in images]
            extra_parts = [
                "[Attached image files]\n"
                + "\n".join(f"- {image_path}" for image_path in image_paths)
            ]

            if current_agent.model.supports_images:
                image_paths_for_model = image_paths
            else:
                _set_phase("tool", "extracting image text")
                ocr_text = _ocr_images_parallel(image_paths)
                if not ocr_text.strip():
                    raise ValueError(
                        f"Model '{current_agent.model.id}' does not support image input "
                        "and no text could be extracted by the external OCR tool. "
                        "Use a vision-capable model or configure an image analysis tool."
                    )
                extra_parts.append(f"[External OCR observation]\n{ocr_text}")

            final_input += "\n\n" + "\n\n".join(extra_parts)
            _set_phase("thinking")

        # Subagent verbosity follows the global ``--debug`` flag (carried
        # via ``tui_state`` since this helper has no direct access to the
        # CLI args): developers debugging a flow want completion + elapsed
        # for every child tool; end users get the tool-first single-line
        # view by default.
        subagent_verbosity = "verbose" if tui_state.get("debug") else "all"
        display_work_dir = Path(work_dir) if work_dir is not None else None
        display = StreamDisplayManager(
            con,
            subagent_verbosity=subagent_verbosity,
            work_dir=display_work_dir,
        )
        # Register live-event callback so the subagent's tool calls and
        # compression events render in real time (instead of being a black
        # box until the parent tool result arrives).
        compact_phase = _make_compact_phase_handler(_set_phase, tui_state)

        def _on_agent_event(event: dict) -> None:
            compact_phase(event)
            _record_main_auto_compaction(event, tui_state)
            _record_main_context_usage(event, tui_state)
            display.handle_event(event)

        current_agent._event_callback = _on_agent_event

        response_stream = current_agent.run_stream_sync(
            final_input,
            config=run_config,
            images=image_paths_for_model,
        )

        shown_tool_count = 0
        # Sequencer aligns parallel tool results with their call lines:
        # backend runs tools concurrently, frontend prints each result exactly
        # once and directly beneath its own tool-call line.
        _tool_seq = _ToolResultSequencer()

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

                        # Snapshot pre-edit content of any file this tool will
                        # write, before Phase 2 execution mutates it. first-touch
                        # dedup in TurnCheckpointer means repeated edits to the
                        # same file in one turn still yield one turn-start capture.
                        for path in extract_rewrite_paths(tool_name, tool_args, work_dir or os.getcwd()):
                            try:
                                turn_checkpointer.snapshot(path)
                            except OSError:
                                pass

                        _tool_seq.on_start(
                            tool_info.get("tool_call_id"), tool_name
                        )
                        display.display_tool(
                            tool_name, tool_args,
                            tool_call_id=tool_info.get("tool_call_id"),
                        )
                        _set_phase("tool", f"🔧 {tool_name}")
                    shown_tool_count = len(chunk.tools)
                continue

            if display_event.kind == RunDisplayEventKind.TOOL_COMPLETED:
                _set_phase("thinking")
                if chunk.tools:
                    for tool_info in chunk.tools:
                        _tool_seq.on_complete(
                            tool_info.get("tool_call_id"), tool_info
                        )
                    # Flush completed results in call order. The front slot
                    # only prints once it is done, so a slow tool never blocks
                    # later tools from being shown — they just queue until the
                    # earlier slot is ready, preserving call→result alignment.
                    for info in _tool_seq.drain():
                        tool_name = info.get("tool_name") or info.get("name", "unknown")
                        result_content = info.get("content", "")
                        is_error = info.get("tool_call_error", False)
                        elapsed = (info.get("metrics") or {}).get("time")
                        tool_args = info.get("tool_args") or info.get("arguments") or {}
                        display_kwargs = {
                            "is_error": is_error,
                            "tool_args": tool_args,
                            "tool_call_id": info.get("tool_call_id"),
                            "tool_display_meta": info.get("tool_display_meta"),
                        }
                        if elapsed is not None:
                            display_kwargs["elapsed"] = float(elapsed)
                        display.display_tool_result(
                            tool_name,
                            str(result_content) if result_content else "",
                            **display_kwargs,
                        )
                continue

            has_content = chunk.content and isinstance(chunk.content, str)
            has_reasoning = chunk.reasoning_content

            if not has_content and not has_reasoning:
                continue

            if has_reasoning and not has_content:
                if tui_state.get("show_reasoning", True):
                    _set_phase("reasoning")
                    display.start_thinking()
                    display.stream_thinking(chunk.reasoning_content)
                continue

            if has_content:
                _set_phase("answering")
                display.stream_response(chunk.content)

        # Compute per-turn cost/token deltas BEFORE closing the turn, so we
        # can hand them to ``display.finalize`` for the closing separator.
        #
        # Note: ``agent.run_response.cost_tracker`` is scoped to a single
        # ``agent.run()`` invocation, so its ``total_*`` fields ARE the
        # per-turn deltas we want — no snapshot/subtract dance needed.
        cost_tracker = current_agent.run_response.cost_tracker
        delta_tokens: int | None = None
        delta_cost_usd: float | None = None
        usage_summary: ProviderUsageSummary | None = None
        if cost_tracker and cost_tracker.turns > 0:
            usage_entries = current_agent.model.usage.request_usage_entries[_usage_entry_baseline:]
            usage_summary = ProviderUsageSummary.from_request_entries(
                usage_entries,
                cost_usd=cost_tracker.total_cost_usd,
            )
            delta_tokens = usage_summary.net_new_tokens
            delta_cost_usd = usage_summary.cost_usd

        # 1-based session-scoped turn counter. Increment BEFORE finalize so
        # the separator shows the turn that just completed.
        tui_state["turn_no"] = tui_state.get("turn_no", 0) + 1
        turn_no = tui_state["turn_no"]
        # Expose the per-turn tool-call count so `/usage` can reconcile with the
        # footer's "N tools" (which counts individual tool calls, incl. parallel).
        tui_state["last_turn_tool_count"] = display.tool_count

        display.finalize(
            turn_no=turn_no,
            delta_tokens=delta_tokens,
            delta_cost_usd=delta_cost_usd,
            usage_summary=usage_summary,
        )
        _set_phase("idle")

        # Surface loop-break reasons (death spiral / max turns / cost budget).
        # These no longer ride inside the response content, so render a notice
        # here to keep the CLI user informed about a truncated run.
        if current_agent.run_response.break_reason:
            con.print(
                f"\n[yellow]⚠ Run stopped early "
                f"({current_agent.run_response.break_reason}): "
                f"{current_agent.run_response.break_message}[/yellow]"
            )

        # Final reconcile of the live status fields. Same idempotent helper
        # the spinner loop calls mid-turn — overwrites with exact end-of-turn
        # totals. Replaces the old ``+=`` accumulation which would double-count
        # (the spinner loop had already been writing ``baseline + delta``
        # continuously throughout the turn).
        _refresh_live_status(
            tui_state, current_agent, request_start,
            _cost_baseline, _active_baseline, _calls_baseline,
            _goal_tokens_baseline,
        )
        # Clear the stashed turn baselines so a stale value can't bleed into
        # the next turn (or into the spinner loop after the agent stops).
        tui_state.pop("_turn_request_start", None)
        tui_state.pop("_turn_cost_baseline", None)
        tui_state.pop("_turn_active_baseline", None)
        tui_state.pop("_turn_calls_baseline", None)
        tui_state.pop("_turn_goal_tokens_baseline", None)

        if not display.has_content_output and display.tool_count == 0 and not display.thinking_shown:
            _set_phase("idle")
            con.print("[info]Agent returned no content.[/info]")

    except KeyboardInterrupt:
        current_agent.cancel()
        _set_phase("idle")
        deadline = time.monotonic() + 3.0
        while current_agent._running and time.monotonic() < deadline:
            time.sleep(0.05)
        current_agent._running = False
        current_agent._cancelled = False
        con.print("\n[yellow]⚡ Agent cancelled.[/yellow] [dim][User interrupted the response][/dim]")
        con.print(
            format_session_summary(
                elapsed_seconds=time.monotonic() - tui_state["session_started_at"],
                usage=current_agent.model.usage,
                session_id=resumable_session_id(current_agent),
                brief=True,
            )
        )
    except AgentCancelledError:
        _set_phase("idle")
        current_agent._running = False
        current_agent._cancelled = False
        con.print("\n[yellow]⚡ Agent cancelled.[/yellow] [dim][User interrupted the response][/dim]")
        con.print(
            format_session_summary(
                elapsed_seconds=time.monotonic() - tui_state["session_started_at"],
                usage=current_agent.model.usage,
                session_id=resumable_session_id(current_agent),
                brief=True,
            )
        )
    except Exception as e:
        _set_phase("idle")
        display_agent_execution_error(con, e)
    finally:
        # Finalize the per-turn rewind checkpoint so /rewind can roll this turn
        # back. Runs on success, cancel, and error alike — a turn that touched
        # files still deserves an undo point even if it failed midway.
        if turn_checkpointer is not None:
            try:
                turn_checkpointer.finalize_turn()
            except Exception as e:
                logger.warning(f"Could not finalize turn checkpoint: {e}")

        # Clear the live-event callback so it doesn't outlive this run.
        current_agent._event_callback = None
        # Strip image payloads from history: the turn already consumed them
        # multimodally; re-encoding and re-sending every local image on each
        # later turn would bloat context and cost. Paths and OCR text remain
        # in the message content, so the model still knows an image exists.
        if images:
            for m in current_agent.working_memory.messages:
                if m.role == "user" and m.images:
                    m.images = None
        # The completed/cancelled answer is now persisted. Re-measure what the
        # next request will carry so the idle bar includes that newest turn.
        _seed_context_tokens(current_agent, tui_state)
        tui_state["git_branch"] = _read_git_branch(tui_state["work_dir"])


__all__ = ['_BRAILLE_SPINNER', '_WAITING_FOR_INPUT_TEXT', '_render_spinner_text', '_refresh_live_status', '_make_compact_phase_handler', '_record_main_auto_compaction', '_record_main_context_usage', '_read_git_branch', '_status_thinking_mode', '_seed_context_tokens', '_process_stream_response']
