# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: run_interactive wiring: process loop, spinner, and app lifecycle
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.patch_stdout import patch_stdout

from agentica.cli.commands.context import (
    CommandContext,
    IMAGE_EXTENSIONS,
    PendingQueue,
)
from agentica.cli.commands.goal import _sync_goal_budget_tui
from agentica.cli.commands.registry import COMMAND_HANDLERS, echo_command_invocation
from agentica.cli.commands.session import (
    display_resumed_transcript,
    hydrate_resumed_session,
)
from agentica.cli.session_resume import prepare_startup_resume
from agentica.cli.display import (
    display_peer_messages,
    display_user_message,
    inject_file_contents,
    parse_file_mentions,
    print_header,
)
from agentica.cli.runtime import (
    configure_tools,
    create_agent,
    get_console,
    set_active_console,
)
from agentica import config
from agentica.cli.setup import session_profile
from agentica.global_config import get_setting
from agentica.peers import PeerSession, format_for_model
from agentica.run_response import AgentCancelledError
from agentica.skills import get_skill_registry, load_skills
from agentica.subagent_loader import load_all_agents
from agentica.tools.ask_user_question_tool import (
    set_default_ask_user_question_callback,
)
from agentica.tools.background_processes import BackgroundProcessCompleted
from agentica.utils.log import logger, suppress_console_logging
from agentica.workspace import Workspace

from .attachments import (
    _deduplicate_image_attachments,
    _detect_file_drop,
    unpack_queue_payload,
)
from .btw import (
    _background_result_for_agent,
    _print_background_completion,
    _run_btw_concurrent,
    hand_to_agent,
    promote_late_steer,
)
from .console_io import (
    ChatConsole,
    _ask_active,
    _ask_state_lock,
    _clear_output_pause,
    _cprint,
    _install_sigquit_escape,
    _open_in_pager,
    _print_interactive_exit_summary,
    _restore_sigquit_escape,
    _tty_write_lock,
)
from .goal_hook import _maybe_continue_goal
from .session_state import (
    SessionState,
    _InputRequest,
)
from .stream_loop import (
    _BRAILLE_SPINNER,
    _WAITING_FOR_INPUT_TEXT,
    _process_stream_response,
    _read_git_branch,
    _refresh_live_status,
    _render_spinner_text,
    _seed_context_tokens,
    _status_thinking_mode,
)
from .tui import _setup_tui

# ==================== Main entry ====================


def _cli_log_file() -> Tuple[Optional[str], Optional[str]]:
    """This CLI process's log file and its level, read at call time.

    ``agentica.cli.main`` assigns the daily ``logs/<date>-<pid>.log`` path onto
    ``agentica.config`` *after* this module is imported, so binding the constant
    with a module-level ``from agentica.config import AGENTICA_LOG_FILE`` freezes
    the SDK default (empty) and every peer would publish "no log file".
    """
    path = config.AGENTICA_LOG_FILE or None
    return path, config.AGENTICA_LOG_LEVEL if path else None


def _maybe_start_cron(state: SessionState, agent_config, extra_tools,
                      workspace, skills_registry) -> None:
    """Start the cron scheduler daemon thread if settings.cron.enabled is true.

    Idempotent: does nothing if a thread is already running. Failures are logged
    but never block CLI startup.
    """
    if state.cron_thread is not None and state.cron_thread.is_alive():
        return
    try:
        from agentica.global_config import get_setting
        if not bool(get_setting("cron.enabled", False)):
            return
        interval = int(get_setting("cron.interval", 60) or 60)
        from agentica.cron.cli_runner import (
            CliAgentRunner, build_cli_agent_factory, start_cron_thread,
        )
        factory = build_cli_agent_factory(
            agent_config, extra_tools, workspace, skills_registry)
        runner = CliAgentRunner(factory)
        thread, stop_event = start_cron_thread(runner, interval=interval)
        state.cron_thread = thread
        state.cron_stop_event = stop_event
        try:
            from agentica.cli.runtime import get_console
            get_console().print(
                f"[dim]cron scheduler started (interval={interval}s). "
                f"Use /cron to manage jobs.[/dim]")
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        from agentica.utils.log import logger
        logger.warning(f"failed to start cron scheduler: {e}")


def _stop_cron(state: SessionState) -> None:
    """Signal the cron daemon thread to stop. Safe to call when not running."""
    if state.cron_stop_event is not None:
        state.cron_stop_event.set()


def run_interactive(
    agent_config: dict,
    extra_tool_names: Optional[List[str]] = None,
    workspace: Optional[Workspace] = None,
    skills_registry=None,
):
    """Run the interactive CLI with fixed-bottom input area TUI."""

    if not agent_config.get("debug"):
        suppress_console_logging()

    perm_mode = agent_config.get("permissions", "allow-all")

    extra_tools = configure_tools(extra_tool_names) if extra_tool_names else None

    # Holder lets the ask_user_question_callback (built now, needed by create_agent)
    # reach `state` / `app`, which are created further below. The agent calls
    # the callback on the background process_loop thread; it parks on a queue
    # while the main prompt_toolkit thread feeds the typed line back via
    # _handle_enter. This replaces the tool's default bare input() which
    # deadlocks against prompt_toolkit's stdin ownership.
    _ui_holder: dict = {}

    def _cli_ask_user_question_callback(prompt: str, options: Optional[List[str]] = None) -> str:
        state_ref = _ui_holder.get("state")
        app_ref = _ui_holder.get("app")
        # Fallback to bare input if the TUI isn't up yet (shouldn't happen in
        # normal flow, but keeps the callback safe).
        if state_ref is None or app_ref is None:
            return input(f"{prompt}\nYour response: ").strip()

        req = _InputRequest(prompt=prompt, options=options)
        logger.info(
            f"[ask] armed: prompt={str(prompt)[:80]!r} options={bool(options)}"
        )

        # Arm the request and repaint. The prompt text itself is rendered by
        # the layout's input_prompt_widget on the main thread (see
        # _get_input_prompt_fragments). We deliberately do NOT print it from
        # this background agent thread: in a non-full-screen app that would go
        # through run_in_terminal (CPR + full redraw) and race the spinner's
        # invalidate(), desyncing the input cursor so the box stops echoing
        # keystrokes while the agent waits for an answer.
        #
        # Whatever the user had half-typed in the input buffer stays exactly
        # where it was — the user can Ctrl+U / backspace it out if they want a
        # clean answer field. Deciding that for them would silently change the
        # meaning of their keystrokes.
        with _ask_state_lock:
            state_ref.input_request = req
            _ask_active[0] = True
            # A slash command (e.g. /resume, /cron) asks between turns, when
            # agent_running is False. Only a question armed *during* a run may
            # be aborted by that run ending, or those prompts would cancel
            # themselves on the first watchdog poll.
            armed_during_run = state_ref.agent_running
        app_ref.invalidate()

        # Block the agent thread until the user submits a line, or Ctrl+C
        # puts the CANCELLED sentinel on the queue to release us. We poll with
        # a short timeout (watchdog) instead of a bare get(): a bare get()
        # parks this worker thread forever, so any desync — input_request
        # pointing at a different req (e.g. a nested vision trial run that
        # re-asked), or the turn returning without resolving us — would hang
        # the turn permanently with no recovery. On each empty poll we
        # re-validate: re-arm if overwritten, abort if the run ended.
        answer = None
        try:
            while True:
                try:
                    answer = req.result.get(timeout=1.0)
                    break
                except queue.Empty:
                    if state_ref.should_exit or (armed_during_run and not state_ref.agent_running):
                        logger.info("[ask] watchdog: run ended, aborting prompt")
                        answer = _InputRequest.CANCELLED
                        break
                    if state_ref.input_request is not req:
                        logger.info("[ask] watchdog: re-arming after overwrite")
                        state_ref.input_request = req
                        app_ref.invalidate()
                    continue
        finally:
            with _ask_state_lock:
                _ask_active[0] = False
                if state_ref.input_request is req:
                    state_ref.input_request = None

        if answer is _InputRequest.CANCELLED:
            # Propagate as AgentCancelledError so the agent runtime unwinds
            # cleanly. Any layer between us and the agent that catches Exception
            # will still respect this because AgentCancelledError subclasses
            # Exception but is explicitly re-raised by the tool infra.
            logger.info("[ask] resolved: CANCELLED")
            raise AgentCancelledError("ask_user_question aborted by user (Ctrl+C)")
        answer_text = str(answer)
        logger.info(f"[ask] resolved: answer={answer_text[:80]!r}")

        # If the user typed an option number, map it back to the option text.
        if options and answer_text:
            try:
                idx = int(answer_text)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            except ValueError:
                pass
        return answer_text

    # The process registry belongs to the CLI session and must exist before
    # the first agent is built because ExecuteTool receives this shared
    # instance during create_agent().
    state = SessionState()
    # Publish this terminal in the peer directory before the first agent exists:
    # the messaging tools are built from this object, and the identity must
    # survive agent rebuilds (/resume, /model) so in-flight messages still land.
    # `agentica resume <id>` may name a session recorded in another directory.
    # Settle that before anything derives state from the cwd — the peer identity,
    # the agent's work_dir and the session storage all depend on the answer.
    if agent_config.get("_resume_requested") and not prepare_startup_resume(
        agent_config,
        user_id=agent_config.get("user_id") or (workspace.user_id if workspace else None),
        printer=get_console().print,
    ):
        return

    peer_cwd = agent_config.get("work_dir") or os.getcwd()
    peer_user_id = agent_config.get("user_id") or (workspace.user_id if workspace else None)
    peer_workspace_path = str(workspace.path) if workspace is not None else None
    peer_memory_path = None
    if workspace is not None:
        peer_memory_path = str(workspace._get_user_memory_md())
    peer_log_file, peer_log_level = _cli_log_file()
    peer_profile, _ = session_profile(agent_config, peer_cwd)
    state.peer_session = PeerSession(
        cwd=peer_cwd,
        git_branch=_read_git_branch(peer_cwd),
        session_id=agent_config.get("session_id"),
        user_id=peer_user_id,
        workspace_path=peer_workspace_path,
        memory_path=peer_memory_path,
        log_file=peer_log_file,
        log_level=peer_log_level,
        profile_name=peer_profile or None,
        model_provider=agent_config.get("model_provider"),
        model_name=agent_config.get("model_name"),
    )
    # Show every accepted peer message in this terminal, whether the idle loop
    # or the running agent drained the mailbox. Set after construction so the
    # callback can close over ``state`` / ``app`` once those exist below.
    state.peer_session.publish()
    current_agent = create_agent(
        agent_config, extra_tools, workspace, skills_registry,
        ask_user_question_callback=_cli_ask_user_question_callback,
        background_process_registry=state.background_processes,
        permission_mode=perm_mode,
        peer_session=state.peer_session,
    )
    # create_agent assigns a session_id when the config did not; publish it so
    # other terminals can address / resume this conversation by that id.
    state.peer_session.publish(session_id=current_agent.session_id)
    state.current_agent = current_agent

    # User/project files fail softly per definition; invalid packaged defaults
    # surface because they indicate a broken installation.
    load_all_agents()

    con = get_console()

    # Print header BEFORE entering TUI
    print_header(
        agent_config["model_provider"],
        agent_config["model_name"],
        work_dir=agent_config.get("work_dir"),
        extra_tools=extra_tool_names,
    )

    if workspace and workspace.exists():
        con.print(f"  Workspace: [green]{workspace.path}[/green]")

    if agent_config.get("_resume_requested"):
        if current_agent._session_log is None or not current_agent._session_log.exists():
            con.print(f"[bold red]No session found: {current_agent.session_id}[/bold red]")
            return
        _, runs_built = hydrate_resumed_session(
            current_agent,
            agent_config.get("_resume_at_uuid"),
        )
        display_stats = display_resumed_transcript(
            current_agent.working_memory.runs,
            current_agent.session_id or "",
        )
        con.print(
            f"[green]Resumed session: {current_agent.session_id}"
            f" — restored {runs_built} runs into context; showing conversation only "
            f"({display_stats.tool_result_count} tool results collapsed)[/green]"
        )

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
    if perm_mode != "allow-all":
        con.print(f"  Permissions: [yellow]{perm_mode}[/yellow]")
    con.print()

    status_work_dir = agent_config.get("work_dir") or os.getcwd()
    tui_state = {
        "model_name": agent_config.get("model_name", ""),
        "model_provider": agent_config.get("model_provider", ""),
        "profile_name": session_profile(agent_config, status_work_dir)[0],
        "thinking_mode": _status_thinking_mode(current_agent, agent_config),
        "work_dir": status_work_dir,
        "git_branch": _read_git_branch(status_work_dir),
        "context_tokens": 0,
        "context_window": current_agent.model.context_window if current_agent.model else 128000,
        "cost_usd": 0.0,
        "active_seconds": 0.0,
        "last_turn_seconds": 0.0,
        "spinner_text": "",
        "show_reasoning": True,
        "statusbar_visible": True,
        "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_started_at": time.monotonic(),
        "total_api_calls": 0,
        "compaction_count": 0,
        "debug": bool(agent_config.get("debug")),
        "goal_token_budget": None,
        "goal_tokens_used": 0,
    }
    _seed_context_tokens(current_agent, tui_state)

    # Cron control surface for the /cron daemon on|off command. We expose
    # start/stop closures via tui_state so the slash command can toggle the
    # live scheduler thread without reaching into module internals.
    def _start_cron():
        _maybe_start_cron(state, agent_config, extra_tools, workspace, skills_registry)
        return state.cron_thread is not None and state.cron_thread.is_alive()

    def _stop_cron_cb():
        _stop_cron(state)
        state.cron_thread = None
        state.cron_stop_event = None

    tui_state["cron_start"] = _start_cron
    tui_state["cron_stop"] = _stop_cron_cb
    tui_state["cron_is_running"] = lambda: (
        state.cron_thread is not None and state.cron_thread.is_alive())

    # Start the cron scheduler in a daemon thread when enabled in settings.
    # Default OFF: scheduled agent runs cost tokens, so the user must opt in
    # (via `agentica setup`, config.yaml settings.cron.enabled, or `/cron daemon on`).
    _maybe_start_cron(state, agent_config, extra_tools, workspace, skills_registry)

    pending_queue = PendingQueue()
    # Keep a mutable list wrapper for image_counter (needed by _try_attach_clipboard_image)
    _image_counter_ref = [0]

    def _open_history_pager(title: str, content: str) -> None:
        """Schedule a blocking pager on prompt_toolkit's terminal owner."""

        def _paged():
            with _tty_write_lock:
                _open_in_pager(title, content)

        def _schedule():
            run_in_terminal(_paged)

        if app.loop is not None:
            app.loop.call_soon_threadsafe(_schedule)

    def _build_ctx() -> CommandContext:
        """Build a CommandContext from current session state."""
        return CommandContext(
            agent_config=agent_config,
            current_agent=state.current_agent,
            extra_tools=extra_tools,
            extra_tool_names=extra_tool_names,
            workspace=workspace,
            skills_registry=skills_registry,
            tui_state=tui_state,
            pending_queue=pending_queue,
            agent_running=state.agent_running,
            attached_images=state.attached_images,
            image_counter=_image_counter_ref,
            bg_tasks=state.bg_tasks,
            bg_task_counter=state.bg_task_counter,
            background_processes=state.background_processes,
            peer_session=state.peer_session,
            goal_manager=state.goal_manager,
            goal_lock=state.goal_lock,
            ask_user_question_callback=_cli_ask_user_question_callback,
            open_pager_callback=_open_history_pager,
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
            # Single source of command-header echo. Individual handlers no
            # longer print their own titles — see commands.echo_command_invocation.
            echo_command_invocation(cmd, cmd_args)
            ctx = _build_ctx()
            result = handler(ctx, cmd_args)
            if isinstance(result, dict):
                _apply_command_result(result)

    def _apply_command_result(result: dict):
        """Apply side effects from command handler results."""
        nonlocal skills_registry, extra_tool_names
        if "work_dir" in result:
            # /resume moved us into the directory the session was started in.
            # The handler already chdir'd; everything downstream reads work_dir
            # from agent_config or the status bar, so both must follow.
            agent_config["work_dir"] = result["work_dir"]
            tui_state["work_dir"] = result["work_dir"]
            tui_state["git_branch"] = _read_git_branch(result["work_dir"])
        if "current_agent" in result:
            state.current_agent = result["current_agent"]
            # Counts only compactions observed for the currently selected CLI
            # agent. A new, cleared, or resumed agent starts a fresh count.
            tui_state["compaction_count"] = 0
            tui_state["model_name"] = agent_config.get("model_name", "")
            tui_state["model_provider"] = agent_config.get("model_provider", "")
            tui_state["profile_name"] = session_profile(
                agent_config, agent_config.get("work_dir") or os.getcwd()
            )[0]
            tui_state["thinking_mode"] = _status_thinking_mode(
                state.current_agent, agent_config
            )
            tui_state["context_window"] = (
                state.current_agent.model.context_window if state.current_agent.model else 128000
            )
            tui_state["cost_usd"] = 0.0
            # A fresh agent (/clear, /model) carries a fresh system prompt and
            # tool set — re-measure rather than dropping the bar to zero.
            _seed_context_tokens(state.current_agent, tui_state)
        if state.peer_session is not None and (
            "current_agent" in result or "work_dir" in result
        ):
            # Keep the live peer record in sync with /resume (new session_id
            # and possibly a new cwd/project dir) so other terminals can still
            # address this process and dig into the right transcript.
            peer_updates = {
                "session_id": state.current_agent.session_id if state.current_agent else None,
                "git_branch": tui_state.get("git_branch"),
            }
            if "work_dir" in result:
                peer_updates["cwd"] = result["work_dir"]
            state.peer_session.publish(**peer_updates)
        if "session_started_at" in result:
            tui_state["session_started_at"] = result["session_started_at"]
            tui_state["active_seconds"] = 0.0
            tui_state["total_api_calls"] = 0
        if result.get("model_switched"):
            # `/model profile <name>` (or `/model provider/name`) changed the
            # active profile and model — sync every status-bar field that
            # derives from them, not just model_name. Without this the bar's
            # ``profile:`` prefix and ``provider/model`` label kept showing the
            # pre-switch values for the rest of the session.
            tui_state["model_name"] = agent_config.get("model_name", "")
            tui_state["model_provider"] = agent_config.get("model_provider", "")
            tui_state["profile_name"] = session_profile(
                agent_config, agent_config.get("work_dir") or os.getcwd()
            )[0]
            tui_state["thinking_mode"] = _status_thinking_mode(
                state.current_agent, agent_config
            )
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
            # Prefer GoalState.tokens_used when a goal is already loaded
            # (e.g. /goal set just wrote a fresh zeroed state, or /resume
            # restored a paused goal with prior spend).
            gs = state.goal_manager.load() if state.goal_manager is not None else None
            state.goal_tokens_baseline = gs.tokens_used if gs is not None and gs.status != "cleared" else 0
            _sync_goal_budget_tui(tui_state, state.goal_manager)

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

    # Wire the ask_user_question callback holder now that state/app/console all exist,
    # so the ask_user_question tool reads via the TUI instead of a blocking input().
    _ui_holder["state"] = state
    _ui_holder["app"] = app

    # Also register the callback as the process-wide default so ANY
    # AskUserQuestionTool without an explicit callback (a subagent spawned
    # mid-turn, a cron job, a regression) routes through the TUI instead of
    # deadlocking on bare input() while pt owns stdin.
    set_default_ask_user_question_callback(_cli_ask_user_question_callback)

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

            # If agent is currently running, re-queue
            if state.agent_running:
                pending_queue.put(payload)
                time.sleep(0.1)
                continue

            # Unpack payload
            queued = unpack_queue_payload(payload)
            user_input = queued.text
            submit_images = list(queued.images)
            is_btw = queued.is_btw
            # Text nobody typed here: already printed on arrival, and none of
            # the input line's affordances apply to it.
            already_shown = queued.is_relayed
            skill_to_invoke = None

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
                    user_input = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"
                else:
                    user_input = f"@{dropped['path']} {dropped['remainder']}".strip()

            # Slash commands
            first_word = user_input.split()[0].lower() if user_input else ""
            skill_cmds = skills_registry.auto_commands() if skills_registry else {}
            # A slash command is a shortcut for the human at this keyboard, so
            # only typed input dispatches one. Relayed text (a peer message, a
            # finished job's report) stays plain text — the single place that
            # enforces what PEER_MESSAGING_POLICY promises the sender. Until now
            # it held only by accident: `format_for_model` happens to prefix an
            # authority header, so the first word was never `/compact`.
            is_command = not already_shown and (
                first_word in COMMAND_HANDLERS or first_word in skill_cmds
            )
            if is_command:
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                handler = COMMAND_HANDLERS.get(cmd)
                if handler:
                    echo_command_invocation(cmd, cmd_args)
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
                    # Skill auto-command dispatch. The rendered body is built
                    # later, against the LLM payload only — ``user_input`` stays
                    # as typed so the transcript shows `/skill-name args` instead
                    # of the whole SKILL.md.
                    matched_skill = skill_cmds.get(cmd)
                    if matched_skill:
                        _cprint(f"  Skill activated: {matched_skill.name}")
                        skill_to_invoke = matched_skill

            # Expand paste references
            _paste_ref_re = re.compile(r"\[Pasted text #\d+: \d+ lines -> (.+?)\]")
            paste_refs = list(_paste_ref_re.finditer(user_input))
            # A pasted block is recorded BOTH as a ``[Pasted text #N: ...]``
            # placeholder in the buffer (matched by ``paste_refs``) and as an
            # entry in ``state.pasted_files`` by the bracketed-paste handler.
            # The two are 1:1, so counting both double-counts — a single pasted
            # block would wrongly render as "2 pasted blocks". Derive both the
            # block count and the line total from ``state.pasted_files`` only
            # (it already carries accurate per-block line counts); ``paste_refs``
            # is used solely to expand the placeholders into the real content.
            n_pasted_blocks = len(state.pasted_files)
            n_pasted_lines = sum(n for _, n in state.pasted_files) if state.pasted_files else 0
            if paste_refs:

                def _expand_ref(m):
                    p = Path(m.group(1))
                    if p.exists():
                        return p.read_text(encoding="utf-8")
                    return m.group(0)

                expanded = _paste_ref_re.sub(_expand_ref, user_input)
                user_input = expanded
            state.pasted_files.clear()

            # Split display from payload: the transcript keeps the typed line,
            # the model gets the skill body. Expanding here (after paste
            # expansion, before @-mention handling) keeps the payload byte-for-
            # byte what it was when it owned ``user_input``.
            llm_input = user_input
            if skill_to_invoke is not None and skills_registry is not None:
                llm_input = skills_registry.expand_invocation(user_input) or user_input

            prompt_text, mentioned_files = parse_file_mentions(llm_input)
            # @-mentioned image files are multimodal attachments, not text to
            # inject — reading a jpg as utf-8 yields garbage / a decode error.
            image_mentions = [f for f in mentioned_files if f.suffix.lower() in IMAGE_EXTENSIONS]
            if image_mentions:
                mentioned_files = [f for f in mentioned_files if f not in image_mentions]
                submit_images.extend(image_mentions)
            final_input = inject_file_contents(prompt_text, mentioned_files)

            submit_images = _deduplicate_image_attachments(submit_images)

            if not is_btw and not already_shown:
                display_user_message(
                    user_input,
                    pasted_blocks=n_pasted_blocks,
                    pasted_lines=n_pasted_lines,
                    images=submit_images,
                )

            turn_images = submit_images if submit_images else None

            # BTW: ephemeral side question (when agent is NOT running, via queue)
            if is_btw:
                _run_btw_concurrent(state.current_agent, final_input, tui_state)
                continue

            # Run agent
            state.agent_running = True
            # Publish what this terminal is working on so another session's
            # agent can pick it as a message target without guessing.
            if state.peer_session is not None:
                state.peer_session.publish(task=" ".join(user_input.split())[:240])
            app.invalidate()
            _process_stream_response(
                state.current_agent,
                final_input,
                tui_state,
                images=turn_images,
                work_dir=agent_config.get("work_dir"),
            )
            state.agent_running = False
            tui_state["spinner_text"] = ""
            # Belt-and-braces: if an ask-user-question request is still armed
            # when the agent turn returns (e.g. an unusual error path in a
            # tool), unblock it so the callback thread can exit and clear the
            # slot before the next turn.
            if state.input_request is not None:
                try:
                    state.input_request.cancel()
                except Exception:
                    pass
                state.input_request = None

            # Steering typed during the run's final inference never reached the
            # model — the window closed before the next drain. Promote it to a
            # queued next turn (never drop it). Runs BEFORE the goal hook so
            # the user's text preempts an automated continuation lap, same as
            # any other queued user input.
            promoted = promote_late_steer(state, pending_queue)
            if promoted:
                count = f"({len(promoted)} messages) " if len(promoted) > 1 else ""
                con.print(
                    f"  [dim]↪ Current task finished before using the guidance "
                    f"{count}· queued next.[/dim]"
                )

            # Standing-goal hook: decide whether to enqueue a continuation
            # for the next turn. Honors user-priority and cancel semantics.
            _maybe_continue_goal(state, pending_queue, tui_state)

            app.invalidate()

    process_thread = threading.Thread(target=process_loop, daemon=True)
    process_thread.start()

    # ── Spinner refresh thread ──
    # One braille spinner cycles through all phases (thinking / reasoning /
    # tool / answering) so the glyph is always turning while the agent is
    # alive — the user can tell a live process (spinner turning, elapsed
    # climbing) from a hung one (spinner frozen) at a glance.
    _frame_idx = [0]

    def spinner_loop():
        while not state.should_exit:
            if not (state.agent_running and app.is_running):
                time.sleep(0.3)
                continue
            # Agent parked on a ask_user_question/confirm tool: stop churning
            # invalidate() (it fights the input renderer and desyncs the
            # cursor) and replace the stale "🔧 tool (Ns)" phase with a
            # steady "waiting" line so the user knows it's their turn.
            if state.input_request is not None:
                if tui_state.get("spinner_text") != _WAITING_FOR_INPUT_TEXT:
                    tui_state["spinner_text"] = _WAITING_FOR_INPUT_TEXT
                    app.invalidate()
                time.sleep(0.2)
                continue
            phase = tui_state.get("_phase", "thinking")
            base = tui_state.get("_spinner_base", "")
            start = tui_state.get("_phase_start") or time.monotonic()
            elapsed = time.monotonic() - start
            tui_state["spinner_text"] = _render_spinner_text(
                _frame_idx[0], phase, base, elapsed
            )
            # Refresh live status-bar fields (tokens / cost / time) every tick.
            # The cost_tracker is updated by the model layer on every LLM call,
            # so this picks up new token/cost totals within ~120ms of any API
            # call returning — including right after each tool completes and the
            # next "decide what to do" call lands. Time fields tick smoothly.
            # Guarded by ``_turn_request_start`` being set (only valid mid-turn);
            # the stream loop clears it at turn end.
            _req_start = tui_state.get("_turn_request_start")
            if _req_start is not None:
                _refresh_live_status(
                    tui_state, state.current_agent, _req_start,
                    tui_state.get("_turn_cost_baseline", 0.0),
                    tui_state.get("_turn_active_baseline", 0.0),
                    tui_state.get("_turn_calls_baseline", 0),
                    tui_state.get("_turn_goal_tokens_baseline", 0),
                )
            _frame_idx[0] = (_frame_idx[0] + 1) % len(_BRAILLE_SPINNER)
            app.invalidate()
            time.sleep(0.12)

    spinner_thread = threading.Thread(target=spinner_loop, daemon=True)
    spinner_thread.start()

    def _hand_to_agent(text: str) -> None:
        hand_to_agent(state, pending_queue, text)

    # A background command's result is the agent's own pending work, so it is
    # delivered by default; set `deliver_background_results: false` in
    # config.yaml for a session that must never wake up on its own.
    deliver_results = bool(get_setting("deliver_background_results", True))

    def background_completion_loop():
        pending_events: List[BackgroundProcessCompleted] = []
        while not state.should_exit:
            try:
                pending_events.append(state.background_processes.wait_completed(timeout=0.2))
            except queue.Empty:
                pass

            with _ask_state_lock:
                if _ask_active[0] or state.input_request is not None:
                    continue

                printed = False
                while pending_events:
                    event = pending_events.pop(0)
                    if event.stop_requested:
                        continue
                    _print_background_completion(event)
                    if deliver_results:
                        _hand_to_agent(_background_result_for_agent(event))
                    printed = True
            if printed and app.is_running:
                app.invalidate()

    background_completion_thread = threading.Thread(
        target=background_completion_loop,
        daemon=True,
        name="background_completion_notifier",
    )
    background_completion_thread.start()

    def peer_message_loop():
        """Keep this terminal discoverable and take messages while idle.

        While a run is active the Runner drains the same mailbox between tool
        batches, so the message reaches the agent mid-task; this loop must not
        race it for those messages, hence the ``agent_running`` skip.

        Accepted messages are printed by ``peer_session.on_drain`` (shared with
        the mid-run path) so both paths show the same receipt.
        """
        peers = state.peer_session
        if peers is None:
            return
        while not state.should_exit:
            time.sleep(1.0)
            agent = state.current_agent
            try:
                # Everything a peer reads about this session that can change
                # under it is published from the same values the status bar
                # renders, on every tick rather than at each mutation site:
                # `/model <name>`, `/model --clear` and `/config set` all change
                # the model, so a record kept current by pushes goes stale the
                # day a fourth path forgets to. Empty strings publish as
                # "unset" rather than being skipped, so clearing works too;
                # heartbeat itself only writes when one of these differs.
                peers.heartbeat(
                    session_id=agent.session_id if agent is not None else None,
                    profile_name=tui_state["profile_name"],
                    model_provider=tui_state["model_provider"],
                    model_name=tui_state["model_name"],
                    busy=state.agent_running,
                    context_tokens=tui_state["context_tokens"],
                    context_window=tui_state["context_window"],
                )
            except OSError:
                logger.warning("peer heartbeat failed", exc_info=True)
            if state.agent_running:
                continue
            try:
                messages = peers.drain()
            except OSError:
                logger.warning("draining the peer mailbox failed", exc_info=True)
                continue
            if not messages:
                continue
            _hand_to_agent(format_for_model(messages))
            if app.is_running:
                app.invalidate()

    def _on_peer_drain(messages) -> None:
        display_peer_messages(messages)
        if app.is_running:
            app.invalidate()

    if state.peer_session is not None:
        state.peer_session.on_drain = _on_peer_drain

    peer_message_thread = threading.Thread(
        target=peer_message_loop,
        daemon=True,
        name="peer_message_notifier",
    )
    peer_message_thread.start()

    # ── Run the TUI ──
    # Install a SIGQUIT hard-escape. When the main prompt_toolkit event loop is
    # blocked (the ask_user_question freeze bug: background run_in_terminal
    # writes starve the loop so Ctrl+C / Ctrl+D keybindings never fire), the
    # only escape that does NOT depend on the loop being responsive is an
    # OS-level signal. SIGQUIT (Ctrl+\) is delivered to the main thread and its
    # handler runs asynchronously between bytecodes, so it works even when the
    # pt keybindings can't. Keybinding-based escapes (double Ctrl+C → app.exit)
    # can't help here because the keybindings themselves won't fire.
    def _hard_exit(signum, frame):
        try:
            os.write(2, b"\n[agentica] hard exit (signal)\n")
        except Exception:
            pass
        os._exit(1)

    sigquit_installation = _install_sigquit_escape(_hard_exit)

    try:
        with patch_stdout():
            app.run()
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        state.should_exit = True
        _stop_cron(state)
        state.background_processes.stop()
        if state.peer_session is not None:
            state.peer_session.unpublish()
        set_active_console(None)
        set_default_ask_user_question_callback(None)
        _restore_sigquit_escape(sigquit_installation)
        _clear_output_pause()

    _print_interactive_exit_summary(state, tui_state)
    get_console().print("\nThank you for using Agentica CLI. Goodbye!", style="bold green")


__all__ = ['_maybe_start_cron', '_stop_cron', 'run_interactive']
