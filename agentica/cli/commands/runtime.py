# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runtime slash commands: queue, steer, bg, fork, peers, checkpoint
"""

from __future__ import annotations

import os
import shlex
import threading
from datetime import datetime
from pathlib import Path

from agentica.cli.runtime import (
    get_console,
    create_agent,
    _generate_session_id,
)
from agentica.cli.display import (
    show_help,
)
from agentica.goals import GoalManager, is_goal_generated_prompt
from agentica.memory.models import AgentRun
from agentica.peers import PeerMessageRefused
from agentica.model.message import Message
from agentica.run_response import RunResponse

from agentica.cli.commands.context import CommandContext, IMAGE_EXTENSIONS
from agentica.cli.commands.session import display_resumed_transcript, hydrate_resumed_session




# ==================== Command Handlers ====================


def _cmd_help(ctx: CommandContext, cmd_args: str = ""):
    show_help(skills_registry=ctx.skills_registry)



def _cmd_exit(ctx: CommandContext, cmd_args: str = ""):
    return "EXIT"



def _cmd_paste(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.attached_images is None or ctx.image_counter is None:
        con.print("[dim]Image paste not available.[/dim]")
        return
    from agentica.cli.clipboard import has_clipboard_image
    from agentica.cli.interactive.attachments import _try_attach_clipboard_image

    if has_clipboard_image():
        if not _try_attach_clipboard_image(ctx.attached_images, ctx.image_counter):
            con.print("  [dim]Clipboard has an image but extraction failed.[/dim]")
    else:
        con.print("  [dim]No image found in clipboard.[/dim]")



def _cmd_image(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    if ctx.attached_images is None or ctx.image_counter is None:
        con.print("[dim]Image attachment not available.[/dim]")
        return
    raw_args = cmd_args.strip()
    if not raw_args:
        con.print("  [dim]Usage: /image <path>  e.g. /image /path/to/image.png[/dim]")
        return

    from agentica.cli.interactive.attachments import _resolve_attachment_path, _split_path_input

    path_token, _ = _split_path_input(raw_args)
    image_path = _resolve_attachment_path(path_token)
    if image_path is None:
        con.print(f"  [dim]File not found: {path_token}[/dim]")
        return
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        con.print(f"  [dim]Not a supported image file: {image_path.name}[/dim]")
        return

    ctx.attached_images.append(image_path)
    ctx.image_counter[0] += 1



def _extract_queue_text(item) -> str:
    """Extract display text from a queue payload (str, tuple, etc.)."""
    if isinstance(item, tuple):
        if item[0] == "__BTW__":
            return str(item[1])
        return str(item[0])  # (text, images)
    return str(item)



def _cmd_queue(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    pq = ctx.pending_queue
    args = cmd_args.strip()

    if not args:
        items = pq.peek_all() if pq else []
        if items:
            con.print(f"  [cyan]Queued messages ({len(items)}):[/cyan]")
            for i, item in enumerate(items):
                preview = _extract_queue_text(item)[:80]
                con.print(f"    {i + 1}. [dim]{preview}[/dim]")
            con.print()
        con.print(
            "  [dim]Usage: /queue <prompt>  |  /queue list  |  /queue edit <n> <text>  |  /queue insert <n> <text>  |  /queue remove <n>  |  /queue clear[/dim]"
        )
        con.print("  [dim]See also: /steer (nudge current run) · /background (run in parallel)[/dim]")
        return

    sub = args.split(maxsplit=1)
    subcommand = sub[0].lower()

    if subcommand == "list":
        if pq is None or pq.empty():
            con.print("  [dim]Queue is empty.[/dim]")
            return
        items = pq.peek_all()
        con.print(f"  [cyan]Queued messages ({len(items)}):[/cyan]")
        for i, item in enumerate(items):
            con.print(f"    {i + 1}. [dim]{_extract_queue_text(item)[:80]}[/dim]")
        return

    if subcommand == "clear":
        if pq is None:
            return
        n = pq.qsize()
        pq.clear()
        con.print(f"  [green]Cleared {n} queued message(s).[/green]")
        return

    if subcommand == "remove":
        if pq is None:
            return
        idx_str = sub[1].strip() if len(sub) > 1 else ""
        if not idx_str.isdigit():
            con.print("  [dim]Usage: /queue remove <number>[/dim]")
            return
        idx = int(idx_str) - 1
        if pq.remove_index(idx):
            con.print(f"  [green]Removed queued message #{idx + 1}.[/green]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1}[/red]")
        return

    if subcommand == "edit":
        if pq is None:
            return
        rest = sub[1].strip() if len(sub) > 1 else ""
        parts = rest.split(maxsplit=1)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].strip():
            con.print("  [dim]Usage: /queue edit <number> <new text>[/dim]")
            return
        idx = int(parts[0]) - 1
        new_text = parts[1]
        if pq.replace_index(idx, new_text):
            preview = new_text[:80] + ("..." if len(new_text) > 80 else "")
            con.print(f"  [green]Edited queued message #{idx + 1}:[/green] [dim]{preview}[/dim]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1}[/red]")
        return

    if subcommand == "insert":
        if pq is None:
            return
        rest = sub[1].strip() if len(sub) > 1 else ""
        parts = rest.split(maxsplit=1)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].strip():
            con.print("  [dim]Usage: /queue insert <number> <text>  (1 = front, qsize+1 = back)[/dim]")
            return
        idx = int(parts[0]) - 1
        new_text = parts[1]
        if pq.insert_index(idx, new_text):
            preview = new_text[:80] + ("..." if len(new_text) > 80 else "")
            con.print(f"  [green]Inserted at position #{idx + 1}:[/green] [dim]{preview}[/dim]")
        else:
            con.print(f"  [red]Invalid index: {idx + 1} (valid range: 1..{pq.qsize() + 1})[/red]")
        return

    # Default: queue a prompt
    pq.put(args)
    preview = args[:80] + ("..." if len(args) > 80 else "")
    if not ctx.agent_running:
        con.print(f"  Queued: {preview}")



def _queue_ahead_of_goal_continuation(pending_queue, text: str) -> None:
    """Enqueue ``text``, but ahead of any prompt the goal loop queued itself.

    A continuation prompt is written by the goal loop, not typed by the user,
    so letting it go first would spend a whole turn before the agent ever sees
    the correction.
    """
    for idx, item in enumerate(pending_queue.peek_all()):
        body = item[0] if isinstance(item, tuple) else item
        if isinstance(body, str) and is_goal_generated_prompt(body):
            if pending_queue.insert_index(idx, text):
                return
            break
    pending_queue.put(text)



def _cmd_steer(ctx: CommandContext, cmd_args: str = ""):
    """Inject guidance into the running agent's tool loop (mid-task).

    Unlike /queue (runs as a fresh turn after the current run finishes), /steer
    is consumed between tool batches of the CURRENT run, so the agent can course-
    correct without being interrupted.

    Guidance is never dropped. When there is no live run to inject into it falls
    back to a queued turn, placed ahead of any goal continuation prompt. Two
    windows make that fallback matter under a standing goal: the seconds the
    loop spends judging a finished turn (the agent is idle but the loop is very
    much alive), and the TOCTOU gap where the run ends between the UI's
    ``agent_running`` check and ``steer()`` — the case ``Agent.steer`` documents
    as "the caller MUST fall back to queuing".
    """
    con = get_console()
    guidance = cmd_args.strip()
    if not guidance:
        con.print("  [dim]Usage: /steer <guidance>  (e.g. /steer don't change the API, keep it compatible)[/dim]")
        return
    if ctx.agent_running and ctx.current_agent.steer(guidance):
        con.print("  [green]Steering queued — the agent will see it on its next step.[/green]")
        return
    if ctx.pending_queue is None:
        con.print("  [yellow]Agent isn't running — use /queue to send this as the next message instead.[/yellow]")
        return
    _queue_ahead_of_goal_continuation(ctx.pending_queue, guidance)
    con.print("  [green]Agent isn't mid-run — queued as the next turn.[/green]")



def _print_fork_points(con, session_log, session_id: str) -> None:
    """List the current session's user messages as branchable points."""
    messages = session_log.list_user_messages(limit=20)
    if not messages:
        con.print("  [dim]This session has no messages yet — nothing to fork.[/dim]")
        return
    con.print(f"  [bold]Fork points in {session_id}[/bold] [dim](newest first)[/dim]\n")
    for i, message in enumerate(messages, 1):
        stamp = (message.get("timestamp") or "")[:16].replace("T", " ")
        preview = " ".join((message.get("content") or "").split())[:76]
        con.print(f"    [cyan]{i:>2}[/cyan]  [dim]{message['uuid'][:8]}[/dim]  {stamp}")
        con.print(f"        {preview}")
    con.print(
        "\n  [dim]/fork <n>       branch off just BEFORE that message, so you can ask it "
        "differently[/dim]"
    )
    con.print("  [dim]/fork <uuid>    same, addressed by the id shown above[/dim]")
    con.print(
        f"  [dim]/resume {session_id[:8]} at <uuid>   re-enter this session from a point "
        "(the uuid is KEPT)[/dim]"
    )



def _cmd_fork(ctx: CommandContext, cmd_args: str = ""):
    """Branch the current conversation into a new session.

    ``/fork`` branches here and now: the whole conversation carries over and you
    keep talking, only in a new session, so whatever you do next does not land in
    the transcript you branched from. ``/fork list`` shows this session's messages
    with the ids to branch at, and ``/fork <n|uuid>`` branches just before one of
    them, putting the model back where it was when you asked — free to answer
    differently. The original session is untouched and stays resumable either way.
    """
    con = get_console()
    agent = ctx.current_agent
    session_log = agent._session_log if agent is not None else None
    if session_log is None or not session_log.exists():
        con.print("  [yellow]This session has nothing on disk yet — nothing to fork.[/yellow]")
        return

    target = (cmd_args or "").strip()
    if target == "list":
        _print_fork_points(con, session_log, agent.session_id)
        return

    chosen = None
    fork_at = None
    if target:
        messages = session_log.list_user_messages(limit=20)
        # A uuid prefix can be all digits, so "is it a number" is not enough to
        # tell the two forms apart — only a number that indexes the list is one.
        if target.isdecimal() and 1 <= int(target) <= len(messages):
            chosen = messages[int(target) - 1]
        else:
            matching = [m for m in messages if m["uuid"].startswith(target)]
            if len(matching) != 1:
                con.print(
                    f"  [red]'{target}' matches {len(matching)} fork points. "
                    "Run /fork list to see them.[/red]"
                )
                return
            chosen = matching[0]

        # `at <uuid>` truncates inclusively, so branching off *before* the chosen
        # question means forking at the entry in front of it — otherwise the
        # branch would end on an unanswered user turn.
        fork_at = session_log.uuid_before(chosen["uuid"])
        if fork_at is None:
            con.print(
                "  [yellow]That is the first message in the session — a branch before it would "
                "be empty. Use /new for a fresh session.[/yellow]"
            )
            return

    source_session_id = agent.session_id
    agent_config = dict(ctx.agent_config)
    agent_config["session_id"] = source_session_id
    # No fork point means "branch at the tip": copy the whole log, which is what
    # `SessionLog.fork(at_uuid=None)` does.
    if fork_at is None:
        agent_config["_fork_session"] = True
    else:
        agent_config["_resume_at_uuid"] = fork_at
    current_agent = create_agent(
        agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
        background_process_registry=ctx.background_processes,
        peer_session=ctx.peer_session,
    )
    if current_agent.session_id == source_session_id:
        con.print("  [red]Fork failed — the session log could not be branched.[/red]")
        return

    _, runs_built = hydrate_resumed_session(current_agent)
    display_stats = display_resumed_transcript(
        current_agent.working_memory.runs, current_agent.session_id or ""
    )
    if chosen is None:
        con.print(
            f"[green]Forked into {current_agent.session_id} — carried over the whole "
            f"conversation ({runs_built} runs, {display_stats.tool_result_count} tool results "
            f"collapsed). Keep going; nothing you say now touches the original.[/green]"
        )
    else:
        dropped = " ".join((chosen.get("content") or "").split())[:60]
        con.print(
            f"[green]Forked into {current_agent.session_id} — dropped '{dropped}' and "
            f"everything after it; restored {runs_built} runs into context "
            f"({display_stats.tool_result_count} tool results collapsed)[/green]"
        )
    con.print(
        f"  [dim]Forked from {source_session_id} — resume it any time with "
        f"`/resume {source_session_id}` here, or `agentica resume {source_session_id}` "
        f"from a shell[/dim]"
    )

    goal_manager = None
    if current_agent._session_log is not None:
        judge_model = current_agent.auxiliary_model or current_agent.model
        goal_manager = GoalManager(current_agent._session_log, judge_model=judge_model)
        state = goal_manager.load()
        if state is not None and state.status == "active":
            # Same reasoning as /resume: continuing a standing goal by itself on
            # a branch the user just created is too surprising.
            goal_manager.force_pause_on_resume()
            con.print(f"  [yellow]⊙ Standing goal paused on the branch:[/yellow] {state.objective}")
    return {"current_agent": current_agent, "goal_manager": goal_manager}



def _cmd_send_message(ctx: CommandContext, cmd_args: str = ""):
    """Send a message from you to one of your other live sessions.

    Named after the ``send_message`` tool the agent calls, the way
    ``/list-agents`` is named after ``list_agents``: same channel, different
    sender. This is for when you want to say something yourself without typing
    it into that terminal. It arrives marked as coming from you, so the
    receiving agent treats it as your instruction rather than another agent's
    information.
    """
    con = get_console()
    peers = ctx.peer_session
    if peers is None:
        con.print("  [yellow]Cross-session messaging is not active in this session.[/yellow]")
        return

    # split() rather than partition(" "): extra spaces after the target are
    # typing, not an empty message.
    parts = (cmd_args or "").strip().split(maxsplit=1)
    target = parts[0] if parts else ""
    text = parts[1].strip() if len(parts) > 1 else ""
    if not target or not text:
        con.print("  [dim]Usage: /send-message <session> <text>   (see /list-agents for names)[/dim]")
        con.print(
            "  [dim]e.g. /send-message benchmarks-b read tmp/handoff.md and take over from there[/dim]"
        )
        return

    try:
        sent = peers.send(target, text, from_kind="user")
    except PeerMessageRefused as exc:
        con.print(f"  [red]Not sent: {exc}[/red]")
        return
    # Name the peer it actually resolved to: '/send-message 7e17' should show
    # which session that prefix picked.
    con.print(
        f"  [green]Queued for {sent.to_name}.[/green] [dim]The other session accepts it "
        f"between tool calls if running, or as its next turn if idle.[/dim]"
    )



def _print_peer_details(con, info, *, indent: str) -> None:
    """Print one session's fields, aligned, from ``PeerInfo.detail_rows()``.

    The rows come from the same place the ``list_agents`` tool renders, so the
    user and the model never see different field sets.
    """
    rows = info.detail_rows()
    width = max((len(label) for label, _ in rows), default=0)
    for label, value in rows:
        con.print(f"{indent}{label + ':':<{width + 1}} [dim]{value}[/dim]")


def _cmd_list_agents(ctx: CommandContext, cmd_args: str = ""):
    """Show the live CLI sessions this one can exchange messages with.

    Purely for the user to inspect: the agent finds its own targets through the
    ``list_agents`` tool, so nothing has to be run before asking it to send.
    """
    con = get_console()
    peers = ctx.peer_session
    if peers is None:
        con.print("  [yellow]Cross-session messaging is not active in this session.[/yellow]")
        return

    con.print(f"  This session: [cyan]{peers.name}[/cyan] [dim]peer={peers.peer_id}[/dim]")
    _print_peer_details(con, peers.info, indent="  ")
    pending = peers.unread_count()
    if pending:
        con.print(f"  [yellow]{pending} message(s) waiting to be read[/yellow]")

    live = peers.list_peers()
    if not live:
        con.print("  [dim]No other live sessions. Start agentica in another terminal to message it.[/dim]")
        return
    con.print(f"\n  [cyan]Other live sessions ({len(live)}):[/cyan]")
    for info in live:
        con.print(f"    [bold]{info.name}[/bold] [dim]peer={info.peer_id}[/dim]  pid={info.pid}")
        _print_peer_details(con, info, indent="      ")
    con.print(
        "  [dim]Ask the agent to message one by name / peer id / session_id "
        "(it calls send_message itself), or say it yourself with "
        "/send-message <name|id> <text>.[/dim]"
    )



def _checkpoint_manager(ctx: CommandContext):
    """Build a disk-backed CheckpointManager scoped to the current session."""
    from agentica.checkpoint import CheckpointManager

    session_id = ctx.current_agent.session_id or "default"
    return CheckpointManager(session_id=session_id)



def _resolve_ckpt_path(ctx: CommandContext, raw: str) -> str:
    """Resolve a user-supplied path against the agent's work_dir."""
    p = os.path.expanduser(raw)
    if os.path.isabs(p):
        return p
    base = ctx.current_agent.work_dir or os.getcwd()
    return os.path.join(str(base), p)



def _work_dir_root(ctx: CommandContext) -> Path:
    return Path(ctx.current_agent.work_dir or os.getcwd()).expanduser().resolve()



def _is_inside_work_dir(path: str, root: Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(root)
        return True
    except ValueError:
        return False



def _cmd_checkpoint(ctx: CommandContext, cmd_args: str = ""):
    """Manual, durable, multi-file checkpoints for the current session.

    /checkpoint [list]                 -> list checkpoints (newest first)
    /checkpoint create <label> <path...> -> snapshot files' current content
    /checkpoint diff <id>              -> unified diff snapshot -> current
    /checkpoint restore <id>           -> roll files back to the snapshot
    """
    con = get_console()
    cm = _checkpoint_manager(ctx)
    args = cmd_args.strip()
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        con.print(f"  [red]Invalid checkpoint command: {exc}[/red]")
        return
    sub = parts[0].lower() if parts else "list"

    def _find(cid_prefix: str):
        ck = cm.get(cid_prefix)
        if ck is not None:
            return ck
        matches = [c for c in cm.list() if c.id.startswith(cid_prefix)]
        return matches[0] if len(matches) == 1 else None

    if sub == "list":
        items = cm.list()
        if not items:
            con.print("  [dim]No checkpoints. Create one: /checkpoint create <label> <path...>[/dim]")
            return
        con.print(f"  [cyan]Checkpoints ({len(items)}):[/cyan]")
        for c in items:
            con.print(f"    {c.id[:18]}  [dim]{c.created_at}[/dim]  {c.label}  ([dim]{len(c.files)} file(s)[/dim])")
        return

    if sub == "create":
        if len(parts) < 3:
            con.print("  [dim]Usage: /checkpoint create <label> <path> [more paths...][/dim]")
            return
        label = parts[1]
        paths = [_resolve_ckpt_path(ctx, p) for p in parts[2:]]
        ck = cm.create(label, paths)
        con.print(f"  [green]Created checkpoint {ck.id[:18]} ({label}) with {len(ck.files)} file(s).[/green]")
        return

    if sub in ("diff", "restore"):
        if len(parts) < 2:
            con.print(f"  [dim]Usage: /checkpoint {sub} <id>{' --yes' if sub == 'restore' else ''}[/dim]")
            return
        ck = _find(parts[1])
        if ck is None:
            con.print(f"  [red]No checkpoint matching '{parts[1]}'.[/red]")
            return
        if sub == "diff":
            con.print(cm.diff(ck.id))
            return

        root = _work_dir_root(ctx)
        outside = [f.path for f in ck.files if not _is_inside_work_dir(f.path, root)]
        if outside:
            con.print(f"  [red]Refusing to restore checkpoint files outside work_dir: {root}[/red]")
            for path in outside[:5]:
                con.print(f"    [dim]{path}[/dim]")
            return

        diff_text = cm.diff(ck.id)
        deletions = [f.path for f in ck.files if not f.existed and Path(f.path).exists()]
        if "--yes" not in parts:
            con.print(diff_text)
            if deletions:
                con.print("  [yellow]Restore will delete file(s) created after the checkpoint:[/yellow]")
                for path in deletions:
                    con.print(f"    [dim]{path}[/dim]")
            con.print("  [yellow]Re-run with --yes to restore this checkpoint.[/yellow]")
            return

        restored = cm.restore(ck.id)
        con.print(f"  [green]Restored {len(restored)} file(s) from {ck.id[:18]} ({ck.label}).[/green]")
        return

    con.print("  [dim]Usage: /checkpoint list | create <label> <path...> | diff <id> | restore <id> --yes[/dim]")



def _cmd_btw(ctx: CommandContext, cmd_args: str = ""):
    """Ephemeral side question — dispatched as concurrent task, no tools, not persisted."""
    con = get_console()
    question = cmd_args.strip()
    if not question:
        con.print("  [dim]Usage: /btw <question>   (quick aside; for a persisted parallel task use /background)[/dim]")
        return
    if ctx.current_agent is None:
        con.print("[yellow]No active agent.[/yellow]")
        return
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(("__BTW__", question))
        con.print(f"  [dim]Side question: {question[:60]}{'...' if len(question) > 60 else ''}[/dim]")



def _cmd_ps(ctx: CommandContext, cmd_args: str = ""):
    """List background agent tasks and background terminal commands."""
    con = get_console()
    active_agents = list(ctx.bg_tasks.items())
    terminals = []
    if ctx.background_processes is not None:
        terminals = ctx.background_processes.list(include_finished=False)

    if not active_agents and not terminals:
        con.print("  [dim]No active background tasks.[/dim]")
        con.print("  [dim]Use /background <prompt> or execute(background=True).[/dim]")
        return

    if terminals:
        con.print(f"  [cyan]Background terminals ({len(terminals)}):[/cyan]")
        for item in terminals:
            kind = " [magenta]delegated session[/magenta]" if item.kind == "delegate" else ""
            con.print(
                f"    #{item.num} [dim]{item.id}[/dim] pid={item.pid} "
                f"elapsed={item.elapsed}{kind}"
            )
            # Full command on its own lines — /ps is the inspection surface,
            # so never truncate here (preview is only for one-line status UI).
            # A delegated session shows its task instead: its command line is a
            # `python -m agentica.cli.main --query <the whole task>`.
            body = item.label if item.kind == "delegate" else (item.command or "")
            for line in body.splitlines() or [""]:
                con.print(f"      {line}")
            con.print(f"      [dim]log: {item.log_path}[/dim]")

    if active_agents:
        con.print(f"  [cyan]Background agents ({len(active_agents)}):[/cyan]")
        for tid, info in active_agents:
            prompt = info["prompt"] or ""
            con.print(f"    #{info['num']} [dim]{tid}[/dim]")
            for line in prompt.splitlines() or [""]:
                con.print(f"      {line}")

    con.print("  [dim]Stop one with /stop <id|pid|#n>, or every one with /stop all.[/dim]")



def _cmd_background(ctx: CommandContext, cmd_args: str = ""):
    """Run a prompt in the background (independent agent with context snapshot)."""
    con = get_console()
    prompt = cmd_args.strip()
    if not prompt:
        _cmd_ps(ctx, "")
        con.print("  [dim]Usage: /background <prompt>[/dim]")
        con.print("  [dim]See also: /queue (next turn, same session) · /btw (quick aside, not persisted)[/dim]")
        return

    ctx.bg_task_counter += 1
    task_num = ctx.bg_task_counter
    task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{task_num}"

    ctx.bg_tasks[task_id] = {"thread": None, "agent": None, "prompt": prompt, "num": task_num}

    # Capture references needed by the background thread
    agent_config = ctx.agent_config
    extra_tools = ctx.extra_tools
    workspace = ctx.workspace
    skills_registry = ctx.skills_registry
    bg_tasks = ctx.bg_tasks

    # Snapshot current conversation context for the background agent.
    # History is loaded via working_memory.runs (not .messages) by the runner,
    # so we must inject a synthetic AgentRun with the snapshot messages.
    context_snapshot = []
    main_agent = ctx.current_agent
    if main_agent and main_agent.working_memory and main_agent.working_memory.messages:
        for msg in main_agent.working_memory.messages:
            if msg.role in ("user", "assistant") and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if len(content) > 500:
                    content = content[:500] + "..."
                context_snapshot.append(Message(role=msg.role, content=content))
        # Keep only last 10 messages to avoid blowing up context
        context_snapshot = context_snapshot[-10:]

    def _run_bg():
        bg_config = dict(agent_config)
        bg_config["session_id"] = _generate_session_id()
        bg_config["debug"] = False
        bg_agent = create_agent(
            bg_config,
            extra_tools,
            workspace,
            skills_registry,
            background_process_registry=ctx.background_processes,
        )
        bg_tasks[task_id]["agent"] = bg_agent

        # Inject context snapshot as a synthetic AgentRun so the runner
        # picks it up via get_messages_from_last_n_runs().
        if context_snapshot:
            synthetic_run = AgentRun(
                response=RunResponse(messages=context_snapshot),
            )
            bg_agent.working_memory.runs.append(synthetic_run)

        result_text = ""
        try:
            response = bg_agent.run_sync(prompt)
            result_text = response.content if response else ""
        except Exception as e:
            if bg_agent._cancelled:
                result_text = "(cancelled)"
            else:
                result_text = f"Error: {e}"
        finally:
            bg_tasks.pop(task_id, None)

        from agentica.cli.interactive.console_io import _print_boxed_result

        _print_boxed_result(
            f"Background #{task_num}",
            prompt,
            result_text or "",
            color="bright_magenta",
        )

    thread = threading.Thread(target=_run_bg, daemon=True, name=task_id)
    bg_tasks[task_id]["thread"] = thread
    thread.start()
    preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
    con.print(f"  [green]Background #{task_num} started:[/green] {preview}")



def _cmd_stop(ctx: CommandContext, cmd_args: str = ""):
    """Stop BACKGROUND work only — an explicit target is required.

    Two deliberate boundaries:

    - ``/stop`` never touches the run you are waiting on. Ctrl+C owns that,
      and it does strictly more than ``Agent.cancel()``: it wakes a thread
      parked in ``ask_user_question``, pauses a standing goal so the post-turn
      hook doesn't immediately re-queue a continuation, and escalates to a
      force exit on a second press. A second, weaker way to cancel the current
      run would only differ from Ctrl+C in the cases that matter — and while
      the agent is blocked on ``ask_user_question`` a typed line is consumed
      as the *answer*, so ``/stop`` cannot even reach this handler there.
    - A bare ``/stop`` prints usage instead of stopping everything. It is one
      keystroke away from ``/stop <id>``, it arrives while other work is
      running, and killing every background task is not a plausible default
      for a missing argument. ``/stop all`` says it on purpose.
    """
    con = get_console()
    target = cmd_args.strip()
    stopped_agents = 0
    stopped_terms = 0

    if not target:
        # "Is anything running" must agree with what /ps shows, so a task whose
        # agent object hasn't been constructed yet (the thread is still starting)
        # still counts — otherwise /ps lists one and /stop says there are none.
        running_agents = list(ctx.bg_tasks)
        running_terms = (
            ctx.background_processes.list(include_finished=False)
            if ctx.background_processes is not None
            else []
        )
        if not running_agents and not running_terms:
            con.print("  [dim]No active background tasks.[/dim]")
            con.print("  [dim]Ctrl+C interrupts the current run; /stop is only for background tasks.[/dim]")
            return
        con.print("  [yellow]/stop needs a target — nothing was stopped.[/yellow]")
        _cmd_ps(ctx, "")
        con.print("  [dim]Ctrl+C interrupts the current run; /stop is only for background tasks.[/dim]")
        return

    def _matches_agent(tid: str, info: dict) -> bool:
        if target.lower() in {"all", "*"}:
            return True
        return target in {tid, str(info.get("num")), f"#{info.get('num')}"}

    for tid, info in list(ctx.bg_tasks.items()):
        if not _matches_agent(tid, info):
            continue
        agent = info.get("agent")
        if agent is not None:
            agent.cancel()
            stopped_agents += 1

    if ctx.background_processes is not None:
        stopped_terms = len(ctx.background_processes.stop(target))

    if stopped_agents == 0 and stopped_terms == 0:
        con.print(f"  [dim]No running background task matched '{target}'.[/dim]")
        return
    parts = []
    if stopped_terms:
        parts.append(f"{stopped_terms} terminal(s)")
    if stopped_agents:
        parts.append(f"{stopped_agents} agent task(s)")
    con.print(f"  [green]Stopped {', '.join(parts)}.[/green]")
