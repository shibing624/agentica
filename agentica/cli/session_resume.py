# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Resolve which session to resume and which directory to resume it in.

Sessions are stored partitioned by project (work_dir), so ``/resume <id>`` used
to only see the sessions of the directory the CLI was started in. This module
lifts that restriction in two steps:

1. Look the id up in the current project first, then across every project of
   the same user (:meth:`SessionLog.find_sessions`).
2. When the match lives in a different directory, ask which one to continue in
   — the session's own directory or the current one — and remember the answer
   in ``settings.resume_cwd`` if the user asks us to.

Both CLI entry points share this: the ``/resume`` command and the
``agentica resume <id>`` startup flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentica.global_config import get_setting, set_setting
from agentica.memory.session_log import SessionLog
from agentica.run_response import AgentCancelledError
from agentica.utils.log import logger

# ``settings.resume_cwd`` in ~/.agentica/config.yaml.
RESUME_CWD_SETTING = "resume_cwd"
RESUME_CWD_ASK = "ask"
RESUME_CWD_SESSION = "session"
RESUME_CWD_CURRENT = "current"
RESUME_CWD_CHOICES = (RESUME_CWD_ASK, RESUME_CWD_SESSION, RESUME_CWD_CURRENT)

# Answer callback: (prompt, options) -> the option the user picked.
Asker = Callable[[str, Optional[List[str]]], str]


@dataclass(frozen=True)
class CwdChoice:
    """Outcome of the "which directory?" question.

    ``work_dir`` is ``None`` when the current directory should be kept, which is
    also the case when there was nothing to decide.
    """

    work_dir: Optional[str] = None
    cancelled: bool = False


def normalize_needle(raw: str) -> str:
    """Strip the ``abc...wxyz`` shorthand a user may copy out of a listing."""
    return (raw or "").split("...", 1)[0].strip()


def same_dir(a: Optional[str], b: Optional[str]) -> bool:
    """Compare two directories through symlinks and trailing separators."""
    if not a or not b:
        return False
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return str(a) == str(b)


def find_sessions_by_id(
    needle: str,
    local_sessions: List[Dict[str, Any]],
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Prefix-match a session id, preferring the current project.

    Searching the whole store only when the current project has no match keeps
    the common case unambiguous: an id prefix that is unique here stays unique
    even if some other project happens to share it.
    """
    needle = normalize_needle(needle)
    if not needle:
        return []
    local = [s for s in local_sessions if s["session_id"].startswith(needle)]
    if local:
        return local
    return SessionLog.find_sessions(needle, user_id=user_id)


def resume_cwd_preference() -> str:
    """Read ``settings.resume_cwd``, falling back to ``ask`` for junk values."""
    value = str(get_setting(RESUME_CWD_SETTING, RESUME_CWD_ASK) or RESUME_CWD_ASK).strip().lower()
    return value if value in RESUME_CWD_CHOICES else RESUME_CWD_ASK


def _match_option(answer: str, options: List[str]) -> Optional[int]:
    """Map a raw answer onto an option index (accepts the number or the text)."""
    answer = (answer or "").strip()
    if not answer:
        return None
    if answer.isdecimal():
        index = int(answer) - 1
        return index if 0 <= index < len(options) else None
    folded = answer.casefold()
    for i, option in enumerate(options):
        if option.casefold() == folded:
            return i
    return None


def _terminal_asker(prompt: str, options: Optional[List[str]]) -> str:
    """Read an answer straight from the terminal (used before the TUI starts)."""
    from prompt_toolkit import prompt as pt_prompt

    lines = [prompt]
    lines.extend(f"  {i}. {option}" for i, option in enumerate(options or [], 1))
    lines.append("> ")
    try:
        return pt_prompt("\n".join(lines))
    except (EOFError, KeyboardInterrupt):
        raise AgentCancelledError("resume directory prompt aborted")


def choose_resume_work_dir(
    session_work_dir: Optional[str],
    current_work_dir: str,
    *,
    asker: Optional[Asker] = None,
    printer: Optional[Callable[[str], None]] = None,
) -> CwdChoice:
    """Decide which directory to resume a session in.

    Returns an empty :class:`CwdChoice` (keep the current directory) when the
    session belongs here already, or when its recorded directory is gone. The
    "always ..." answers are persisted to ``settings.resume_cwd`` so the
    question is asked at most once per user.
    """
    say = printer or (lambda _msg: None)

    if not session_work_dir or same_dir(session_work_dir, current_work_dir):
        return CwdChoice()
    if not Path(session_work_dir).is_dir():
        say(
            f"[yellow]Session directory no longer exists ({session_work_dir}); "
            f"resuming in {current_work_dir}.[/yellow]"
        )
        return CwdChoice()

    preference = resume_cwd_preference()
    if preference == RESUME_CWD_SESSION:
        return CwdChoice(work_dir=session_work_dir)
    if preference == RESUME_CWD_CURRENT:
        return CwdChoice()

    options = [
        f"Use session directory ({session_work_dir})",
        f"Use current directory ({current_work_dir})",
        "Always use session directory",
        "Always use current directory",
    ]
    prompt = (
        "This session was started in another directory. "
        "Choose the working directory to resume it in."
    )
    ask = asker or _terminal_asker
    try:
        answer = ask(prompt, options)
    except AgentCancelledError:
        return CwdChoice(cancelled=True)

    choice = _match_option(answer, options)
    if choice is None:
        # No usable answer (no TTY, blank line): continuing the session where it
        # was written is the reason the user typed its id in the first place.
        say(f"[dim]No answer given — using the session directory {session_work_dir}.[/dim]")
        return CwdChoice(work_dir=session_work_dir)
    if choice in (2, 3):
        remembered = RESUME_CWD_SESSION if choice == 2 else RESUME_CWD_CURRENT
        set_setting(RESUME_CWD_SETTING, remembered)
        say(f"[dim]Saved settings.{RESUME_CWD_SETTING}={remembered} in config.yaml.[/dim]")
    return CwdChoice(work_dir=session_work_dir if choice in (0, 2) else None)


def prepare_startup_resume(
    agent_config: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    printer: Optional[Callable[[str], None]] = None,
    asker: Optional[Asker] = None,
) -> bool:
    """Settle `agentica resume <id>` before the first agent is built.

    Rewrites ``agent_config`` in place with the full session id, the directory
    its transcript lives in, and the work_dir the user chose. Returns ``False``
    when the resume cannot go ahead, in which case the caller should exit rather
    than silently start an unrelated session.
    """
    say = printer or (lambda _msg: None)

    needle = normalize_needle(str(agent_config.get("session_id") or ""))
    matches = SessionLog.find_sessions(needle, user_id=user_id)
    if not matches:
        say(f"[bold red]No session found: {needle}[/bold red]")
        return False
    if len(matches) > 1:
        say(
            f"[bold red]'{needle}' matches {len(matches)} sessions. "
            f"Use a longer id prefix.[/bold red]"
        )
        return False

    session = matches[0]
    current_work_dir = agent_config.get("work_dir") or os.getcwd()
    choice = choose_resume_work_dir(
        session["work_dir"], current_work_dir, asker=asker, printer=say
    )
    if choice.cancelled:
        say("[yellow]Resume cancelled.[/yellow]")
        return False

    agent_config["session_id"] = session["session_id"]
    agent_config["session_base_dir"] = session["base_dir"]
    if session.get("profile_name"):
        agent_config["_resume_session_profile_name"] = session["profile_name"]
        agent_config["_resume_session_profile_source"] = session.get("profile_source") or "session"
    if choice.work_dir:
        if not enter_work_dir(choice.work_dir):
            say(f"[red]Cannot enter {choice.work_dir}; resume aborted.[/red]")
            return False
        agent_config["work_dir"] = choice.work_dir
        say(f"[dim]Working directory: {choice.work_dir}[/dim]")
    return True


def enter_work_dir(work_dir: str) -> bool:
    """Move the process into ``work_dir``.

    Tools already honour the agent's ``work_dir``, but git status, ``@file``
    completion and shell-out commands read the process cwd, so both have to
    move together or the session resumes into a split-brain directory.
    """
    try:
        os.chdir(work_dir)
        return True
    except OSError as e:
        logger.warning(f"Could not enter session directory {work_dir}: {e}")
        return False
