"""Turn-loop checkpoint helpers for the CLI (Clue-Code-style rewind).

Glue between the per-turn ``TurnCheckpointer`` primitive and the interactive
CLI: path extraction from file-write tool calls, conversation truncation, and
the session-scoped checkpointer accessor shared by the turn loop and the
``/rewind`` command.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from agentica.checkpoint import TurnCheckpointer
from agentica.utils.log import logger

# Tool names whose tool_args carry a write target. These are the ones we
# snapshot before execution so a turn rewind can restore turn-start content.
_REWRITE_TOOLS = {"write_file", "apply_patch"}

# ``*** Update File: path`` / ``*** Add File: path`` / ``*** Delete File: path``
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


def _resolve(raw: str, work_dir: str) -> str:
    """Resolve a tool path the same way BuiltinFileTool does: ~ expand, absolute
    passthrough, relative against work_dir."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(work_dir) / p
    return str(p.resolve())


def extract_rewrite_paths(tool_name: str, tool_args: dict, work_dir: str) -> List[str]:
    """Return the absolute paths a file-write tool call will touch.

    Empty for non-write tools. ``apply_patch`` yields every file in the envelope.
    """
    if tool_name not in _REWRITE_TOOLS:
        return []
    args = tool_args or {}
    raw_paths: List[str] = []
    if tool_name == "apply_patch":
        patch = args.get("patch") or ""
        raw_paths = [m.group(1).strip() for m in _PATCH_FILE_RE.finditer(patch)]
    else:
        fp = args.get("file_path")
        if fp:
            raw_paths = [str(fp)]
    seen: set = set()
    resolved: List[str] = []
    for raw in raw_paths:
        try:
            p = _resolve(raw, work_dir)
        except (OSError, ValueError) as e:
            logger.warning(f"rewind: could not resolve tool path {raw!r}: {e}")
            continue
        if p not in seen:
            seen.add(p)
            resolved.append(p)
    return resolved


def truncate_conversation(agent, msg_index: Optional[int]) -> int:
    """Truncate working_memory to before ``msg_index``. Returns removed count.

    Rebuilds ``runs`` from the surviving flat ``messages`` via ``collapse_runs``
    so the prompt builder (``get_messages_from_last_n_runs``) stops seeing the
    undone turns. System messages sit at the front and are preserved.
    """
    wm = agent.working_memory
    if msg_index is None or msg_index < 0:
        return 0
    removed = len(wm.messages) - msg_index
    if removed <= 0:
        return 0
    wm.messages = wm.messages[:msg_index]
    wm.collapse_runs(wm.messages)
    return removed


def get_turn_checkpointer(tui_state: dict, session_id: str) -> TurnCheckpointer:
    """Get-or-create the session-scoped TurnCheckpointer cached in tui_state.

    The instance must outlive a single turn (begin/snapshot/finalize span one
    ``_process_stream_response`` call), and be recreated when the session id
    changes (``/newchat``).
    """
    tc = tui_state.get("_turn_checkpointer")
    if tc is None or getattr(tc, "session_id", None) != session_id:
        tc = TurnCheckpointer(session_id=session_id)
        tui_state["_turn_checkpointer"] = tc
    return tc


def print_turn_list(con, turns) -> None:
    """Render rewindable turns, shared by ``/rewind list`` and the ``/undo`` redirect."""
    if not turns:
        con.print(
            "  [dim]No turns to rewind yet. Finish a turn, then /rewind to roll it back.[/dim]"
        )
        return
    con.print(f"  [cyan]Rewindable turns ({len(turns)}):[/cyan]")
    for c in turns:
        prompt_preview = (c.prompt or "").strip().replace("\n", " ")
        if len(prompt_preview) > 48:
            prompt_preview = prompt_preview[:48] + "…"
        con.print(
            f"    [bold]{c.turn}[/bold]  [dim]{c.created_at}[/dim]  "
            f"{prompt_preview}  ([dim]{len(c.files)} file(s)[/dim])"
        )
    con.print(
        "  [dim]Usage: /rewind <n> --yes  (restores code + conversation to that turn's start)[/dim]"
    )
