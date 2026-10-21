"""Harness-authored session notes, written on every window cut.

The notes file is standing state for the next window: goals, constraints,
IDs, decisions. It used to be the model's own job, prompted by a 95%
token-budget nudge with the window cut postponed behind it. That failed in
three ways: the model often did not write it, it had to be told where the
file lived, and a cut that fell inside a long tool loop could not deliver the
nudge at all.

The harness writes it instead, from one request on the same model over the
same messages, right before the cut. System prompt, transcript, tools,
tool_choice and thinking configuration are exactly what the window's last
request used, so the request reads the prompt cache instead of paying for
the context again; only the trailing instruction is new. Tools are not run:
this pass reads the window and writes prose, it does not continue the task.
"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from agentica.compression.new_window import _NOTES_HEAD_RATIO, _NOTES_INJECT_LIMIT
from agentica.compression.notes import clip_head_tail
from agentica.model.message import Message
from agentica.utils.log import logger

_INSTRUCTIONS = """Write the session notes file for the next context window.

This context window is about to be replaced by a fresh one. Do not continue
the task and do not call tools: your whole reply is saved as the notes file
and injected at the top of the next window.

Keep what is needed to continue this session cold:
- the user's goal in their own terms, and how far it got
- constraints and preferences the user stated
- identifiers that cannot be re-derived: paths, ports, branch names, ticket
  or order numbers, commit hashes, service names
- decisions already made and why, so they are not re-litigated
- what is still open, and the next concrete step

Facts only, as markdown bullets, under 3000 characters. No narration of the
steps taken: the full transcript stays in the session log and search_session
can find it. Reply with the file contents only."""

_REVISE = """

The notes file already has this content. Your reply replaces it, so carry
forward what is still true, drop what is done or superseded, and fold in what
happened since:

<current_notes>
{existing}
</current_notes>"""


def notes_instruction(notes_path: str) -> str:
    path = Path(notes_path)
    existing = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if not existing:
        return _INSTRUCTIONS
    # An append-only file from an older version can be 98KB. Sent whole, at
    # 95% of the window, the request overflows and the file never gets the
    # rewrite that would shrink it. Clipped the same way as its injection.
    existing = clip_head_tail(
        existing, _NOTES_INJECT_LIMIT, marker="\n…[truncated]…\n", head_ratio=_NOTES_HEAD_RATIO,
    )
    return _INSTRUCTIONS + _REVISE.format(existing=existing)


def write_notes_atomically(notes_path: str, text: str) -> None:
    """Write via one rename, so a crash cannot leave a half-written file.

    The file is injected as ``<session_notes>`` at every later cut; a torn
    write would be carried into each following window.
    """
    path = Path(notes_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".notes-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


async def author_session_notes(
    messages: List[Message],
    model,
    notes_path: Optional[str],
) -> Optional[str]:
    """Have ``model`` rewrite the notes file from ``messages``; returns the text.

    Returns None and leaves the file untouched when there is no notes file to
    write (SDK agent without a session), no user turn to take notes on, or
    the request fails or comes back empty. The cut never waits on this: the
    ``<dropped_span>`` skim still rides along as the fallback carrier.
    """
    if not notes_path or not any(m.role == "user" for m in messages):
        return None

    request = [*messages, Message(role="user", content=notes_instruction(notes_path))]
    run_tools = model.run_tools
    model.run_tools = False
    parts: List[str] = []
    try:
        # Streamed like every turn of the loop: the Anthropic SDK refuses a
        # non-streaming request whose max_tokens could run past ten minutes.
        async for chunk in model.response_stream(request):
            if chunk.content:
                parts.append(chunk.content)
    except Exception as exc:
        logger.warning(f"Session notes not written ({type(exc).__name__}: {exc}); the digest carries the window")
        return None
    finally:
        model.run_tools = run_tools

    text = "".join(parts).strip()
    if not text:
        logger.warning("Session notes request returned no text; the digest carries the window")
        return None
    write_notes_atomically(notes_path, text + "\n")
    logger.info(f"Session notes written by the harness: {len(text)} chars -> {notes_path}")
    return text
