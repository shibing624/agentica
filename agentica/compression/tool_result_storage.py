# -*- coding: utf-8 -*-
"""
@description: Layer 0 — bound how much one turn's tool output may inject.

This is not compression but an output policy, and it runs at the moment a
result is produced rather than before a request. Two rules:

    1. Per-result: a single result over ``Function.max_result_size_chars``
       is shrunk on the spot, so a 5 MB output never enters the context once.
    2. Per-batch: when the whole batch of fresh results would occupy more than
       ``TOOL_BATCH_BUDGET_RATIO`` of the window, the largest ones are shrunk
       until it fits. Layer 1 deliberately never touches the trailing batch,
       so without this a single wide parallel round has nothing to save it.

**Shrinking to a file path is only useful when someone can open the path.**
An agent with ``read_file`` (or ``execute``) can pull the full copy back for
one tool call; an agent assembled from business tools alone cannot, and for it
the path is a handle nobody in the session can hold — the data is gone and the
message invites the model to read a file it has no way to read. So the form of
the shrink follows the session's actual capability (``can_recover_spill``):
spill to disk and hand over the path when it can be read back, otherwise
truncate honestly and say so.

Accumulated *history* is not this layer's problem. That is Layer 1's, gated on
real window pressure; a second budget here in fixed chars would be a second
threshold governing the same decision, and it fired long before the window was
under any pressure at all.

Path structure:
    ~/.agentica/projects/<user>/<sanitized-cwd>/<session-id>/tool-results/<tool-use-id>.txt

Usage (automatic - called from Model.run_function_calls):
    from agentica.compression.tool_result_storage import maybe_persist_result
    content = maybe_persist_result(
        tool_name="execute", tool_use_id="call_abc123",
        content=huge_bash_output, session_id="sess_xyz",
        recoverable=can_recover_spill(model.functions),
    )
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from agentica.security.redact import redact_sensitive_text
from agentica.utils.log import logger
from agentica.utils.tokens import count_text_tokens

if TYPE_CHECKING:
    from agentica.model.message import Message

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max chars to keep inline in the context (preview)
PREVIEW_CHARS = 2000

# Default max result size before persisting to disk (single tool).
# Individual tools can override via Function.max_result_size_chars.
DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000

# Share of the context window one turn's fresh tool output may occupy. Tied to
# the window rather than to a char count so this layer and Layer 1 measure
# pressure against the same number: a 200K-char budget fired on a 512K-token
# window that had room to spare, and never fired on a 8K-token window that had
# none.
TOOL_BATCH_BUDGET_RATIO = 0.25

# Tools that can read an arbitrary local path back into the context. Handing
# out a spill path is only better than truncating when one of these is
# registered for the call.
RECOVERY_TOOL_NAMES = frozenset({"read_file", "execute"})


def can_recover_spill(function_names: Iterable[str]) -> bool:
    """Whether this session could read a persisted result back into context."""
    return bool(RECOVERY_TOOL_NAMES.intersection(function_names))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_MAX_SANITIZED_LENGTH = 200


def sanitize_path(raw: str) -> str:
    """Convert a filesystem path into a readable, safe, collision-free directory name.

    Non-alphanumeric characters become ``-``, which is lossy: structurally
    different paths can slugify to the same string (e.g. ``/a/b`` and
    ``/a-b`` both become ``-a-b``). An md5 hash suffix is therefore ALWAYS
    appended (not just when truncating long paths) so two different inputs
    can never collide on the same directory. Long paths (>200 chars) are
    truncated before the suffix so the final name stays filesystem-safe.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', raw)
    hash_suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{sanitized[:_MAX_SANITIZED_LENGTH]}-{hash_suffix}"


def safe_user_segment(user_id: Optional[str]) -> str:
    """Path segment for the per-user spill directory.

    Delegates to ``Workspace.sanitize_user_id`` so persisted tool-result
    paths line up with ``users/{user_id}/`` exactly, including the
    "default" sentinel for None / blank input.
    """
    from agentica.workspace import Workspace
    return Workspace.sanitize_user_id(user_id)


def get_projects_root(user_id: Optional[str] = None) -> str:
    """Return ``<AGENTICA_PROJECTS_DIR>/<user>/`` — the parent of every project dir.

    Delegates to :func:`agentica.project_store.projects_root` so sessions,
    profiles, and tool-result spill share one layout.
    """
    from agentica.project_store import projects_root

    return projects_root(user_id)


def get_project_dir(cwd: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Return ``<AGENTICA_PROJECTS_DIR>/<user>/<sanitized-cwd>/`` for the given user + cwd.

    Thin wrapper around :func:`agentica.project_store.project_base_dir`.
    """
    from agentica.project_store import project_base_dir

    return project_base_dir(cwd, user_id=user_id)


def get_tool_results_dir(
    cwd: Optional[str] = None,
    session_id: str = "default",
    user_id: Optional[str] = None,
) -> str:
    """Return ``~/.agentica/projects/<user>/<project-hash>/<session-id>/tool-results/``."""
    return os.path.join(get_project_dir(cwd, user_id=user_id), session_id, "tool-results")


def get_tool_result_path(
    tool_use_id: str,
    cwd: Optional[str] = None,
    session_id: str = "default",
    is_json: bool = False,
    user_id: Optional[str] = None,
) -> str:
    """Return full path for a persisted tool result file."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_use_id)
    ext = "json" if is_json else "txt"
    return os.path.join(
        get_tool_results_dir(cwd, session_id, user_id=user_id),
        f"{safe_id}.{ext}",
    )


# ---------------------------------------------------------------------------
# Persistence message builder
# ---------------------------------------------------------------------------

def _preview(content: str) -> str:
    """40% head + 60% tail — keeps both the early context (command echo,
    headers) and the final results (exit codes, summaries)."""
    if len(content) <= PREVIEW_CHARS:
        return content
    head_chars = int(PREVIEW_CHARS * 0.4)
    tail_chars = PREVIEW_CHARS - head_chars
    omitted = len(content) - head_chars - tail_chars
    return (
        content[:head_chars]
        + f"\n\n... [{omitted} chars omitted] ...\n\n"
        + content[-tail_chars:]
    )


def _size_kb(content: str) -> float:
    return len(content.encode("utf-8", errors="ignore")) / 1024


def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def _already_shrunk(content: str) -> bool:
    return "<persisted-output>" in content or "<truncated-output>" in content


def _build_persisted_message(
    file_path: str,
    content: str,
    *,
    size_bytes: Optional[int] = None,
    n_lines: Optional[int] = None,
) -> str:
    """Preview + the path holding the full copy, for a session that can read it."""
    kb = (size_bytes / 1024) if size_bytes is not None else _size_kb(content)
    lines = n_lines if n_lines is not None else _line_count(content)
    return (
        f"<persisted-output>\n"
        f"Output too large ({kb:.1f} KB, {lines} lines). "
        f"Full output saved to:\n"
        f"{file_path}\n\n"
        f"Use read_file (tail or offset/limit) or grep on that path for the rest.\n\n"
        f"Preview ({PREVIEW_CHARS} chars, 40%head+60%tail):\n"
        f"{_preview(content)}"
        f"\n</persisted-output>"
    )


def _build_truncated_message(content: str) -> str:
    """Preview only, for a session with no way to read a saved copy.

    Naming the missing capability matters: the model's next move should be to
    narrow the call, not to look for a file, and not to assume it saw
    everything.
    """
    return (
        f"<truncated-output>\n"
        f"Output too large ({_size_kb(content):.1f} KB) and no tool in this session can read "
        f"a saved copy, so it was truncated.\n\n"
        f"Preview ({PREVIEW_CHARS} chars, 40%head+60%tail):\n"
        f"{_preview(content)}\n\n"
        f"Narrow the call (a filter, fewer items, a smaller range) if you need the rest."
        f"\n</truncated-output>"
    )


def _shrink_one_result(
    tool_use_id: str,
    content: str,
    *,
    session_id: str,
    cwd: Optional[str],
    user_id: Optional[str],
    recoverable: bool,
) -> str:
    """Shrink one oversized result to a bounded message.

    Spills to disk and hands over the path only when the session can read it
    back; otherwise nothing is written — a file no one can open is landfill,
    and the debug log already records that a truncation happened.
    """
    redacted = redact_sensitive_text(content)
    if recoverable:
        file_path = get_tool_result_path(
            tool_use_id, cwd=cwd, session_id=session_id, user_id=user_id,
        )
        if _persist_to_disk(file_path, redacted):
            return _build_persisted_message(file_path, redacted)
    return _build_truncated_message(redacted)


def _persist_to_disk(file_path: str, content: str) -> bool:
    """Write content to disk. Returns True on success."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        Path(file_path).write_text(content, encoding="utf-8")
        return True
    except OSError as e:
        logger.warning(f"Failed to persist tool result to {file_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Layer 1: Per-tool persistence
# ---------------------------------------------------------------------------

def maybe_persist_result(
    tool_name: str,
    tool_use_id: str,
    content: str,
    session_id: str = "default",
    cwd: Optional[str] = None,
    max_result_size_chars: Optional[int] = DEFAULT_MAX_RESULT_SIZE_CHARS,
    user_id: Optional[str] = None,
    recoverable: bool = True,
) -> str:
    """Shrink one oversized result at the moment it is produced.

    Args:
        tool_name:              Name of the tool that produced the result.
        tool_use_id:            Unique call ID (used as filename).
        content:                Full tool output string.
        session_id:             Session identifier for directory isolation.
        cwd:                    Project working directory (for path generation).
        max_result_size_chars:  Threshold in chars. None = never shrink.
        recoverable:            Whether the session has a tool that could read a
                                spilled copy back (see ``can_recover_spill``).
                                False truncates instead of writing a file.

    Returns:
        Original content (if under threshold) or a bounded preview.
        Never raises: a failed shrink still returns a truncated preview, so a
        50 MB ``execute`` dump cannot survive into the next model request.
    """
    if max_result_size_chars is None:
        return content
    if _already_shrunk(content):
        return content
    try:
        return _maybe_persist_result_inner(
            tool_name, tool_use_id, content,
            session_id=session_id, cwd=cwd,
            max_result_size_chars=max_result_size_chars,
            user_id=user_id, recoverable=recoverable,
        )
    except Exception as persist_err:
        logger.warning(f"Tool result persistence failed: {persist_err}")
        return _build_truncated_message(content[: PREVIEW_CHARS * 3])


def _maybe_persist_result_inner(
    tool_name: str,
    tool_use_id: str,
    content: str,
    *,
    session_id: str,
    cwd: Optional[str],
    max_result_size_chars: int,
    user_id: Optional[str],
    recoverable: bool,
) -> str:
    """Shrink one oversized result. Caller wraps this so it cannot leak."""
    # ── Classify first: image/binary should never sit raw in context, even
    #    when under the size threshold (a 5 KB base64 image is still noise). ──
    from agentica.compression.tool_result_classification import (
        classify_tool_result, describe_media, ToolResultClass,
    )
    cls = classify_tool_result(content, large_threshold=max_result_size_chars)
    if cls in (ToolResultClass.IMAGE, ToolResultClass.BINARY) and len(content) > PREVIEW_CHARS:
        descriptor = describe_media(content, cls)
        if not recoverable:
            return descriptor
        file_path = get_tool_result_path(
            tool_use_id, cwd=cwd, session_id=session_id, user_id=user_id,
        )
        if _persist_to_disk(file_path, content):
            logger.debug(f"Persisted {cls.value} {tool_name} result to {file_path}")
            return f"{descriptor}\nFull {cls.value} output saved to: {file_path}"
        return descriptor

    if len(content) <= max_result_size_chars:
        return content

    logger.debug(
        f"Layer 0: {tool_name} result is {len(content):,} chars "
        f"(recoverable={recoverable})"
    )
    return _shrink_one_result(
        tool_use_id,
        content,
        session_id=session_id,
        cwd=cwd,
        user_id=user_id,
        recoverable=recoverable,
    )


# ---------------------------------------------------------------------------
# Per-batch budget enforcement
# ---------------------------------------------------------------------------

def enforce_tool_batch_budget(
    tool_results: List["Message"],
    *,
    context_window: int,
    model_id: str = "gpt-4o",
    session_id: str = "default",
    cwd: Optional[str] = None,
    user_id: Optional[str] = None,
    recoverable: bool = True,
    budget_ratio: float = TOOL_BATCH_BUDGET_RATIO,
) -> int:
    """Bound one turn's fresh tool results to a share of the context window.

    Called where the batch is produced, so it sees plain result messages
    whatever the provider will later pack them into. Shrinks the largest
    results first, in place, until the batch fits.

    Args:
        tool_results:   Result messages from one assistant turn.
        context_window: Model window. Zero disables the check — without a
                        window there is no way to tell a big batch from a
                        batch this model has ample room for.
        model_id:       Tokenizer selection.
        recoverable:    See ``can_recover_spill``. Decides whether a shrunk
                        result keeps a readable path or is simply truncated.

    Returns:
        Number of results shrunk by this call.
    """
    if not tool_results or context_window <= 0:
        return 0
    budget = int(context_window * budget_ratio)
    if budget <= 0:
        return 0

    # An already-shrunk result is bounded, and re-shrinking a persisted one
    # would throw away the path that is the only handle on its full content.
    sizes = []
    for msg in tool_results:
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        already_shrunk = _already_shrunk(content)
        sizes.append((count_text_tokens(content, model_id), already_shrunk))

    total = sum(t for t, _ in sizes)
    if total <= budget:
        return 0

    candidates = [(i, sizes[i][0]) for i in range(len(tool_results)) if not sizes[i][1]]
    candidates.sort(key=lambda x: x[1], reverse=True)

    shrunk_count = 0
    for idx, tokens in candidates:
        if total <= budget:
            break
        msg = tool_results[idx]
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        new_content = _shrink_one_result(
            msg.tool_call_id or f"batch_{idx}",
            content,
            session_id=session_id,
            cwd=cwd,
            user_id=user_id,
            recoverable=recoverable,
        )
        msg.content = new_content
        total -= tokens - count_text_tokens(new_content, model_id)
        shrunk_count += 1

    if shrunk_count:
        logger.debug(
            f"Layer 0 batch budget: shrunk {shrunk_count} tool result(s), "
            f"batch now ~{total:,} tokens (budget={budget:,}, recoverable={recoverable})"
        )
    return shrunk_count
