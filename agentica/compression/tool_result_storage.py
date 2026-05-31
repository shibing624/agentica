# -*- coding: utf-8 -*-
"""
@description: Tool result storage - persist large tool outputs to disk.

When a tool result exceeds a threshold, the full content is saved to disk
and the context message is replaced with a short preview + file path.

Path structure (mirrors CC's toolResultStorage.ts):
    ~/.agentica/projects/<sanitized-cwd>/<session-id>/tool-results/<tool-use-id>.txt

Two-layer budget:
    1. Per-tool: single result > DEFAULT_MAX_RESULT_SIZE_CHARS -> persist
    2. Per-message: all tool_results in one message > MAX_TOOL_RESULTS_PER_MESSAGE_CHARS
       -> persist the largest ones until under budget

Usage (automatic - called from Model.run_function_calls):
    from agentica.compression.tool_result_storage import maybe_persist_result
    content = maybe_persist_result(
        tool_name="execute", tool_use_id="call_abc123",
        content=huge_bash_output, session_id="sess_xyz",
    )
"""
import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from agentica.config import AGENTICA_PROJECTS_DIR
from agentica.security.redact import redact_sensitive_text
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.model.message import Message

# ---------------------------------------------------------------------------
# Constants (aligned with CC's toolLimits.ts)
# ---------------------------------------------------------------------------

# Max chars to keep inline in the context (preview)
PREVIEW_CHARS = 2000

# Default max result size before persisting to disk (single tool).
# Individual tools can override via Function.max_result_size_chars.
DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000

# Per-message budget: total chars across all tool_results in one assistant response.
# When exceeded, the largest fresh results are persisted until under budget.
MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_MAX_SANITIZED_LENGTH = 200


def _sanitize_path(raw: str) -> str:
    """Convert a filesystem path into a readable, safe directory name (mirrors CC's sanitizePath).

    Long paths (>200 chars): truncate + md5 hash suffix for uniqueness.
    """
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', raw)
    if len(sanitized) <= _MAX_SANITIZED_LENGTH:
        return sanitized
    hash_suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{sanitized[:_MAX_SANITIZED_LENGTH]}-{hash_suffix}"


def _safe_user_segment(user_id: Optional[str]) -> str:
    """Path segment for the per-user spill directory.

    Delegates to ``Workspace.sanitize_user_id`` so persisted tool-result
    paths line up with ``users/{user_id}/`` exactly, including the
    "default" sentinel for None / blank input.
    """
    from agentica.workspace import Workspace
    return Workspace.sanitize_user_id(user_id)


def get_project_dir(cwd: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Return ``<AGENTICA_PROJECTS_DIR>/<user>/<sanitized-cwd>/`` for the given user + cwd.

    The ``user_id`` segment was added to prevent multi-tenant collisions:
    two requests from different tenants that happen to share the same cwd
    (e.g. both running from ``/tmp`` in a worker) used to spill to the same
    directory and read each other's persisted tool outputs.
    """
    cwd = cwd or os.getcwd()
    return os.path.join(AGENTICA_PROJECTS_DIR, _safe_user_segment(user_id), _sanitize_path(cwd))


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

def _build_persisted_message(file_path: str, content: str) -> str:
    """Build the preview message returned to the model after persisting.

    Uses 40% head + 60% tail strategy to preserve both early context
    (command echo, headers) and final results (exit codes, summaries).
    """
    max_preview = PREVIEW_CHARS
    size_kb = len(content.encode("utf-8", errors="ignore")) / 1024

    if len(content) <= max_preview:
        preview = content
    else:
        # 40% head + 60% tail — preserves both early context and final results
        head_chars = int(max_preview * 0.4)
        tail_chars = max_preview - head_chars
        omitted = len(content) - head_chars - tail_chars
        preview = (
            content[:head_chars]
            + f"\n\n... [{omitted} chars omitted] ...\n\n"
            + content[-tail_chars:]
        )

    msg = (
        f"<persisted-output>\n"
        f"Output too large ({size_kb:.1f} KB). Full output saved to:\n"
        f"{file_path}\n\n"
        f"Preview ({max_preview} chars, 40%head+60%tail):\n"
        f"{preview}"
        f"\n</persisted-output>"
    )
    return msg


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
) -> str:
    """If content exceeds threshold, persist to disk and return preview.

    Args:
        tool_name:              Name of the tool that produced the result.
        tool_use_id:            Unique call ID (used as filename).
        content:                Full tool output string.
        session_id:             Session identifier for directory isolation.
        cwd:                    Project working directory (for path generation).
        max_result_size_chars:  Threshold in chars. None = never persist.

    Returns:
        Original content (if under threshold) or preview + disk path.
    """
    if max_result_size_chars is None:
        return content

    # ── Classify first: image/binary should never sit raw in context, even
    #    when under the size threshold (a 5 KB base64 image is still noise). ──
    from agentica.compression.tool_result_classification import (
        classify_tool_result, describe_media, ToolResultClass,
    )
    cls = classify_tool_result(content, large_threshold=max_result_size_chars)
    if cls in (ToolResultClass.IMAGE, ToolResultClass.BINARY) and len(content) > PREVIEW_CHARS:
        descriptor = describe_media(content, cls)
        file_path = get_tool_result_path(
            tool_use_id, cwd=cwd, session_id=session_id, user_id=user_id,
        )
        if _persist_to_disk(file_path, content):
            logger.debug(f"Persisted {cls.value} {tool_name} result to {file_path}")
            return f"{descriptor}\nFull {cls.value} output saved to: {file_path}"
        return descriptor

    if len(content) <= max_result_size_chars:
        return content

    redacted_content = redact_sensitive_text(content)
    file_path = get_tool_result_path(
        tool_use_id, cwd=cwd, session_id=session_id, user_id=user_id,
    )
    if not _persist_to_disk(file_path, redacted_content):
        # Fallback: truncate in-place
        return redacted_content[:max_result_size_chars] + "\n... (output truncated)"

    logger.debug(
        f"Persisted {tool_name} result ({len(content):,} chars) to {file_path}"
    )
    return _build_persisted_message(file_path, redacted_content)


# ---------------------------------------------------------------------------
# Layer 2: Per-message budget enforcement
# ---------------------------------------------------------------------------

def enforce_tool_result_budget(
    tool_results: List["Message"],
    session_id: str = "default",
    cwd: Optional[str] = None,
    budget: int = MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
    user_id: Optional[str] = None,
) -> int:
    """Enforce per-message budget on a batch of tool result messages.

    If the total chars across all tool_results exceed `budget`, persist the
    largest results to disk (biggest first) until total is under budget.
    Modifies messages in-place.

    Args:
        tool_results: List of tool result Message objects from one assistant turn.
        session_id:   Session ID for path generation.
        cwd:          Project working directory.
        budget:       Max total chars allowed across all tool results.

    Returns:
        Number of results that were persisted by this call.
    """
    if not tool_results or budget <= 0:
        return 0

    # Compute sizes, skip already-persisted results
    sizes = []
    for msg in tool_results:
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        # Already persisted results contain the <persisted-output> tag
        already_persisted = "<persisted-output>" in content
        sizes.append((len(content), already_persisted))

    total = sum(s for s, _ in sizes)
    if total <= budget:
        return 0

    # Sort indices by size descending, skip already-persisted
    candidates = [
        (i, sizes[i][0])
        for i in range(len(tool_results))
        if not sizes[i][1]
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    persisted_count = 0
    for idx, size in candidates:
        if total <= budget:
            break
        msg = tool_results[idx]
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        tool_use_id = msg.tool_call_id or f"budget_{idx}"
        file_path = get_tool_result_path(
            tool_use_id, cwd=cwd, session_id=session_id, user_id=user_id,
        )

        redacted_content = redact_sensitive_text(content)
        if _persist_to_disk(file_path, redacted_content):
            new_content = _build_persisted_message(file_path, redacted_content)
            msg.content = new_content
            saved = size - len(new_content)
            total -= saved
            persisted_count += 1
            logger.debug(
                f"Budget enforcement: persisted tool result [{idx}] "
                f"({size:,} -> {len(new_content):,} chars, saved {saved:,})"
            )

    if persisted_count:
        logger.debug(
            f"Budget enforcement: persisted {persisted_count} tool results, "
            f"total now {total:,} chars (budget={budget:,})"
        )
    return persisted_count
