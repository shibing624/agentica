# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Classify tool results to drive smarter context handling.

Pure size-based truncation is blunt: a 30 KB base64 image and a 30 KB log file
get the same treatment, and a small inline image still floods the model context.
Classifying the result first lets the storage layer decide what to keep inline,
what to divert to disk, and what to replace with a compact descriptor.

Classes:
    NORMAL  - plain text, keep inline
    LARGE   - big text, persist to disk + preview (handled by storage layer)
    BINARY  - contains NUL / mostly non-printable bytes -> never inline raw
    IMAGE   - data:image/... URI or large bare base64 -> replace with a note
    ERROR   - tool failed (caller-supplied flag)
    EMPTY   - no meaningful content
"""
import re
from enum import Enum
from typing import Optional


class ToolResultClass(str, Enum):
    NORMAL = "normal"
    LARGE = "large"
    BINARY = "binary"
    IMAGE = "image"
    ERROR = "error"
    EMPTY = "empty"


# A bare base64 blob this long (with no whitespace) is almost certainly encoded
# media/binary rather than something the model benefits from reading verbatim.
_BASE64_BLOB_MIN = 1024
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
_DATA_IMAGE_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _looks_binary(content: str) -> bool:
    """Heuristic: a high share of control / non-printable characters.

    NUL is counted toward the control-char ratio below, but is deliberately
    NOT an instant "binary" trigger on its own. A single stray NUL inside
    otherwise-readable text (e.g. a shell command whose stdout echoed a
    literal ``\\x00``, or a ``tail`` of a log that happens to contain one)
    must not flip a 13 KB text output to BINARY and divert it to disk as an
    opaque ``<binary .../>`` note. Genuine binary content carries a *dense*
    run of control bytes and still trips the 30% ratio.
    """
    sample = content[:4096]
    if not sample:
        return False
    nonprintable = sum(1 for c in sample if ord(c) < 9 or (13 < ord(c) < 32))
    return nonprintable / len(sample) > 0.3


def _looks_like_bare_base64(content: str) -> bool:
    """Detect a large, contiguous base64 blob (encoded media/binary).

    Deliberately conservative to avoid misclassifying repetitive or long plain
    text: requires a single whitespace-free token, base64 alphabet, length a
    multiple of 4, and high character diversity (real base64 of binary data).
    """
    stripped = content.strip()
    if len(stripped) < _BASE64_BLOB_MIN:
        return False
    if any(c.isspace() for c in stripped):
        return False
    if len(stripped) % 4 != 0:
        return False
    if not _BASE64_RE.match(stripped):
        return False
    # Repetitive plain text (e.g. "xxxx...") has very few distinct characters;
    # genuine base64-encoded binary uses most of the alphabet.
    return len(set(stripped[:512])) >= 16


def classify_tool_result(
    content: str,
    *,
    is_error: bool = False,
    large_threshold: Optional[int] = None,
) -> ToolResultClass:
    """Classify a tool result string.

    Args:
        content: The tool output.
        is_error: True if the tool call failed (short-circuits to ERROR).
        large_threshold: Char count above which plain text is LARGE.
    """
    if is_error:
        return ToolResultClass.ERROR
    if content is None or not content.strip():
        return ToolResultClass.EMPTY
    if _DATA_IMAGE_RE.search(content[:256]) or _looks_like_bare_base64(content):
        return ToolResultClass.IMAGE
    if _looks_binary(content):
        return ToolResultClass.BINARY
    if large_threshold is not None and len(content) > large_threshold:
        return ToolResultClass.LARGE
    return ToolResultClass.NORMAL


def describe_media(content: str, cls: ToolResultClass) -> str:
    """Build a compact, model-friendly descriptor for media/binary content.

    Keeps the context clean (no megabytes of base64) while telling the model
    what was produced and how big it was.
    """
    size_kb = len(content.encode("utf-8", errors="ignore")) / 1024
    if cls == ToolResultClass.IMAGE:
        mime = "image"
        m = _DATA_IMAGE_RE.search(content[:256])
        if m:
            mime = m.group(0).split("data:")[1].split(";")[0]
        return f"<media kind=\"{mime}\" size=\"{size_kb:.1f}KB\" note=\"image data omitted from context\"/>"
    return f"<binary size=\"{size_kb:.1f}KB\" note=\"binary data omitted from context\"/>"
