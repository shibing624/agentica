# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: V4A patch envelope format shared by tools, agent core, and CLI.

Pure string-level library: parses the ``*** Begin Patch`` envelope into
FilePatch operations and applies update hunks to text. No disk I/O —
BuiltinFileTool.apply_patch adds path resolution, sandbox checks, locking,
atomic writes, and diagnostics on top. agent.approvals and the CLI display
reuse the parser to list which files a patch touches.

The envelope is extracted the way Codex (lenient) and OpenCode do: find
Begin/End anywhere, unwrap a markdown fence or heredoc, and wrap a bare
``*** Update/Add/Delete File:`` body. Hunk context matching stays exact.
"""
import json
import re
from typing import List, Literal, Optional
from dataclasses import dataclass


ApplyDiffMode = Literal["default", "create"]


@dataclass
class Chunk:
    """Represents a chunk of changes in a diff."""
    orig_index: int
    del_lines: List[str]
    ins_lines: List[str]


@dataclass
class ParsedUpdateDiff:
    """Result of parsing an update diff."""
    chunks: List[Chunk]
    fuzz: int


@dataclass
class ReadSectionResult:
    """Result of reading a section from a diff."""
    next_context: List[str]
    section_chunks: List[Chunk]
    end_index: int
    eof: bool


@dataclass
class ParserState:
    """State of the diff parser."""
    lines: List[str]
    index: int = 0
    fuzz: int = 0


@dataclass
class ContextMatch:
    """Result of finding context in source."""
    new_index: int
    fuzz: int


@dataclass(frozen=True)
class ContextFailure:
    """One update hunk whose context was not found in the current file."""

    hunk_number: int
    eof: bool = False
    unmatched: str = ""
    contiguous: bool = False

    def render(self) -> str:
        location = "EOF context" if self.eof else "context"
        prefix = f"Hunk {self.hunk_number}: {location} not found"
        if self.contiguous:
            return f"{prefix} as a contiguous block{self.unmatched}"
        if not self.unmatched:
            return f"{prefix}."
        tip = self.unmatched if len(self.unmatched) <= 120 else self.unmatched[:119] + "…"
        return f"{prefix}: {tip!r}."


class PatchContextError(ValueError):
    """All context mismatches found while preflighting one file patch."""

    def __init__(self, failures: List[ContextFailure]):
        self.failures = tuple(failures)
        super().__init__("\n".join(failure.render() for failure in self.failures))


class PatchNoChangeError(ValueError):
    """An update hunk matched the file but would write the same bytes back."""


PatchAction = Literal["add", "update", "delete"]


@dataclass(frozen=True)
class FilePatch:
    """One file operation parsed from an apply_patch envelope."""

    action: PatchAction
    path: str
    diff: str = ""


# V4A diff markers
BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]


_FILE_MARKER_RE = re.compile(
    r"^\*\*\*\s*(Add|Update|Delete)\s+File:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_BEGIN_RE = re.compile(r"^\*\*\*\s*Begin Patch\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\*\*\*\s*End Patch\s*$", re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:patch|diff|apply_patch)?\s*$", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_HEREDOC_OPEN_RE = re.compile(r"^<<(['\"]?)(\w+)\1\s*$")


def _unwrap_json_patch(text: str) -> str:
    """Peel ``{"patch": "..."}`` / ``{"command":["apply_patch","..."]}`` wrappers."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    try:
        data = json.loads(stripped)
    except ValueError:
        return text
    if not isinstance(data, dict):
        return text
    inner = data.get("patch")
    if isinstance(inner, str) and inner.strip():
        return inner
    command = data.get("command")
    if (
        isinstance(command, list)
        and len(command) >= 2
        and "apply_patch" in str(command[0])
        and isinstance(command[1], str)
    ):
        return command[1]
    return text


def _strip_markdown_fence(text: str) -> str:
    lines = text.strip("\n").split("\n")
    start = 0
    end = len(lines) - 1
    while start <= end and not lines[start].strip():
        start += 1
    while end >= start and not lines[end].strip():
        end -= 1
    if start >= end:
        return text
    if _FENCE_OPEN_RE.fullmatch(lines[start].strip()) and _FENCE_CLOSE_RE.fullmatch(
        lines[end].strip()
    ):
        return "\n".join(lines[start + 1:end])
    return text


def _strip_heredoc(text: str) -> str:
    """Codex lenient mode: ``<<EOF`` / ``<<'EOF'`` … ``EOF`` around the patch."""
    lines = text.strip("\n").split("\n")
    if len(lines) < 4:
        return text
    opener = _HEREDOC_OPEN_RE.fullmatch(lines[0].strip())
    if opener is None:
        return text
    token = opener.group(2)
    if lines[-1].strip() != token:
        return text
    return "\n".join(lines[1:-1])


def _file_marker(line: str) -> Optional[re.Match]:
    return _FILE_MARKER_RE.fullmatch(line.rstrip())


def _extract_envelope_lines(text: str) -> List[str]:
    """Return canonical ``Begin … End`` lines, or raise a model-facing error."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    raw = _unwrap_json_patch(raw)
    raw = _strip_markdown_fence(raw)
    raw = _strip_heredoc(raw)
    lines = raw.strip("\n").split("\n") if raw.strip() else []

    begin_idx = next((i for i, line in enumerate(lines) if _BEGIN_RE.fullmatch(line.strip())), None)
    end_idx = None
    if begin_idx is not None:
        for i in range(len(lines) - 1, begin_idx, -1):
            if _END_RE.fullmatch(lines[i].strip()):
                end_idx = i
                break
        inner = lines[begin_idx + 1:end_idx] if end_idx is not None else lines[begin_idx + 1:]
    else:
        file_idx = next((i for i, line in enumerate(lines) if _file_marker(line)), None)
        if file_idx is None:
            first = lines[0] if lines else ""
            raise ValueError(
                "Patch must contain '*** Begin Patch' / '*** End Patch', "
                "or start with '*** Update/Add/Delete File:'. "
                f"Got first line: {first!r}."
            )
        inner = lines[file_idx:]
        if inner and _END_RE.fullmatch(inner[-1].strip()):
            inner = inner[:-1]

    while inner and _file_marker(inner[0]) is None:
        inner = inner[1:]

    if not inner or _file_marker(inner[0]) is None:
        raise ValueError(
            "Patch contains no '*** Update/Add/Delete File:' operation."
        )
    return [BEGIN_PATCH, *inner, END_PATCH]


def parse_patch_envelope(patch: str) -> List[FilePatch]:
    """Parse a multi-file ``*** Begin Patch`` envelope.

    Envelope wrapping is lenient (fence / heredoc / preamble / omitted
    Begin-End around a File header). Hunk bodies stay exact. Per-file
    update bodies are validated later against current file content by
    ``apply_diff`` before any filesystem mutation occurs.
    """
    lines = _extract_envelope_lines(patch)

    operations: List[FilePatch] = []
    seen_paths = set()
    index = 1
    while index < len(lines) - 1:
        while index < len(lines) - 1 and not lines[index].strip():
            index += 1
        if index >= len(lines) - 1:
            break
        marker = _file_marker(lines[index])
        if marker is None:
            raise ValueError(f"Invalid patch line {index + 1}: {lines[index]!r}")

        action = marker.group(1).lower()
        path = marker.group(2).strip()
        if not path:
            raise ValueError(f"Missing file path on patch line {index + 1}.")
        if path in seen_paths:
            raise ValueError(f"Duplicate file operation for {path!r}.")
        seen_paths.add(path)

        index += 1
        body_start = index
        while index < len(lines) - 1 and _file_marker(lines[index]) is None:
            if _BEGIN_RE.fullmatch(lines[index].strip()):
                raise ValueError(f"Unexpected nested patch marker on line {index + 1}.")
            index += 1
        body = "\n".join(lines[body_start:index])

        if action == "delete":
            if body:
                raise ValueError(f"Delete File {path!r} must not contain a diff body.")
        elif not body:
            raise ValueError(f"{action.title()} File {path!r} requires a diff body.")

        operations.append(FilePatch(action=action, path=path, diff=body))

    if not operations:
        raise ValueError("Patch contains no file operations.")
    return operations


def apply_diff(input_text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    """Apply a V4A diff to the provided text.

    This parser understands both the create-file syntax (only "+" prefixed
    lines) and the default update syntax that includes context hunks.

    Args:
        input_text: The original text content.
        diff: The diff to apply in V4A format.
        mode: "create" for new files, "default" for updates.

    Returns:
        The patched text content.
    """
    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines)

    parsed = _parse_update_diff(diff_lines, input_text)
    return _apply_chunks(input_text, parsed.chunks)


def _normalize_diff_lines(diff: str) -> List[str]:
    """Normalize diff lines by removing carriage returns."""
    lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _is_done(state: ParserState, prefixes: List[str]) -> bool:
    """Check if parser is done processing."""
    if state.index >= len(state.lines):
        return True
    if any(state.lines[state.index].startswith(prefix) for prefix in prefixes):
        return True
    return False


def _read_str(state: ParserState, prefix: str) -> str:
    """Read a string with the given prefix from current position."""
    if state.index >= len(state.lines):
        return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix):]
    return ""


def _parse_create_diff(lines: List[str]) -> str:
    """Parse a create-file diff (all lines start with +)."""
    parser = ParserState(lines=[*lines, END_PATCH])
    output: List[str] = []

    while not _is_done(parser, SECTION_TERMINATORS):
        if parser.index >= len(parser.lines):
            break
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"):
            raise ValueError(f"Invalid Add File Line: {line}")
        output.append(line[1:])

    return "\n".join(output)


def _trim_context_tip(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _contiguous_break_note(
    context: List[str],
    input_lines: List[str],
    start: int,
) -> str:
    """Where a hunk whose lines all exist fails to match as one block."""
    if not context:
        return ""
    first = context[0]
    search_from = start
    if not any(line == first for line in input_lines[start:]):
        search_from = 0
    for i in range(search_from, len(input_lines)):
        if input_lines[i] != first:
            continue
        for j, expected in enumerate(context):
            pos = i + j
            actual = input_lines[pos] if pos < len(input_lines) else None
            if actual != expected:
                got = "EOF" if actual is None else repr(_trim_context_tip(actual))
                want = repr(_trim_context_tip(expected))
                return f" (line {i + 1}). file: {got}, hunk: {want}."
        break
    return ""


def _context_failure(
    hunk_number: int,
    eof: bool,
    context: List[str],
    input_lines: List[str],
    start: int,
) -> ContextFailure:
    missing = ""
    present = set(input_lines)
    for line in context:
        if line not in present:
            missing = line
            break
    if missing:
        return ContextFailure(hunk_number=hunk_number, eof=eof, unmatched=missing)
    note = _contiguous_break_note(context, input_lines, start)
    if note:
        return ContextFailure(
            hunk_number=hunk_number, eof=eof, unmatched=note, contiguous=True
        )
    return ContextFailure(
        hunk_number=hunk_number,
        eof=eof,
        unmatched=context[0] if context else "",
    )


def _recover_unprefixed_keep_lines(
    diff_lines: List[str],
    input_lines: List[str],
) -> List[str]:
    """Treat an unprefixed line as keep when it uniquely matches the file.

    Exact file line → keep as written. Else one distinct line that matches
    after rstrip → keep using the file's bytes. Ambiguous or not in the
    file stays unprefixed so ``_read_section`` still raises Malformed.
    """
    exact = set(input_lines)
    rstrip_originals: dict[str, List[str]] = {}
    for line in input_lines:
        key = line.rstrip()
        seen = rstrip_originals.setdefault(key, [])
        if line not in seen:
            seen.append(line)

    recovered: List[str] = []
    for raw in diff_lines:
        if (
            not raw
            or raw.startswith(("+", "-", " ", "@@"))
            or raw.startswith("***")
        ):
            recovered.append(raw)
            continue
        if raw in exact:
            recovered.append(" " + raw)
            continue
        originals = rstrip_originals.get(raw.rstrip(), [])
        if len(originals) == 1:
            recovered.append(" " + originals[0])
            continue
        recovered.append(raw)
    return recovered


def _parse_update_diff(lines: List[str], input_text: str) -> ParsedUpdateDiff:
    """Parse an update diff with context hunks."""
    input_lines = input_text.split("\n")
    lines = _recover_unprefixed_keep_lines(lines, input_lines)
    parser = ParserState(lines=[*lines, END_PATCH])
    chunks: List[Chunk] = []
    failures: List[ContextFailure] = []
    cursor = 0
    hunk_number = 0

    while not _is_done(parser, END_SECTION_MARKERS):
        hunk_number += 1
        anchor = _read_str(parser, "@@ ")
        has_bare_anchor = (
            anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@"
        )
        if has_bare_anchor:
            parser.index += 1

        if not (anchor or has_bare_anchor or cursor == 0):
            current_line = parser.lines[parser.index] if parser.index < len(parser.lines) else ""
            raise ValueError(
                "Malformed patch: each hunk after the first must start with @@; "
                f"got {current_line!r}"
            )

        if anchor.strip():
            cursor = _advance_cursor_to_anchor(anchor, input_lines, cursor)

        section = _read_section(parser.lines, parser.index)
        find_result = _find_context(input_lines, section.next_context, cursor, section.eof)
        parser.index = section.end_index
        if find_result.new_index == -1:
            failures.append(
                _context_failure(
                    hunk_number, section.eof, section.next_context, input_lines, cursor
                )
            )
            continue

        cursor = find_result.new_index + len(section.next_context)
        parser.fuzz += find_result.fuzz

        for ch in section.section_chunks:
            chunks.append(
                Chunk(
                    orig_index=ch.orig_index + find_result.new_index,
                    del_lines=list(ch.del_lines),
                    ins_lines=list(ch.ins_lines),
                )
            )

    if failures:
        raise PatchContextError(failures)
    return ParsedUpdateDiff(chunks=chunks, fuzz=parser.fuzz)


def _advance_cursor_to_anchor(
    anchor: str,
    input_lines: List[str],
    cursor: int,
) -> int:
    """Advance cursor to the first exact match of the @@ anchor line."""
    if any(line == anchor for line in input_lines[:cursor]):
        return cursor
    for i in range(cursor, len(input_lines)):
        if input_lines[i] == anchor:
            return i + 1
    return cursor


def _read_section(lines: List[str], start_index: int) -> ReadSectionResult:
    """Read a section of the diff."""
    context: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    section_chunks: List[Chunk] = []
    mode: Literal["keep", "add", "delete"] = "keep"
    index = start_index
    orig_index = index

    while index < len(lines):
        raw = lines[index]
        if (
            raw.startswith("@@")
            or raw.startswith(END_PATCH)
            or raw.startswith("*** Update File:")
            or raw.startswith("*** Delete File:")
            or raw.startswith("*** Add File:")
            or raw.startswith(END_FILE)
        ):
            break
        if raw == "***":
            break
        if raw.startswith("***"):
            raise ValueError(f"Invalid Line: {raw}")

        index += 1
        last_mode = mode
        line = raw if raw else " "
        prefix = line[0]
        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            raise ValueError(
                "Malformed patch: each line after @@ must start with "
                f"' ' (keep), '-' (delete), or '+' (add); got {line!r}. "
                "Keep lines need a leading space."
            )

        line_content = line[1:]
        switching_to_context = mode == "keep" and last_mode != mode
        if switching_to_context and (del_lines or ins_lines):
            section_chunks.append(
                Chunk(
                    orig_index=len(context) - len(del_lines),
                    del_lines=list(del_lines),
                    ins_lines=list(ins_lines),
                )
            )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(line_content)
            context.append(line_content)
        elif mode == "add":
            ins_lines.append(line_content)
        else:
            context.append(line_content)

    if del_lines or ins_lines:
        section_chunks.append(
            Chunk(
                orig_index=len(context) - len(del_lines),
                del_lines=list(del_lines),
                ins_lines=list(ins_lines),
            )
        )

    if index < len(lines) and lines[index] == END_FILE:
        return ReadSectionResult(context, section_chunks, index + 1, True)

    if index == orig_index:
        next_line = lines[index] if index < len(lines) else ""
        raise ValueError(f"Nothing in this section - index={index} {next_line}")

    return ReadSectionResult(context, section_chunks, index, False)


def _find_context(lines: List[str], context: List[str], start: int, eof: bool) -> ContextMatch:
    """Find context lines in the source."""
    if eof:
        end_start = max(0, len(lines) - len(context))
        end_match = _find_context_core(lines, context, end_start)
        if end_match.new_index != -1:
            return end_match
        fallback = _find_context_core(lines, context, start)
        return ContextMatch(new_index=fallback.new_index, fuzz=fallback.fuzz + 10000)
    return _find_context_core(lines, context, start)


def _find_context_core(lines: List[str], context: List[str], start: int) -> ContextMatch:
    """Find an exact context match. No whitespace or quote rewriting."""
    if not context:
        return ContextMatch(new_index=start, fuzz=0)

    n = len(context)
    for i in range(start, len(lines) - n + 1):
        if lines[i:i + n] == context:
            return ContextMatch(new_index=i, fuzz=0)
    return ContextMatch(new_index=-1, fuzz=0)


def _apply_chunks(input_text: str, chunks: List[Chunk]) -> str:
    """Apply parsed chunks to the input text."""
    orig_lines = input_text.split("\n")
    dest_lines: List[str] = []
    cursor = 0

    for chunk in chunks:
        if chunk.orig_index > len(orig_lines):
            raise ValueError(
                f"applyDiff: chunk.origIndex {chunk.orig_index} > input length {len(orig_lines)}"
            )
        if cursor > chunk.orig_index:
            raise ValueError(
                f"applyDiff: overlapping chunk at {chunk.orig_index} (cursor {cursor})"
            )

        dest_lines.extend(orig_lines[cursor:chunk.orig_index])
        cursor = chunk.orig_index

        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)

        cursor += len(chunk.del_lines)

    dest_lines.extend(orig_lines[cursor:])
    return "\n".join(dest_lines)
