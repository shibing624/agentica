# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: V4A patch apply helpers used by BuiltinFileTool.apply_patch.

Context matching is exact: a hunk must match the file byte-for-byte aside
from the required `` `` / ``-`` / ``+`` prefixes. Whitespace and quotes are
not rewritten.
"""
import re
from typing import List, Literal
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

    def render(self) -> str:
        location = "EOF context" if self.eof else "context"
        return f"Hunk {self.hunk_number}: {location} not found."


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
END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]


_FILE_MARKER_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")


def parse_patch_envelope(patch: str) -> List[FilePatch]:
    """Parse a strict multi-file ``*** Begin Patch`` envelope.

    If the first line is already ``*** Update/Add/Delete File:``, the
    Begin/End markers are inserted. Other omitted-envelope shapes are
    rejected. Per-file update bodies are validated later against current
    file content by ``apply_diff`` before any filesystem mutation occurs.
    """
    normalized = patch.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    lines = normalized.split("\n") if normalized else []
    # Models often emit a valid body starting at "*** Update File:" and
    # omit the envelope. That is the same grammar, not a second one.
    if lines and _FILE_MARKER_RE.fullmatch(lines[0]):
        if lines[-1] != END_PATCH:
            lines.append(END_PATCH)
        lines.insert(0, "*** Begin Patch")
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != END_PATCH:
        raise ValueError(
            "Patch must start with '*** Begin Patch' and end with '*** End Patch'."
        )

    operations: List[FilePatch] = []
    seen_paths = set()
    index = 1
    while index < len(lines) - 1:
        marker = _FILE_MARKER_RE.fullmatch(lines[index])
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
        while index < len(lines) - 1 and _FILE_MARKER_RE.fullmatch(lines[index]) is None:
            if lines[index] == "*** Begin Patch":
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


def _parse_update_diff(lines: List[str], input_text: str) -> ParsedUpdateDiff:
    """Parse an update diff with context hunks."""
    parser = ParserState(lines=[*lines, END_PATCH])
    input_lines = input_text.split("\n")
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
                ContextFailure(hunk_number=hunk_number, eof=section.eof)
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
