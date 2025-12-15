# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools for editing files - useful for code editors and AI coding assistants.

Supports V4A diff format (used by GPT-5.1) and unified diff format.
V4A Diff Parser (compatible with GPT-5.1 ApplyPatchTool)
"""
import os
import re
import difflib
from pathlib import Path
from typing import Optional, List, Literal, Callable
from dataclasses import dataclass

from agentica.tools.base import Tool
from agentica.utils.log import logger


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
    cursor = 0

    while not _is_done(parser, END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        has_bare_anchor = (
            anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@"
        )
        if has_bare_anchor:
            parser.index += 1

        if not (anchor or has_bare_anchor or cursor == 0):
            current_line = parser.lines[parser.index] if parser.index < len(parser.lines) else ""
            raise ValueError(f"Invalid Line:\n{current_line}")

        if anchor.strip():
            cursor = _advance_cursor_to_anchor(anchor, input_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        find_result = _find_context(input_lines, section.next_context, cursor, section.eof)
        if find_result.new_index == -1:
            ctx_text = "\n".join(section.next_context)
            if section.eof:
                raise ValueError(f"Invalid EOF Context {cursor}:\n{ctx_text}")
            raise ValueError(f"Invalid Context {cursor}:\n{ctx_text}")

        cursor = find_result.new_index + len(section.next_context)
        parser.fuzz += find_result.fuzz
        parser.index = section.end_index

        for ch in section.section_chunks:
            chunks.append(
                Chunk(
                    orig_index=ch.orig_index + find_result.new_index,
                    del_lines=list(ch.del_lines),
                    ins_lines=list(ch.ins_lines),
                )
            )

    return ParsedUpdateDiff(chunks=chunks, fuzz=parser.fuzz)


def _advance_cursor_to_anchor(
    anchor: str,
    input_lines: List[str],
    cursor: int,
    parser: ParserState,
) -> int:
    """Advance cursor to find the anchor line."""
    found = False

    if not any(line == anchor for line in input_lines[:cursor]):
        for i in range(cursor, len(input_lines)):
            if input_lines[i] == anchor:
                cursor = i + 1
                found = True
                break

    if not found and not any(line.strip() == anchor.strip() for line in input_lines[:cursor]):
        for i in range(cursor, len(input_lines)):
            if input_lines[i].strip() == anchor.strip():
                cursor = i + 1
                parser.fuzz += 1
                found = True
                break

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
            raise ValueError(f"Invalid Line: {line}")

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
    """Core context finding with fuzzy matching."""
    if not context:
        return ContextMatch(new_index=start, fuzz=0)

    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value):
            return ContextMatch(new_index=i, fuzz=0)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.rstrip()):
            return ContextMatch(new_index=i, fuzz=1)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.strip()):
            return ContextMatch(new_index=i, fuzz=100)

    return ContextMatch(new_index=-1, fuzz=0)


def _equals_slice(
    source: List[str], target: List[str], start: int, map_fn: Callable[[str], str]
) -> bool:
    """Check if a slice of source equals target after applying map_fn."""
    if start + len(target) > len(source):
        return False
    for offset, target_value in enumerate(target):
        if map_fn(source[start + offset]) != map_fn(target_value):
            return False
    return True


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


# ============================================================================
# EditTool Class
# ============================================================================

class EditTool(Tool):
    """
    A toolkit for editing files, particularly useful for code editing features.
    Provides functionality similar to modern code editors.

    Supports both V4A diff format (used by GPT-5.1) and unified diff format.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            edit_file: bool = True,
            apply_patch: bool = True,
            compare_files: bool = True,
            search_replace: bool = True,
    ):
        """
        Initialize the EditTool.

        Args:
            work_dir: The working directory for file operations. Defaults to current directory.
            edit_file: Whether to include the edit_file function.
            apply_patch: Whether to include the apply_patch function.
            compare_files: Whether to include the compare_files function.
            search_replace: Whether to include the search_replace function.
        """
        super().__init__(name="edit_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()

        if edit_file:
            self.register(self.edit_file)
        if apply_patch:
            self.register(self.apply_patch)
        if compare_files:
            self.register(self.compare_files)
        if search_replace:
            self.register(self.search_replace)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolves a file path, making it absolute if it's relative."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.work_dir.joinpath(path)
        return path

    def _detect_diff_format(self, patch_content: str) -> str:
        """Detect the format of the diff.

        Returns:
            "v4a" for V4A format, "unified" for unified diff format.
        """
        # Check for V4A wrapper markers
        if "*** Begin Patch" in patch_content or "*** Add File:" in patch_content or "*** Update File:" in patch_content:
            return "v4a"

        # Check for unified diff format: @@ -line,count +line,count @@
        if re.search(r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', patch_content):
            return "unified"

        # V4A uses bare @@ or @@ with context (not line numbers)
        # Check if @@ is followed by content lines (V4A style)
        lines = patch_content.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("@@"):
                # V4A: @@ alone or @@ followed by context text (not line numbers)
                if line == "@@" or (line.startswith("@@ ") and not re.match(r'@@ -\d+', line)):
                    return "v4a"

        # Check for V4A style lines (starting with +, -, or space after @@)
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if in_hunk and line and line[0] in "+- ":
                return "v4a"

        return "unified"

    def edit_file(self, target_file: str, code_edit: str) -> str:
        """Edit a file or create a new one with the specified content.

        Args:
            target_file (str): The path of the file to edit.
            code_edit (str): The new content for the file or a patch in V4A/unified diff format.

        Returns:
            str: A message describing the result of the operation.
        """
        try:
            file_path = self._resolve_path(target_file)

            # Check if this is a patch by looking for patch format markers
            if "*** Begin Patch" in code_edit or "@@" in code_edit[:100]:
                return self.apply_patch(target_file, code_edit)

            # Ensure directory exists
            os.makedirs(file_path.parent, exist_ok=True)

            # Determine if we're creating or modifying
            action = "Created" if not file_path.exists() else "Updated"

            # Write the new content
            file_path.write_text(code_edit, encoding='utf-8')
            logger.info(f"{action} file: {file_path}")

            return f"{action} file: {target_file}"

        except Exception as e:
            error_msg = f"Error editing file {target_file}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def apply_patch(self, target_file: str, patch_content: str) -> str:
        """Apply a patch to a file.

        Supports both V4A diff format (used by GPT-5.1) and unified diff format.

        V4A format example:
        ```
        *** Begin Patch
        *** Update File: path/to/file.py
        @@ context line
         unchanged line
        -removed line
        +added line
         unchanged line
        *** End Patch
        ```

        Args:
            target_file (str): The path of the file to patch.
            patch_content (str): The patch content in V4A or unified diff format.

        Returns:
            str: A message describing the result of the operation.
        """
        try:
            file_path = self._resolve_path(target_file)
            diff_format = self._detect_diff_format(patch_content)

            # Extract the actual diff content from V4A wrapper
            if "*** Begin Patch" in patch_content:
                match = re.search(
                    r'(?:\*{3} Begin Patch[\s\S]*?\*{3} (?:Update|Add) File:[^\n]*\n)([\s\S]+?)(?:\*{3} End Patch|$)',
                    patch_content
                )
                if match:
                    patch_content = match.group(1).strip()

            # Handle file creation
            if not file_path.exists():
                if "*** Add File:" in patch_content or diff_format == "v4a":
                    try:
                        # Create file with V4A create mode
                        new_content = apply_diff("", patch_content, mode="create")
                        os.makedirs(file_path.parent, exist_ok=True)
                        file_path.write_text(new_content, encoding='utf-8')
                        logger.info(f"Created file: {file_path}")
                        return f"Created file: {target_file}"
                    except ValueError as e:
                        return f"Error creating file: {str(e)}"
                else:
                    return f"Error: Cannot apply patch to non-existent file {target_file}"

            # Read the original file
            original_content = file_path.read_text(encoding='utf-8')

            # Apply the patch based on format
            if diff_format == "v4a":
                try:
                    new_content = apply_diff(original_content, patch_content, mode="default")
                    file_path.write_text(new_content, encoding='utf-8')
                    logger.info(f"Applied V4A patch to file: {file_path}")
                    return f"Successfully patched file: {target_file}"
                except ValueError as e:
                    return f"Error applying V4A patch: {str(e)}"
            else:
                # Fall back to unified diff format
                return self._apply_unified_patch(file_path, original_content, patch_content)

        except Exception as e:
            error_msg = f"Error applying patch to {target_file}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _apply_unified_patch(self, file_path: Path, original_content: str, patch_content: str) -> str:
        """Apply a unified diff format patch."""
        try:
            original_lines = original_content.splitlines()
            patch_lines = patch_content.splitlines()
            patched_content = list(original_lines)

            i = 0
            while i < len(patch_lines):
                line = patch_lines[i]
                if line.startswith("@@"):
                    # Parse the hunk header
                    hunk_match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    if hunk_match:
                        orig_start = int(hunk_match.group(1)) - 1  # 0-based indexing

                        # Process the hunk
                        j = i + 1
                        orig_lines_list = []
                        new_lines = []

                        while j < len(patch_lines) and not patch_lines[j].startswith("@@"):
                            hunk_line = patch_lines[j]
                            if hunk_line.startswith("+"):
                                new_lines.append(hunk_line[1:])
                            elif hunk_line.startswith("-"):
                                orig_lines_list.append(hunk_line[1:])
                            else:
                                line_content = hunk_line[1:] if hunk_line.startswith(" ") else hunk_line
                                orig_lines_list.append(line_content)
                                new_lines.append(line_content)
                            j += 1

                        # Apply the changes
                        patched_content = patched_content[:orig_start] + new_lines + patched_content[orig_start + len(orig_lines_list):]

                        i = j
                    else:
                        i += 1
                else:
                    i += 1

            # Write the patched content
            file_path.write_text("\n".join(patched_content), encoding='utf-8')
            logger.info(f"Applied unified patch to file: {file_path}")

            return f"Successfully patched file: {file_path}"

        except ValueError as ve:
            return f"Error applying patch: {str(ve)}"

    def compare_files(self, file1: str, file2: str, context_lines: int = 3) -> str:
        """Compare two files and return the differences.

        Args:
            file1 (str): The path to the first file.
            file2 (str): The path to the second file.
            context_lines (int): Number of context lines to show around differences.

        Returns:
            str: A unified diff of the two files.
        """
        try:
            path1 = self._resolve_path(file1)
            path2 = self._resolve_path(file2)

            if not path1.exists():
                return f"Error: File not found: {file1}"
            if not path2.exists():
                return f"Error: File not found: {file2}"

            # Read file contents
            content1 = path1.read_text(encoding='utf-8').splitlines()
            content2 = path2.read_text(encoding='utf-8').splitlines()

            # Generate unified diff
            diff = difflib.unified_diff(
                content1, content2,
                fromfile=str(file1), tofile=str(file2),
                n=context_lines
            )

            # Convert the diff generator to a string
            diff_text = "\n".join(diff)

            return diff_text if diff_text else "Files are identical."

        except Exception as e:
            error_msg = f"Error comparing files {file1} and {file2}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def search_replace(self, target_file: str, search_pattern: str, replacement: str, use_regex: bool = False) -> str:
        """Search and replace text in a file.

        Args:
            target_file (str): The path of the file to modify.
            search_pattern (str): The text or pattern to search for.
            replacement (str): The text to replace with.
            use_regex (bool): Whether to treat search_pattern as a regular expression.

        Returns:
            str: A message describing the result of the operation.
        """
        try:
            file_path = self._resolve_path(target_file)

            if not file_path.exists():
                return f"Error: File not found: {target_file}"

            # Read the file content
            content = file_path.read_text(encoding='utf-8')

            # Perform the replacement
            if use_regex:
                try:
                    pattern = re.compile(search_pattern, re.MULTILINE)
                    new_content, count = re.subn(pattern, replacement, content)
                except re.error as e:
                    return f"Error: Invalid regular expression pattern: {str(e)}"
            else:
                count = content.count(search_pattern)
                new_content = content.replace(search_pattern, replacement)

            # Only update the file if changes were made
            if count > 0:
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Replaced {count} occurrences in file: {file_path}")
                return f"Replaced {count} occurrences in file: {target_file}"
            else:
                return f"No occurrences of '{search_pattern}' found in {target_file}"

        except Exception as e:
            error_msg = f"Error performing search and replace in {target_file}: {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == '__main__':
    # Simple test
    editor = EditTool()

    # Test edit_file
    print(editor.edit_file("test_edit.py", "print('Hello, World!')"))

    # Test search_replace
    print(editor.search_replace("test_edit.py", "Hello", "Greetings"))

    # Test V4A diff format
    v4a_patch = """@@
 print('Greetings, World!')
-print('Greetings, World!')
+print('Hello, Universe!')
+print('This is a new line!')
"""
    print("\n--- Testing V4A Patch ---")
    print(editor.apply_patch("test_edit.py", v4a_patch))

    # Read the result
    with open("test_edit.py", "r") as f:
        print("File content after V4A patch:")
        print(f.read())

    # Clean up
    if os.path.exists("test_edit.py"):
        os.remove("test_edit.py")
