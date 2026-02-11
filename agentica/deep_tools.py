# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in tools for DeepAgent

Built-in tool set for DeepAgent, including:
- ls: List directory contents
- read_file: Read file content
- write_file: Write file content
- edit_file: Edit file (string replacement)
- glob: File pattern matching
- grep: Search file content
- execute: Execute command
- web_search: Web search (implemented using BaiduSearch)
- fetch_url: Fetch URL content (implemented using UrlCrawler)
- write_todos: Create and manage task list
- read_todos: Read current task list
- task: Launch subagent to handle complex tasks
"""
import asyncio
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
import time
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING, Union

import aiofiles

from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.utils.string import truncate_if_too_long

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model
    from agentica.tools.skill_tool import SkillTool


class BuiltinFileTool(Tool):
    """
    Built-in file system tool providing ls, read_file, write_file, edit_file, glob, grep functions.
    """

    def __init__(
            self,
            base_dir: Optional[str] = None,
            max_read_lines: int = 500,
            max_line_length: int = 2000,
    ):
        """
        Initialize BuiltinFileTool.

        Args:
            base_dir: Base directory for file operations, defaults to current working directory
            max_read_lines: Maximum number of lines to read by default
            max_line_length: Maximum length per line, longer lines will be truncated
        """
        super().__init__(name="builtin_file_tool")
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.max_read_lines = max_read_lines
        self.max_line_length = max_line_length

        # Register all file operation functions
        self.register(self.ls)
        self.register(self.read_file)
        self.register(self.write_file, sanitize_arguments=False)
        self.register(self.edit_file, sanitize_arguments=False)
        self.register(self.glob)
        self.register(self.grep)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path, supporting absolute, relative, and ~ paths.
        
        - ~ paths are expanded to user home directory
        - Absolute paths are used directly
        - Relative paths are resolved relative to base_dir
        """
        # Expand ~ to user home directory
        if path.startswith("~"):
            return Path(path).expanduser()
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def _validate_path(self, path: str) -> str:
        """Validate path - currently no restrictions, base_dir is just the default."""
        # Allow all paths: absolute, relative, with .. or ~
        # base_dir is only used as the default starting point for relative paths
        return path

    async def ls(self, directory: str = ".") -> str:
        """Lists all files in the directory.

        Usage:
        - The directory parameter can be an absolute or relative path
        - The ls tool will return a list of all files in the specified directory.
        - This is very useful for exploring the file system and finding the right file to read or edit.
        - You should almost ALWAYS use this tool before using the Read or Edit tools.

        Args:
            directory: Directory path to list files, defaults to current directory

        Returns:
            str, JSON formatted file list
        """
        try:
            self._validate_path(directory)
            dir_path = self._resolve_path(directory)

            if not dir_path.exists():
                return f"Error: Directory not found: {directory}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {directory}"

            def _ls_sync():
                items = []
                for item in sorted(dir_path.iterdir()):
                    item_type = "dir" if item.is_dir() else "file"
                    items.append({
                        "name": item.name,
                        "path": str(item),
                        "type": item_type,
                    })
                return items

            items = await asyncio.get_event_loop().run_in_executor(None, _ls_sync)

            logger.debug(f"Listed {len(items)} items in {dir_path}")
            result = json.dumps(items, ensure_ascii=False, indent=2)
            result = truncate_if_too_long(result)
            return str(result)
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")
            return f"Error listing directory: {e}"

    async def read_file(
            self,
            file_path: str,
            offset: int = 0,
            limit: Optional[int] = 500,
    ) -> str:
        """Reads a file from the filesystem. You can access any file directly by using this tool.
        Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

        Usage:
        - The file_path parameter must be an absolute path, not a relative path
        - By default, it reads up to 500 lines starting from the beginning of the file
        - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
        - First scan: read_file(path, limit=100) to see file structure
        - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
        - Only omit limit (read full file) when necessary for editing
        - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
        - Any lines longer than 2000 characters will be truncated
        - Results are returned using cat -n format, with line numbers starting at 1
        - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
        - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
        - You should ALWAYS make sure a file has been read before editing it.

        Args:
            file_path: File path, support md, txt, py, etc. absolute path
            offset: Starting line number (0-based)
            limit: Maximum number of lines to read, defaults to 500

        Returns:
            File content with line numbers
        """
        try:
            self._validate_path(file_path)
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"
            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            limit = limit if limit is not None else self.max_read_lines
            max_line_len = self.max_line_length

            # Async streaming read — only read the lines we need
            output_lines = []
            total_lines = 0
            end_line = offset + limit
            async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                async for line in f:
                    total_lines += 1
                    if total_lines > offset and total_lines <= end_line:
                        line = line.rstrip('\n\r')
                        if len(line) > max_line_len:
                            line = line[:max_line_len] + "..."
                        output_lines.append(f"{total_lines:6d}\t{line}")

            result = "\n".join(output_lines)

            # Add file info if truncated
            actual_end = min(offset + len(output_lines), total_lines)
            if actual_end < total_lines:
                result += f"\n\n[Showing lines {offset + 1}-{actual_end} of {total_lines} total lines]"

            logger.debug(f"Read file {file_path}: lines {offset + 1}-{actual_end}, total {total_lines} lines")
            return result
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"

    async def write_file(self, file_path: str, content: str) -> str:
        """Writes content to a file in the filesystem.

        Usage:
        - The file_path can be relative (e.g., "tmp/script.py", "./outputs/data.txt") or absolute path
        - Relative paths are resolved relative to the base working directory
        - The tool returns the actual absolute path of the created file - ALWAYS use this returned path for subsequent operations (read_file, execute, etc.)
        - The content parameter must be a string
        - The write_file tool will create a new file or overwrite existing file
        - Parent directories will be created automatically if they don't exist
        - Prefer to edit existing files over creating new ones when possible

        IMPORTANT: After calling write_file, use the absolute path returned in the result for any follow-up operations like execute or read_file. Do NOT guess or construct the path yourself.

        Args:
            file_path: File path (relative or absolute). Examples: "tmp/script.py", "outputs/result.txt", "./tmp/main.py", use './tmp/' prefix file path for temporary files
            content: File content to write

        Returns:
            Operation result message containing the actual absolute path of the file
        """
        try:
            self._validate_path(file_path)
            path = self._resolve_path(file_path)
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            action = "Created" if not path.exists() else "Updated"

            # Atomic write: write to temp file then rename to avoid partial writes
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                os.close(tmp_fd)
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
                # Atomic rename
                os.replace(tmp_path, str(path))
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Return absolute path to help LLM use correct path in subsequent operations
            absolute_path = str(path.resolve())
            logger.debug(f"{action} file: {absolute_path}, file content length: {len(content)} characters")
            return f"{action} file, absolute path: {absolute_path}"
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return f"Error writing file: {e}"

    async def edit_file(
            self,
            file_path: str,
            edit: Union[dict, List[dict]],
    ) -> str:
        """Replace specific strings within a specified file.

        **Tips:**
        - Only use this tool on text files.
        - Multi-line strings are supported.
        - Can specify a single edit or a list of edits in one call.
        - You should prefer this tool over write_file tool and shell `sed` command.

        Core logic:
        1. Read entire file content
        2. Apply edit(s) sequentially (each edit sees the result of previous edits)
        3. Write modified content back (only if all edits succeed)

        Important:
        - Edits are applied sequentially: each edit operates on the result of previous edits
        - If any edit fails when providing multiple edits, NO changes are written (all-or-nothing)
        - Uses literal string matching (NOT regex)

        Args:
            file_path: The path to the file to edit. Absolute paths are required when editing
                      files outside the working directory.
            edit: The edit(s) to apply to the file. You can provide a single edit object
                  or a list of edit objects.
                  Each edit has the following fields:
                  - old (str, required): The old string to replace. Can be multi-line.
                  - new (str, required): The new string to replace with. Can be multi-line.
                  - replace_all (bool, optional): Whether to replace all occurrences.
                    Default: False (only replace first match).

        Returns:
            Operation result message

        Examples:
            # Single edit:
            edit_file("file.py", {"old": "def foo():", "new": "def bar():"})

            # Multiple edits:
            edit_file("file.py", [
                {"old": "x = 1", "new": "x = 2"},
                {"old": "print(x)", "new": "print(f'x = {x}')", "replace_all": True}
            ])
        """
        try:
            self._validate_path(file_path)
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"
            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            # Normalize edit to list
            if isinstance(edit, dict):
                edits = [edit]
            elif isinstance(edit, list):
                edits = edit
            else:
                return "Error: edit must be a dict or list of dicts"

            if not edits:
                return "Error: No edits provided"

            # Async read
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()

            results = []

            # Apply edits sequentially (pure CPU, no I/O)
            for i, edit_item in enumerate(edits):
                old_string = edit_item.get("old") or edit_item.get("old_string")
                new_string = edit_item.get("new") or edit_item.get("new_string")
                replace_all = edit_item.get("replace_all", False)

                if old_string is None or new_string is None:
                    return f"Error: Edit {i+1} missing 'old' or 'new' field"

                result = self._str_replace(content, old_string, new_string, replace_all)

                if not result["success"]:
                    if len(edits) == 1:
                        return f"Error: {result['error']}"
                    else:
                        return (
                            f"Error in edit {i+1}: {result['error']}\n"
                            f"No changes have been applied to the file."
                        )

                content = result["new_content"]
                results.append({
                    "edit_num": i + 1,
                    "replaced_count": result["count"],
                })

            # All edits succeeded — atomic write back
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                os.close(tmp_fd)
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
                os.replace(tmp_path, str(path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            total_replacements = sum(r["replaced_count"] for r in results)

            if len(results) == 1:
                logger.debug(f"Replaced {total_replacements} occurrence(s) in {file_path}")
                return f"Successfully replaced {total_replacements} occurrence(s) in '{file_path}'"
            else:
                logger.debug(f"Applied {len(results)} edits, {total_replacements} replacements in {file_path}")
                return f"Successfully applied {len(results)} edits ({total_replacements} total replacements) in '{file_path}'"

        except Exception as e:
            logger.error(f"Error editing file {file_path}: {e}")
            return f"Error editing file: {e}"

    def _str_replace(
            self,
            content: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> dict:
        """Internal string replacement logic.

        Returns:
            {"success": bool, "new_content": str, "count": int, "error": str}
        """
        # Find all match positions
        matches = []
        start = 0
        while True:
            idx = content.find(old_string, start)
            if idx == -1:
                break
            matches.append(idx)
            start = idx + len(old_string)

        if not matches:
            display_old = old_string[:100] + "..." if len(old_string) > 100 else old_string
            return {
                "success": False,
                "error": f"String not found: '{display_old}'",
                "new_content": content,
                "count": 0,
            }

        # If not replace_all and multiple matches, show context for each match
        if not replace_all and len(matches) > 1:
            contexts = []
            for idx in matches[:3]:  # Show first 3 matches
                line_num = content[:idx].count('\n') + 1
                # Get surrounding context (up to 50 chars around the match)
                context_start = max(0, idx - 20)
                context_end = min(len(content), idx + len(old_string) + 30)
                context = content[context_start:context_end].replace('\n', '\\n')
                contexts.append(f"  Line {line_num}: ...{context}...")

            error_msg = (
                f"Found {len(matches)} occurrences of the string.\n"
                f"Use replace_all=True to replace all, or provide more context to make it unique.\n"
                f"Matches found at:\n" + '\n'.join(contexts)
            )
            if len(matches) > 3:
                error_msg += f"\n  ... and {len(matches) - 3} more"

            return {
                "success": False,
                "error": error_msg,
                "new_content": content,
                "count": len(matches),
            }

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            count = len(matches)
        else:
            # Replace only the first match (leftmost)
            idx = matches[0]
            new_content = content[:idx] + new_string + content[idx + len(old_string):]
            count = 1

        return {
            "success": True,
            "new_content": new_content,
            "count": count,
            "error": None,
        }

    async def glob(self, pattern: str, path: str = ".") -> str:
        """Find files matching a glob pattern (supports recursive search with `**`).

        Usage:
        - This tool searches for files by matching standard glob wildcards, returns JSON formatted absolute file paths
        - Core glob wildcards (key differences):
        1. `*`: Matches any files in the **current specified single directory** (non-recursive, no deep subdirectories)
        2. `**`: Matches any directories recursively (penetrates all deep subdirectories for cross-level search)
        3. `?`: Matches any single character (e.g., "file?.txt" matches "file1.txt", "filea.txt")
        - Patterns can be absolute (starting with `/`, e.g., "/home/user/*.py") or relative (e.g., "docs/*.md")
        - Automatically excludes common useless directories (.git, __pycache__, etc.) to filter valid files
        - Returns empty JSON list if no matching files are found

        Examples (clear parameter correspondence and function explanation):
        - pattern: `*.py`, path: "." - Find all Python files in the current working directory (non-recursive)
        - pattern: `*.txt`, path: "." - Find all text files in the current working directory (non-recursive)
        - pattern: `**/*.md`, path: "/path/to/subdir/" - Find all markdown files in all levels under /path/to/subdir/ (recursive)
        - pattern: `subdir/*.md`, path: "." - Find all markdown files directly in the "subdir" folder (non-recursive, no deep subdirs)

        Args:
            pattern: Valid glob search pattern, e.g., "*.py", "**/*.md", "src/?*.js"
            path: Starting search directory (relative or absolute), defaults to current working directory (".").

        Returns:
            JSON formatted string of sorted absolute file paths (filtered to exclude ignored directories).
            Error message string if directory not found or other exceptions occur.
        """
        try:
            self._validate_path(path)
            base_path = self._resolve_path(path)

            if not base_path.exists():
                return f"Error: Directory not found: {path}"

            # Run glob in executor to avoid blocking on large directory trees
            def _glob_sync():
                matches = list(base_path.glob(pattern))
                ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
                return sorted(
                    str(m) for m in matches
                    if not set(m.parts).intersection(ignore_dirs)
                )

            filtered = await asyncio.get_event_loop().run_in_executor(None, _glob_sync)

            logger.debug(f"Glob found {len(filtered)} files matching pattern '{pattern}' in directory '{path}'")
            # Convert to formatted JSON string
            result = json.dumps(filtered, ensure_ascii=False, indent=2)
            # Truncate if content exceeds the limit to avoid excessive output
            result = truncate_if_too_long(result)
            return str(result)
        except Exception as e:
            logger.error(f"Exception occurred during glob search (pattern: '{pattern}', path: '{path}'): {str(e)}")
            return f"Error in glob search: {str(e)}"

    async def grep(
            self,
            pattern: str,
            path: str = ".",
            *,
            include: Optional[str] = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
            case_insensitive: bool = False,
            multiline: bool = False,
            context_lines: int = 0,
            before_context: int = 0,
            after_context: int = 0,
            max_results: int = 100,
            fixed_strings: bool = False,
    ) -> str:
        """Search for a pattern in files using ripgrep (rg).

        Usage:
        - Searches text patterns across files, powered by ripgrep for maximum speed
        - The pattern parameter supports regex by default (e.g., 'class \\w+', 'def \\w+')
        - Use fixed_strings=True to treat pattern as literal text (no regex interpretation)
        - The path parameter specifies the search directory (default: current working directory)
        - The include parameter filters files by glob (e.g., "*.py", "*.{ts,tsx}")
        - The output_mode parameter controls output format:
          - "files_with_matches": List only file paths (default)
          - "content": Show matching lines with file path, line numbers, and optional context
          - "count": Show match count per file
        - Automatically falls back to pure Python search if ripgrep is not installed

        Args:
            pattern: Text/regex to search for
            path: Starting directory for search (default: ".")
            include: File glob filter, e.g., "*.py", "*.{js,ts}" (maps to rg --glob)
            output_mode: Output format — "files_with_matches", "content", or "count"
            case_insensitive: Ignore case when matching (default: False)
            multiline: Enable multiline matching where . matches newlines (default: False)
            context_lines: Show N lines before and after each match (default: 0, content mode only)
            before_context: Show N lines before each match (default: 0, content mode only)
            after_context: Show N lines after each match (default: 0, content mode only)
            max_results: Maximum results to return (default: 100)
            fixed_strings: Treat pattern as literal text, not regex (default: False)

        Returns:
            Search results as formatted string
        """
        # Resolve and validate path
        self._validate_path(path)
        base_path = self._resolve_path(path)
        if not base_path.exists():
            return f"Error: Directory not found: {path}"

        # Check if rg is available
        rg_path = shutil.which("rg")
        if rg_path is None:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._grep_fallback, pattern, path, include, output_mode,
                max_results, fixed_strings, case_insensitive,
            )

        # Build rg command arguments
        cmd: List[str] = [rg_path]

        # Output mode flags
        if output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count")
        else:  # content
            cmd.append("--line-number")

        # Matching options
        if fixed_strings:
            cmd.append("--fixed-strings")
        if case_insensitive:
            cmd.append("--ignore-case")
        if multiline:
            cmd.extend(["--multiline", "--multiline-dotall"])

        # Context lines (content mode only)
        if output_mode == "content":
            if context_lines > 0:
                cmd.extend(["--context", str(context_lines)])
            else:
                if before_context > 0:
                    cmd.extend(["--before-context", str(before_context)])
                if after_context > 0:
                    cmd.extend(["--after-context", str(after_context)])

        # File filter
        if include:
            cmd.extend(["--glob", include])

        # Result limit: for content mode, limit matches per file
        if output_mode == "content":
            cmd.extend(["--max-count", str(max_results)])

        # Exclude common irrelevant directories (rg already ignores .git via .gitignore)
        for d in ["__pycache__", "node_modules", ".venv", "venv", ".idea", ".pytest_cache"]:
            cmd.extend(["--glob", f"!{d}/"])

        # Pattern and path
        cmd.append("--")
        cmd.append(pattern)
        cmd.append(str(base_path))

        # Execute asynchronously
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            return "Error: grep timed out after 30 seconds"
        except FileNotFoundError:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._grep_fallback, pattern, path, include, output_mode,
                max_results, fixed_strings, case_insensitive,
            )

        # rg exit codes: 0=matches found, 1=no matches, 2=error
        if proc.returncode == 2:
            err = stderr.decode("utf-8", errors="replace").strip()
            return f"Error: {err}"

        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return f"No matches found for '{pattern}'"

        # Truncate result lines for files_with_matches / count
        if output_mode in ("files_with_matches", "count"):
            lines = output.split("\n")
            if len(lines) > max_results:
                output = "\n".join(lines[:max_results])
                output += f"\n... ({len(lines) - max_results} more results truncated)"

        result = truncate_if_too_long(output)
        logger.debug(f"Grep(rg) for '{pattern}': result length {len(result)} chars")
        return result

    def _grep_fallback(
            self,
            pattern: str,
            path: str,
            include: Optional[str],
            output_mode: str,
            max_results: int,
            fixed_strings: bool,
            case_insensitive: bool = False,
    ) -> str:
        """Fallback grep using pure Python when ripgrep is not available."""
        try:
            base_path = self._resolve_path(path)

            # Compile regex
            regex_pattern = None
            if not fixed_strings:
                try:
                    flags = re.IGNORECASE if case_insensitive else 0
                    regex_pattern = re.compile(pattern, flags)
                except re.error as e:
                    return f"Error: Invalid regex pattern '{pattern}': {e}"

            # Determine files to search
            if include:
                files = list(base_path.glob(f"**/{include}"))
            else:
                files = list(base_path.glob("**/*"))

            # Exclude directories and ignored paths
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            files = [f for f in files if f.is_file() and not set(f.parts).intersection(ignore_dirs)]

            results = []
            file_counts = {}

            match_pattern = pattern.lower() if (case_insensitive and fixed_strings) else pattern

            for fp in files:
                if len(results) >= max_results:
                    break

                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    file_matches = []
                    for line_num, line in enumerate(lines, 1):
                        if fixed_strings:
                            check_line = line.lower() if case_insensitive else line
                            matched = match_pattern in check_line
                        else:
                            matched = regex_pattern.search(line)
                        if matched:
                            file_matches.append({
                                "line_num": line_num,
                                "content": line.strip()[:200],
                            })

                    if file_matches:
                        file_counts[str(fp)] = len(file_matches)
                        if output_mode == "content":
                            for match in file_matches[:max_results - len(results)]:
                                results.append(f"{fp}:{match['line_num']}: {match['content']}")
                        elif output_mode == "files_with_matches":
                            results.append(str(fp))
                except Exception:
                    continue

            # Format output
            if output_mode == "count":
                output_lines = [f"{p}:{c}" for p, c in file_counts.items()]
                result = "\n".join(output_lines) if output_lines else f"No matches found for '{pattern}'"
            elif output_mode == "files_with_matches":
                result = "\n".join(sorted(set(results))) if results else f"No matches found for '{pattern}'"
            else:  # content
                result = "\n".join(results) if results else f"No matches found for '{pattern}'"

            result = truncate_if_too_long(result)
            logger.debug(f"Grep(fallback) for '{pattern}': found {len(file_counts)} files, result length: {len(result)} chars")
            return result

        except Exception as e:
            logger.error(f"Error in grep fallback: {e}")
            return f"Error in grep: {e}"


class BuiltinExecuteTool(Tool):
    """
    Built-in command execution tool using async subprocess.
    Exposed as execute function for consistent naming in DeepAgent.
    """

    def __init__(self, base_dir: Optional[str] = None, timeout: int = 120, max_output_length: int = 20000):
        """
        Initialize BuiltinExecuteTool.

        Args:
            base_dir: Base directory for command execution
            timeout: Command execution timeout in seconds
            max_output_length: Maximum length of output to return
        """
        super().__init__(name="builtin_execute_tool")
        self._base_dir: Optional[Path] = Path(base_dir) if base_dir else None
        self._timeout = timeout
        self._max_output_length = max_output_length
        # Import ShellTool for its syntax-fix helpers
        from agentica.tools.shell_tool import ShellTool
        self._shell = ShellTool(base_dir=base_dir, timeout=timeout)
        self.register(self.execute)

    async def execute(self, command: str) -> str:
        """Executes a given command, capturing both stdout and stderr.

        Before executing the command, please follow these steps:

        1. Directory Verification:
        - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
        - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory

        2. Command Execution:
        - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
        - Examples of proper quoting:
            - cd "/Users/name/My Documents" (correct)
            - cd /Users/name/My Documents (incorrect - will fail)
            - python3 "/path/with spaces/script.py" (correct)
            - python3 /path/with spaces/script.py (incorrect - will fail)
        - After ensuring proper quoting, execute the command
        - Capture the output of the command

        Usage notes:
        - The command parameter is required
        - Returns combined stdout/stderr output with exit code
        - If the output is very large, it may be truncated
        - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
        - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
            - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
            - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
        - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd
        - For multi-line Python code, the tool automatically converts `python3 -c "..."` to heredoc format for better handling. Use Python syntax correctly, use `None`, `True`/`False`, verify the syntax is correct Python.

        Examples:
        Good examples:
            - execute(command="python3 /path/to/script.py")
            - execute(command="pytest /path/to/tests/test.py")
            - execute(command="python3 -c 'print(33333**2 + 332.2 / 12)'")
            - execute(command="npm install && npm test")

        Bad examples (avoid these):
            - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
            - execute(command="cat file.txt")  # Use read_file tool instead
            - execute(command="find . -name '*.py'")  # Use glob tool instead
            - execute(command="grep -r 'pattern' .")  # Use grep tool instead

        Args:
            command: command to execute

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        # Use ShellTool's syntax fixers (python -c → heredoc conversion, null/true/false fix)
        command = self._shell._convert_python_c_to_heredoc(command)

        logger.debug(f"Executing command: {command}")
        cwd = str(self._base_dir) if self._base_dir else None
        proc = None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # Graceful termination: SIGTERM first, then SIGKILL
            if proc is not None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                except ProcessLookupError:
                    pass
            logger.warning(f"Command timed out after {self._timeout}s: {command}")
            return f"Error: Command timed out after {self._timeout} seconds"
        except Exception as e:
            logger.warning(f"Failed to run shell command: {e}")
            return f"Error: {e}"

        # Combine stdout and stderr
        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")

        output = "\n".join(output_parts).strip()

        # Truncate if too long
        if len(output) > self._max_output_length:
            output = output[:self._max_output_length] + "\n... (output truncated)"

        # Add exit code info
        if proc.returncode != 0:
            output = f"{output}\n\n[Exit code: {proc.returncode}]"

        logger.debug(f"Command exit code: {proc.returncode}")
        return output if output else f"Command executed successfully (exit code: {proc.returncode})"


class BuiltinWebSearchTool(Tool):
    """
    Built-in web search tool using Baidu search.
    Exposed as web_search function.
    """

    def __init__(self):
        """
        Initialize BuiltinWebSearchTool.
        """
        super().__init__(name="builtin_web_search_tool")
        from agentica.tools.baidu_search_tool import BaiduSearchTool
        self._search = BaiduSearchTool()
        self.register(self.web_search)

    async def web_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
        """Search the web using Baidu for multiple queries and return results

        Args:
            queries (Union[str, List[str]]): Search keyword(s), can be a single string or a list of strings
            max_results (int, optional): Number of results to return for each query, default 5

        Returns:
            str: A JSON formatted string containing the search results.

        IMPORTANT: After using this tool:
        1. Read through the 'content' field of each result
        2. Extract relevant information that answers the user's question
        3. Synthesize this into a clear, natural language response
        4. Cite sources by mentioning the page titles or URLs
        5. NEVER show the raw JSON to the user - always provide a formatted response
        """

        try:
            result = await self._search.baidu_search(queries, max_results=max_results)
            logger.debug(f"Web search for '{queries}', result length: {len(result)} characters.")
            return result
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return json.dumps({"error": f"Web search error: {e}", "queries": queries}, ensure_ascii=False)


class BuiltinFetchUrlTool(Tool):
    """
    Built-in URL fetching tool that wraps UrlCrawlerTool.
    Exposed as fetch_url function for consistent naming in DeepAgent.
    """

    def __init__(self, max_content_length: int = 16000):
        """
        Initialize BuiltinFetchUrlTool.

        Args:
            max_content_length: Maximum length of returned content
        """
        super().__init__(name="builtin_fetch_url_tool")
        self.max_content_length = max_content_length
        # Import and initialize UrlCrawlerTool (uses default cache dir ~/.cache/agentica/web_cache/)
        from agentica.tools.url_crawler_tool import UrlCrawlerTool
        self._crawler = UrlCrawlerTool(max_content_length=max_content_length)
        self.register(self.fetch_url)

    async def fetch_url(self, url: str) -> str:
        """Fetch URL content and convert to clean text format.

        Args:
            url: URL to fetch, url starts with http:// or https://

        Returns:
            str, JSON formatted fetch result containing url, content, and save_path

        IMPORTANT: After using this tool:
        1. Read through the return content
        2. Extract relevant information that answers the user's question
        3. Synthesize this into a clear, natural language response
        4. NEVER show the raw JSON to the user unless specifically requested
        """
        result = await self._crawler.url_crawl(url)
        logger.debug(f"Fetched URL: {url}, result length: {len(result)} characters.")
        return result


class BuiltinTodoTool(Tool):
    """
    Built-in task management tool providing write_todos and read_todos functions.
    Used for tracking progress of complex tasks.
    """
    # System prompt for todo tool usage guidance
    WRITE_TODOS_SYSTEM_PROMPT = dedent("""## `write_todos`

    You have access to the `write_todos` tool to help you manage and plan complex objectives.
    Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
    This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

    It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
    For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
    Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

    ## Important To-Do List Usage Notes to Remember
    - The `write_todos` tool should never be called multiple times in parallel.
    - Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.""")

    def __init__(self):
        """Initialize BuiltinTodoTool."""
        super().__init__(name="builtin_todo_tool")
        self._todos: List[Dict[str, Any]] = []
        self.register(self.write_todos)
        self.register(self.read_todos)

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for todo tool usage guidance."""
        return self.WRITE_TODOS_SYSTEM_PROMPT

    def write_todos(self, todos: Optional[List[Dict[str, str]]] = None) -> str:
        """Create and manage a structured task list.

        Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

        Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

        ## When to Use This Tool
        Use this tool in these scenarios:
        1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
        2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
        3. User explicitly requests todo list - When the user directly asks you to use the todo list
        4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
        5. The plan may need future revisions or updates based on results from the first few steps

        ## How to Use This Tool
        1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
        2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
        3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
        4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

        ## When NOT to Use This Tool
        It is important to skip using this tool when:
        1. There is only a single, straightforward task
        2. The task is trivial and tracking it provides no benefit
        3. The task can be completed in less than 3 trivial steps
        4. The task is purely conversational or informational

        ## Task States and Management

        1. **Task States**: Use these states to track progress:
        - pending: Task not yet started
        - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
        - completed: Task finished successfully

        2. **Task Management**:
        - Update task status in real-time as you work
        - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
        - Complete current tasks before starting new ones
        - Remove tasks that are no longer relevant from the list entirely
        - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
        - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

        3. **Task Completion Requirements**:
        - ONLY mark a task as completed when you have FULLY accomplished it
        - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
        - When blocked, create a new task describing what needs to be resolved
        - Never mark a task as completed if:
            - There are unresolved issues or errors
            - Work is partial or incomplete
            - You encountered blockers that prevent completion
            - You couldn't find necessary resources or dependencies
            - Quality standards haven't been met

        4. **Task Breakdown**:
        - Create specific, actionable items
        - Break complex tasks into smaller, manageable steps
        - Use clear, descriptive task names

        Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
        Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

        Each task item should contain:
        - content: Task description
        - status: Task status ("pending", "in_progress", "completed")

        Args:
            todos: Task list, each task is a dict with content and status. Required.
            Example: [{"content": "Write a report", "status": "pending"}, {"content": "Review report", "status": "pending"}]

        Returns:
            Updated task list
        """
        try:
            # Validate todos parameter
            if todos is None:
                return "Error: 'todos' parameter is required. Please provide a list of tasks with 'content' and 'status' fields."
            if len(todos) == 0:
                return "Error: 'todos' list cannot be empty. Please provide at least one task."
            valid_statuses = {"pending", "in_progress", "completed"}
            validated_todos = []

            for i, todo in enumerate(todos):
                if not isinstance(todo, dict):
                    return f"Error: Todo item {i} must be a dictionary"

                content = todo.get("content", "")
                status = todo.get("status", "pending")

                if not content:
                    return f"Error: Todo item {i} must have 'content' field"
                if status not in valid_statuses:
                    return f"Error: Invalid status '{status}' for todo item {i}. Must be one of: {valid_statuses}"

                validated_todos.append({
                    "id": str(i + 1),
                    "content": content,
                    "status": status,
                })

            self._todos = validated_todos
            logger.debug(f"Updated todo list: {len(self._todos)} items, todos: {self._todos}")

            return json.dumps({
                "message": f"Updated todo list with {len(self._todos)} items",
                "todos": self._todos,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error updating todos: {e}")
            return f"Error updating todos: {e}"

    def read_todos(self) -> str:
        """Read current task list status.

        Use this tool to check the current state of your task list. This is useful for:
        1. Reviewing progress before continuing work
        2. Checking which tasks are completed, in progress, or pending
        3. Deciding which task to work on next
        4. Verifying task completion before reporting to user

        This tool works together with write_todos:
        - Use write_todos to create/update tasks
        - Use read_todos to check current status

        Returns:
            JSON formatted current task list with summary statistics
        """
        if not self._todos:
            return json.dumps({
                "message": "No todos found. Use write_todos to create a task list.",
                "todos": [],
                "summary": {"pending": 0, "in_progress": 0, "completed": 0, "total": 0}
            }, ensure_ascii=False, indent=2)

        # Count status statistics
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0}
        for todo in self._todos:
            status = todo.get("status", "pending")
            if status in status_counts:
                status_counts[status] += 1

        total = len(self._todos)
        progress_pct = round(status_counts["completed"] / total * 100) if total > 0 else 0

        logger.debug(f"Todo list: {status_counts}, progress: {progress_pct}%")
        return json.dumps({
            "summary": {
                **status_counts,
                "total": total,
                "progress": f"{progress_pct}%"
            },
            "todos": self._todos
        }, ensure_ascii=False, indent=2)


class BuiltinTaskTool(Tool):
    """
    Built-in task tool for launching subagents to handle complex, multi-step tasks.
    
    This tool allows the main agent to delegate complex tasks to ephemeral subagents
    that can work independently with isolated context windows.
    
    Supports multiple subagent types:
    - explore: Read-only codebase exploration (fastest, lowest context)
    - general: Full capabilities for complex tasks
    - research: Web search and document analysis
    - code: Code generation and execution
    - Custom user-defined subagent types (via register_custom_subagent)
    """

    # Base system prompt template for task tool usage guidance
    TASK_SYSTEM_PROMPT_TEMPLATE = dedent("""## task Tool (Subagent Spawner)

    You have access to a `task` function to launch short-lived subagents that handle isolated tasks.

    ### How to Call

    Use your standard function calling mechanism to invoke `task(description="...", subagent_type="...")`.

    ### Available Subagent Types

    {subagent_table}

    ### Usage Guidelines

    1. **Parallel Execution**: Launch multiple agents in a single message for independent tasks
    2. **Clear Instructions**: Provide detailed task descriptions and expected output format
    3. **Right Tool for Job**: Choose the most appropriate subagent type
    4. **Isolated Context**: Each subagent has its own context window - include all necessary info

    ### When NOT to Use

    - Task is trivial (1-3 tool calls)
    - You need to see intermediate steps
    - Task depends on main conversation context
    - Simple questions that don't need delegation""")

    def __init__(
            self,
            model: Optional["Model"] = None,
            tools: Optional[List[Any]] = None,
            base_dir: Optional[str] = None,
            max_iterations: int = 15,
            custom_subagent_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize BuiltinTaskTool.

        Args:
            model: Model to use for subagents. If None, will use the parent agent's model.
            tools: Default tools for general subagent. If None, will use basic tools.
            base_dir: Base directory for file operations.
            max_iterations: Default max iterations for subagent execution.
            custom_subagent_configs: Custom subagent configurations to add/override defaults.
        """
        super().__init__(name="builtin_task_tool")
        self._model = model
        self._tools = tools
        self._base_dir = base_dir
        self._max_iterations = max_iterations
        self._parent_agent: Optional["Agent"] = None
        self._custom_configs = custom_subagent_configs or {}
        self.register(self.task)

    def _build_subagent_table(self) -> str:
        """Build a markdown table of available subagent types."""
        from agentica.subagent import get_available_subagent_types
        
        available_types = get_available_subagent_types()
        
        lines = ["| Type | Name | Description |", "|------|------|-------------|"]
        for st in available_types:
            # Truncate description for table
            desc = st['description'].split('\n')[0][:60]
            if len(st['description'].split('\n')[0]) > 60:
                desc += "..."
            type_name = st['type']
            name = st['name']
            lines.append(f"| `{type_name}` | {name} | {desc} |")
        
        return "\n".join(lines)

    def get_system_prompt(self) -> Optional[str]:
        """
        Get the dynamically generated system prompt for task tool.
        
        This prompt is regenerated each time to include any custom subagent types
        that may have been registered since initialization.
        """
        subagent_table = self._build_subagent_table()
        return self.TASK_SYSTEM_PROMPT_TEMPLATE.format(subagent_table=subagent_table)

    def set_parent_agent(self, agent: "Agent") -> None:
        """Set the parent agent reference for accessing model and tools."""
        self._parent_agent = agent
        # Also set base_dir from parent agent if available
        if self._base_dir is None and hasattr(agent, 'work_dir') and agent.work_dir:
            self._base_dir = agent.work_dir

    def _get_tools_for_subagent(self, subagent_type: str) -> List[Any]:
        """Get appropriate tools for a subagent type."""
        from agentica.subagent import get_subagent_config, SubagentType
        
        config = get_subagent_config(subagent_type)
        
        # Default tools if no config found
        if config is None:
            return self._tools or [
                BuiltinFileTool(base_dir=self._base_dir),
                BuiltinWebSearchTool(),
                BuiltinFetchUrlTool(),
            ]
        
        # Build tool list based on config type
        tools = []
        
        # For explore type, only include file tools (read-only)
        if config.type == SubagentType.EXPLORE:
            tools.append(BuiltinFileTool(base_dir=self._base_dir))
            return tools
        
        # For research type, include search and fetch tools
        if config.type == SubagentType.RESEARCH:
            tools.append(BuiltinFileTool(base_dir=self._base_dir))
            tools.append(BuiltinWebSearchTool())
            tools.append(BuiltinFetchUrlTool())
            return tools
        
        # For code type, include file and execute tools
        if config.type == SubagentType.CODE:
            tools.append(BuiltinFileTool(base_dir=self._base_dir))
            tools.append(BuiltinExecuteTool(base_dir=self._base_dir))
            return tools
        
        # For general type, include all basic tools (but not task to prevent nesting)
        tools.append(BuiltinFileTool(base_dir=self._base_dir))
        tools.append(BuiltinExecuteTool(base_dir=self._base_dir))
        tools.append(BuiltinWebSearchTool())
        tools.append(BuiltinFetchUrlTool())
        
        return tools

    async def task(self, description: str, subagent_type: str = "general") -> str:
        """Launch a subagent to handle a complex task.

        Args:
            description: Detailed description of the task to perform.
                Include what you want the subagent to do and what information to return.
            subagent_type: Type of subagent to use. Options:
                - "explore": Read-only codebase exploration (fast, low context)
                - "general": Full capabilities for complex tasks (default)
                - "research": Web search and document analysis
                - "code": Code generation and execution

        Returns:
            The result from the subagent after completing the task.
        """
        from agentica.subagent import (
            SubagentRegistry, SubagentRun, SubagentType,
            get_subagent_config, generate_subagent_session_key, is_subagent_session
        )
        
        # Get registry
        registry = SubagentRegistry()
        
        # Check if we're already in a subagent (prevent nesting)
        if self._parent_agent is not None:
            parent_session_id = getattr(self._parent_agent, 'session_id', None)
            if parent_session_id and is_subagent_session(parent_session_id):
                return json.dumps({
                    "success": False,
                    "error": "Nested subagent spawning is not allowed. Complete your task without delegating.",
                }, ensure_ascii=False)
        
        try:
            # Get subagent configuration
            config = get_subagent_config(subagent_type)
            if config is None:
                # Default to general if unknown type
                config = get_subagent_config("general")
                logger.warning(f"Unknown subagent type '{subagent_type}', using 'general'")
            
            # Get model from parent agent or use configured model
            model = self._model
            if model is None and self._parent_agent is not None:
                model = self._parent_agent.model

            if model is None:
                return json.dumps({
                    "success": False,
                    "error": "No model available for subagent. Please configure a model.",
                }, ensure_ascii=False)

            # Generate unique session key for subagent
            parent_session_id = getattr(self._parent_agent, 'session_id', 'main') if self._parent_agent else 'main'
            subagent_session_key = generate_subagent_session_key(parent_session_id, config.type)
            run_id = str(uuid.uuid4())
            
            # Create subagent run entry
            run = SubagentRun(
                run_id=run_id,
                subagent_type=config.type,
                session_key=subagent_session_key,
                parent_session_key=parent_session_id,
                task_label=description[:50] + "..." if len(description) > 50 else description,
                task_description=description,
                started_at=datetime.now(),
                status="running",
            )
            registry.register(run)

            # Get tools for this subagent type
            subagent_tools = self._get_tools_for_subagent(subagent_type)

            # Import Agent here to avoid circular imports
            from agentica.agent import Agent

            # Create subagent with isolated session
            subagent = Agent(
                model=model,
                name=f"{config.name}",
                description=config.description,
                system_prompt=config.system_prompt,
                session_id=subagent_session_key,  # Isolated session
                add_datetime_to_instructions=True,
                tools=subagent_tools,
                markdown=True,
                tool_call_limit=config.max_iterations,
            )

            logger.debug(f"Launching {config.name} [{config.type.value}] for task: {description[:100]}...")
            
            # Run subagent with async streaming to collect tool usage info
            start_time = time.time()
            tool_calls_log = []
            final_content = ""

            async for chunk in subagent.run_stream(description, stream_intermediate_steps=True):
                if chunk is None:
                    continue
                # Collect tool call info from intermediate events
                if chunk.event in ("ToolCallStarted", "ToolCallCompleted", 
                                   "MultiRoundToolCall", "MultiRoundToolResult"):
                    if chunk.tools:
                        for tool_info in chunk.tools:
                            tool_name = tool_info.get("tool_name") or tool_info.get("name", "")
                            if not tool_name:
                                continue
                            # Build a brief info string
                            tool_args = tool_info.get("tool_args") or tool_info.get("arguments", {})
                            content = tool_info.get("content")
                            brief = self._format_tool_brief(tool_name, tool_args, content)
                            entry = {"name": tool_name, "info": brief}
                            # Deduplicate: update existing entry for same tool or add new
                            if chunk.event in ("ToolCallCompleted", "MultiRoundToolResult"):
                                # Update the last entry with same tool name (add result info)
                                for i in range(len(tool_calls_log) - 1, -1, -1):
                                    if tool_calls_log[i]["name"] == tool_name and "result" not in tool_calls_log[i]:
                                        tool_calls_log[i]["info"] = brief
                                        tool_calls_log[i]["result"] = True
                                        break
                            else:
                                tool_calls_log.append(entry)
                # Accumulate final content
                if chunk.event in ("RunResponse",) and chunk.content:
                    final_content += str(chunk.content)
            
            elapsed = time.time() - start_time
            result = final_content if final_content else "Subagent completed but returned no content."
            
            # Build tool calls summary for display
            tool_summary = []
            for tc in tool_calls_log:
                tool_summary.append({"name": tc["name"], "info": tc.get("info", "")})
            
            # Update registry with success
            registry.update_status(
                run_id=run_id,
                status="completed",
                result=result,
            )
            
            logger.debug(f"{config.name} [{config.type.value}] completed task.")
            
            return json.dumps({
                "success": True,
                "subagent_type": config.type.value,
                "subagent_name": config.name,
                "result": result,
                "tool_calls_summary": tool_summary,
                "execution_time": round(elapsed, 3),
                "tool_count": len(tool_summary),
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Subagent task error: {e}")
            
            # Update registry with error if we have a run_id
            if 'run_id' in locals():
                registry.update_status(
                    run_id=run_id,
                    status="error",
                    error=str(e),
                )
            
            return json.dumps({
                "success": False,
                "error": f"Subagent task error: {e}",
                "description": description[:300],
            }, ensure_ascii=False)

    @staticmethod
    def _format_tool_brief(tool_name: str, tool_args, content=None) -> str:
        """Format a brief description for a subagent tool call."""
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        if not isinstance(tool_args, dict):
            tool_args = {}
        
        if tool_name == "read_file":
            fp = tool_args.get("file_path", "")
            if fp:
                fname = fp.rsplit("/", 1)[-1] if "/" in fp else fp
                lines = ""
                if tool_args.get("offset") or tool_args.get("limit"):
                    start = (tool_args.get("offset", 0) or 0) + 1
                    end = start + (tool_args.get("limit", 500) or 500) - 1
                    lines = f" (L{start}-{end})"
                if content:
                    line_count = str(content).count("\n") + 1
                    return f"Read {line_count} line(s) from {fname}"
                return f"{fname}{lines}"
        elif tool_name in ("grep", "search_content"):
            pattern = tool_args.get("pattern", "")
            if content and isinstance(content, str):
                match_count = content.count("\n") + 1 if content.strip() else 0
                return f'Found {match_count} match(es) for "{pattern[:40]}"'
            return f'"{pattern[:40]}"'
        elif tool_name in ("glob", "search_file"):
            pattern = tool_args.get("pattern", "")
            return f'pattern: {pattern}'
        elif tool_name == "ls":
            directory = tool_args.get("directory", ".")
            return directory.rsplit("/", 1)[-1] if "/" in directory else directory
        elif tool_name == "execute":
            cmd = tool_args.get("command", "")
            return cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif tool_name == "write_file":
            fp = tool_args.get("file_path", "")
            return fp.rsplit("/", 1)[-1] if "/" in fp else fp
        elif tool_name == "edit_file":
            fp = tool_args.get("file_path", "")
            return fp.rsplit("/", 1)[-1] if "/" in fp else fp
        elif tool_name == "web_search":
            queries = tool_args.get("queries", "")
            if isinstance(queries, list):
                return ", ".join(str(q)[:30] for q in queries[:2])
            return str(queries)[:60]
        elif tool_name == "fetch_url":
            url = tool_args.get("url", "")
            return url[:60] + ("..." if len(url) > 60 else "")
        
        # Default: show first arg value briefly
        for k, v in tool_args.items():
            return f"{k}={str(v)[:50]}"
        return ""


class BuiltinMemoryTool(Tool):
    """
    Built-in memory tool for saving important information to workspace.
    
    This tool allows the agent to persist important user information, preferences,
    and conversation highlights to the workspace memory system.
    
    Memory types:
    - Daily memory: Temporary notes for the current day (auto-cleared after 7 days)
    - Long-term memory: Persistent information (user preferences, important facts)
    """

    MEMORY_SYSTEM_PROMPT = dedent("""## save_memory Tool

    You have access to a `save_memory` tool to persist important information for future conversations.

    ### When to Use This Tool

    Use this tool to save:
    1. **User Preferences**: Language, communication style, technical level
    2. **Personal Information**: Name, occupation, interests (when shared by user)
    3. **Important Facts**: Key project details, decisions made, important context
    4. **User Requests**: When user explicitly asks to "remember this" or "save this"

    ### Memory Types

    - `long_term=False` (default): Daily memory - temporary notes, cleared after 7 days
    - `long_term=True`: Permanent memory - important preferences and facts that persist

    ### Guidelines

    1. **Be Selective**: Only save truly important or explicitly requested information
    2. **Be Concise**: Write clear, brief memory entries (1-2 sentences)
    3. **User Intent**: If user says "remember", "save", "note this" → use this tool
    4. **Privacy Aware**: Don't save sensitive information unless explicitly asked

    ### Examples

    - User says "I prefer Python over JavaScript" → save_memory("User prefers Python over JavaScript", long_term=True)
    - User says "Remember to check the API docs tomorrow" → save_memory("Check API docs", long_term=False)
    - User shares "My name is Alice, I'm a data scientist" → save_memory("User: Alice, data scientist", long_term=True)
    """)

    def __init__(self, workspace=None):
        """Initialize BuiltinMemoryTool.
        
        Args:
            workspace: Workspace instance for storing memories. If None, memories won't be persisted.
        """
        super().__init__(name="builtin_memory_tool")
        self._workspace = workspace
        self.register(self.save_memory)

    def set_workspace(self, workspace):
        """Set or update the workspace instance.
        
        Args:
            workspace: Workspace instance
        """
        self._workspace = workspace

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for memory tool usage."""
        return self.MEMORY_SYSTEM_PROMPT

    async def save_memory(self, content: str, long_term: bool = False) -> str:
        """Save important information to memory for future conversations.

        Use this tool to remember important user preferences, personal information,
        or any details that should be recalled in future conversations.

        Args:
            content: The information to remember. Should be a concise, clear statement.
                Examples:
                - "User prefers concise responses"
                - "User is working on a Python web project"
                - "User's name is Alice, she is a data scientist"
            long_term: If True, save to permanent long-term memory.
                If False (default), save to daily memory (cleared after 7 days).
                Use long_term=True for: user preferences, personal info, important facts.
                Use long_term=False for: temporary notes, daily tasks, short-term context.

        Returns:
            Confirmation message indicating where the memory was saved.
        """
        if not content or not content.strip():
            return "Error: Memory content cannot be empty."

        content = content.strip()
        
        if self._workspace is None:
            logger.warning("No workspace configured, memory not saved to disk.")
            return json.dumps({
                "success": False,
                "error": "No workspace configured. Memory not persisted.",
                "content": content,
            }, ensure_ascii=False)

        try:
            await self._workspace.save_memory(content, long_term=long_term)
            memory_type = "long-term" if long_term else "daily"
            logger.debug(f"Saved {memory_type} memory: {content[:50]}...")
            
            return json.dumps({
                "success": True,
                "memory_type": memory_type,
                "content": content,
                "message": f"Memory saved to {memory_type} storage.",
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to save memory: {e}",
                "content": content,
            }, ensure_ascii=False)


def get_builtin_tools(
        base_dir: Optional[str] = None,
        include_file_tools: bool = True,
        include_execute: bool = True,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        include_todos: bool = True,
        include_task: bool = True,
        include_skills: bool = True,
        include_memory: bool = True,
        task_model: Optional["Model"] = None,
        task_tools: Optional[List[Any]] = None,
        custom_skill_dirs: Optional[List[str]] = None,
        workspace=None,
    ) -> List[Tool]:
    """
    Get the list of built-in tools for DeepAgent.

    Args:
        base_dir: Base directory for file operations
        include_file_tools: Whether to include file tools (ls, read_file, write_file, edit_file, glob, grep)
        include_execute: Whether to include code execution tool
        include_web_search: Whether to include web search tool
        include_fetch_url: Whether to include URL fetching tool
        include_todos: Whether to include task management tools
        include_task: Whether to include subagent task tool
        include_skills: Whether to include skill tool for executing skills
        include_memory: Whether to include memory save tool
        task_model: Model for subagent tasks (optional, will use parent agent's model if not set)
        task_tools: Tools for subagent tasks (optional)
        custom_skill_dirs: Custom skill directories to load (optional)
        workspace: Workspace instance for memory tool (optional)

    Returns:
        List of tools
    """
    tools = []

    if include_file_tools:
        tools.append(BuiltinFileTool(base_dir=base_dir))

    if include_execute:
        tools.append(BuiltinExecuteTool(base_dir=base_dir))

    if include_web_search:
        tools.append(BuiltinWebSearchTool())

    if include_fetch_url:
        tools.append(BuiltinFetchUrlTool())

    if include_todos:
        tools.append(BuiltinTodoTool())

    if include_task:
        tools.append(BuiltinTaskTool(model=task_model, tools=task_tools))

    if include_memory:
        tools.append(BuiltinMemoryTool(workspace=workspace))

    if include_skills:
        from agentica.tools.skill_tool import SkillTool
        tools.append(SkillTool(custom_skill_dirs=custom_skill_dirs))

    return tools


if __name__ == '__main__':
    # Test file tool
    file_tool = BuiltinFileTool()
    print("=== ls test ===")
    print(file_tool.ls("."))

    print("\n=== glob test ===")
    print(file_tool.glob("*.py", "."))

    # Test search tool
    search_tool = BuiltinWebSearchTool()
    print("\n=== web_search test ===")
    print(search_tool.web_search("Python programming", max_results=2))

    # Test todo tool
    todo_tool = BuiltinTodoTool()
    print("\n=== write_todos test ===")
    print(todo_tool.write_todos([
        {"content": "Task 1", "status": "in_progress"},
        {"content": "Task 2", "status": "pending"},
    ]))
    print("\n=== read_todos test ===")
    print(todo_tool.read_todos())
