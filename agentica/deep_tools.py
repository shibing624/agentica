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
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING, Union

from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.utils.string import truncate_if_too_long

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model


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
        """Resolve path, supporting both absolute and relative paths.
        
        If path is absolute but not under base_dir, treat it as relative to base_dir.
        This prevents writing to system directories like root (/).
        """
        p = Path(path)
        if p.is_absolute():
            # Check if the path is under base_dir
            try:
                p.relative_to(self.base_dir)
                return p
            except ValueError:
                # Path is not under base_dir, treat the path name as relative
                # e.g., /research_xxx -> base_dir/research_xxx
                relative_path = str(p).lstrip("/")
                return self.base_dir / relative_path
        return self.base_dir / p

    def _validate_path(self, path: str) -> str:
        """Validate path security to prevent path traversal attacks."""
        if ".." in path or path.startswith("~"):
            raise ValueError(f"Path traversal not allowed: {path}")
        return path

    def ls(self, directory: str = ".") -> str:
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

            items = []
            for item in sorted(dir_path.iterdir()):
                item_type = "dir" if item.is_dir() else "file"
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": item_type,
                })

            logger.info(f"Listed {len(items)} items in {dir_path}")
            result = json.dumps(items, ensure_ascii=False, indent=2)
            result = truncate_if_too_long(result)
            return str(result)
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return f"Error listing directory: {e}"

    def read_file(
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

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            total_lines = len(lines)
            end_line = min(offset + limit, total_lines)
            selected_lines = lines[offset:end_line]

            # Format output with line numbers
            output_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                line = line.rstrip('\n\r')
                if len(line) > self.max_line_length:
                    line = line[:self.max_line_length] + "..."
                output_lines.append(f"{i:6d}\t{line}")

            result = "\n".join(output_lines)

            # Add file info if truncated
            if end_line < total_lines:
                result += f"\n\n[Showing lines {offset + 1}-{end_line} of {total_lines} total lines]"

            logger.info(f"Read file {file_path}: lines {offset + 1}-{end_line}, total {total_lines} lines")
            return result
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"

    def write_file(self, file_path: str, content: str) -> str:
        """Writes to a new file in the filesystem.

        Usage:
        - The file_path parameter must be an absolute path, not a relative path
        - The content parameter must be a string
        - The write_file tool will create the a new file.
        - Prefer to edit existing files over creating new ones when possible.
        
        Args:
            file_path: File absolute path
            content: File content

        Returns:
            Operation result message
        """
        try:
            self._validate_path(file_path)
            path = self._resolve_path(file_path)

            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            action = "Created" if not path.exists() else "Updated"

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"{action} file: {path}")
            return f"{action} file: {file_path}"
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return f"Error writing file: {e}"

    def edit_file(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> str:
        """Perform exact string replacement in a file.

        Usage:
        - You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
        - When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
        - ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
        - Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
        - The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
        - Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

        Args:
            file_path: File absolute path
            old_string: Original string to replace
            new_string: New string to replace with
            replace_all: Whether to replace all occurrences, defaults to replacing only the first

        Returns:
            Operation result message
        """
        try:
            self._validate_path(file_path)
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"
            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if old_string exists
            count = content.count(old_string)
            if count == 0:
                return f"Error: String not found in file: '{old_string[:50]}...'"

            # If not replace_all, check for uniqueness
            if not replace_all and count > 1:
                return (f"Error: Found {count} occurrences of the string. "
                        f"Use replace_all=True to replace all, or provide more context to make it unique.")

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            replaced_count = count if replace_all else 1
            logger.info(f"Replaced {replaced_count} occurrence(s) in {file_path}")
            return f"Successfully replaced {replaced_count} occurrence(s) in '{file_path}'"
        except Exception as e:
            logger.error(f"Error editing file {file_path}: {e}")
            return f"Error editing file: {e}"

    def glob(self, pattern: str, path: str = ".") -> str:
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

            # Get all files matching the glob pattern
            matches = list(base_path.glob(pattern))

            # Exclude common ignored directories to avoid invalid files
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            filtered = [
                str(m) for m in matches
                if not set(m.parts).intersection(ignore_dirs)
            ]

            logger.info(f"Glob found {len(filtered)} files matching pattern '{pattern}' in directory '{path}'")
            # Convert to formatted JSON string
            result = json.dumps(sorted(filtered), ensure_ascii=False, indent=2)
            # Truncate if content exceeds the limit to avoid excessive output
            result = truncate_if_too_long(result)
            return str(result)
        except Exception as e:
            logger.error(f"Exception occurred during glob search (pattern: '{pattern}', path: '{path}'): {str(e)}")
            return f"Error in glob search: {str(e)}"

    def grep(
            self,
            pattern: str,
            path: str = ".",
            glob_pattern: Optional[str] = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
            max_results: int = 50,
    ) -> str:
        """Search for a pattern in files.

        Usage:
        - The grep tool searches for text patterns across files
        - The pattern parameter is the text to search for (literal string, not regex)
        - The path parameter filters which directory to search in (default is the current working directory)
        - The glob parameter accepts a glob pattern to filter which files to search (e.g., `*.py`)
        - The output_mode parameter controls the output format:
        - `files_with_matches`: List only file paths containing matches (default)
        - `content`: Show matching lines with file path and line numbers
        - `count`: Show count of matches per file

        Args:
            pattern: Text to search for (literal string)
            path: Starting directory for search
            glob_pattern: File filter pattern, e.g., "*.py"
            output_mode: Output mode
                - "files_with_matches": Only list file paths containing matches
                - "content": Show matching lines with context
                - "count": Show match count per file
            max_results: Maximum number of results

        Returns:
            Search results
        """
        try:
            self._validate_path(path)
            base_path = self._resolve_path(path)
            if not base_path.exists():
                return f"Error: Directory not found: {path}"

            # Determine files to search
            if glob_pattern:
                files = list(base_path.glob(f"**/{glob_pattern}"))
            else:
                files = list(base_path.glob("**/*"))

            # Exclude directories and ignored paths
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            files = [f for f in files if f.is_file() and not set(f.parts).intersection(ignore_dirs)]

            results = []
            file_counts = {}

            for file_path in files:
                if len(results) >= max_results:
                    break

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    file_matches = []
                    for line_num, line in enumerate(lines, 1):
                        if pattern in line:
                            file_matches.append({
                                "line_num": line_num,
                                "content": line.strip()[:200],
                            })

                    if file_matches:
                        file_counts[str(file_path)] = len(file_matches)
                        if output_mode == "content":
                            for match in file_matches[:max_results - len(results)]:
                                results.append({
                                    "file": str(file_path),
                                    "line": match["line_num"],
                                    "content": match["content"],
                                })
                        elif output_mode == "files_with_matches":
                            results.append(str(file_path))
                except Exception:
                    continue

            # Format output
            if output_mode == "count":
                output = [f"{p}: {c}" for p, c in file_counts.items()]
                result = "\n".join(output) if output else f"No matches found for '{pattern}'"
            elif output_mode == "files_with_matches":
                result = json.dumps(list(set(results)), ensure_ascii=False, indent=2)
            else:  # content
                output_lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results]
                result = "\n".join(output_lines) if output_lines else f"No matches found for '{pattern}'"

            # Truncate if too long and log result
            result = truncate_if_too_long(result)
            logger.info(f"Grep for '{pattern}': found {len(file_counts)} files, result length: {len(result)} characters.")
            return str(result)

        except Exception as e:
            logger.error(f"Error in grep: {e}")
            return f"Error in grep: {e}"

class BuiltinExecuteTool(Tool):
    """
    Built-in command execution tool that wraps ShellTool.
    Exposed as execute function for consistent naming in DeepAgent.
    """

    def __init__(self, base_dir: Optional[str] = None, timeout: int = 120):
        """
        Initialize BuiltinExecuteTool.

        Args:
            base_dir: Base directory for command execution
            timeout: Command execution timeout in seconds
        """
        super().__init__(name="builtin_execute_tool")
        # Import and initialize ShellTool
        from agentica.tools.shell_tool import ShellTool
        self._shell = ShellTool(base_dir=base_dir, timeout=timeout)
        self.register(self.execute)

    def execute(self, command: str) -> str:
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
            - python "/path/with spaces/script.py" (correct)
            - python /path/with spaces/script.py (incorrect - will fail)
        - After ensuring proper quoting, execute the command
        - Capture the output of the command

        Usage notes:
        - The command parameter is required
        - Commands run in an isolated sandbox environment
        - Returns combined stdout/stderr output with exit code
        - If the output is very large, it may be truncated
        - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
        - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
            - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
            - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
        - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

        Examples:
        Good examples:
            - execute(command="python /path/to/script.py")
            - execute(command="pytest /path/to/tests/test.py")
            - execute(command="python -c 'print(33333**2 + 332.2 / 12)'")
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
        return self._shell.execute(command)


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

    def web_search(self, queries: Union[str, List[str]], max_results: int = 5) -> str:
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
            result = self._search.baidu_search(queries, max_results=max_results)
            logger.info(f"Web search for '{queries}', result length: {len(result)} characters.")
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
        # Import and initialize UrlCrawlerTool
        from agentica.tools.url_crawler_tool import UrlCrawlerTool
        self._crawler = UrlCrawlerTool(base_dir='./tmp/', max_content_length=max_content_length)
        self.register(self.fetch_url)

    def fetch_url(self, url: str) -> str:
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
        result = self._crawler.url_crawl(url)
        logger.info(f"Fetched URL: {url}, result length: {len(result)} characters.")
        return result


class BuiltinTodoTool(Tool):
    """
    Built-in task management tool providing write_todos and read_todos functions.
    Used for tracking progress of complex tasks.
    """

    # System prompt for todo tool usage guidance
    WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""

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
            logger.info(f"Updated todo list: {len(self._todos)} items, todos: {self._todos}")

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

        logger.info(f"Todo list: {status_counts}, progress: {progress_pct}%")
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
    """

    # Default system prompt for subagents
    DEFAULT_SUBAGENT_PROMPT = """You are a helpful assistant that completes tasks autonomously.
Focus on the specific task given to you and provide a clear, concise result.
Use the available tools to accomplish your task efficiently."""

    # System prompt for task tool usage guidance (injected into main agent)
    TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""

    # Task tool description for the LLM
    TASK_TOOL_DESCRIPTION = """Launch a subagent to handle complex, multi-step tasks with isolated context.

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, don't use the task tool
</commentary>
</example>

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit
"""

    def __init__(
            self,
            model: Optional["Model"] = None,
            tools: Optional[List[Any]] = None,
            system_prompt: Optional[str] = None,
            max_iterations: int = 10,
    ):
        """
        Initialize BuiltinTaskTool.

        Args:
            model: Model to use for subagents. If None, will use the parent agent's model.
            tools: Tools available to subagents. If None, will use basic tools.
            system_prompt: System prompt for subagents.
            max_iterations: Maximum iterations for subagent execution.
        """
        super().__init__(name="builtin_task_tool")
        self._model = model
        self._tools = tools
        self._system_prompt = system_prompt or self.DEFAULT_SUBAGENT_PROMPT
        self._max_iterations = max_iterations
        self._parent_agent: Optional["Agent"] = None
        self.register(self.task)

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for task tool usage guidance."""
        return self.TASK_SYSTEM_PROMPT

    def set_parent_agent(self, agent: "Agent") -> None:
        """Set the parent agent reference for accessing model and tools."""
        self._parent_agent = agent

    def task(self, description: str, subagent_type: str = "general-purpose") -> str:
        """Launch a subagent to handle a complex task.

        Args:
            description: Detailed description of the task to perform.
                Include what you want the subagent to do and what information to return.
            subagent_type: Type of subagent to use. Currently supports "general-purpose".
                Default is "general-purpose" which has access to all standard tools.

        Returns:
            The result from the subagent after completing the task.
        """
        try:
            # Get model from parent agent or use configured model
            model = self._model
            if model is None and self._parent_agent is not None:
                model = self._parent_agent.model

            if model is None:
                return "Error: No model available for subagent. Please configure a model."

            # Get tools for subagent
            subagent_tools = self._tools
            if subagent_tools is None:
                # Use basic file and search tools for subagent
                subagent_tools = [
                    BuiltinFileTool(),
                    BuiltinWebSearchTool(),
                    BuiltinFetchUrlTool(),
                ]

            # Import Agent here to avoid circular imports
            from agentica.agent import Agent

            # Create subagent
            subagent = Agent(
                model=model,
                name=f"Subagent-{subagent_type}",
                description=f"Subagent for handling: {description[:100]}",
                system_prompt=self._system_prompt,
                add_datetime_to_instructions=True,
                tools=subagent_tools,
                markdown=True,
            )

            logger.info(f"Launching subagent [{subagent_type}] for task: {description}")
            # Run subagent with the task description
            response = subagent.run(description)

            # Extract result
            result = response.content if response else "Subagent completed but returned no content."
            # logger.info(f"Subagent [{subagent_type}] completed task.")
            return json.dumps({
                "success": True,
                "subagent_type": subagent_type,
                "result": result,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Subagent task error: {e}")
            return json.dumps({
                "success": False,
                "error": f"Subagent task error: {e}",
                "description": description[:300],
            }, ensure_ascii=False)


def get_builtin_tools(
        base_dir: Optional[str] = None,
        include_file_tools: bool = True,
        include_execute: bool = True,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        include_todos: bool = True,
        include_task: bool = True,
        task_model: Optional["Model"] = None,
        task_tools: Optional[List[Any]] = None,
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
        task_model: Model for subagent tasks (optional, will use parent agent's model if not set)
        task_tools: Tools for subagent tasks (optional)

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
