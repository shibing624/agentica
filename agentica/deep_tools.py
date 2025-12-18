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
- execute: Execute Python code
- web_search: Web search (implemented using BaiduSearch)
- fetch_url: Fetch URL content (implemented using UrlCrawler)
- write_todos: Create and manage task list
- read_todos: Read current task list
- task: Launch subagent to handle complex tasks
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Callable, Union, TYPE_CHECKING

from agentica.tools.base import Tool
from agentica.utils.log import logger

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
        """Resolve path, supporting both absolute and relative paths."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def _validate_path(self, path: str) -> str:
        """Validate path security to prevent path traversal attacks."""
        if ".." in path or path.startswith("~"):
            raise ValueError(f"Path traversal not allowed: {path}")
        return path

    def ls(self, path: str = ".") -> str:
        """List all files and subdirectories in a directory.

        Args:
            path: Directory path to list, defaults to current directory

        Returns:
            JSON formatted file list
        """
        try:
            self._validate_path(path)
            dir_path = self._resolve_path(path)

            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                item_type = "dir" if item.is_dir() else "file"
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": item_type,
                })

            logger.info(f"Listed {len(items)} items in {dir_path}")
            return json.dumps(items, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return f"Error listing directory: {e}"

    def read_file(
            self,
            file_path: str,
            offset: int = 0,
            limit: Optional[int] = None,
    ) -> str:
        """Read file content.

        Args:
            file_path: File path, support md, txt, py, etc. absolute or relative path
            offset: Starting line number (0-based)
            limit: Maximum number of lines to read, defaults to max_read_lines

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

            logger.info(f"Read file {file_path}: lines {offset + 1}-{end_line}")
            return result
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"

    def write_file(self, file_path: str, content: str) -> str:
        """Create a new file or completely overwrite an existing file.

        Args:
            file_path: File path
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

        Args:
            file_path: File path
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
        """Find files matching a pattern.

        Args:
            pattern: Glob pattern, e.g., "*.py", "**/*.md"
            path: Starting directory for search

        Returns:
            JSON formatted list of matching files
        """
        try:
            self._validate_path(path)
            base_path = self._resolve_path(path)

            if not base_path.exists():
                return f"Error: Directory not found: {path}"

            matches = list(base_path.glob(pattern))

            # Exclude common ignored directories
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.pytest_cache'}
            filtered = [
                str(m) for m in matches
                if not set(m.parts).intersection(ignore_dirs)
            ]

            logger.info(f"Found {len(filtered)} files matching '{pattern}'")
            return json.dumps(sorted(filtered), ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error in glob {pattern}: {e}")
            return f"Error in glob: {e}"

    def grep(
            self,
            pattern: str,
            path: str = ".",
            glob_pattern: Optional[str] = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
            max_results: int = 50,
    ) -> str:
        """Search for text pattern in files.

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
            logger.info(f"Searching for '{pattern}' in '{path}'")

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
                return "\n".join(output) if output else f"No matches found for '{pattern}'"
            elif output_mode == "files_with_matches":
                return json.dumps(list(set(results)), ensure_ascii=False, indent=2)
            else:  # content
                output_lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results]
                return "\n".join(output_lines) if output_lines else f"No matches found for '{pattern}'"

        except Exception as e:
            logger.error(f"Error in grep: {e}")
            return f"Error in grep: {e}"


class BuiltinExecuteTool(Tool):
    """
    Built-in code execution tool using Python interpreter.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize BuiltinExecuteTool.

        Args:
            base_dir: Base directory for code execution
        """
        super().__init__(name="builtin_execute_tool")
        self.base_dir = base_dir or os.getcwd()
        self.register(self.execute)

    def execute(self, code: str) -> str:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Execution result (stdout output or error message)
        """
        import io
        import sys
        import traceback

        logger.info(f"Executing code:\n{code}")

        old_stdout, old_stderr = sys.stdout, sys.stderr
        new_stdout, new_stderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = new_stdout, new_stderr

        try:
            compiled_code = compile(code, '<execute>', 'exec')
            namespace = {'__name__': '__main__'}
            exec(compiled_code, namespace)

            stdout_output = new_stdout.getvalue()
            stderr_output = new_stderr.getvalue()

            result_parts = []
            if stdout_output:
                result_parts.append(stdout_output.strip())
            if stderr_output:
                result_parts.append(f"[stderr]\n{stderr_output.strip()}")

            return "\n".join(result_parts) if result_parts else "Code executed successfully (no output)"
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Execution error: {e}")
            return f"Error: {e}\n\nTraceback:\n{error_traceback}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            new_stdout.close()
            new_stderr.close()


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
        """Execute Baidu search for multiple queries and return results

        Args:
            queries (Union[str, List[str]]): Search keyword(s), can be a single string or a list of strings
            max_results (int, optional): Maximum number of results to return for each query, default 5

        Returns:
            str: A JSON formatted string containing the search results.
        """

        try:
            result = self._search.baidu_search(queries, max_results=max_results)
            logger.info(f"Web search for '{queries}', result length: {len(result)}, prewiview: {result[:200]}...")
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
        self._crawler = UrlCrawlerTool(base_dir='/tmp/', max_content_length=max_content_length)
        self.register(self.fetch_url)

    def fetch_url(self, url: str) -> str:
        """Fetch URL content and convert to clean text format.

        Args:
            url: URL to fetch, url starts with http:// or https://

        Returns:
            JSON formatted fetch result containing url, content, and save_path
        """
        result = self._crawler.url_crawl(url)
        logger.info(f"Fetched URL: {url}, result length: {len(result)}, preview: {result[:200]}...")
        return result


class BuiltinTodoTool(Tool):
    """
    Built-in task management tool providing write_todos and read_todos functions.
    Used for tracking progress of complex tasks.
    """

    def __init__(self):
        """Initialize BuiltinTodoTool."""
        super().__init__(name="builtin_todo_tool")
        self._todos: List[Dict[str, Any]] = []
        self.register(self.write_todos)
        self.register(self.read_todos)

    def write_todos(self, todos: List[Dict[str, str]]) -> str:
        """Create and manage a structured task list.

        Each task item should contain:
        - content: Task description
        - status: Task status ("pending", "in_progress", "completed")

        Args:
            todos: Task list, each task is a dict with content and status

        Returns:
            Updated task list
        """
        try:
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
            logger.info(f"Updated todo list: {len(self._todos)} items")

            return json.dumps({
                "message": f"Updated todo list with {len(self._todos)} items",
                "todos": self._todos,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error updating todos: {e}")
            return f"Error updating todos: {e}"

    def read_todos(self) -> str:
        """Read current task list.

        Returns:
            JSON formatted current task list
        """
        if not self._todos:
            return json.dumps({"message": "No todos found", "todos": []}, ensure_ascii=False, indent=2)

        # Count status statistics
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0}
        for todo in self._todos:
            status = todo.get("status", "pending")
            if status in status_counts:
                status_counts[status] += 1
        logger.info(f"Todo list summary: {status_counts}")
        return json.dumps({"summary": status_counts, "todos": self._todos}, ensure_ascii=False, indent=2)


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

    # Task tool description for the LLM
    TASK_TOOL_DESCRIPTION = """Launch a subagent to handle complex, multi-step tasks with isolated context.

## When to use this tool:
1. Complex multi-step tasks that require focused work
2. Tasks that can run independently without needing intermediate feedback
3. Research or analysis tasks that would benefit from isolated context
4. When you want to parallelize work by launching multiple subagents

## When NOT to use this tool:
1. Simple tasks that can be done in 1-2 tool calls
2. Tasks that require continuous user interaction
3. Tasks where you need to see intermediate steps

## Usage:
- Provide a clear, detailed description of the task
- Specify what information should be returned
- The subagent will complete the task and return a single result

## Examples:
- "Research the top 5 Python web frameworks and summarize their pros and cons"
- "Analyze the code in src/ directory and identify potential security issues"
- "Search for recent news about AI and create a summary report"
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
            from agentica.agent import Agent

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

            # Create subagent
            subagent = Agent(
                model=model,
                name=f"Subagent-{subagent_type}",
                description=f"Subagent for handling: {description[:50]}...",
                system_prompt=self._system_prompt,
                tools=subagent_tools,
                markdown=True,
            )

            logger.info(f"Launching subagent [{subagent_type}] for task: {description[:100]}...")

            # Run subagent with the task description
            response = subagent.run(description)

            # Extract result
            result = response.content if response else "Subagent completed but returned no content."

            logger.info(f"Subagent [{subagent_type}] completed task.")

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
                "description": description[:200],
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
