# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description: Load and save files with search capabilities.
"""
import os
import re
import json
from pathlib import Path
from typing import Optional
from agentica.utils.markdown_converter import MarkdownConverter
from agentica.utils.log import logger
from agentica.tools.base import Tool


class FileTool(Tool):
    def __init__(
            self,
            data_dir: Optional[str] = None,
            save_file: bool = True,
            read_file: bool = True,
            list_files: bool = False,
            read_files: bool = False,
            search_files: bool = False,
            search_content: bool = False,
    ):
        """
        Initialize the FileTool.

        Args:
            data_dir: The base directory for file operations.
            save_file: Whether to include the save_file function.
            read_file: Whether to include the read_file function.
            list_files: Whether to include the list_files function.
            read_files: Whether to include the read_files function.
            search_files: Whether to include the search_files function (search by filename).
            search_content: Whether to include the search_content function (search file contents).
        """
        super().__init__(name="file_tool")

        self.data_dir: Path = Path(data_dir) if data_dir else Path.cwd()
        if save_file:
            self.register(self.save_file, sanitize_arguments=False)
        if read_file:
            self.register(self.read_file)
        if list_files:
            self.register(self.list_files)
        if read_files:
            self.register(self.read_files)
        if search_files:
            self.register(self.search_files)
        if search_content:
            self.register(self.search_content)

    def save_file(self, contents: str, file_name: str, overwrite: bool = True, save_dir: str = "") -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        Args:
            contents (str): The contents to save.
            file_name (str): The name of the file to save to.
            overwrite (bool): Overwrite the file if it already exists.
            save_dir (str): The directory to save the file to, defaults to the base directory.

        Returns:
            str: The file name if successful, otherwise returns an error message.
        """
        try:
            if save_dir:
                save_dir = Path(save_dir)
            else:
                save_dir = self.data_dir
            file_path = save_dir.joinpath(file_name)
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents)
            logger.info(f"Saved contents to file: {file_path}")
            return str(file_name)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_file(self, file_name: str) -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.

        Args:
            file_name (str): The name of the file to read.

        Returns:
            str: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            if os.path.exists(file_name):
                path = Path(file_name)
            else:
                path = self.data_dir.joinpath(file_name)
            logger.info(f"Reading file: {path}")

            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")
            return MarkdownConverter().convert(str(path)).text_content
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self, dir_path: str = "") -> str:
        """Returns a list of files in the base directory

        Args:
            dir_path (str): The directory to list files from, defaults to the base directory.

        Returns:
            str: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            if dir_path:
                data_dir = Path(dir_path)
            else:
                data_dir = self.data_dir
            logger.info(f"Reading files in : {data_dir}")
            return json.dumps([str(file_path) for file_path in data_dir.iterdir()],
                              indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

    def read_files(self, dir_path: str = "") -> str:
        """Reads the contents of all files in the base directory and returns the contents.

        Args:
            dir_path (str): The directory to read files from, defaults to the base directory.

        Returns:
            str: The contents of all files if successful, otherwise returns an error message.
        """
        try:
            if dir_path:
                data_dir = Path(dir_path)
            else:
                data_dir = self.data_dir
            logger.info(f"Reading all files in: {data_dir}")
            all_contents = []
            for file_path in data_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"Reading file: {file_path}")
                    contents = self.read_file(str(file_path))
                    all_contents.append(f"Contents of {file_path.name}:\n{contents}")
            return json.dumps(all_contents, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

    def search_files(
            self,
            pattern: str,
            dir_path: str = "",
            recursive: bool = True,
            file_extensions: Optional[str] = None,
    ) -> str:
        """Search for files by name pattern.

        Args:
            pattern (str): The pattern to search for in file names (supports wildcards like *.py, test*.txt).
            dir_path (str): The directory to search in, defaults to the base directory.
            recursive (bool): Whether to search recursively in subdirectories.
            file_extensions (str): Comma-separated list of file extensions to filter (e.g., ".py,.txt,.md").

        Returns:
            str: A JSON list of matching file paths.
        """
        try:
            if dir_path:
                search_dir = Path(dir_path)
            else:
                search_dir = self.data_dir

            if not search_dir.exists():
                return f"Error: Directory '{search_dir}' does not exist."

            # Parse file extensions filter
            ext_filter = None
            if file_extensions:
                ext_filter = [ext.strip().lower() for ext in file_extensions.split(",")]
                ext_filter = [ext if ext.startswith(".") else f".{ext}" for ext in ext_filter]

            # Convert wildcard pattern to regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
            regex = re.compile(regex_pattern, re.IGNORECASE)

            matching_files = []
            if recursive:
                for root, dirs, files in os.walk(search_dir):
                    # Skip hidden directories and common ignore patterns
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "__pycache__", ".git"]]
                    for file in files:
                        if regex.search(file):
                            file_path = Path(root) / file
                            if ext_filter is None or file_path.suffix.lower() in ext_filter:
                                matching_files.append(str(file_path))
            else:
                for file_path in search_dir.iterdir():
                    if file_path.is_file() and regex.search(file_path.name):
                        if ext_filter is None or file_path.suffix.lower() in ext_filter:
                            matching_files.append(str(file_path))

            logger.info(f"Found {len(matching_files)} files matching pattern '{pattern}'")
            return json.dumps(matching_files, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return f"Error searching files: {e}"

    def search_content(
            self,
            query: str,
            dir_path: str = "",
            file_extensions: str = ".py,.txt,.md,.json,.yaml,.yml,.js,.ts,.html,.css",
            recursive: bool = True,
            max_results: int = 50,
            context_lines: int = 2,
            use_regex: bool = False,
    ) -> str:
        """Search for content within files.

        Args:
            query (str): The text or regex pattern to search for.
            dir_path (str): The directory to search in, defaults to the base directory.
            file_extensions (str): Comma-separated list of file extensions to search in.
            recursive (bool): Whether to search recursively in subdirectories.
            max_results (int): Maximum number of results to return.
            context_lines (int): Number of context lines to show before and after each match.
            use_regex (bool): Whether to treat query as a regular expression.

        Returns:
            str: A formatted string containing search results with file paths, line numbers, and context.
        """
        try:
            if dir_path:
                search_dir = Path(dir_path)
            else:
                search_dir = self.data_dir

            if not search_dir.exists():
                return f"Error: Directory '{search_dir}' does not exist."

            # Parse file extensions
            ext_filter = [ext.strip().lower() for ext in file_extensions.split(",")]
            ext_filter = [ext if ext.startswith(".") else f".{ext}" for ext in ext_filter]

            # Compile regex pattern
            if use_regex:
                try:
                    pattern = re.compile(query, re.IGNORECASE | re.MULTILINE)
                except re.error as e:
                    return f"Error: Invalid regex pattern: {e}"
            else:
                # Escape special regex characters for literal search
                pattern = re.compile(re.escape(query), re.IGNORECASE)

            results = []
            files_searched = 0

            def search_file(file_path: Path):
                nonlocal files_searched
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    files_searched += 1

                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            # Get context lines
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context = []
                            for j in range(start, end):
                                prefix = ">>>" if j == i else "   "
                                context.append(f"{prefix} {j + 1}: {lines[j]}")

                            results.append({
                                "file": str(file_path),
                                "line": i + 1,
                                "match": line.strip(),
                                "context": "\n".join(context),
                            })

                            if len(results) >= max_results:
                                return True  # Stop searching
                except Exception:
                    pass  # Skip files that can't be read
                return False

            # Walk through files
            if recursive:
                for root, dirs, files in os.walk(search_dir):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "__pycache__", ".git"]]
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix.lower() in ext_filter:
                            if search_file(file_path):
                                break
                    if len(results) >= max_results:
                        break
            else:
                for file_path in search_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ext_filter:
                        if search_file(file_path):
                            break

            # Format results
            if not results:
                return f"No matches found for '{query}' in {files_searched} files."

            output = [f"Found {len(results)} matches for '{query}' in {files_searched} files:\n"]
            for r in results:
                output.append(f"\n{'=' * 60}")
                output.append(f"File: {r['file']}")
                output.append(f"Line {r['line']}: {r['match']}")
                output.append(f"\nContext:\n{r['context']}")

            logger.info(f"Found {len(results)} matches for '{query}'")
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return f"Error searching content: {e}"


if __name__ == '__main__':
    m = FileTool(search_files=True, search_content=True)
    print(m.save_file(contents="Hello, world!", file_name="hello.txt"))
    print(m.read_file(file_name="hello.txt"))
    print(m.list_files())
    print(m.read_files())
    print(m.read_files(dir_path="."))

    # Test search_files
    print("\n--- Search Files ---")
    print(m.search_files(pattern="*.py", dir_path=".", recursive=False))

    # Test search_content
    print("\n--- Search Content ---")
    print(m.search_content(query="FileTool", dir_path=".", recursive=False))

    if os.path.exists("hello.txt"):
        os.remove("hello.txt")
