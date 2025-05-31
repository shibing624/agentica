# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools for workspace management, project operations, and file navigation.
"""
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from agentica.tools.base import Tool
from agentica.utils.log import logger


class WorkspaceTool(Tool):
    """
    A toolkit for workspace management and file navigation.
    Essential for building a code editor's project management functionality.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            list_files: bool = True,
            find_files: bool = True,
            create_directory: bool = True,
            move_file: bool = True,
            copy_file: bool = True,
            delete_file: bool = True,
            get_workspace_info: bool = True,
    ):
        """
        Initialize the WorkspaceTool.

        Args:
            work_dir: The root directory for workspace operations. Defaults to current directory.
            list_files: Whether to include the list_files function.
            find_files: Whether to include the find_files function.
            create_directory: Whether to include the create_directory function.
            move_file: Whether to include the move_file function.
            copy_file: Whether to include the copy_file function.
            delete_file: Whether to include the delete_file function.
            get_workspace_info: Whether to include the get_workspace_info function.
        """
        super().__init__(name="workspace_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()

        if list_files:
            self.register(self.list_files)
        if find_files:
            self.register(self.find_files)
        if create_directory:
            self.register(self.create_directory)
        if move_file:
            self.register(self.move_file)
        if copy_file:
            self.register(self.copy_file)
        if delete_file:
            self.register(self.delete_file)
        if get_workspace_info:
            self.register(self.get_workspace_info)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolves a file path, making it absolute if it's relative."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.work_dir.joinpath(path)
        return path

    def _get_file_info(self, path: Path, include_content_preview: bool = False) -> Dict[str, Any]:
        """Get detailed information about a file or directory."""
        is_dir = path.is_dir()

        info = {
            'name': path.name,
            'path': str(path),
            'is_directory': is_dir,
            'size_bytes': 0,
            'last_modified': '',
        }

        try:
            stat = path.stat()
            info['size_bytes'] = stat.st_size if not is_dir else sum(
                f.stat().st_size for f in path.glob('**/*') if f.is_file())
            info['last_modified'] = str(stat.st_mtime)

            # Format size for display
            size_bytes = info['size_bytes']
            if size_bytes < 1024:
                info['size_display'] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                info['size_display'] = f"{size_bytes / 1024:.1f} KB"
            else:
                info['size_display'] = f"{size_bytes / (1024 * 1024):.1f} MB"

            # Add content preview for text files if requested
            if include_content_preview and not is_dir and path.suffix.lower() in ['.txt', '.py', '.js', '.html', '.css',
                                                                                  '.json', '.md', '.xml']:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read(500)  # Read first 500 chars
                    if len(content) >= 500:
                        content = content[:497] + "..."
                    info['content_preview'] = content
                except Exception:
                    info['content_preview'] = "[Binary or inaccessible content]"
        except Exception as e:
            info['error'] = str(e)

        return info

    def list_files(self, directory: str = "", recursive: bool = False, pattern: str = "*",
                   include_hidden: bool = False, show_details: bool = False) -> str:
        """List files and directories in the specified directory.

        Args:
            directory (str): Directory path to list. Default is workspace root.
            recursive (bool): Whether to list files recursively.
            pattern (str): File glob pattern to filter by, e.g., "*.py".
            include_hidden (bool): Whether to include hidden files (starting with '.').
            show_details (bool): Whether to show detailed file information.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.list_files(directory="src", pattern="*.py", show_details=True)
            print(result)

        Returns:
            str: JSON-formatted list of files and directories.
        """
        try:
            dir_path = self._resolve_path(directory) if directory else self.work_dir

            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Directory not found: {directory or str(self.work_dir)}"

            # Prepare glob pattern
            if pattern and '*' not in pattern and '?' not in pattern:
                pattern = f"*{pattern}*"  # Make it more forgiving if no wildcards

            # Process files
            files = []

            if recursive:
                glob_pattern = '**/' + (pattern or '*')
                items = list(dir_path.glob(glob_pattern))
            else:
                glob_pattern = pattern or '*'
                items = list(dir_path.glob(glob_pattern))

            # Sort items: directories first, then files
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for item in items:
                # Skip hidden files if not included
                if not include_hidden and item.name.startswith('.'):
                    continue

                # Skip parent directory
                if item.name == '..' or item.name == '.':
                    continue

                if show_details:
                    files.append(self._get_file_info(item, include_content_preview=False))
                else:
                    files.append({
                        'name': item.name,
                        'path': str(item.relative_to(self.work_dir) if item.is_relative_to(self.work_dir) else item),
                        'is_directory': item.is_dir()
                    })

            result = {
                'directory': str(dir_path),
                'item_count': len(files),
                'items': files
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error listing files: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def find_files(self, pattern: str, directory: str = "", search_content: bool = False,
                   content_pattern: str = "", max_results: int = 100) -> str:
        """Find files matching a pattern, optionally searching file contents.

        Args:
            pattern (str): File glob pattern to match, e.g., "*.py".
            directory (str): Directory to search in. Default is workspace root.
            search_content (bool): Whether to search inside file contents.
            content_pattern (str): Pattern to search for in file contents.
            max_results (int): Maximum number of results to return.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.find_files(pattern="*.py", search_content=True, content_pattern="import")
            print(result)

        Returns:
            str: JSON-formatted search results.
        """
        try:
            dir_path = self._resolve_path(directory) if directory else self.work_dir

            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Directory not found: {directory or str(self.work_dir)}"

            # Find matching files
            matches = []
            count = 0

            # Handle special case where pattern is a direct path
            if (dir_path / pattern).exists():
                item = dir_path / pattern
                matches.append({
                    'path': str(item.relative_to(self.work_dir) if item.is_relative_to(self.work_dir) else item),
                    'type': 'directory' if item.is_dir() else 'file'
                })
                count += 1

            # Otherwise do a glob search
            else:
                # Make the pattern more forgiving if no wildcards
                if '*' not in pattern and '?' not in pattern and not pattern.startswith('.'):
                    search_pattern = f"**/*{pattern}*"
                else:
                    search_pattern = f"**/{pattern}"

                for item in dir_path.glob(search_pattern):
                    if count >= max_results:
                        break

                    match_info = {
                        'path': str(item.relative_to(self.work_dir) if item.is_relative_to(self.work_dir) else item),
                        'type': 'directory' if item.is_dir() else 'file'
                    }

                    # Search in file content if requested
                    if search_content and content_pattern and item.is_file():
                        try:
                            # Skip binary files
                            if item.suffix.lower() in ['.jpg', '.png', '.gif', '.pdf', '.zip', '.exe', '.bin']:
                                continue

                            with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                            # Search for pattern in content
                            matches_in_file = []
                            try:
                                for i, line in enumerate(content.splitlines(), 1):
                                    if content_pattern in line:
                                        matches_in_file.append({
                                            'line': i,
                                            'content': line.strip()[:100]  # Limit line length
                                        })

                                if matches_in_file:
                                    match_info['content_matches'] = matches_in_file[:5]  # Limit to 5 matches per file
                                    matches.append(match_info)
                                    count += 1
                            except Exception:
                                # Skip on errors processing specific lines
                                continue
                        except Exception:
                            # Skip files that can't be read as text
                            continue
                    else:
                        matches.append(match_info)
                        count += 1

            result = {
                'search_pattern': pattern,
                'search_directory': str(dir_path),
                'match_count': len(matches),
                'matches': matches,
                'content_search': content_pattern if search_content else None
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error finding files: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def create_directory(self, directory: str) -> str:
        """Create a new directory.

        Args:
            directory (str): Path of the directory to create.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.create_directory("new_project/src")
            print(result)

        Returns:
            str: Success or error message.
        """
        try:
            dir_path = self._resolve_path(directory)

            if dir_path.exists():
                return f"Directory already exists: {directory}"

            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

            return f"Successfully created directory: {directory}"

        except Exception as e:
            error_msg = f"Error creating directory {directory}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def move_file(self, source: str, destination: str, overwrite: bool = False) -> str:
        """Move a file or directory to a new location.

        Args:
            source (str): Path of the file or directory to move.
            destination (str): Destination path.
            overwrite (bool): Whether to overwrite if destination exists.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.move_file("old_name.py", "new_name.py")
            print(result)

        Returns:
            str: Success or error message.
        """
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)

            if not source_path.exists():
                return f"Error: Source not found: {source}"

            if dest_path.exists() and not overwrite:
                return f"Error: Destination already exists: {destination}. Use overwrite=True to replace."

            # Create parent directories if they don't exist
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            # If destination is a directory and exists, move into it with original filename
            if dest_path.exists() and dest_path.is_dir():
                dest_path = dest_path / source_path.name

            # Use shutil for directories or os.rename for files
            if source_path.is_dir():
                shutil.move(str(source_path), str(dest_path))
            else:
                os.rename(str(source_path), str(dest_path))

            logger.info(f"Moved {source_path} to {dest_path}")

            return f"Successfully moved {source} to {destination}"

        except Exception as e:
            error_msg = f"Error moving {source} to {destination}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def copy_file(self, source: str, destination: str, overwrite: bool = False) -> str:
        """Copy a file or directory to a new location.

        Args:
            source (str): Path of the file or directory to copy.
            destination (str): Destination path.
            overwrite (bool): Whether to overwrite if destination exists.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.copy_file("module.py", "backup/module.py")
            print(result)

        Returns:
            str: Success or error message.
        """
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)

            if not source_path.exists():
                return f"Error: Source not found: {source}"

            if dest_path.exists() and not overwrite:
                return f"Error: Destination already exists: {destination}. Use overwrite=True to replace."

            # Create parent directories if they don't exist
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            # If destination is a directory and exists, copy into it with original filename
            if dest_path.exists() and dest_path.is_dir():
                dest_path = dest_path / source_path.name

            # Use shutil for copying
            if source_path.is_dir():
                shutil.copytree(str(source_path), str(dest_path), dirs_exist_ok=overwrite)
            else:
                shutil.copy2(str(source_path), str(dest_path))

            logger.info(f"Copied {source_path} to {dest_path}")

            return f"Successfully copied {source} to {destination}"

        except Exception as e:
            error_msg = f"Error copying {source} to {destination}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def delete_file(self, path: str, recursive: bool = False) -> str:
        """Delete a file or directory.

        Args:
            path (str): Path of the file or directory to delete.
            recursive (bool): Whether to recursively delete directories.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.delete_file("old_file.py")
            print(result)

        Returns:
            str: Success or error message.
        """
        try:
            file_path = self._resolve_path(path)

            if not file_path.exists():
                return f"Error: Path not found: {path}"

            if file_path.is_dir():
                if not recursive:
                    return f"Error: {path} is a directory. Use recursive=True to delete directories."
                shutil.rmtree(str(file_path))
            else:
                os.remove(str(file_path))

            logger.info(f"Deleted {file_path}")

            return f"Successfully deleted {path}"

        except Exception as e:
            error_msg = f"Error deleting {path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_workspace_info(self, directory: str = "") -> str:
        """Get information about the workspace or a specific directory.

        Args:
            directory (str): Directory to analyze. Default is workspace root.

        Example:
            from agentica.tools.workspace_tool import WorkspaceTool
            ws = WorkspaceTool()
            result = ws.get_workspace_info()
            print(result)

        Returns:
            str: JSON-formatted workspace information.
        """
        try:
            dir_path = self._resolve_path(directory) if directory else self.work_dir

            if not dir_path.exists() or not dir_path.is_dir():
                return f"Error: Directory not found: {directory or str(self.work_dir)}"

            # Get basic info
            info = {
                'path': str(dir_path),
                'name': dir_path.name,
                'parent': str(dir_path.parent),
                'stats': {},
                'structure': {},
            }

            # Collect statistics
            all_files = list(dir_path.glob('**/*'))
            files = [f for f in all_files if f.is_file()]
            dirs = [d for d in all_files if d.is_dir()]

            info['stats'] = {
                'total_files': len(files),
                'total_directories': len(dirs),
                'total_size_bytes': sum(f.stat().st_size for f in files),
                'extensions': {},
            }

            # Format total size
            size_bytes = info['stats']['total_size_bytes']
            if size_bytes < 1024:
                info['stats']['total_size_display'] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                info['stats']['total_size_display'] = f"{size_bytes / 1024:.1f} KB"
            else:
                info['stats']['total_size_display'] = f"{size_bytes / (1024 * 1024):.1f} MB"

            # Count files by extension
            extensions = {}
            for file in files:
                ext = file.suffix.lower() or "(no extension)"
                extensions[ext] = extensions.get(ext, 0) + 1

            # Sort by count
            info['stats']['extensions'] = dict(sorted(extensions.items(), key=lambda x: x[1], reverse=True))

            # Detect project type
            project_type = "unknown"
            if (dir_path / "package.json").exists():
                project_type = "node"
            elif (dir_path / "requirements.txt").exists() or list(dir_path.glob("**/*.py")):
                project_type = "python"
            elif (dir_path / "pom.xml").exists():
                project_type = "java-maven"
            elif (dir_path / "build.gradle").exists():
                project_type = "java-gradle"
            elif (dir_path / "Cargo.toml").exists():
                project_type = "rust"
            elif (dir_path / "go.mod").exists():
                project_type = "go"

            info['project_type'] = project_type

            # Check for git
            if (dir_path / ".git").exists():
                info['git'] = True

            # Check for common directories and files
            common_dirs = ["src", "lib", "test", "tests", "docs", "examples"]
            found_dirs = [d.name for d in dir_path.iterdir() if d.is_dir() and d.name in common_dirs]
            if found_dirs:
                info['common_directories'] = found_dirs

            # Get simplified directory structure (1 level deep)
            structure = {}
            for item in dir_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    subfiles = len(list(item.glob('*')))
                    structure[item.name] = f"Directory ({subfiles} items)"
                elif item.is_file():
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    structure[item.name] = f"File ({size_str})"

            info['structure'] = structure

            return json.dumps(info, indent=2)

        except Exception as e:
            error_msg = f"Error getting workspace info: {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == '__main__':
    # Simple test
    workspace = WorkspaceTool()

    # Test directory operations
    print(workspace.create_directory("test_workspace"))

    # Create some files for testing
    test_dir = Path("test_workspace")
    (test_dir / "file1.txt").write_text("Test file 1")
    (test_dir / "file2.py").write_text("print('Hello world')")
    (test_dir / "subdir").mkdir(exist_ok=True)
    (test_dir / "subdir" / "file3.md").write_text("# Test Markdown")

    # Test listing
    print("\nListing files:")
    print(workspace.list_files("test_workspace", show_details=True))

    # Test finding
    print("\nFinding files:")
    print(workspace.find_files(pattern="*.py", directory="test_workspace"))

    # Test copying
    print("\nCopying file:")
    print(workspace.copy_file("test_workspace/file1.txt", "test_workspace/file1_copy.txt"))

    # Test workspace info
    print("\nWorkspace info:")
    print(workspace.get_workspace_info("test_workspace"))

    if os.path.exists("test_workspace"):
        shutil.rmtree("test_workspace")
