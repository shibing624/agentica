# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools for editing files - useful for code editors and AI coding assistants.
"""
import os
import re
import difflib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from agentica.tools.base import Tool
from agentica.utils.log import logger


class EditTool(Tool):
    """
    A toolkit for editing files, particularly useful for code editing features.
    Provides functionality similar to modern code editors.
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

    def edit_file(self, target_file: str, code_edit: str) -> str:
        """Edit a file or create a new one with the specified content.

        Args:
            target_file (str): The path of the file to edit.
            code_edit (str): The new content for the file or a patch in the format '*** Begin Patch\n...'

        Example:
            from agentica.tools.edit_tool import EditTool
            editor = EditTool()
            result = editor.edit_file("example.py", "print('Hello, World!')")
            print(result)

        Returns:
            str: A message describing the result of the operation.
        """
        try:
            file_path = self._resolve_path(target_file)
            
            # Check if this is a patch by looking for patch format markers
            if code_edit.startswith("*** Begin Patch") or "@@" in code_edit[:100]:
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

        Args:
            target_file (str): The path of the file to patch.
            patch_content (str): The patch content in unified diff format.

        Example:
            from agentica.tools.edit_tool import EditTool
            editor = EditTool()
            patch = "*** Begin Patch ***\\n@@ -1,1 +1,2 @@\\n print('Hello')\\n+print('World')\\n"
            result = editor.apply_patch("example.py", patch)
            print(result)

        Returns:
            str: A message describing the result of the operation.
        """
        try:
            file_path = self._resolve_path(target_file)
            
            # Clean up the patch - remove any wrapper markers
            if "*** Begin Patch" in patch_content:
                # Extract just the diff part
                match = re.search(r'(?:\*{3} Begin Patch \*{3}[\r\n]+)([\s\S]+?)(?:[\r\n]+\*{3} End Patch \*{3}|$)', patch_content)
                if match:
                    patch_content = match.group(1)
            
            # Check if file exists
            if not file_path.exists():
                # For new files, just create with the content after the patch header
                # Extract the content after the patch header (anything after +++ line)
                content_match = re.search(r'\+{3}[^\n]*\n([\s\S]+)', patch_content)
                if content_match:
                    content = content_match.group(1)
                    # Remove the '+' markers at the beginning of lines
                    content = re.sub(r'^\+', '', content, flags=re.MULTILINE)
                    return self.edit_file(target_file, content)
                else:
                    return f"Error: Cannot apply patch to non-existent file {target_file}"
            
            # Read the original file
            original_content = file_path.read_text(encoding='utf-8')
            original_lines = original_content.splitlines()
            
            # Parse the patch into a list of diff objects
            patch_lines = patch_content.splitlines()
            
            # Apply the patch
            # This is a simplified patch application; for complex patches you might
            # want to use a proper diff/patch library
            try:
                patched_content = []
                i = 0
                while i < len(patch_lines):
                    line = patch_lines[i]
                    if line.startswith("@@"):
                        # Parse the hunk header
                        hunk_match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                        if hunk_match:
                            orig_start = int(hunk_match.group(1)) - 1  # 0-based indexing
                            new_start = int(hunk_match.group(2)) - 1   # 0-based indexing
                            
                            # Process the hunk
                            j = i + 1
                            orig_lines = []
                            new_lines = []
                            
                            while j < len(patch_lines) and not patch_lines[j].startswith("@@"):
                                hunk_line = patch_lines[j]
                                if hunk_line.startswith("+"):
                                    new_lines.append(hunk_line[1:])
                                elif hunk_line.startswith("-"):
                                    orig_lines.append(hunk_line[1:])
                                else:
                                    orig_lines.append(hunk_line[1:] if hunk_line.startswith(" ") else hunk_line)
                                    new_lines.append(hunk_line[1:] if hunk_line.startswith(" ") else hunk_line)
                                j += 1
                            
                            # Verify the original lines match
                            for k, orig_line in enumerate(orig_lines):
                                if orig_start + k >= len(original_lines) or original_lines[orig_start + k] != orig_line:
                                    raise ValueError(f"Patch does not match original file at line {orig_start + k + 1}")
                            
                            # Apply the changes
                            patched_content = original_lines[:orig_start] + new_lines + original_lines[orig_start + len(orig_lines):]
                            original_lines = patched_content
                            
                            i = j
                        else:
                            i += 1
                    else:
                        i += 1
                
                # Write the patched content
                file_path.write_text("\n".join(patched_content), encoding='utf-8')
                logger.info(f"Applied patch to file: {file_path}")
                
                return f"Successfully patched file: {target_file}"
                
            except ValueError as ve:
                return f"Error applying patch: {str(ve)}"
            
        except Exception as e:
            error_msg = f"Error applying patch to {target_file}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def compare_files(self, file1: str, file2: str, context_lines: int = 3) -> str:
        """Compare two files and return the differences.

        Args:
            file1 (str): The path to the first file.
            file2 (str): The path to the second file.
            context_lines (int): Number of context lines to show around differences.

        Example:
            from agentica.tools.edit_tool import EditTool
            editor = EditTool()
            result = editor.compare_files("file1.py", "file2.py")
            print(result)

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

        Example:
            from agentica.tools.edit_tool import EditTool
            editor = EditTool()
            result = editor.search_replace("example.py", "Hello", "Hi")
            print(result)

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
                new_content, count = content.replace(search_pattern, replacement), content.count(search_pattern)
            
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
    
    # Clean up
    import os
    if os.path.exists("test_edit.py"):
        os.remove("test_edit.py") 