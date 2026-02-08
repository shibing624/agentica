# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import re
import subprocess
from pathlib import Path
from typing import Optional, Union
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentica.tools.base import Tool
from agentica.utils.log import logger


class ShellTool(Tool):
    def __init__(
            self,
            base_dir: Optional[Union[Path, str]] = None,
            timeout: int = 120,
            max_output_length: int = 20000,
    ):
        """
        Initialize ShellTool.

        Args:
            base_dir: Base directory for command execution
            timeout: Command execution timeout in seconds
            max_output_length: Maximum length of output to return
        """
        super().__init__(name="shell_tool")

        self.base_dir: Optional[Path] = None
        if base_dir is not None:
            self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self.timeout = timeout
        self.max_output_length = max_output_length

        self.register(self.execute)

    def _fix_python_syntax(self, code: str) -> str:
        """Fix common Python syntax errors from LLM output.
        
        LLMs sometimes generate JavaScript-style syntax in Python code.
        This method fixes common issues:
        - null -> None
        - true -> True  
        - false -> False
        - undefined -> None
        
        Args:
            code: Python code string
            
        Returns:
            Fixed Python code
        """
        # Only fix standalone keywords, not parts of strings or variable names
        # Use word boundaries to avoid replacing inside strings or identifiers
        fixes = [
            (r'\bnull\b', 'None'),
            (r'\bundefined\b', 'None'),
            (r'\btrue\b', 'True'),
            (r'\bfalse\b', 'False'),
        ]
        
        fixed_code = code
        for pattern, replacement in fixes:
            fixed_code = re.sub(pattern, replacement, fixed_code)
        
        if fixed_code != code:
            logger.debug("Auto-fixed Python syntax (null/true/false -> None/True/False)")
        
        return fixed_code

    def _convert_python_c_to_heredoc(self, command: str) -> str:
        """Convert python -c with multi-line code to heredoc format.
        
        If the command is `python -c "..."` or `python3 -c "..."` with multi-line code,
        convert it to heredoc format for better handling of newlines and special characters.
        Also applies Python syntax fixes (null -> None, etc.).
        
        Args:
            command: Original command string
            
        Returns:
            Converted command using heredoc if applicable, otherwise original command
        """
        # Pattern to match python -c or python3 -c with quoted code
        pattern = r'^(python3?)\s+-c\s+(["\'])(.*)\2\s*$'
        match = re.match(pattern, command, re.DOTALL)
        
        if match:
            python_cmd = match.group(1)
            code = match.group(3)
            
            # Fix common Python syntax errors from LLM
            code = self._fix_python_syntax(code)
            
            # Check if code contains newlines or is complex (has function definitions, etc.)
            if '\n' in code or '\\n' in code or 'def ' in code or 'class ' in code:
                # Unescape \\n to actual newlines if present
                code = code.replace('\\n', '\n')
                # Use heredoc format
                heredoc_cmd = f"{python_cmd} << 'PYTHON_EOF'\n{code}\nPYTHON_EOF"
                logger.debug(f"Converted python -c to heredoc format")
                return heredoc_cmd
            else:
                # Return with fixed code but original format
                return f"{python_cmd} -c '{code}'"
        
        # Check if it's already a heredoc format with Python
        heredoc_pattern = r"^(python3?)\s*<<\s*['\"]?(\w+)['\"]?\s*\n(.*)\n\2\s*$"
        heredoc_match = re.match(heredoc_pattern, command, re.DOTALL)
        if heredoc_match:
            python_cmd = heredoc_match.group(1)
            delimiter = heredoc_match.group(2)
            code = heredoc_match.group(3)
            # Fix Python syntax in heredoc code
            fixed_code = self._fix_python_syntax(code)
            if fixed_code != code:
                return f"{python_cmd} << '{delimiter}'\n{fixed_code}\n{delimiter}"
        
        return command

    def execute(self, command: str) -> str:
        """Executes a given command in the specified base directory.

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
        - Commands run in an isolated sandbox environment
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
            - execute(command="pytest /foo/bar/tests")
            - execute(command="python3 /path/to/script.py")
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
        try:
            # Convert python -c with multi-line code to heredoc format
            command = self._convert_python_c_to_heredoc(command)
            
            logger.debug(f"Executing command: {command}")

            # Execute command using shell
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.base_dir) if self.base_dir else None,
            )

            # Combine stdout and stderr
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            output = "\n".join(output_parts).strip()

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"

            # Add exit code info
            if result.returncode != 0:
                output = f"{output}\n\n[Exit code: {result.returncode}]"

            logger.debug(f"Command exit code: {result.returncode}")
            return output if output else f"Command executed successfully (exit code: {result.returncode})"

        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {self.timeout}s: {command}")
            return f"Error: Command timed out after {self.timeout} seconds"
        except Exception as e:
            logger.warning(f"Failed to run shell command: {e}")
            return f"Error: {e}"


if __name__ == '__main__':
    m = ShellTool()
    r = m.execute("ls -l /tmp")
    print(r)
    
    # Test heredoc conversion
    print("\n--- Test heredoc conversion ---")
    code = '''python3 -c "def fib(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

fib(10)"'''
    r = m.execute(code)
    print(r)
    
    # Test auto-fix null/true/false
    print("\n--- Test auto-fix Python syntax ---")
    code_with_null = '''python3 << 'EOF'
def test():
    if null:
        return false
    return true
print(test())
EOF'''
    r = m.execute(code_with_null)
    print(r)
