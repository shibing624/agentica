# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import subprocess
from pathlib import Path
from typing import Optional, Union
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
            - execute(command="pytest /foo/bar/tests")
            - execute(command="python /path/to/script.py")
            - execute(command="npm install && npm test")

        Bad examples (avoid these):
            - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
            - execute(command="cat file.txt")  # Use read_file tool instead
            - execute(command="find . -name '*.py'")  # Use glob tool instead
            - execute(command="grep -r 'pattern' .")  # Use grep tool instead

        Args:
            command: Shell command to execute

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        try:
            logger.info(f"Executing shell command: {command}")

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
