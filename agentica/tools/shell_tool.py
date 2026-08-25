# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import asyncio
from pathlib import Path
from typing import Optional, Union
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentica.tools.base import Tool
from agentica.utils.async_utils import close_subprocess_transport, terminate_subprocess
from agentica.utils.log import logger


class ShellTool(Tool):
    def __init__(
            self,
            work_dir: Optional[Union[Path, str]] = None,
            timeout: int = 120,
            max_output_length: int = 20000,
    ):
        """
        Initialize ShellTool.

        Args:
            work_dir: Work directory for command execution
            timeout: Command execution timeout in seconds
            max_output_length: Maximum length of output to return
        """
        super().__init__(name="shell_tool")

        self.work_dir: Optional[Path] = None
        if work_dir is not None:
            self.work_dir = Path(work_dir) if isinstance(work_dir, str) else work_dir
        self.timeout = timeout
        self.max_output_length = max_output_length

        self.register(self.execute)

    async def execute(self, command: str) -> str:
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
        - The command string is passed unchanged to the system shell. This tool
          does not normalize quotes, convert heredocs, or repair source code.
        - Returns combined stdout/stderr output with exit code
        - Invalid UTF-8 bytes are replaced while decoding. Output beyond
          max_output_length is explicitly marked as truncated.
        - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
            - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
            - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
        - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

        Examples:
        Good examples:
            - execute(command="pytest /foo/bar/tests")
            - execute(command="python3 /path/to/script.py")
            - execute(command="python3 -c 'print(33333**2 + 332.2 / 12)'")
            - execute(command="npm install && npm test")

        Bad examples (avoid these):
            - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead

        Args:
            command: command to execute

        Returns:
            str: The output of the command (stdout + stderr) with exit code
        """
        logger.debug(f"Executing command: {command}")

        # Execute command using async subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.work_dir) if self.work_dir else None,
            start_new_session=os.name != "nt",
        )

        drained = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            drained = True
        except asyncio.TimeoutError:
            logger.warning(f"Command timed out after {self.timeout}s: {command}")
            raise TimeoutError(
                f"Command timed out after {self.timeout} seconds"
            ) from None
        finally:
            if not drained:
                await terminate_subprocess(process, process_group=True)
            close_subprocess_transport(process)

        # Decode output
        stdout_str = stdout.decode(errors='replace') if stdout else ""
        stderr_str = stderr.decode(errors='replace') if stderr else ""

        # Combine stdout and stderr
        output_parts = []
        if stdout_str:
            output_parts.append(stdout_str)
        if stderr_str:
            output_parts.append(f"[stderr]\n{stderr_str}")

        output = "\n".join(output_parts).strip()

        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "\n... (output truncated)"

        # Add exit code info
        returncode = process.returncode
        if returncode != 0:
            output = f"{output}\n\n[Exit code: {returncode}]"

        logger.debug(f"Command exit code: {returncode}")
        return output if output else f"Command executed successfully (exit code: {returncode})"


if __name__ == '__main__':
    import asyncio

    m = ShellTool()
    r = asyncio.run(m.execute("ls -l /tmp"))
    print(r)
