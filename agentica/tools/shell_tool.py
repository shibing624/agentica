# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
from pathlib import Path
from typing import List, Optional, Union
from agentica.tools.base import Tool
from agentica.utils.log import logger


class ShellTool(Tool):
    def __init__(self, data_dir: Optional[Union[Path, str]] = None):
        super().__init__(name="shell_tool")

        self.data_dir: Optional[Path] = None
        if data_dir is not None:
            self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        self.register(self.run_shell_command)

    def run_shell_command(self, args: List[str], tail: int = 100) -> str:
        """Runs a shell command and returns the output or error.

        Args:
            args (List[str]): The command to run as a list of strings.
            tail (int): The number of lines to return from the output.

        Example:
            from agentica.tools.shell_tool import ShellTool
            m = ShellTool()
            result = m.run_shell_command(["ls", "-l", "/tmp"])
            print(result)

        Returns:
            str: The output of the command.
        """
        import subprocess

        try:
            logger.info(f"Running shell command: {args}")
            if self.data_dir:
                args = ["cd", str(self.data_dir), ";"] + args
            result = subprocess.run(args, capture_output=True, text=True)
            logger.debug(f"Result: {result}")
            logger.debug(f"Return code: {result.returncode}")
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            # return only the last n lines of the output
            return "\n".join(result.stdout.split("\n")[-tail:])
        except Exception as e:
            logger.warning(f"Failed to run shell command: {e}")
            return f"Error: {e}"


if __name__ == '__main__':
    m = ShellTool()
    r = m.run_shell_command(["ls", "-l", "/tmp"])
    print(r)
