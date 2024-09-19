# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
from __future__ import annotations

import functools
import os
import runpy
from typing import Optional

from agentica.tools.base import Toolkit
from agentica.utils.log import logger


@functools.lru_cache(maxsize=None)
def warn() -> None:
    logger.warning("RunPythonCode can run arbitrary code, please provide human supervision.")


class RunPythonCodeTool(Toolkit):
    """impl of RunPythonCodeTool, which can run python code, install package, read file, list files, read files.
        We call it code interpreter tool.
    """

    def __init__(
            self,
            data_dir: Optional[str] = None,
            run_code: bool = True,
            pip_install: bool = False,
            safe_globals: Optional[dict] = None
    ):
        super().__init__(name="run_python_code_tool")
        self.data_dir: str = data_dir if data_dir else os.path.curdir
        # Restricted global and local scope
        self.safe_globals: dict = safe_globals or globals()
        if run_code:
            self.register(self.save_and_run_python_code, sanitize_arguments=False)
        if pip_install:
            self.register(self.pip_install_package)

    def save_and_run_python_code(
            self, file_name: str, code: str, variable_to_return: Optional[str] = None, overwrite: bool = True
    ) -> str:
        """
        Saves Python code to a specified file, then runs the file.

        :param file_name: The name of the file to save and run, e.g., "test_script.py", required.
        :param code: The code to save and run, e.g., "a = 5\nb = 110\nc = a + b\nprint(c)", required.
        :param variable_to_return: Optional[str], The variable to return the value of after execution, default is None.
        :param overwrite: Whether to overwrite the file if it already exists, default is True.
        :return: The value of `variable_to_return` if provided and available, otherwise a success message or error message.
        """
        try:
            warn()

            os.makedirs(self.data_dir, exist_ok=True)
            file_path = os.path.join(self.data_dir, file_name)
            logger.debug(f"Saving code to {file_path}")
            if os.path.exists(file_path) and not overwrite:
                return f"File {file_name} already exists"
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(code)
            logger.debug(f"Saved: {file_path}")
            logger.info(f"Running {file_path}")

            globals_after_run = runpy.run_path(str(file_path), init_globals=self.safe_globals, run_name="__main__")

            if variable_to_return:
                variable_value = globals_after_run.get(variable_to_return)
                if variable_value is None:
                    return f"Variable {variable_to_return} not found"
                logger.debug(f"Variable {variable_to_return} value: {variable_value}")
                return str(variable_value)
            else:
                return f"successfully ran {str(file_path)}"
        except Exception as e:
            logger.error(f"Error saving and running code: {e}")
            return f"Error saving and running code: {e}"

    def pip_install_package(self, package_name: str) -> str:
        """This function installs a package using pip in the current environment.
        :param package_name: The name of the package to install.
        :return: success message if successful, otherwise returns an error message.
        """
        try:
            warn()

            logger.debug(f"Installing package {package_name}")
            import sys
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"successfully installed package {package_name}"
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return f"Error installing package {package_name}: {e}"


if __name__ == '__main__':
    tool = RunPythonCodeTool()

    result = tool.save_and_run_python_code(
        file_name="test_script.py",
        code="a = 5\nb = 110\nc = a + b\nprint(c)",
        variable_to_return="c"
    )
    print(result)
