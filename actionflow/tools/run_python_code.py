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

from actionflow.utils.log import logger

from actionflow.tool import Toolkit


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
        self.data_dir: str = data_dir or os.path.curdir
        # Restricted global and local scope
        self.safe_globals: dict = safe_globals or globals()
        if run_code:
            self.register(self.run_python_code, sanitize_arguments=False)
        if pip_install:
            self.register(self.pip_install_package)

    def run_python_code(
            self, file_name: str, code: str, variable_to_return: Optional[str] = None, overwrite: bool = True
    ) -> str:
        """
        This function saves Python code to a file called `file_name` and then runs it.
            If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
            If failed, returns an error message.

            Make sure the file_name ends with `.py`

        :param file_name: The name of the file the code will be saved to.
        :param code: The code to save and run.
        :param variable_to_return: Optional[str], The variable to return.
        :param overwrite: Overwrite the file if it already exists.
        :return: if run is successful, the value of `variable_to_return` if provided else file name.
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
        If successful, returns a success message.
        If failed, returns an error message.

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

    result = tool.run_python_code(
        file_name="test_script.py",
        code="a = 5\nb = 110\nc = a + b\nprint(c)",
        variable_to_return="c"
    )
    print(result)
