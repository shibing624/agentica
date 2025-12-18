# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
from __future__ import annotations

import io
import os
import runpy
import sys
import traceback
from typing import Optional

from agentica.tools.base import Tool
from agentica.utils.log import logger



class RunPythonCodeTool(Tool):
    """impl of RunPythonCodeTool,
        which can run python code, install package by pip.
        We call it code interpreter tool.
    """

    def __init__(
            self,
            base_dir: Optional[str] = None,
            save_and_run: bool = True,
            pip_install: bool = False,
            run_code: bool = False,
            run_file: bool = False,
    ):
        super().__init__(name="run_python_code_tool")
        self.base_dir: str = base_dir if base_dir else os.path.curdir
        
        if run_code:
            self.register(self.run_python_code, sanitize_arguments=False)
        if save_and_run:
            self.register(self.save_to_file_and_run, sanitize_arguments=False)
        if pip_install:
            self.register(self.pip_install_package)
        if run_file:
            self.register(self.run_python_file)

    def run_python_code(self, code: str) -> str:
        """This function runs Python code in the current environment.
        If successful, returns the stdout output.
        If failed, returns an error message with traceback.

        :param code: The code to run.
        :return: stdout output or error message.
        """
        logger.info(f"Running code:\n\n{code}\n\n")
        
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            # compile the code to check for syntax errors
            compiled_code = compile(code, '<string>', 'exec')
            namespace = {}
            # execute the code
            exec(compiled_code, namespace)
            
            # get stdout output
            output = new_stdout.getvalue().strip()
            result = output if output else "successfully ran python code"
            return result
        except Exception as e:
            error = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Error: {error}\n\nTraceback:\n{error_traceback}")
            return f"Error: {error}\n\nTraceback:\n{error_traceback}"
        finally:
            sys.stdout = old_stdout
            new_stdout.close()

    def save_to_file_and_run(self, file_name: str, code: str, overwrite: bool = True) -> str:
        """Saves Python code to a specified file, then runs the file.

        Args:
            file_name (str): The name of the file to save and run, e.g., "test_script.py", required.
            code (str): The code to save and run, e.g., "a = 5\nb = 110\nc = a + b\nprint(c)", required.
            overwrite (bool): Whether to overwrite the file if it already exists, default is True.

        Returns:
            str: stdout output or success message.
        """
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            file_path = os.path.join(self.base_dir, file_name)
            logger.debug(f"Saving code to {file_path}")
            
            if os.path.exists(file_path) and not overwrite:
                return f"File {file_name} already exists"
                
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Saved: {file_path}")
            logger.info(f"Running {file_path}")
            
            runpy.run_path(str(file_path), init_globals={}, run_name="__main__")
            
            # get stdout output
            output = new_stdout.getvalue().strip()
            result = output if output else f"successfully ran {str(file_path)}"
            return result
        except Exception as e:
            error = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Error: {error}\n\nTraceback:\n{error_traceback}")
            return f"Error: {error}\n\nTraceback:\n{error_traceback}"
        finally:
            sys.stdout = old_stdout
            new_stdout.close()

    def pip_install_package(self, package_name: str) -> str:
        """This function installs a package using pip in the current environment.
        :param package_name: The name of the package to install.
        :return: success message if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Installing package {package_name}")
            import sys
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return f"successfully installed package {package_name}"
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {e}")
            return f"Error installing package {package_name}: {e}"

    def run_python_file(self, file_name: str) -> str:
        """This function runs code in a Python file.
        If successful, returns the stdout output.
        If failed, returns an error message with traceback.

        :param file_name: The name of the file to run.
        :return: stdout output or success message.
        """
        file_path = os.path.join(self.base_dir, file_name)
        
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            logger.info(f"Running {file_path}")
            runpy.run_path(str(file_path), init_globals={}, run_name="__main__")
            
            # get stdout output
            output = new_stdout.getvalue().strip()
            result = output if output else f"successfully ran {str(file_path)}"
            return result
        except Exception as e:
            error = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Error: {error}\n\nTraceback:\n{error_traceback}")
            return f"Error: {error}\n\nTraceback:\n{error_traceback}"
        finally:
            sys.stdout = old_stdout
            new_stdout.close()


if __name__ == '__main__':
    tool = RunPythonCodeTool()

    result = tool.save_to_file_and_run(
        file_name="calc_add.py",
        code="a = 5\nb = 110\nc = a + b\nprint(c)"
    )
    print(result)
