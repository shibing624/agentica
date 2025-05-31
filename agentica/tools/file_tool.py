# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description: Load and save files.
"""
import os
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
    ):
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

    def save_file(self, contents: str, file_name: str, overwrite: bool = True, save_dir: str = "") -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        Args:
            contents (str): The contents to save.
            file_name (str): The name of the file to save to.
            overwrite (bool): Overwrite the file if it already exists.
            save_dir (str): The directory to save the file to, defaults to the base directory.

        Example:
            from agentica.tools.file_tool import FileTool
            m = FileTool()
            print(m.save_file(contents="Hello, world!", file_name="hello.txt"))

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

        Example:
            from agentica.tools.file_tool import FileTool
            m = FileTool()
            print(m.read_file(file_name="hello.txt"))

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

        Example:
            from agentica.tools.file_tool import FileTool
            m = FileTool()
            print(m.list_files('/home/user/data'))

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

        Example:
            from agentica.tools.file_tool import FileTool
            m = FileTool()
            print(m.read_files(dir_path="/home/user/data"))

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


if __name__ == '__main__':
    m = FileTool()
    print(m.save_file(contents="Hello, world!", file_name="hello.txt"))
    print(m.read_file(file_name="hello.txt"))
    print(m.list_files())
    print(m.read_files())
    print(m.read_files(dir_path="."))
    if os.path.exists("hello.txt"):
        os.remove("hello.txt")
