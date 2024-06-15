# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
from pathlib import Path
from typing import Optional

from actionflow.utils.log import logger
from actionflow.tool import Toolkit


class FileTool(Toolkit):
    def __init__(
            self,
            data_dir: Optional[str] = None,
            save_file: bool = True,
            read_file: bool = True,
            list_files: bool = True,
            read_files: bool = True,

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

    def save_file(self, contents: str, file_name: str, overwrite: bool = True) -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        :param contents: The contents to save.
        :param file_name: The name of the file to save to.
        :param overwrite: Overwrite the file if it already exists.
        :return: The file name if successful, otherwise returns an error message.
        """
        try:
            file_path = self.data_dir.joinpath(file_name)
            logger.debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and not overwrite:
                return f"File {file_name} already exists"
            file_path.write_text(contents)
            logger.info(f"Saved: {file_path}")
            return str(file_name)
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_file(self, file_name: str) -> str:
        """Reads the contents of the file `file_name` and returns the contents if successful.

        :param file_name: The name of the file to read.
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            file_path = self.data_dir.joinpath(file_name)
            logger.info(f"Reading file: {file_path}")
            contents = file_path.read_text()
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        """Returns a list of files in the base directory

        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Reading files in : {self.data_dir}")
            return json.dumps([str(file_path) for file_path in self.data_dir.iterdir()],
                              indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"

    def read_files(self) -> str:
        """Reads the contents of all files in the base directory and returns the contents.

        :return: The contents of all files if successful, otherwise returns an error message.
        """
        try:
            logger.info(f"Reading all files in: {self.data_dir}")
            all_contents = []
            for file_path in self.data_dir.iterdir():
                if file_path.is_file():
                    logger.info(f"Reading file: {file_path}")
                    contents = file_path.read_text()
                    all_contents.append(f"Contents of {file_path.name}:\n{contents}")
            return json.dumps(all_contents, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error reading files: {e}")
            return f"Error reading files: {e}"
