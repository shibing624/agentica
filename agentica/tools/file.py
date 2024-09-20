# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description: Load and save files.
"""
import os
import json
from pathlib import Path
from typing import Optional
from agentica.utils.file_parser import (
    read_json_file,
    read_csv_file,
    read_txt_file,
    read_pdf_file,
    read_docx_file,
    read_excel_file
)
from agentica.utils.log import logger
from agentica.tools.base import Toolkit


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

    def save_file(self, contents: str, file_name: str, overwrite: bool = True, save_dir: str = "") -> str:
        """Saves the contents to a file called `file_name` and returns the file name if successful.

        :param contents: The contents to save.
        :param file_name: The name of the file to save to.
        :param overwrite: Overwrite the file if it already exists.
        :param save_dir: The directory to save the file to, defaults to the base directory.
        :return: The file name if successful, otherwise returns an error message.
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

        :param file_name: The name of the file to read.
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            if os.path.exists(file_name):
                path = Path(file_name)
            else:
                path = self.data_dir.joinpath(file_name)
            logger.info(f"Reading file: {path}")

            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")

            if path.suffix in [".json", ".jsonl"]:
                file_contents = read_json_file(path)
            elif path.suffix in [".csv"]:
                file_contents = read_csv_file(path)
            elif path.suffix in [".txt", ".md"]:
                file_contents = read_txt_file(path)
            elif path.suffix in [".pdf"]:
                file_contents = read_pdf_file(path)
            elif path.suffix in [".doc", ".docx"]:
                if path.suffix == ".doc":
                    raise ValueError("Unsupported doc format. Please convert to docx.")
                file_contents = read_docx_file(path)
            elif path.suffix in [".xls", ".xlsx"]:
                file_contents = read_excel_file(path)
            else:
                logger.warning(f"Unknown file format: {path.suffix}, reading as text")
                file_contents = read_txt_file(path)

            return str(file_contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

    def list_files(self, dir_path: str = "") -> str:
        """Returns a list of files in the base directory

        :param dir_path: The directory to list files from, defaults to the base directory.
        :return: The contents of the file if successful, otherwise returns an error message.
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

        :param dir_path: The directory to read files from, defaults to the base directory.
        :return: The contents of all files if successful, otherwise returns an error message.
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
