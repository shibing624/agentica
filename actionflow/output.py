# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module provides a class for managing output files.
It creates a unique directory for each flow and allows saving files to that directory.
"""

import json
import os

from typing import Union


class Output:
    """
    This class is responsible for managing output files.
    It creates a unique directory for each flow and provides a method to save files to that directory.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the Output object with a unique directory for the flow.

        :param output_dir: str, The save dir of the action flow process result.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, file_name: str, file_contents: Union[str, list, dict]) -> str:
        """
        Saves the file contents to a file in the flow's directory.

        :param file_name: The name of the file.
        :type file_name: str
        :param file_contents: The contents of the file.
        :type file_contents: Union[str, list, dict]
        :return: The path to the saved file.
        :rtype: str
        """
        file_path = os.path.join(self.output_dir, file_name)
        if isinstance(file_contents, str):
            data_to_write = file_contents
        elif isinstance(file_contents, (list, dict)):
            data_to_write = json.dumps(file_contents, indent=4, ensure_ascii=False)
        else:
            raise TypeError("file_contents must be of type str, list, or dict")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data_to_write)

        return os.path.abspath(file_path)

    def __repr__(self):
        return f'Output({self.output_dir})'
