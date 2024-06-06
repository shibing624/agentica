# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a class for saving a file to the output directory. The file's name and contents are provided as parameters.
"""

from actionflow.tool import BaseTool


class SaveFile(BaseTool):
    """
    This class inherits from the BaseFunction class. It defines a function for saving a file to the output directory.
    """

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",  # "type": "function" indicates that this is a function definition.
            "function": {
                "name": "save_file",
                "description": "Saves a file. Returns the path to the file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file, including its extension. For example, test.txt.",
                        },
                        "file_contents": {
                            "type": "string",
                            "description": "The contents of the file.",
                        },
                    },
                    "required": ["file_name", "file_contents"],
                },
            }
        }

    def execute(self, file_name: str, file_contents: str) -> str:
        """
        Saves a file to the output directory. The file's name and contents are provided as parameters.

        :param file_name: The name of the file, including its extension. For example, test.txt.
        :type file_name: str
        :param file_contents: The contents of the file.
        :type file_contents: str
        :return: The name of the saved file.
        :rtype: str
        """
        return self.output.save(file_name, file_contents)
