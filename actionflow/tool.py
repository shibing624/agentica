# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module provides classes for managing tools.
It includes an abstract base class for tools and a class for managing tool instances.
"""

import importlib
import json
from abc import ABC, abstractmethod

from actionflow.output import Output


class BaseTool(ABC):
    """
    This abstract base class defines the interface for tools.
    """

    def __init__(self, output: Output):
        """
        Initializes the BaseTool object with an output object.

        :param output: The output object.
        :type output: Output
        """
        self.output = output

    @abstractmethod
    def get_definition(self) -> dict:
        """
        Returns the definition of the tool.

        :return: The definition of the tool.
        :rtype: dict
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """
        Executes the tool with the given arguments.

        :param args: The positional arguments.
        :param kwargs: The keyword arguments.
        :return: The result of the tool execution.
        :rtype: str
        """
        pass


class Tool:
    """
    This class is responsible for managing tool instances.
    """

    def __init__(self, tool_name: str, output: Output):
        """
        Initializes the tool object by importing the tool module and creating an instance of the tool class.

        :param tool_name: The name of the tool.
        :type tool_name: str
        :param output: The output object.
        :type output: Output
        """
        self.module = importlib.import_module(f"actionflow.tools.{tool_name}")
        self.tool_class = getattr(self.module, tool_name.replace("_", " ").title().replace(" ", ""))
        self.instance = self.tool_class(output)

    @property
    def definition(self) -> dict:
        """
        Returns the definition of the tool instance.

        :return: The definition of the tool instance.
        :rtype: dict
        """
        return self.instance.get_definition()

    def execute(self, args_json: str) -> str:
        """
        Executes the tool instance with the given arguments.

        :param args_json: The arguments in JSON format as a string.
        :type args_json: str
        :return: The result of the tool execution.
        :rtype: str
        """
        args_dict = json.loads(args_json)
        return self.instance.execute(**args_dict)
