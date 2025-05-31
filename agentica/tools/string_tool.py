# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: String manipulation tools
"""
import json
from agentica.tools.base import Tool
from agentica.utils.log import logger


class StringTool(Tool):
    def __init__(
            self,
            enable_reverse_string=True,
            enable_text_length=True,
    ):
        super().__init__(name="string_tool")
        if enable_reverse_string:
            self.register(self.reverse_string)
        if enable_text_length:
            self.register(self.text_length)

    def reverse_string(self, s: str) -> str:
        """Reverse the input string.

        Args:
            s (str): Input string.

        Returns:
            str: JSON string of the result.
        """
        try:
            reversed_str = s[::-1]
            logger.debug(f"Reversing string: {s}, Reversed string: {reversed_str}")
            return json.dumps({"operation": "reverse_string", "result": reversed_str})
        except Exception as e:
            logger.error(f"Error in reversing string: {str(e)}")
            return f"Error: {str(e)}"

    def text_length(self, s: str) -> str:
        """Calculate the length of the input string.

        Args:
            s (str): Input string.

        Returns:
            str: JSON string of the result.
        """
        try:
            length = len(s)
            logger.debug(f"Calculating text length: {s}, Text length: {length}")
            return json.dumps({"operation": "text_length", "result": str(length)})
        except Exception as e:
            logger.error(f"Error in calculating text length: {str(e)}")
            return f"Error: {str(e)}"


if __name__ == '__main__':
    tool = StringTool()
    print(tool.reverse_string("Hello, World!"))
    print(tool.text_length("Hello, World!"))
