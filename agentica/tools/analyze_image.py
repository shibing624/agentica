# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description:
This module contains a class for reading and understanding image content using OpenAI's API.
It uploads an image, sends it to the API, and retrieves a description of the image content.
"""
import base64
import os
from typing import Optional, cast

from agentica.llm import LLM, OpenAILLM
from agentica.tools.base import Toolkit
from agentica.utils.log import logger


class AnalyzeImageTool(Toolkit):
    """
    This class inherits from the Toolkit class.
    It defines a function for analyzing and understanding image content using OpenAI's API.
    """

    def __init__(
            self,
            data_dir: Optional[str] = None,
            llm: Optional[LLM] = None,
            prompt: str = "详细描述图片内容",
            model_name: str = "gpt-4o"
    ):
        super().__init__(name="read_image_tool")
        self.data_dir: str = data_dir or os.path.curdir
        self.llm = llm
        self.prompt = prompt
        self.model_name = model_name

        self.register(self.analyze_image_content)

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAILLM()

    def analyze_image_content(self, image_path_or_url: str) -> str:
        """
        Reads and understands the content of an image using OpenAI's API.

        :param image_path_or_url: The path to the image or the URL of the image.
        :type image_path_or_url: str
        :return: The description of the image content.
        :rtype: str
        """

        # Update the LLM (set defaults, add logit etc.)
        self.update_llm()
        if image_path_or_url.startswith("http"):
            description = self._analyze_image_url(image_path_or_url)
        else:
            description = self._analyze_image_path(image_path_or_url)
        logger.debug(f"Read Image: {image_path_or_url}, model: {self.model_name}, Result description: {description}")
        return description

    def _analyze_image_url(self, image_url: str) -> str:
        """
        Analyzes the image content using OpenAI's API.

        :param image_url: The URL of the image.
        :type image_url: str
        :return: The description of the image content.
        :rtype: str
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        self.llm = cast(LLM, self.llm)
        response = self.llm.get_client().chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=1000
        )

        return response.choices[0].message.content

    def _analyze_image_path(self, image_path: str) -> str:
        """
        Analyzes the image content using OpenAI's API.

        :param image_path: The path to the image.
        :type image_path: str
        :return: The description of the image content.
        :rtype: str
        """
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        if not base64_image:
            logger.error("Failed to encode the image to base64.")
            return ""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        self.llm = cast(LLM, self.llm)
        response = self.llm.get_client().chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=1000
        )

        return response.choices[0].message.content


# Example usage:
if __name__ == "__main__":
    tool = AnalyzeImageTool()
    image_description = tool.analyze_image_content("../../examples/data/chinese.jpg")
    print(image_description)
