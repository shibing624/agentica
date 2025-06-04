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

from agentica.model.base import Model
from agentica.model.openai.chat import OpenAIChat
from agentica.tools.base import Tool
from agentica.utils.log import logger


class ImageAnalysisTool(Tool):
    """
    This class inherits from the Toolkit class.
    It defines a function for analyzing and understanding image content using OpenAI's API.
    """

    def __init__(
            self,
            data_dir: Optional[str] = None,
            llm: Optional[Model] = None,
            prompt: str = "详细描述图片内容",
            model_name: str = "gpt-4o"
    ):
        super().__init__(name="read_image_tool")
        self.data_dir = data_dir or os.path.curdir
        self.llm = llm
        self.prompt = prompt
        self.model_name = model_name

        self.register(self.analyze_image_content)

    def update_llm(self) -> None:
        if self.llm is None:
            self.llm = OpenAIChat(id=self.model_name)

    def analyze_image_content(self, image_path_or_url: str, prompt: str = '') -> str:
        """Reads and understands the content of an image using image understand model API.

        Args:
            image_path_or_url (str): The path to the image or the URL of the image.
            prompt (str, optional): The prompt to use for the image analysis. Default is "详细描述图片内容".

        Example:
            ```python
            from agentica.tools.analyze_image_tool import AnalyzeImageTool
            tool = AnalyzeImageTool()
            image_description = tool.analyze_image_content("../../examples/data/chinese.jpg")
            print(image_description)
            ```

        Returns:
            str: The description of the image content.
        """

        # Update the Model (set defaults, add logit etc.)
        self.update_llm()
        if image_path_or_url.startswith("http"):
            description = self._analyze_image_url(image_path_or_url, prompt)
        else:
            description = self._analyze_image_path(image_path_or_url, prompt)
        logger.debug(f"Read Image: {image_path_or_url}, model: {self.model_name}, Result description: {description}")
        return description

    def _analyze_image_url(self, image_url: str, prompt: str = '') -> str:
        """
        Analyzes the image content using OpenAI's API.

        :param image_url: The URL of the image.
        :type image_url: str
        :param prompt: The prompt to use for the image analysis.
        :type prompt: str
        :return: The description of the image content.
        :rtype: str
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt or self.prompt
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
        self.llm = cast(Model, self.llm)
        response = self.llm.get_client().chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=4000
        )

        return response.choices[0].message.content

    def _analyze_image_path(self, image_path: str, prompt: str = '') -> str:
        """
        Analyzes the image content using OpenAI's API.

        :param image_path: The path to the image.
        :type image_path: str
        :param prompt: The prompt to use for the image analysis.
        :type prompt: str
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
                        "text": prompt or self.prompt
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
        self.llm = cast(Model, self.llm)
        response = self.llm.get_client().chat.completions.create(
            model=self.model_name, messages=messages, max_tokens=4000
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    tool = ImageAnalysisTool()
    image_description = tool.analyze_image_content("../../examples/data/chinese.jpg")
    print(image_description)
