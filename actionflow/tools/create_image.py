# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a class for creating an image from a description using OpenAI's API.
It generates a unique image name based on the prompt and the current time, downloads the image, and saves it to a specified output path.
"""

import hashlib
import os
import time

import requests

from actionflow.llm import LLM
from actionflow.output import Output
from actionflow.tool import BaseTool


class CreateImage(BaseTool):
    """
    This class inherits from the BaseFunction class. It defines a function for creating an image from a description using OpenAI's API.
    """

    def __init__(self, output: Output):
        """
        Initializes the SummarizeText object.
        """
        super().__init__(output)
        self.llm = LLM()

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",  # "type": "function" indicates that this is a function definition.
            "function": {
                "name": "create_image",
                "description": "Creates an image from a description. Returns the path to the image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt that describes the image. Be specific and detailed about the content and style of the image.",
                        }
                    },
                    "required": ["prompt"],
                },
            }
        }

    def execute(self, prompt: str, n: int = 1, size: str = "1024x1024", model: str = 'dall-e-3') -> str:
        """
        Creates an image from a description using OpenAI's API, generates a unique image name based on the prompt
            and the current time, downloads the image, and saves it to a specified output path.

        :param prompt: The prompt that describes the image.
        :type prompt: str
        :param n: The number of images to generate. Defaults to 1. Currently, only 1 is supported.
        :type n: int, optional
        :param size: The size of the image. Defaults to "1024x1024". Currently, only "1024x1024" is supported.
        :type size: str, optional
        :param model: The model to use for image generation. Defaults to 'dall-e-3'.
        :return: The path to the image.
        :rtype: str
        """
        image_name = self._generate_image_name(prompt)
        image_url = self._create_image(prompt, n, size, model)
        image_path = f"{self.output.output_dir}/{image_name}"
        self._download_and_save_image(image_url, image_path)
        return os.path.abspath(image_path)

    def _generate_image_name(self, prompt: str) -> str:
        """
        Generates a unique image name based on the prompt and the current time.

        :param prompt: The prompt that describes the image.
        :type prompt: str
        :return: The name of the image file.
        :rtype: str
        """
        timestamp = str(time.time())
        file_name = str(hashlib.sha256((prompt + timestamp).encode()).hexdigest())[:16]
        return file_name + ".png"

    def _create_image(self, prompt: str, n: int, size: str, model: str = 'dall-e-3') -> str:
        """
        Creates an image from a description using OpenAI's API.

        :param prompt: The prompt that describes the image.
        :type prompt: str
        :param n: The number of images to generate.
        :type n: int
        :param size: The size of the image.
        :type size: str
        :param model: The model to use for image generation. Defaults to 'dall-e-3'.
        :type model: str
        :return: The URL of the image.
        :rtype: str
        """
        response = self.llm.client.images.generate(prompt=prompt, n=n, size=size, model=model)
        return response.data[0].url

    def _download_and_save_image(self, image_url: str, image_path: str) -> None:
        """
        Downloads the image and saves it to the specified path.

        :param image_url: The URL of the image.
        :type image_url: str
        :param image_path: The path to the image.
        :type image_path: str
        """
        image_data = requests.get(image_url).content
        with open(image_path, "wb") as handler:
            handler.write(image_data)
