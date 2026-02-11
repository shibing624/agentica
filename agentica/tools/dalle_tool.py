# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains a class for creating an image from a description using OpenAI's API.
It generates a unique image name based on the prompt and the current time, downloads the image,
and saves it to a specified output path.
"""
import asyncio
import hashlib
import os
import time
from typing import Optional, cast

import httpx

from agentica.model.base import Model
from agentica.model.openai import OpenAIChat
from agentica.tools.base import Tool


class DalleTool(Tool):
    """
    This class inherits from the Toolkit class.
        It defines a function for creating an image from a description using OpenAI's API.
    """

    def __init__(
            self,
            data_dir: Optional[str] = None,
            model: Optional[Model] = None

    ):
        super().__init__(name="create_image_from_dalle_tool")
        self.data_dir: str = data_dir or "outputs"
        self.model = model
        self.register(self.create_dalle_image)

    def update_llm(self) -> None:
        if self.model is None:
            self.model = OpenAIChat()

    async def create_dalle_image(self, prompt: str, n: int = 1, size: str = "1024x1024", model: str = 'dall-e-3') -> str:
        """Creates an image from a description using dalle API, generates a unique image name based on the prompt,
        downloads the image, and saves it to a specified output path.

        Args:
            prompt (str): The prompt that describes the image.
            n (int, optional): The number of images to generate. Defaults to 1. Currently, only 1 is supported.
            size (str, optional): The size of the image. Defaults to "1024x1024". only 1024x1024 is supported.
            model (str, optional): The model use for image generation, can be dall-e-2 or dall-e-3. Defaults 'dall-e-3'.

        Returns:
            str: The path to the image.
        """
        self.update_llm()

        image_name = self._generate_image_name(prompt)
        image_url = await self._create_image(prompt, n, size, model)
        image_path = f"{self.data_dir}/{image_name}"
        await self._download_and_save_image(image_url, image_path)
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

    async def _create_image(self, prompt: str, n: int, size: str, model: str = 'dall-e-3') -> str:
        """
        Creates an image from a description using OpenAI's API.

        :param prompt: The prompt that describes the image.
        :param n: The number of images to generate.
        :param size: The size of the image.
        :param model: The model to use for image generation. Defaults to 'dall-e-3'.
        :return: The URL of the image.
        """
        self.model = cast(Model, self.model)

        # OpenAI sync client - wrap in executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.get_client().images.generate(prompt=prompt, n=n, size=size, model=model)
        )
        return response.data[0].url

    async def _download_and_save_image(self, image_url: str, image_path: str) -> None:
        """
        Downloads the image and saves it to the specified path.

        :param image_url: The URL of the image.
        :param image_path: The path to the image.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            image_data = response.content

        if os.path.dirname(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_file, image_path, image_data)

    @staticmethod
    def _write_file(path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)


if __name__ == '__main__':
    import asyncio

    m = DalleTool()
    prompt = "A painting of a beautiful sunset over the ocean."
    r = asyncio.run(m.create_dalle_image(prompt))
    print(r)
