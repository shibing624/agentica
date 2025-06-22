# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
CogView-3
模型编码:cogview-3-plus、cogview-3、cogview-3-flash(免费)

适用于图像生成任务，通过对用户文字描述快速、精准的理解，让AI的图像表达更加精确和个性化。
"""
import hashlib
import json
import os
from os import getenv
import time
from typing import Optional

import requests

try:
    from zhipuai import ZhipuAI
except ImportError:
    raise ImportError("`zhipuai` not installed. Please install using `pip install zhipuai`.")

from agentica.tools.base import Tool
from agentica.utils.log import logger


class CogViewTool(Tool):
    """
    This class inherits from the Toolkit class.
        It defines a function for creating an image from a description using ZhipuAI's API.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            data_dir: Optional[str] = None,
    ):
        super().__init__(name="create_image_from_cogview_tool")
        self.api_key = api_key or getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            logger.error("ZHIPUAI_API_KEY not set. Please set the ZHIPUAI_API_KEY environment variable.")
        self.client = ZhipuAI(api_key=self.api_key)
        self.data_dir: str = data_dir or "outputs"
        self.register(self.create_cogview_image)

    def create_cogview_image(self, prompt: str, size: str = "1024x1024", model: str = "cogview-3-flash") -> str:
        """Creates an image from a description using ZhipuAI API, generates a unique image name based on the prompt,
        downloads the image, and saves it to a specified output path.

        Args:
            prompt (str): The prompt that describes the image.
            size (str): 图片尺寸， cogview-3-plus 、cogview-3-flash支持该参数。
                可选范围：[1024x1024,768x1344,864x1152,1344x768,1152x864,1440x720,720x1440]，默认是1024x1024。
            model (str, optional): The model use for image generation. Defaults 'cogview-3-flash'.

        Example:
            from agentica.tools.cogview_tool import CogViewTool
            tool = CogViewTool()
            image_path = tool.create_cogview_image("A painting of a beautiful sunset over the ocean.")
            print(image_path)

        Returns:
            str: The path to the image.
        """
        image_name = self._generate_image_name(prompt)
        image_url = self._create_image(prompt, size, model)
        image_path = f"{self.data_dir}/{image_name}"
        self._download_and_save_image(image_url, image_path)
        saved_img_path = os.path.abspath(image_path)
        result = {"action": "create_cogview_image", "result_image_url": image_url, "result_image_path": saved_img_path}
        return json.dumps(result, ensure_ascii=False)

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

    def _create_image(self, prompt: str, size: str, model: str = 'cogview-3-flash') -> str:
        """
        Creates an image from a description using ZhipuAI's API.

        :param prompt: The prompt that describes the image.
        :type prompt: str
        :param size: The size of the image.
        :type size: str
        :param model: The model to use for image generation. Defaults to 'cogview-3-flash'.
        :type model: str
        :return: The URL of the image.
        :rtype: str
        """
        response = self.client.images.generations(
            model=model,
            prompt=prompt,
            size=size
        )
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
        if os.path.dirname(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(image_data)


if __name__ == '__main__':
    # from agentica.tools.cogview_tool import CogViewTool
    m = CogViewTool()
    prompt = "A painting of a beautiful sunset over the ocean."
    r = m.create_cogview_image(prompt)
    print(r)
