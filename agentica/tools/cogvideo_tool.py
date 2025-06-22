# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
CogVideoX
模型编码：cogvideox、cogvideox-flash

CogVideoX 是由智谱AI开发的视频生成大模型，具备强大的视频生成能力，只需输入文本或图片就可以轻松完成视频制作。
"""
import hashlib
import json
import os
from os import getenv
import time
import requests
from typing import Optional

try:
    from zhipuai import ZhipuAI
except ImportError:
    raise ImportError("`zhipuai` not installed. Please install using `pip install zhipuai`.")

from agentica.tools.base import Tool
from agentica.utils.log import logger


class CogVideoTool(Tool):
    """
    This class inherits from the Toolkit class.
        It defines a function for creating an image from a description using ZhipuAI's API.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            data_dir: Optional[str] = None,

    ):
        super().__init__(name="create_video_from_cogvideo_tool")
        self.api_key = api_key or getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            logger.error("ZHIPUAI_API_KEY not set. Please set the ZHIPUAI_API_KEY environment variable.")
        self.client = ZhipuAI(api_key=self.api_key)
        self.data_dir: str = data_dir or "outputs"
        self.register(self.create_video)

    def _generate_name(self, prompt: Optional[str] = None) -> str:
        """
        Generates a unique name based on the prompt and the current time.

        :param prompt: The prompt that describes the video.
        :type prompt: str
        :return: The name of the video file.
        :rtype: str
        """
        timestamp = str(time.time())
        prompt = prompt if prompt else ""
        file_name = str(hashlib.sha256((prompt + timestamp).encode()).hexdigest())[:16]
        return file_name + ".mp4"

    def create_video(
            self,
            prompt: Optional[str] = None,
            image_url: Optional[str] = None,
            model: str = "cogvideox-flash",
            with_audio: bool = False,
            size: Optional[str] = None,
            duration: int = 5,
            fps: int = 30
    ) -> str:
        """Creates a video from a description or image using ZhipuAI API, generates a unique video name based on the prompt or image URL,
        and saves it to a specified output path.

        Args:
            prompt (str, optional): The prompt that describes the video.
            image_url (str, optional): The URL of the image to base the video on.
            model (str, optional): The model to use for video generation. Defaults to 'cogvideox'.
            with_audio (bool, optional): Whether to generate AI audio. Defaults to False.
            size (str, optional): The size of the video. Defaults to '1920x1080'.
            duration (int, optional): The duration of the video in seconds. Defaults to 5.
            fps (int, optional): The frame rate of the video. Defaults to 30.

        Example:
            from agentica.tools.cogvideo_tool import CogVideoTool
            tool = CogVideoTool()
            image_path = tool.create_video("A painting of a beautiful sunset over the ocean.")
            print(image_path)

        Returns:
            str: The json result for the video generation.
        """
        params = {"model": model}
        if prompt:
            params["prompt"] = prompt
        if image_url:
            params["image_url"] = image_url
            if size:
                params["size"] = size
            if with_audio:
                params["with_audio"] = with_audio
            if duration:
                params["duration"] = duration
            if fps:
                params["fps"] = fps
        logger.info(f"params: {params}")
        response = self.client.videos.generations(**params)
        # 异步接口，轮询获取视频生成结果
        video_id = response.id
        while True:
            result = self.client.videos.retrieve_videos_result(id=video_id)
            if result.task_status == "SUCCESS":
                logger.info("Video generation succeeded. URL:", result)
                break
            elif result.task_status == "FAIL":
                logger.debug("Video generation failed.")
                break
            elif result.task_status == "PROCESSING":
                logger.debug("Video generation in progress. Checking again in 10 seconds...")
                time.sleep(10)
        video_url = result.video_result[0].url
        cover_image_url = result.video_result[0].cover_image_url
        video_path = f"{self.data_dir}/{self._generate_name(prompt)}"
        self._download_and_save_video(video_url, video_path)
        saved_video_path = os.path.abspath(video_path)
        result = {"action": "create_video", "result_video_url": video_url, "result_video_path": saved_video_path,
                  "cover_image_url": cover_image_url}
        return json.dumps(result, ensure_ascii=False)

    def _download_and_save_video(self, video_url: str, saved_video_path: str) -> None:
        """
        Downloads the video from the given URL and saves it to the specified path.

        :param video_url: The URL of the video.
        :type video_url: str
        :param saved_video_path: The path to save the video.
        :type saved_video_path: str
        """
        try:
            video_data = requests.get(video_url).content
            if os.path.dirname(saved_video_path):
                os.makedirs(os.path.dirname(saved_video_path), exist_ok=True)
            with open(saved_video_path, "wb") as f:
                f.write(video_data)
            logger.info(f"Video saved to: {saved_video_path}")
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")


if __name__ == '__main__':
    # from agentica.tools.cogview_tool import CogViewTool
    m = CogVideoTool()
    prompt = "比得兔开小汽车，游走在马路上，脸上的表情充满开心喜悦"
    r = m.create_video(prompt)
    print(r)
