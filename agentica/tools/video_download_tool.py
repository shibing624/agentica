# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# pip install yt-dlp ffmpeg-python Pillow

import io
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from PIL import Image
from agentica.tools.base import Tool
from agentica.utils.log import logger


def capture_screenshot(video_file: str, timestamp: float) -> Image.Image:
    r"""Capture a screenshot from a video file at a specific timestamp.

    Args:
        video_file (str): The path to the video file.
        timestamp (float): The time in seconds from which to capture the
          screenshot.

    Returns:
        Image.Image: The captured screenshot in the form of Image.Image.
    """
    import ffmpeg

    try:
        out, _ = (
            ffmpeg.input(video_file, ss=timestamp)
            .filter('scale', 320, -1)
            .output('pipe:', vframes=1, format='image2', vcodec='png')
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to capture screenshot: {e.stderr}")

    return Image.open(io.BytesIO(out))


class VideoDownloaderTool(Tool):
    r"""A class for downloading videos and optionally splitting them into
    chunks.

    Args:
        download_directory (Optional[str], optional): The directory where the
            video will be downloaded to. If not provided, video will be stored
            in a temporary directory and will be cleaned up after use.
            (default: :obj:`None`)
        cookies_path (Optional[str], optional): The path to the cookies file
            for the video service in Netscape format. (default: :obj:`None`)
    """

    def __init__(
            self,
            download_directory: Optional[str] = None,
            cookies_path: Optional[str] = None,
    ) -> None:
        super().__init__(name="video_downloader_tool")
        self._cleanup = download_directory is None
        self._cookies_path = cookies_path

        self._download_directory = Path(
            download_directory or tempfile.mkdtemp()
        ).resolve()

        try:
            self._download_directory.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            raise ValueError(
                f"{self._download_directory} is not a valid directory."
            )
        except OSError as e:
            raise ValueError(
                f"Error creating directory {self._download_directory}: {e}"
            )

        logger.debug(f"Video will be downloaded to {self._download_directory}")

        self.register(self.download_video)
        self.register(self.get_video_bytes)
        self.register(self.get_video_screenshots)

    def __del__(self) -> None:
        r"""Deconstructor for the VideoDownloaderToolkit class.

        Cleans up the downloaded video if they are stored in a temporary
        directory.
        """
        if self._cleanup:
            try:
                import sys

                if getattr(sys, 'modules', None) is not None:
                    import shutil

                    shutil.rmtree(self._download_directory, ignore_errors=True)
            except (ImportError, AttributeError):
                # Skip cleanup if interpreter is shutting down
                pass

    def download_video(self, url: str, browser_name: str = None) -> str:
        r"""Download the video and optionally split it into chunks.

        yt-dlp will detect if the video is downloaded automatically so there
        is no need to check if the video exists.

        Args:
            url (str): The URL of the video to download.
            browser_name (str, optional): Browser name to extract cookies from.
                Options include: 'chrome', 'firefox', 'safari', 'edge', etc.
                If None, will use cookiefile if provided, otherwise no cookies.

        Returns:
            str: The path to the downloaded video file.
        """
        import yt_dlp

        video_template = self._download_directory / "%(title)s.%(ext)s"
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # 优先选择mp4格式
            # 'format': 'bestvideo+bestaudio/best',
            'outtmpl': str(video_template),
            'force_generic_extractor': True,
            'ignoreerrors': False,
            'quiet': False,
            'no_warnings': False,
        }
        # Add cookies handling
        if browser_name:
            ydl_opts['cookiesfrombrowser'] = (browser_name, None, None, None)
        elif self._cookies_path:
            ydl_opts['cookiefile'] = self._cookies_path

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Checking available formats for {url}")
                meta_info = ydl.extract_info(url, download=False)

                # 如果只有图片可用，则调整为下载缩略图
                if 'formats' in meta_info and len(meta_info['formats']) == 0:
                    logger.warning("No video formats available, downloading thumbnail instead")
                    ydl_opts['writethumbnail'] = True
                    ydl_opts['skip_download'] = True
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_thumb:
                        info = ydl_thumb.extract_info(url, download=True)
                        # 返回缩略图路径
                        filename = ydl_thumb.prepare_filename(info)
                        thumbnail_path = filename.rsplit(".", 1)[0] + ".jpg"
                        return thumbnail_path
                logger.info(f"Downloading video from {url}")
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as e:
            logger.warning(f"First download attempt failed: {e}. Trying with simpler options...")
            ydl_opts['format'] = 'best'  # 简化格式选择
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return ydl.prepare_filename(info)
            except Exception as e2:
                raise RuntimeError(f"Failed to download video from {url}: {e2}")
        except Exception as e:
            raise RuntimeError(f"Failed to download video from {url}: {e}")

    def get_video_bytes(
            self,
            video_path: str,
    ) -> bytes:
        r"""Download video by the path, and return the content in bytes.

        Args:
            video_path (str): The path to the video file.

        Returns:
            bytes: The video file content in bytes.
        """
        parsed_url = urlparse(video_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        if is_url:
            video_path = self.download_video(video_path)
        video_file = video_path

        with open(video_file, 'rb') as f:
            video_bytes = f.read()

        return video_bytes

    def get_video_screenshots(
            self, video_path: str, amount: int
    ) -> List[Image.Image]:
        r"""Capture screenshots from the video at specified timestamps or by
        dividing the video into equal parts if an integer is provided.

        Args:
            video_url (str): The URL of the video to take screenshots.
            amount (int): the amount of evenly split screenshots to capture.

        Returns:
            List[Image.Image]: A list of screenshots as Image.Image.
        """
        import ffmpeg

        parsed_url = urlparse(video_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        if is_url:
            video_path = self.download_video(video_path)
        video_file = video_path

        # Get the video length
        try:
            probe = ffmpeg.probe(video_file)
            video_length = float(probe['format']['duration'])
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to determine video length: {e.stderr}")

        interval = video_length / (amount + 1)
        timestamps = [i * interval for i in range(1, amount + 1)]

        images = [capture_screenshot(video_file, ts) for ts in timestamps]

        return images

if __name__ == "__main__":
    tool = VideoDownloaderTool()
    video_url = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    r = tool.download_video(video_url)
    print(r)
    video_bytes = tool.get_video_bytes(video_url)
    print(f"Video bytes length: {len(video_bytes)}")
    # get video screenshots
    screenshots = tool.get_video_screenshots(r, 3)
    print(f"Captured {len(screenshots)} screenshots from the video.")
    for i, img in enumerate(screenshots):
        img.show(title=f"Screenshot {i+1}")
        # Save the screenshot if needed
        img.save(f"screenshot_{i+1}.png")
    # Clean up screenshots
    for i in range(len(screenshots)):
        Path(f"screenshot_{i+1}.png").unlink(missing_ok=True)
    # Clean up video
    Path(r).unlink(missing_ok=True)
    print("Cleaned up screenshots and video file.")
