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
import base64
import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional, cast
from PIL import Image

from agentica.model.openai.chat import OpenAIChat, OpenAIImageType
from agentica.model.openai.audio import OpenAIAudioModel
from agentica.model.base import Model
from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.tools.video_download_tool import (
    VideoDownloaderTool,
    capture_screenshot,
)

VIDEO_QA_PROMPT = """
Analyze the provided video frames and corresponding audio transcription to \
answer the given question(s) thoroughly and accurately.

Instructions:
    1. Visual Analysis:
        - Examine the video frames to identify visible entities.
        - Differentiate objects, species, or features based on key attributes \
such as size, color, shape, texture, or behavior.
        - Note significant groupings, interactions, or contextual patterns \
relevant to the analysis.

    2. Audio Integration:
        - Use the audio transcription to complement or clarify your visual \
observations.
        - Identify names, descriptions, or contextual hints in the \
transcription that help confirm or refine your visual analysis.

    3. Detailed Reasoning and Justification:
        - Provide a brief explanation of how you identified and distinguished \
each species or object.
        - Highlight specific features or contextual clues that informed \
your reasoning.

    4. Comprehensive Answer:
        - Specify the total number of distinct species or object types \
identified in the video.
        - Describe the defining characteristics and any supporting evidence \
from the video and transcription.

    5. Important Considerations:
        - Pay close attention to subtle differences that could distinguish \
similar-looking species or objects 
          (e.g., juveniles vs. adults, closely related species).
        - Provide concise yet complete explanations to ensure clarity.

output language: {output_language} 

**Audio Transcription:**
{audio_transcription}

**Question:**
{question}
"""


class VideoAnalysisTool(Tool):
    r"""A class for analysing videos with vision-language model.

    Args:
        download_directory (Optional[str], optional): The directory where the
            video will be downloaded to. If not provided, video will be stored
            in a temporary directory and will be cleaned up after use.
            (default: :obj:`None`)
        model (Optional[BaseModelBackend], optional): The model to use for
            visual analysis. (default: :obj:`None`)
        use_audio_transcription (bool, optional): Whether to enable audio
            transcription using OpenAI's audio models. Requires a valid OpenAI
            API key. When disabled, video analysis will be based solely on
            visual content. (default: :obj:`False`)
        frame_interval (float, optional): Interval in seconds between frames
            to extract from the video. (default: :obj:`4.0`)
        output_language (str, optional): The language for output responses.
            (default: :obj:`"English"`)
        cookies_path (Optional[str]): The path to the cookies file
            for the video service in Netscape format. (default: :obj:`None`)
        model_name (str, optional): The name of the model to use for video analysis
            image. Defaults to "gpt-4o".
    """

    def __init__(
            self,
            download_directory: Optional[str] = None,
            model: Optional[Model] = None,
            use_audio_transcription: bool = False,
            frame_interval: float = 4.0,
            output_language: str = "en",
            cookies_path: Optional[str] = None,
            model_name: str = "gpt-4o"
    ) -> None:
        super().__init__(name="video_analysis_tool")
        self._cleanup = download_directory is None
        self._temp_files: list[str] = []  # Track temporary files for cleanup
        self._use_audio_transcription = use_audio_transcription
        self.output_language = output_language
        self.frame_interval = frame_interval

        self._download_directory = Path(
            download_directory or tempfile.mkdtemp()
        ).resolve()

        self.video_downloader_toolkit = VideoDownloaderTool(
            download_directory=str(self._download_directory),
            cookies_path=cookies_path,
        )

        try:
            self._download_directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Error creating directory {self._download_directory}: {e}"
            )

        logger.debug(f"Video will be downloaded to {self._download_directory}")

        self.vl_model = model
        self.model_name = model_name
        if self.vl_model is None:
            self.vl_model = OpenAIChat(id=self.model_name)

        # Initialize audio models only if audio transcription is enabled
        self.audio_model = None
        if self._use_audio_transcription:
            try:
                self.audio_model = OpenAIAudioModel()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize OpenAIAudioModels: {e}. "
                    "Audio transcription will be disabled."
                )
                self._use_audio_transcription = False

        # Register the tool methods
        self.register(self.ask_question_about_video)

    def __del__(self):
        r"""Clean up temporary directories and files when the object is
        destroyed.
        """
        # Clean up temporary files
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
                except OSError as e:
                    logger.warning(
                        f"Failed to remove temporary file {temp_file}: {e}"
                    )

        # Clean up temporary directory if needed
        if self._cleanup and os.path.exists(self._download_directory):
            try:
                import sys

                if getattr(sys, 'modules', None) is not None:
                    import shutil

                    shutil.rmtree(self._download_directory)
                    logger.debug(
                        f"Removed temp directory: {self._download_directory}"
                    )
            except (ImportError, AttributeError):
                # Skip cleanup if interpreter is shutting down
                pass
            except OSError as e:
                logger.warning(
                    f"Failed to remove temporary directory "
                    f"{self._download_directory}: {e}"
                )

    def _extract_audio_from_video(
            self, video_path: str, output_format: str = "mp3"
    ) -> str:
        r"""Extract audio from the video.

        Args:
            video_path (str): The path to the video file.
            output_format (str): The format of the audio file to be saved.
                (default: :obj:`"mp3"`)

        Returns:
            str: The path to the audio file.
        """
        import ffmpeg

        # Handle case where video file doesn't have an extension
        base_path = os.path.splitext(video_path)[0]
        output_path = f"{base_path}.{output_format}"

        try:
            (
                ffmpeg.input(video_path)
                .output(output_path, vn=None, acodec="libmp3lame")
                .run(quiet=True)
            )
            # Track the audio file for cleanup
            self._temp_files.append(output_path)
            return output_path
        except ffmpeg.Error as e:
            error_message = f"FFmpeg-Python failed: {e}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    def _transcribe_audio(self, audio_path: str) -> str:
        r"""Transcribe the audio of the video."""
        # Check if audio transcription is enabled and audio models are
        # available
        if not self._use_audio_transcription or self.audio_model is None:
            logger.warning("Audio transcription is disabled or not available")
            return "No audio transcription available."

        try:
            audio_transcript = self.audio_model.speech_to_text(audio_path)
            if not audio_transcript:
                logger.warning("Audio transcription returned empty result")
                return "No audio transcription available."
            return audio_transcript
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return "Audio transcription failed."

    def _extract_keyframes(self, video_path: str) -> List[Image.Image]:
        r"""Extract keyframes from a video based on scene changes and
        regular intervals,and return them as PIL.Image.Image objects.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[Image.Image]: A list of PIL.Image.Image objects representing
                the extracted keyframes.

        Raises:
            ValueError: If no frames could be extracted from the video.
        """
        import cv2
        import numpy as np
        from scenedetect import (  # type: ignore[import-untyped]
            SceneManager,
            open_video,
        )
        from scenedetect.detectors import (  # type: ignore[import-untyped]
            ContentDetector,
        )

        # Get video information
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        frame_interval = self.frame_interval  # seconds
        # Maximum number of frames to extract to avoid memory issues
        MAX_FRAMES = 100
        # Minimum time difference (in seconds) to consider frames as distinct
        TIME_THRESHOLD = 1.0

        # Calculate the total number of frames to extract
        if duration <= 0 or fps <= 0:
            logger.warning(
                "Invalid video duration or fps, using default frame count"
            )
            num_frames = 10
        else:
            num_frames = max(int(duration / frame_interval), 1)

            if num_frames > MAX_FRAMES:
                frame_interval = duration / MAX_FRAMES
                num_frames = MAX_FRAMES

            logger.info(
                f"Video duration: {duration:.2f}s, target frames: {num_frames}"
                f"at {frame_interval:.2f}s intervals"
            )

        # Use scene detection to extract keyframes
        # Use open_video instead of VideoManager
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        # Detect scenes using the modern API
        scene_manager.detect_scenes(video)

        scenes = scene_manager.get_scene_list()
        keyframes: List[Image.Image] = []

        # If scene detection is successful, prioritize scene change points
        if scenes:
            logger.info(f"Detected {len(scenes)} scene changes")

            if len(scenes) > num_frames:
                scene_indices = np.linspace(
                    0, len(scenes) - 1, num_frames, dtype=int
                )
                selected_scenes = [scenes[i] for i in scene_indices]
            else:
                selected_scenes = scenes

            # Extract frames from scenes
            for scene in selected_scenes:
                try:
                    # Get start time in seconds
                    start_time = scene[0].get_seconds()
                    frame = capture_screenshot(video_path, start_time)
                    keyframes.append(frame)
                except Exception as e:
                    logger.warning(
                        f"Failed to capture frame at scene change"
                        f" {scene[0].get_seconds()}s: {e}"
                    )

        if len(keyframes) < num_frames and duration > 0:
            logger.info(
                f"Scene detection provided {len(keyframes)} frames, "
                f"supplementing with regular interval frames"
            )

            existing_times = []
            if scenes:
                existing_times = [scene[0].get_seconds() for scene in scenes]

            regular_frames = []
            for i in range(num_frames):
                time_sec = i * frame_interval

                is_duplicate = False
                for existing_time in existing_times:
                    if abs(existing_time - time_sec) < TIME_THRESHOLD:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    try:
                        frame = capture_screenshot(video_path, time_sec)
                        regular_frames.append(frame)
                    except Exception as e:
                        logger.warning(
                            f"Failed to capture frame at {time_sec}s: {e}"
                        )

            frames_needed = num_frames - len(keyframes)
            if frames_needed > 0 and regular_frames:
                if len(regular_frames) > frames_needed:
                    indices = np.linspace(
                        0, len(regular_frames) - 1, frames_needed, dtype=int
                    )
                    selected_frames = [regular_frames[i] for i in indices]
                else:
                    selected_frames = regular_frames

                keyframes.extend(selected_frames)

        if not keyframes:
            logger.warning(
                "No frames extracted, falling back to simple interval"
                "extraction"
            )
            for i in range(
                    min(num_frames, 10)
            ):  # Limit to a maximum of 10 frames to avoid infinite loops
                time_sec = i * (duration / 10 if duration > 0 else 6.0)
                try:
                    frame = capture_screenshot(video_path, time_sec)
                    keyframes.append(frame)
                except Exception as e:
                    logger.warning(
                        f"Failed to capture frame at {time_sec}s: {e}"
                    )

        if not keyframes:
            error_msg = (
                f"Failed to extract any keyframes from video: {video_path}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize image sizes
        normalized_keyframes = self._normalize_frames(keyframes)

        logger.info(
            f"Extracted and normalized {len(normalized_keyframes)} keyframes"
        )
        return normalized_keyframes

    def _normalize_frames(
            self, frames: List[Image.Image], target_width: int = 512
    ) -> List[Image.Image]:
        r"""Normalize the size of extracted frames.

        Args:
            frames (List[Image.Image]): List of frames to normalize.
            target_width (int): Target width for normalized frames.

        Returns:
            List[Image.Image]: List of normalized frames.
        """
        normalized_frames: List[Image.Image] = []

        for frame in frames:
            # Get original dimensions
            width, height = frame.size

            # Calculate new height, maintaining aspect ratio
            aspect_ratio = width / height
            new_height = int(target_width / aspect_ratio)

            # Resize image
            resized_frame = frame.resize(
                (target_width, new_height), Image.Resampling.LANCZOS
            )

            # Ensure the image has a proper format
            if resized_frame.mode != 'RGB':
                resized_frame = resized_frame.convert('RGB')

            # Create a new image with explicit format
            with io.BytesIO() as buffer:
                resized_frame.save(buffer, format='JPEG')
                buffer.seek(0)
                formatted_frame = Image.open(buffer)
                formatted_frame.load()  # Load the image data

            normalized_frames.append(formatted_frame)

        return normalized_frames

    def ask_question_about_video(
            self,
            video_path: str,
            question: str,
    ) -> str:
        r"""Ask a question about the video.

        Args:
            video_path (str): The path to the video file.
                It can be a local file or a URL (such as Youtube website).
            question (str): The question to ask about the video.

        Returns:
            str: The answer to the question.
        """
        from urllib.parse import urlparse

        parsed_url = urlparse(video_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])

        downloaded_video_path = None
        try:
            if is_url:
                downloaded_video_path = (
                    self.video_downloader_toolkit.download_video(video_path)
                )
                if not downloaded_video_path or not os.path.exists(
                        downloaded_video_path
                ):
                    raise ValueError(
                        f"Failed to download video from {video_path}"
                    )
                video_path = downloaded_video_path

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            audio_transcript = "No audio transcription available."
            if self._use_audio_transcription:
                audio_path = self._extract_audio_from_video(video_path)
                audio_transcript = self._transcribe_audio(audio_path)

            video_frames = self._extract_keyframes(video_path)

            self.vl_model = cast(Model, self.vl_model)
            prompt = VIDEO_QA_PROMPT.format(
                audio_transcription=audio_transcript,
                question=question,
                output_language=self.output_language,
            )

            hybrid_content = []
            hybrid_content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )
            if video_frames and len(video_frames) > 0:
                for image in video_frames:
                    if image.format is None:
                        raise ValueError(
                            f"Image's `format` is `None`, please "
                            f"transform the `PIL.Image.Image` to  one of "
                            f"following supported formats, such as "
                            f"{list(OpenAIImageType)}"
                        )

                    image_type: str = image.format.lower()
                    if image_type not in OpenAIImageType:
                        raise ValueError(
                            f"Image type {image.format} "
                            f"is not supported by OpenAI vision model"
                        )
                    with io.BytesIO() as buffer:
                        image.save(fp=buffer, format=image.format)
                        encoded_image = base64.b64encode(buffer.getvalue()).decode(
                            "utf-8"
                        )
                    image_prefix = f"data:image/{image_type};base64,"
                    hybrid_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{image_prefix}{encoded_image}"
                            }
                        }
                    )

            messages = [
                {
                    "role": "user",
                    "content": hybrid_content
                }
            ]
            response = self.vl_model.get_client().chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=4000
            )
            if not response or not response.choices:
                logger.error("Model returned empty response")
                return (
                    "Failed to generate an answer. "
                    "The model returned an empty response."
                )

            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            error_message = f"Error processing video: {e}"
            logger.error(error_message)
            return f"Error: {error_message}"

if __name__ == '__main__':
    # demo case
    video_url = (
        "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
    )
    video_analysis_tool = VideoAnalysisTool(
        download_directory=None,
        use_audio_transcription=False,
        frame_interval=4.0,
        output_language="zh",
    )
    question = "What is the main subject of this video?"
    answer = video_analysis_tool.ask_question_about_video(
        video_path=video_url,
        question=question,
    )
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    # Clean up temporary files
    del video_analysis_tool
    import gc
    gc.collect()  # Ensure cleanup happens immediately
    logger.info("Video analysis tool demo completed.")
