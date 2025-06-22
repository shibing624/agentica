# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Text-to-Speech tool using Volcengine TTS API.
"""
import os
import json
import uuid
import base64
from typing import Optional, Dict, Any, List
import requests

from agentica.tools.base import Tool
from agentica.utils.log import logger


class VolcTtsTool(Tool):
    """
    Tool for text-to-speech conversion using Volcengine TTS API.
    """

    def __init__(
            self,
            appid: Optional[str] = None,
            access_token: Optional[str] = None,
            cluster: str = "volcano_tts",
            voice_type: str = "BV700_V2_streaming",
            host: str = "openspeech.bytedance.com",
    ):
        """
        Initialize the Volcengine TTS tool.

        Args:
            appid: Platform application ID. If None, uses VOLCENGINE_TTS_APPID env variable.
            access_token: Access token for authentication. If None, uses VOLCENGINE_TTS_ACCESS_TOKEN env variable.
            cluster: TTS cluster name
            voice_type: Voice type to use
            host: API host
        """
        super().__init__(name="volc_tts_tool")

        self.appid = appid or os.getenv("VOLCENGINE_TTS_APPID")
        self.access_token = access_token or os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN")
        self.cluster = cluster or os.getenv("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        self.voice_type = voice_type
        self.host = host
        self.api_url = f"https://{host}/api/v1/tts"

        if not self.appid or not self.access_token:
            logger.warning("Volcengine TTS credentials not provided. Some functions may not work.")
        else:
            self.header = {"Authorization": f"Bearer;{self.access_token}"}

        # Register the functions with the tool
        self.register(self.text_to_speech)
        self.register(self.text_to_speech_file)
        self.register(self.list_voice_types)

    def text_to_speech(
            self,
            text: str,
            encoding: str = "mp3",
            speed_ratio: float = 1.0,
            volume_ratio: float = 1.0,
            pitch_ratio: float = 1.0,
            text_type: str = "plain",
            voice_type: Optional[str] = None,
            with_frontend: int = 1,
            frontend_type: str = "unitTson",
    ) -> Dict[str, Any]:
        """
        Convert text to speech using Volcengine TTS API and return the base64-encoded audio data.

        Args:
            text: Text to convert to speech
            encoding: Audio encoding format (mp3, wav, etc.)
            speed_ratio: Speech speed ratio (0.5-2.0)
            volume_ratio: Speech volume ratio (0.5-2.0)
            pitch_ratio: Speech pitch ratio (0.5-2.0)
            text_type: Text type (plain or ssml)
            voice_type: Voice type to use (overrides the default)
            with_frontend: Whether to use frontend processing
            frontend_type: Frontend type for text processing

        Returns:
            Dictionary containing success status, response data, and base64-encoded audio

        Example:
            from agentica.tools.volc_tts_tool import VolcTtsTool
            
            tts_tool = VolcTtsTool()
            result = tts_tool.text_to_speech("Hello, how are you today?")
            if result["success"]:
                # Base64 encoded audio is in result["audio_data"]
                print("TTS successful")
            else:
                print(f"TTS failed: {result['error']}")
        """
        if not self.appid or not self.access_token:
            return {
                "success": False,
                "error": "Volcengine TTS credentials not configured",
                "audio_data": None,
            }

        uid = str(uuid.uuid4())
        active_voice_type = voice_type or self.voice_type

        request_json = {
            "app": {
                "appid": self.appid,
                "token": self.access_token,
                "cluster": self.cluster,
            },
            "user": {"uid": uid},
            "audio": {
                "voice_type": active_voice_type,
                "encoding": encoding,
                "speed_ratio": speed_ratio,
                "volume_ratio": volume_ratio,
                "pitch_ratio": pitch_ratio,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": text_type,
                "operation": "query",
                "with_frontend": with_frontend,
                "frontend_type": frontend_type,
            },
        }

        try:
            logger.debug(f"Sending TTS request for text: {text[:50]}...")
            response = requests.post(
                self.api_url, json.dumps(request_json), headers=self.header
            )
            response_json = response.json()

            if response.status_code != 200:
                logger.error(f"TTS API error: {response_json}")
                return {"success": False, "error": response_json, "audio_data": None}

            if "data" not in response_json:
                logger.error(f"TTS API returned no data: {response_json}")
                return {
                    "success": False,
                    "error": "No audio data returned",
                    "audio_data": None,
                }

            return {
                "success": True,
                "response": response_json,
                "audio_data": response_json["data"],  # Base64 encoded audio data
            }

        except Exception as e:
            logger.exception(f"Error in TTS API call: {str(e)}")
            return {"success": False, "error": str(e), "audio_data": None}

    def text_to_speech_file(
            self,
            text: str,
            output_file: str,
            encoding: str = "mp3",
            speed_ratio: float = 1.0,
            volume_ratio: float = 1.0,
            pitch_ratio: float = 1.0,
            voice_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert text to speech and save to a file.

        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
            encoding: Audio encoding format (should match the file extension)
            speed_ratio: Speech speed ratio (0.5-2.0)
            volume_ratio: Speech volume ratio (0.5-2.0)
            pitch_ratio: Speech pitch ratio (0.5-2.0)
            voice_type: Voice type to use (overrides the default)

        Returns:
            Dictionary containing success status and file path

        Example:
            from agentica.tools.volc_tts_tool import VolcTtsTool
            
            tts_tool = VolcTtsTool()
            result = tts_tool.text_to_speech_file(
                "Hello, this is a test.", 
                "output.mp3"
            )
            
            if result["success"]:
                print(f"Audio saved to {result['file_path']}")
            else:
                print(f"Failed: {result['error']}")
        """
        result = self.text_to_speech(
            text=text,
            encoding=encoding,
            speed_ratio=speed_ratio,
            volume_ratio=volume_ratio,
            pitch_ratio=pitch_ratio,
            voice_type=voice_type,
        )

        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "file_path": None,
            }

        try:
            # Ensure the directory exists
            path = os.path.abspath(output_file)
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Decode and save the audio data
            audio_data = base64.b64decode(result["audio_data"])

            with open(output_file, "wb") as f:
                f.write(audio_data)

            logger.info(f"TTS output saved to: {output_file}")

            return {
                "success": True,
                "file_path": output_file,
            }
        except Exception as e:
            logger.exception(f"Error saving TTS output to file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": None,
            }

    def list_voice_types(self) -> List[Dict[str, str]]:
        """
        List available voice types for Volcengine TTS.

        Returns:
            List of dictionaries containing voice types and their descriptions

        Example:
            from agentica.tools.volc_tts_tool import VolcTtsTool
            
            tts_tool = VolcTtsTool()
            voices = tts_tool.list_voice_types()
            for voice in voices:
                print(f"{voice['id']}: {voice['description']}")
        """
        # This is a static list of popular voice types
        # In a real implementation, this might come from an API call
        return [
            {
                "id": "zh_male_M392_conversation_wvae_bigtts",
                "description": "Chinese male voice, conversational style",
                "language": "Chinese",
                "gender": "Male"
            },
            {
                "id": "zh_female_F158_conversation_wvae_bigtts",
                "description": "Chinese female voice, conversational style",
                "language": "Chinese",
                "gender": "Female"
            },
            {
                "id": "BV700_V2_streaming",
                "description": "Default voice (English)",
                "language": "English",
                "gender": "Male"
            },
            {
                "id": "en_female_D101_streaming",
                "description": "English female voice, streaming capable",
                "language": "English",
                "gender": "Female"
            },
            {
                "id": "en_male_M902_streaming",
                "description": "English male voice, streaming capable",
                "language": "English",
                "gender": "Male"
            },
            {
                "id": "jp_female_J008_common",
                "description": "Japanese female voice",
                "language": "Japanese",
                "gender": "Female"
            },
            {
                "id": "kr_female_K002_common",
                "description": "Korean female voice",
                "language": "Korean",
                "gender": "Female"
            }
        ]


if __name__ == '__main__':
    # Example usage
    import os

    # Get credentials from environment variables
    appid = os.getenv("VOLCENGINE_TTS_APPID")
    access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN")

    # Initialize the tool
    tts_tool = VolcTtsTool(appid=appid, access_token=access_token)

    # List available voice types
    voices = tts_tool.list_voice_types()
    print("Available voice types:")
    for voice in voices:
        print(f"- {voice['id']}: {voice['description']} ({voice['language']}, {voice['gender']})")

    # Select a Chinese voice type
    voice_type = "zh_male_M392_conversation_wvae_bigtts"
    test_text = "欢迎使用火山引擎语音合成服务，这是一个测试。"
    output_file = "tts_demo_output.mp3"

    # Convert text to speech and save to file
    result = tts_tool.text_to_speech_file(
        text=test_text,
        output_file=output_file,
        voice_type=voice_type
    )

    if result["success"]:
        print(f"语音合成成功！已保存到文件: {result['file_path']}")
    else:
        print(f"语音合成失败: {result['error']}")
