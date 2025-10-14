from time import time
from enum import Enum
from typing import Optional, Any, Dict

from dataclasses import dataclass


class ModelResponseEvent(str, Enum):
    """Events that can be sent by the Model.response() method"""

    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"
    assistant_response = "AssistantResponse"
    reasoning_response = "ReasoningResponse"


@dataclass
class ModelResponse:
    """Response returned by Model.response()"""

    content: Optional[str] = None
    parsed: Optional[Any] = None
    audio: Optional[Dict[str, Any]] = None
    tool_call: Optional[Dict[str, Any]] = None
    event: str = ModelResponseEvent.assistant_response.value
    reasoning_content: Optional[str] = None
    created_at: int = int(time())


class FileType(str, Enum):
    MP4 = "mp4"
    GIF = "gif"
    MP3 = "mp3"
