from dataclasses import dataclass
from typing import Optional
from agentica.model.openai.chat import OpenAIChat


@dataclass
class OpenAILike(OpenAIChat):
    id: str = "not-provided"
    name: str = "OpenAILike"
    api_key: Optional[str] = "not-provided"
