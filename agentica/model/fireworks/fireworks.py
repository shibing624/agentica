from os import getenv
from typing import Optional, List, Iterator

from agentica.model.message import Message
from agentica.model.openai import OpenAILike
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class Fireworks(OpenAILike):
    """
    Fireworks model

    Attributes:
        id (str): The model name to use. Defaults to "accounts/fireworks/models/firefunction-v2".
        name (str): The model name to use. Defaults to "Fireworks: " + id.
        provider (str): The provider to use. Defaults to "Fireworks".
        api_key (Optional[str]): The API key to use. Defaults to getenv("FIREWORKS_API_KEY").
        base_url (str): The base URL to use. Defaults to "https://api.fireworks.ai/inference/v1".
    """

    id: str = "accounts/fireworks/models/firefunction-v2"
    name: str = "Fireworks: " + id
    provider: str = "Fireworks"

    api_key: Optional[str] = getenv("FIREWORKS_API_KEY", None)
    base_url: str = "https://api.fireworks.ai/inference/v1"
