from os import getenv
from typing import Optional

from agentica.model.openai.like import OpenAILike


class Grok(OpenAILike):
    """
    Class for interacting with the xAI API.

    Attributes:
        id (str): The ID of the language model.
        name (str): The name of the API.
        provider (str): The provider of the API.
        api_key (Optional[str]): The API key for the xAI API.
        base_url (Optional[str]): The base URL for the xAI API.
    """

    id: str = "grok-beta"
    name: str = "Grok"
    provider: str = "xAI"

    api_key: Optional[str] = getenv("XAI_API_KEY")
    base_url: Optional[str] = "https://api.x.ai/v1"
