from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class DoubaoChat(OpenAILike):
    """
    A model class for Doubao Chat API.

    Attributes:
    - id: str: The unique identifier of the model, model_name.
    - name: str: The name of the class.
    - provider: str: The provider of the model.
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model.
    """

    id: str = "ep-20241012172611-btlgr"
    name: str = "DoubaoChat"
    provider: str = "Doubao"

    api_key: Optional[str] = getenv("ARK_API_KEY", None)
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
