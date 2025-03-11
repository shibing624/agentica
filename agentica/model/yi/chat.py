from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class Yi(OpenAILike):
    """
    A model class for YI Chat API.

    Attributes:
    - id: str: The unique identifier of the model, model_name.
    - name: str: The name of the class.
    - provider: str: The provider of the model.
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model.
    """

    id: str = "yi-lightning"
    name: str = "Yi"
    provider: str = "01.ai"

    api_key: Optional[str] = getenv("YI_API_KEY", None)
    base_url: str = "https://api.lingyiwanwu.com/v1"
