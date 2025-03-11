from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class Moonshot(OpenAILike):
    """
    A model class for Moonshot Chat API.

    Attributes:
    - id: str: The unique identifier of the model. Default: "moonshot-v1-auto".
    - name: str: The name of the model. Default: "MoonshotChat".
    - provider: str: The provider of the model. Default: "MoonShot".
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model. Default: "https://api.moonshot.cn/v1".
    """

    id: str = "moonshot-v1-8k"
    name: str = "Moonshot"
    provider: str = "MoonShot"

    api_key: Optional[str] = getenv("MOONSHOT_API_KEY", None)
    base_url: str = "https://api.moonshot.cn/v1"
