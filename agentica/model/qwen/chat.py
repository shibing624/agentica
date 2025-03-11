from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class Qwen(OpenAILike):
    """
    A model class for YI Chat API.

    Attributes:
    - id: str: The unique identifier of the model, model_name.
    - name: str: The name of the class.
    - provider: str: The provider of the model.
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model.
    """

    id: str = "qwen-max"
    name: str = "Qwen"
    provider: str = "Alibaba"

    api_key: Optional[str] = getenv("DASHSCOPE_API_KEY", None)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
