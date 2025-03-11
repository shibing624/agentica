from typing import Optional
from os import getenv

from agentica.model.openai.like import OpenAILike


class ZhipuAI(OpenAILike):
    """
    A model class for ZhipuAI Chat API.

    Attributes:
    - id: str: The unique identifier of the model, model_name.
    - name: str: The name of the class.
    - provider: str: The provider of the model.
    - api_key: Optional[str]: The API key for the model.
    - base_url: str: The base URL for the model.
    """

    id: str = "glm-4-flash"
    name: str = "ZhipuAI"
    provider: str = "ZhipuAI"

    api_key: Optional[str] = getenv("ZHIPUAI_API_KEY", None)
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
