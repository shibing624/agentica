"""
This module provides a class for interacting with OpenAI's LLMs. It includes a dataclass for settings and a class for managing the interaction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from actionflow.config import api_key, base_url, default_model


@dataclass
class Settings:
    """
    This dataclass holds the settings for interacting with OpenAI's LLMs.
    """

    model: str = default_model
    tool_name: Optional[str] = None
    tool_choice: Optional[str] = "auto"  # tool_choice="none" means no tool, default "auto"
    temperature: float = 1.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    def to_dict(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "tool_choice": self.tool_choice,
        }


class LLM:
    """
    This class is responsible for managing the interaction with OpenAI's LLMs.
    """

    def __init__(self, api_key: str = api_key, base_url: str = base_url):
        """
        Initializes the LLM object by loading the environment variables and setting the OpenAI API key.

        :param api_key: The OpenAI API key.
        :type api_key: str
        :param base_url: The base URL for the OpenAI API.
        :type base_url: str
        """
        if not api_key:
            raise ValueError("LLM `api_key` is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
        self.base_url = base_url

    def __repr__(self):
        show_api_key = "*" * 6 + self.api_key[-4:]
        return f"LLM(api_key={show_api_key}, base_url={base_url}, client={self.client})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        retry=retry_if_exception_type(ConnectionError),
    )
    def respond(
            self,
            settings: Settings,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """
        Sends a request to OpenAI's LLM API and returns the response.

        :param settings: The settings for the interaction.
        :type settings: Settings
        :param messages: The messages to be processed by the language model.
        :type messages: List[Dict[str, str]]
        :param tools: The tools to be processed by the language model.
        :type tools: Optional[List[Dict[str, str]]]
        :return: The response from the language model.
        :rtype: Any
        """
        openai_args = {k: v for k, v in settings.to_dict().items() if v is not None}
        openai_args["messages"] = messages
        if tools:
            openai_args["tools"] = tools
        # logger.debug(f"openai_args={openai_args}")
        response = self.client.chat.completions.create(**openai_args)
        return response.choices[0].message
