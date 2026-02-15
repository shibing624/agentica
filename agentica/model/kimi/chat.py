# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Kimi for Coding model, compatible with Anthropic Claude API protocol.

Uses anthropic SDK with custom base_url pointing to Kimi API.
"""
from typing import Optional
from os import getenv

from agentica.model.anthropic.claude import Claude
from agentica.utils.log import logger

try:
    from anthropic import AsyncAnthropic as AnthropicClient
except (ModuleNotFoundError, ImportError):
    raise ImportError("`anthropic` not installed. Please install using `pip install anthropic`")


class KimiChat(Claude):
    """Kimi for Coding model, compatible with Anthropic Claude API.

    Attributes:
        id: Model identifier. Default: "k2p5".
        name: Display name. Default: "KimiChat".
        provider: Provider name. Default: "Kimi".
        api_key: Kimi API key (or set KIMI_API_KEY env var).
        base_url: Kimi API endpoint.
    """

    id: str = "k2p5"
    name: str = "KimiChat"
    provider: str = "Kimi"

    api_key: Optional[str] = None
    base_url: str = "https://api.kimi.com/coding/"

    def get_client(self) -> AnthropicClient:
        if self.client:
            return self.client

        self.api_key = self.api_key or getenv("KIMI_API_KEY")
        if not self.api_key:
            logger.error("KIMI_API_KEY not set. Please set the KIMI_API_KEY environment variable.")

        _client_params = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        _client_params["base_url"] = self.base_url
        if self.client_params:
            _client_params.update(self.client_params)
        return AnthropicClient(**_client_params)
