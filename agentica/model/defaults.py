# -*- coding: utf-8 -*-
"""
Default model resolution for callers that did not pass an explicit model.
"""
from os import getenv
from typing import Optional

from agentica.model.base import Model
from agentica.model.providers import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    create_provider,
)


DEFAULT_PROVIDER_ORDER = (
    "openai",
    "anthropic",
    "deepseek",
    "doubao",
    "zhipuai",
    "qwen",
    "moonshot",
    "yi",
    "xai",
    "openrouter",
    "together",
    "fireworks",
    "sambanova",
    "nvidia",
    "internlm",
)


def _configured_registry_api_key(config: ProviderConfig) -> Optional[str]:
    api_key = getenv(config.api_key_env)
    if api_key is None and config.api_key_env_fallback:
        api_key = getenv(config.api_key_env_fallback)
    return api_key


def _configured_api_key(provider_key: str) -> Optional[str]:
    if provider_key == "openai":
        return getenv("OPENAI_API_KEY")
    if provider_key == "anthropic":
        return getenv("ANTHROPIC_API_KEY")
    return _configured_registry_api_key(PROVIDER_REGISTRY[provider_key])


def _create_model(provider_key: str) -> Model:
    if provider_key == "openai":
        from agentica.model.openai import OpenAIChat

        return OpenAIChat(api_key=getenv("OPENAI_API_KEY"))
    if provider_key == "anthropic":
        from agentica.model.anthropic.claude import Claude

        return Claude(api_key=getenv("ANTHROPIC_API_KEY"))
    return create_provider(provider_key)


def create_default_model() -> Model:
    """Create a default model from configured provider credentials.

    Preserve the historical OpenAI default when ``OPENAI_API_KEY`` is present,
    then fall through to other configured providers in a deterministic order.
    """
    for provider_key in DEFAULT_PROVIDER_ORDER:
        if _configured_api_key(provider_key):
            return _create_model(provider_key)

    env_names = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for provider_key in DEFAULT_PROVIDER_ORDER:
        if provider_key in ("openai", "anthropic"):
            continue
        config = PROVIDER_REGISTRY[provider_key]
        env_names.append(config.api_key_env)
        if config.api_key_env_fallback:
            env_names.append(config.api_key_env_fallback)

    raise RuntimeError(
        "No default LLM provider is configured. Pass model=... explicitly or set "
        f"one of: {', '.join(env_names)}."
    )
