# -*- coding: utf-8 -*-
"""
Default model resolution for callers that did not pass an explicit model.
"""
from os import getenv
from typing import Optional, Tuple

from agentica.model.base import Model


# Provider key → (primary env var, optional fallback env var)
_PROVIDER_ENV_VARS: dict[str, Tuple[str, Optional[str]]] = {
    "openai": ("OPENAI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "deepseek": ("DEEPSEEK_API_KEY", None),
    "ark": ("ARK_API_KEY", None),
    "zhipuai": ("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
    "qwen": ("DASHSCOPE_API_KEY", None),
    "moonshot": ("MOONSHOT_API_KEY", None),
    "yi": ("YI_API_KEY", None),
    "xai": ("XAI_API_KEY", None),
    "openrouter": ("OPENROUTER_API_KEY", None),
    "together": ("TOGETHER_API_KEY", None),
    "fireworks": ("FIREWORKS_API_KEY", None),
    "sambanova": ("SAMBANOVA_API_KEY", None),
    "nvidia": ("NVIDIA_API_KEY", None),
    "internlm": ("INTERNLM_API_KEY", None),
}

DEFAULT_PROVIDER_ORDER = tuple(_PROVIDER_ENV_VARS.keys())


def _configured_api_key(provider_key: str) -> Optional[str]:
    primary, fallback = _PROVIDER_ENV_VARS[provider_key]
    value = getenv(primary)
    if value is None and fallback:
        value = getenv(fallback)
    return value


def _create_model(provider_key: str) -> Model:
    if provider_key == "openai":
        from agentica.model.openai import OpenAIChat
        return OpenAIChat(api_key=getenv("OPENAI_API_KEY"))
    if provider_key == "anthropic":
        from agentica.model.anthropic.claude import Claude
        return Claude(api_key=getenv("ANTHROPIC_API_KEY"))
    from agentica import PROVIDER_FACTORIES
    return PROVIDER_FACTORIES[provider_key]()


def create_default_model() -> Model:
    """Create a default model from configured provider credentials.

    Preserve the historical OpenAI default when ``OPENAI_API_KEY`` is present,
    then fall through to other configured providers in a deterministic order.
    """
    for provider_key in DEFAULT_PROVIDER_ORDER:
        if _configured_api_key(provider_key):
            return _create_model(provider_key)

    env_names = []
    for provider_key in DEFAULT_PROVIDER_ORDER:
        primary, fallback = _PROVIDER_ENV_VARS[provider_key]
        env_names.append(primary)
        if fallback:
            env_names.append(fallback)

    raise RuntimeError(
        "No default LLM provider is configured. Pass model=... explicitly or set "
        f"one of: {', '.join(env_names)}."
    )
