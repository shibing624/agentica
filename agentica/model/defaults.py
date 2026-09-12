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


def provider_env_var(provider_key: str) -> Optional[str]:
    """Primary env var that holds this provider's API key, or None if unknown.

    The child of a delegated run inherits its credentials this way: the key is
    never put on the command line (``ps`` would show it), only in the process
    environment, and each provider's model client reads its own variable.
    """
    pair = _PROVIDER_ENV_VARS.get(provider_key)
    return pair[0] if pair else None


# Model class name → provider key. Only the base classes appear: the
# agentica.DeepSeekChat / MoonshotChat / ... factories all return plain
# OpenAIChat instances (with their own base_url), so every OpenAI-compatible
# provider lands on "openai" and a delegated child gets the base_url + api_key
# as one pair. AzureOpenAIChat subclasses OpenAIChat but stands EARLIER in the
# MRO, so it is detected first and correctly refused (Azure credentials have no
# environment variable a child process could read). A third-party Model class
# agentica does not know maps to nothing.
_MODEL_CLASS_PROVIDERS = {
    "OpenAIChat": "openai",
    "AzureOpenAIChat": "azure",
    "Claude": "anthropic",
}


def provider_for_model(model: Model) -> Optional[str]:
    """The provider key a Model instance belongs to, or None if unrecognized."""
    for klass in type(model).__mro__:
        provider = _MODEL_CLASS_PROVIDERS.get(klass.__name__)
        if provider:
            return provider
    return None


def model_display_label(model: Optional[Model]) -> Optional[str]:
    """Short ``provider/id`` label for a Model, or None when there is none.

    Used where the user needs to see which model is about to run rather than a
    Model object: ``task`` prints the subagent's tier model on the call line.
    An id that already carries a slash (a proxy-style ``openai/glm-5``) is left
    whole instead of gaining a second provider prefix.
    """
    if model is None:
        return None
    model_id = str(getattr(model, "id", "") or "").strip()
    if not model_id or model_id == "not-provided":
        return None
    provider = provider_for_model(model)
    if provider and "/" not in model_id:
        return f"{provider}/{model_id}"
    return model_id


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
