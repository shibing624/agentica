# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
OpenAI-Compatible provider preset configuration registry.

Replaces 13 thin wrapper directories (deepseek/, qwen/, etc.) with a single
registry + factory function. Each provider is just a configuration entry.

Usage:
    from agentica.model.providers import create_provider
    model = create_provider("deepseek")
    model = create_provider("deepseek", id="deepseek-reasoner")
"""

from dataclasses import dataclass
from typing import Optional, Dict
from os import getenv


@dataclass
class ProviderConfig:
    """Configuration for an OpenAI-Compatible provider."""
    name: str
    default_model: str
    base_url: str
    api_key_env: str
    api_key_env_fallback: Optional[str] = None
    provider: Optional[str] = None
    context_window: int = 128000
    max_output_tokens: Optional[int] = None


# Global registry
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}


def register_provider(key: str, config: ProviderConfig) -> None:
    """Register an OpenAI-Compatible provider configuration."""
    PROVIDER_REGISTRY[key] = config


def get_provider_config(key: str) -> ProviderConfig:
    """Get a registered provider configuration."""
    if key not in PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown provider '{key}'. "
            f"Available: {sorted(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[key]


def list_providers() -> list:
    """List all registered provider names."""
    return sorted(PROVIDER_REGISTRY.keys())


def create_provider(key: str, **overrides):
    """Factory: create an OpenAILike instance from the registry.

    Args:
        key: Provider name in the registry, e.g. "deepseek", "qwen"
        **overrides: Override default config params, e.g. id="deepseek-reasoner"
            Supports both id= and model= (Pydantic alias) for model specification.

    Returns:
        OpenAILike instance

    Example:
        model = create_provider("deepseek")
        model = create_provider("deepseek", id="deepseek-reasoner")
        model = create_provider("qwen", id="qwen-turbo")
    """
    from agentica.model.openai.like import OpenAILike

    config = get_provider_config(key)
    api_key = getenv(config.api_key_env)
    if api_key is None and config.api_key_env_fallback:
        api_key = getenv(config.api_key_env_fallback)

    params = {
        "id": config.default_model,
        "name": config.name,
        "provider": config.provider or config.name,
        "api_key": api_key,
        "base_url": config.base_url,
    }
    if config.max_output_tokens is not None:
        params["max_output_tokens"] = config.max_output_tokens

    # Handle "model=" as alias for "id=" (Pydantic Field alias compatibility)
    if "model" in overrides:
        overrides["id"] = overrides.pop("model")
    params.update(overrides)
    return OpenAILike(**params)


# ── Built-in provider registrations ────────────────────────────────────────

register_provider("deepseek", ProviderConfig(
    name="DeepSeek",
    default_model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
))

register_provider("doubao", ProviderConfig(
    name="Doubao",
    default_model=getenv("ARK_MODEL_NAME", "doubao-1.5-pro-32k"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key_env="ARK_API_KEY",
    provider="ByteDance",
))

register_provider("qwen", ProviderConfig(
    name="Qwen",
    default_model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="DASHSCOPE_API_KEY",
    provider="Alibaba",
))

register_provider("xai", ProviderConfig(
    name="Grok",
    default_model="grok-beta",
    base_url="https://api.x.ai/v1",
    api_key_env="XAI_API_KEY",
    provider="xAI",
))

register_provider("yi", ProviderConfig(
    name="Yi",
    default_model="yi-lightning",
    base_url="https://api.lingyiwanwu.com/v1",
    api_key_env="YI_API_KEY",
    provider="01.ai",
))

register_provider("nvidia", ProviderConfig(
    name="Nvidia",
    default_model="nvidia/llama-3.1-nemotron-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="NVIDIA_API_KEY",
))

register_provider("internlm", ProviderConfig(
    name="InternLM",
    default_model="internlm2.5-latest",
    base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions",
    api_key_env="INTERNLM_API_KEY",
))

register_provider("moonshot", ProviderConfig(
    name="MoonShot",
    default_model="kimi-k2.5",
    base_url="https://api.moonshot.cn/v1",
    api_key_env="MOONSHOT_API_KEY",
))

register_provider("zhipuai", ProviderConfig(
    name="ZhipuAI",
    default_model="glm-4.7-flash",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key_env="ZHIPUAI_API_KEY",
    api_key_env_fallback="ZAI_API_KEY",
))

register_provider("sambanova", ProviderConfig(
    name="Sambanova",
    default_model="Meta-Llama-3.1-8B-Instruct",
    base_url="https://api.sambanova.ai/v1",
    api_key_env="SAMBANOVA_API_KEY",
))

register_provider("openrouter", ProviderConfig(
    name="OpenRouter",
    default_model="gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    max_output_tokens=16384,
))

register_provider("fireworks", ProviderConfig(
    name="Fireworks",
    default_model="accounts/fireworks/models/firefunction-v2",
    base_url="https://api.fireworks.ai/inference/v1",
    api_key_env="FIREWORKS_API_KEY",
))

register_provider("together", ProviderConfig(
    name="Together",
    default_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    api_key_env="TOGETHER_API_KEY",
))
