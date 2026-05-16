# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model providers for Agentica.

Core providers:
    - OpenAIChat: OpenAI Chat Completions API
    - Claude: Anthropic Claude
    - AzureOpenAIChat: Azure OpenAI
    - OllamaChat: Ollama local models
    - LiteLLMChat: Universal gateway (100+ providers)
    - KimiChat: Kimi for Coding (Anthropic protocol)

OpenAI-compatible providers (DeepSeek, Qwen, ZhipuAI, Moonshot, Ark, Together,
Grok, Yi, Nvidia, Sambanova, OpenRouter, Fireworks, InternLM): use the
top-level factories from ``agentica`` directly, e.g.::

    from agentica import DeepSeekChat, ZhipuAIChat
    model = DeepSeekChat(id="deepseek-v4-pro")
"""
