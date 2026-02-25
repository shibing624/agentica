# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model providers for Agentica.

Core providers:
    - OpenAIChat, OpenAILike: OpenAI and compatible APIs
    - Claude: Anthropic Claude
    - AzureOpenAIChat: Azure OpenAI
    - OllamaChat: Ollama local models
    - LiteLLMChat: Universal gateway (100+ providers)
    - KimiChat: Kimi for Coding (Anthropic protocol)

OpenAI-compatible providers via registry:
    from agentica.model.providers import create_provider
    model = create_provider("deepseek")
"""
