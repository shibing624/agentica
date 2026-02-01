# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Model-specific prompt modules

Model-specific prompts optimized for different LLM providers:
- claude: Claude models (Opus, Sonnet, Haiku)
- openai: GPT models (GPT-4o, GPT-4, o1)
- zhipu: Zhipu GLM models
- deepseek: DeepSeek models
- default: Fallback for other models
"""

from agentica.prompts.models.claude import (
    CLAUDE_SPECIFIC_PROMPT,
    get_claude_prompt,
)
from agentica.prompts.models.openai import (
    OPENAI_SPECIFIC_PROMPT,
    get_openai_prompt,
)
from agentica.prompts.models.zhipu import (
    ZHIPU_SPECIFIC_PROMPT,
    get_zhipu_prompt,
)
from agentica.prompts.models.deepseek import (
    DEEPSEEK_SPECIFIC_PROMPT,
    get_deepseek_prompt,
)
from agentica.prompts.models.default import (
    DEFAULT_PROMPT,
    get_default_prompt,
)

__all__ = [
    "CLAUDE_SPECIFIC_PROMPT",
    "get_claude_prompt",
    "OPENAI_SPECIFIC_PROMPT",
    "get_openai_prompt",
    "ZHIPU_SPECIFIC_PROMPT",
    "get_zhipu_prompt",
    "DEEPSEEK_SPECIFIC_PROMPT",
    "get_deepseek_prompt",
    "DEFAULT_PROMPT",
    "get_default_prompt",
]
