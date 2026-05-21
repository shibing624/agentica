# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Prompts for ``agentica.critic`` (critic/revise loop).

Both templates contain literal ``{placeholder}`` markers. They are loaded
without substitution here; ``critic.py`` calls ``.format(...)`` at use
time with ``style_guidance``, ``task``, ``draft``, ``approval_token``, and
``feedback`` as appropriate.
"""
from agentica.prompts.utils import load_prompt as _load_prompt


def _load(filename: str) -> str:
    return _load_prompt(__file__, filename)


CRITIC_PROMPT_TEMPLATE: str = _load("critic.md")
REVISE_PROMPT_TEMPLATE: str = _load("revise.md")


__all__ = ["CRITIC_PROMPT_TEMPLATE", "REVISE_PROMPT_TEMPLATE"]
