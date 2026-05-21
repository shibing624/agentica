# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Prompts for ``agentica.swarm`` (peer-to-peer multi-agent
collaboration).

Both templates contain literal ``{placeholder}`` markers that the caller
fills via ``str.format`` at run time, so we return the raw markdown here
without performing substitution.
"""
from agentica.prompts.utils import load_prompt as _load_prompt


def _load(filename: str) -> str:
    return _load_prompt(__file__, filename)


COORDINATOR_SYSTEM_PROMPT: str = _load("coordinator.md")
SYNTHESIZER_PROMPT: str = _load("synthesizer.md")


__all__ = ["COORDINATOR_SYSTEM_PROMPT", "SYNTHESIZER_PROMPT"]
