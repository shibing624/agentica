# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Prompts for ``agentica.compression`` (context compression).

- ``DEFAULT_COMPRESSION_PROMPT``: one-shot tool-result summarisation.
- ``ITERATIVE_COMPRESSION_PROMPT``: incremental summary update template;
  callers fill ``{previous_summary}`` and ``{new_messages}`` via
  ``str.format``.
"""
from agentica.prompts.utils import load_prompt as _load_prompt


def _load(filename: str) -> str:
    return _load_prompt(__file__, filename)


DEFAULT_COMPRESSION_PROMPT: str = _load("default.md")
ITERATIVE_COMPRESSION_PROMPT: str = _load("iterative.md")


__all__ = ["DEFAULT_COMPRESSION_PROMPT", "ITERATIVE_COMPRESSION_PROMPT"]
