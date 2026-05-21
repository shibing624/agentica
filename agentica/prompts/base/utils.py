# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: base/ shim around the package-level load_prompt.

Existing base/*.py modules call ``load_prompt("name.md")`` with no caller
argument. We keep that ergonomic by binding ``__file__`` here so the
underlying loader (``agentica.prompts.utils.load_prompt``) stays generic.
"""
from agentica.prompts.utils import load_prompt as _load_prompt


def load_prompt(filename: str, **kwargs) -> str:
    """Load a prompt from ``agentica/prompts/base/md/<filename>``."""
    return _load_prompt(__file__, filename, **kwargs)
