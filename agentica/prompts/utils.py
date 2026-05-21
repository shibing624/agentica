# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared utility for loading prompt markdown files from any
sub-package under ``agentica/prompts/``.

Each sub-package keeps its raw prompt text under a sibling ``md/`` directory;
the ``.py`` loader file calls ``load_prompt(__file__, "<name>.md")`` so the
markdown can be edited without touching Python and so multiple sub-packages
can share this single loader.
"""
from pathlib import Path


def load_prompt(caller_file: str, filename: str, **kwargs) -> str:
    """Load prompt content from the ``md/`` directory next to ``caller_file``.

    Args:
        caller_file: Pass ``__file__`` from the loader module. We resolve the
            sibling ``md/`` directory relative to it so each prompt subpackage
            (memory/, compression/, swarm/, ...) keeps its own md/ folder.
        filename: Name of the .md file (e.g. ``"extract.md"``).
        **kwargs: Optional ``str.format`` variables to substitute.

    Returns:
        Stripped file content, or empty string if the file is missing.
    """
    filepath = Path(caller_file).parent / "md" / filename
    if not filepath.exists():
        return ""
    content = filepath.read_text(encoding="utf-8").strip()
    if kwargs:
        content = content.format(**kwargs)
    return content
