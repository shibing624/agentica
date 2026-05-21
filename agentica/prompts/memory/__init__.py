# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Memory-related prompts.

Single source of truth for:
- Memory type taxonomy (user / project / reference) shared between the
  built-in `save_memory` tool and the boundary-triggered memory extraction
  hook (`MemoryExtractHooks`).
- The extraction sub-call prompt.
- The single-turn and batched correction-judge prompts used by
  `ExperienceCaptureHooks`.

All markdown lives under ``md/`` so prompts can be edited without touching
Python and so the same building blocks (type_spec, exclusion_spec) can be
composed into multiple final prompts.
"""
from agentica.prompts.utils import load_prompt as _load_prompt


def _load(filename: str, **kwargs) -> str:
    return _load_prompt(__file__, filename, **kwargs)


# Shared building blocks — composed into multiple final prompts.
MEMORY_TYPE_SPEC: str = _load("type_spec.md")
MEMORY_EXCLUSION_SPEC: str = _load("exclusion_spec.md")

# Final prompts.
# Trailing newline so callers can directly concatenate the conversation text.
MEMORY_EXTRACT_PROMPT: str = _load(
    "extract.md",
    type_spec=MEMORY_TYPE_SPEC,
    exclusion_spec=MEMORY_EXCLUSION_SPEC,
) + "\n"
MEMORY_SYSTEM_PROMPT: str = _load(
    "system.md",
    type_spec=MEMORY_TYPE_SPEC,
    exclusion_spec=MEMORY_EXCLUSION_SPEC,
)
FEEDBACK_CLASSIFY_PROMPT: str = _load("feedback_classify.md")
BATCH_JUDGE_PROMPT: str = _load("batch_judge.md") + "\n"


__all__ = [
    "MEMORY_TYPE_SPEC",
    "MEMORY_EXCLUSION_SPEC",
    "MEMORY_EXTRACT_PROMPT",
    "MEMORY_SYSTEM_PROMPT",
    "FEEDBACK_CLASSIFY_PROMPT",
    "BATCH_JUDGE_PROMPT",
]
