# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Modular prompt system for Agentica.

All non-trivial natural-language prompts live under this package so they
can be edited (markdown files in each ``<topic>/md/`` directory) and
composed without touching Python control flow.

Sub-packages:

- ``base/``         — Core agent system-prompt modules (heartbeat, soul,
                      tools, self-verification, goal).
- ``experience/``   — Skill upgrade lifecycle prompts.
- ``memory/``       — Memory taxonomy, extraction, and correction-judge
                      prompts (shared between ``BuiltinMemoryTool`` and
                      ``MemoryExtractHooks`` / ``ExperienceCaptureHooks``).
- ``compression/``  — Context compression prompts used by
                      ``CompressionManager``.
- ``swarm/``        — Coordinator and synthesizer prompts used by
                      ``Swarm``.
- ``critic/``       — Critic/revise loop templates used by ``critic.refine``.

Shared loader: ``agentica.prompts.utils.load_prompt``.
"""

from agentica.prompts.builder import PromptBuilder

__all__ = ["PromptBuilder"]
