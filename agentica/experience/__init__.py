# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Experience system for self-evolution.

Four-layer decomposition:
- ExperienceEventStore: append-only raw event persistence (events.jsonl)
- ExperienceCompiler: pure/stateless compiler (raw events/errors -> compiled cards)
- CompiledExperienceStore: compiled card CRUD, lifecycle, retrieval, sync
- SkillEvolutionManager: experience -> skill upgrade pipeline
"""
from agentica.experience.event_store import ExperienceEventStore
from agentica.experience.compiler import ExperienceCompiler
from agentica.experience.compiled_store import CompiledExperienceStore
from agentica.experience.skill_upgrade import SkillEvolutionManager
from agentica.experience.skill_lifecycle_hooks import (
    SkillLifecycleHooks,
    NoopSkillLifecycleHooks,
)

__all__ = [
    "ExperienceEventStore",
    "ExperienceCompiler",
    "CompiledExperienceStore",
    "SkillEvolutionManager",
    "SkillLifecycleHooks",
    "NoopSkillLifecycleHooks",
]
