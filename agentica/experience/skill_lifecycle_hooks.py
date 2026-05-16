# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Extension hooks for the experience -> skill upgrade lifecycle.

The SDK runs a minimal, opinionated skill lifecycle:

    spawn (judge candidate) -> install shadow
    runtime episodes -> checkpoint judge -> promote / rollback / revise

Anything beyond that — multi-critic admission gates, append-only provenance
audit logs, LLM-driven repair-or-discard maintenance, fingerprinting, etc.
— is research / paper-grade extension and lives outside the SDK
(see ``evaluation/vag/lifecycle/``).

External code injects those extensions through ``SkillLifecycleHooks``,
passed to ``SkillUpgradeConfig.lifecycle_hooks``. The default
``NoopSkillLifecycleHooks`` keeps the lifecycle pure-deterministic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class SkillLifecycleHooks(Protocol):
    """Optional extension points called by ``SkillEvolutionManager``.

    Returning ``False`` from a gate aborts the corresponding transition
    (the skill stays in its previous state). Returning ``None`` from
    ``on_failure_threshold`` falls back to the SDK's default deterministic
    rollback behaviour.

    Implementations may write side-channel logs (e.g. provenance JSONL)
    inside any hook; the SDK does not care.
    """

    async def gate_admission(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        """Run before a generated SKILL.md is installed as a shadow skill."""
        ...

    async def gate_promotion(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        """Run before a shadow skill is promoted to ``auto`` status."""
        ...

    async def on_failure_threshold(
        self,
        skill_dir: Path,
        meta: Dict[str, Any],
        model: Any,
    ) -> Optional[str]:
        """Run when ``consecutive_failures`` crosses the rollback threshold.

        Return one of:
          - ``"repair"`` — the hook rewrote SKILL.md and updated meta.json.
            The SDK keeps the skill in its current status.
          - ``"discard"`` — the hook retired the skill (status change + disable).
          - ``None`` — fall back to the SDK's default deterministic rollback.
        """
        ...


class NoopSkillLifecycleHooks:
    """Default hooks: approve everything, never intervene on failures."""

    async def gate_admission(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        return True

    async def gate_promotion(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        return True

    async def on_failure_threshold(
        self,
        skill_dir: Path,
        meta: Dict[str, Any],
        model: Any,
    ) -> Optional[str]:
        return None


__all__ = ["SkillLifecycleHooks", "NoopSkillLifecycleHooks"]
