# -*- coding: utf-8 -*-
"""VaG (Verifier-as-Gatekeeper) lifecycle extension for SkillEvolutionManager.

This package is paper / research grade and lives outside the SDK proper. It
plugs into ``agentica.experience.skill_upgrade.SkillEvolutionManager`` via
``SkillLifecycleHooks``.

Public surface:
    - ``SkillCandidate``      — Pydantic schema for a generated SKILL.md
    - ``SkillAdmissionGate``  — conjunctive multi-critic gate
    - ``SkillGateResult``     — gate verdict aggregate
    - ``GateVerdict``         — per-critic verdict
    - ``skill_fingerprint``   — deterministic SHA-256 of a generated SKILL.md
    - ``append_provenance_event`` / ``read_provenance_events`` — JSONL log
    - ``VaGLifecycleHooks``   — drop-in ``SkillLifecycleHooks`` implementation
                                that wires gates + provenance + repair-or-discard
                                onto the SDK lifecycle.
"""
from evaluation.vag.lifecycle.candidate import SkillCandidate
from evaluation.vag.lifecycle.gate import (
    GateVerdict,
    SkillAdmissionGate,
    SkillGateResult,
    skill_fingerprint,
)
from evaluation.vag.lifecycle.hooks import VaGLifecycleHooks
from evaluation.vag.lifecycle.maintenance import repair_or_discard
from evaluation.vag.lifecycle.provenance import (
    PROVENANCE_FILENAME,
    append_provenance_event,
    get_provenance_path,
    read_provenance_events,
)

__all__ = [
    "SkillCandidate",
    "SkillAdmissionGate",
    "SkillGateResult",
    "GateVerdict",
    "skill_fingerprint",
    "VaGLifecycleHooks",
    "repair_or_discard",
    "PROVENANCE_FILENAME",
    "append_provenance_event",
    "get_provenance_path",
    "read_provenance_events",
]
