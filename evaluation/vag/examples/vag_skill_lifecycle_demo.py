# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: VaG(Verifier-as-Gatekeeper) skill lifecycle demo: admission, promotion, repair, discard.

This demo uses a real LLM by default and real SkillEvolutionManager code
paths to show how generated skills move through the Verifier-as-Gatekeeper
lifecycle:

1. Admission gate decides whether a generated SKILL.md may be installed.
2. Promotion gate decides whether a shadow skill may become auto/active.
3. Repeated failures trigger LLM-style maintenance when explicitly enabled.
4. A repaired skill must validate and pass the repair gate.
5. A discarded skill is marked retired and removed from runtime loading.

Run:
    export ARK_API_KEY=...
    export VAG_DEMO_PROVIDER=ark
    export VAG_DEMO_MODEL=deepseek-v3.2
    python examples/self_evolution/02_vag_skill_lifecycle.py

Offline smoke mode:
    VAG_DEMO_USE_FAKE_LLM=1 python3 examples/self_evolution/02_vag_skill_lifecycle.py
"""
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from agentica import Agent, PROVIDER_FACTORIES
from agentica.agent.config import SkillUpgradeConfig
from agentica.critic import AgentCritic, CritiqueResult, CritiqueStyle, ExecCritic, SchemaCritic
from agentica.experience.skill_upgrade import SkillEvolutionManager
from evaluation.vag.lifecycle import (
    SkillCandidate,
    VaGLifecycleHooks,
    read_provenance_events,
)


DEMO_DIR = Path("./tmp/vag_skill_lifecycle_demo").resolve()
GENERATED_SKILLS_DIR = DEMO_DIR / "generated_skills"


GOOD_SKILL_MD = """---
name: check-path-before-read
description: Check paths before reading files
when-to-use: file operations, path validation
---

Check that a path exists before reading it.

## Gotchas
- Missing files trigger noisy retries; list the parent directory first.
- Similar filenames are easy to confuse; inspect the directory before choosing.

## Minimal Example
```python
from pathlib import Path
path = Path("data/input.txt")
if path.exists():
    content = path.read_text()
```
"""


REPAIRED_SKILL_MD = GOOD_SKILL_MD.replace(
    "Check that a path exists before reading it.",
    "Check that a path exists, list the parent directory, then read the exact file.",
)


class SemanticSkillCritic:
    name = "agent"

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        lowered = answer.lower()
        risky = ["rm -rf", "api token", "every task", "always ignore"]
        hits = [marker for marker in risky if marker in lowered]
        if hits:
            return CritiqueResult(
                approved=False,
                issues=f"semantic risk markers: {hits}",
                critic_name=self.name,
            )
        return CritiqueResult(approved=True, critic_name=self.name)


def no_destructive_command(_task: str, answer: str) -> CritiqueResult:
    risky = ["rm -rf", "git reset --hard", "chmod -R 777"]
    hits = [marker for marker in risky if marker in answer]
    if hits:
        return CritiqueResult(
            approved=False,
            issues=f"destructive command markers: {hits}",
            critic_name="exec",
        )
    return CritiqueResult(approved=True, critic_name="exec")


def build_gate_critics(real_model) -> List:
    critics = [
        SchemaCritic(SkillCandidate),
        ExecCritic(no_destructive_command, name="exec"),
    ]
    if real_model is None:
        critics.append(SemanticSkillCritic())
        return critics

    reviewer = Agent(
        name="vag-skill-reviewer",
        model=real_model,
        instructions=(
            "Review candidate SKILL.md files for VaG admission. Reject skills "
            "that are destructive, overgeneralized, contradictory, privacy-leaking, "
            "task-specific, or unsupported by the provided evidence. Reply APPROVED "
            "only when the skill is safe, reusable, and appropriately scoped."
        ),
    )
    critics.append(AgentCritic(reviewer, name="agent", style=CritiqueStyle.STRICT))
    return critics


def fake_spawn_model(skill_md: str = GOOD_SKILL_MD) -> MagicMock:
    model = MagicMock()
    model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
        "action": "install_shadow",
        "skill_name": "check-path-before-read",
        "source_experience": "path-check-card",
        "reason": "Repeated file-read failures came from guessed paths.",
        "skill_md": skill_md,
    })))
    return model


def fake_judge_model(decision: str) -> MagicMock:
    model = MagicMock()
    model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
        "decision": decision,
        "reason": f"demo judge says {decision}",
    })))
    return model


def fake_maintenance_model_repair() -> MagicMock:
    model = MagicMock()
    model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
        "decision": "repair",
        "reason": "Failure trace shows the parent directory was not listed first.",
        "revised_skill_md": REPAIRED_SKILL_MD,
    })))
    return model


def fake_maintenance_model_discard() -> MagicMock:
    model = MagicMock()
    model.response = AsyncMock(return_value=MagicMock(
        content="DISCARD: the underlying tool/API no longer exists",
    ))
    return model


def build_demo_model():
    """Build the real LLM used by this example.

    Examples should exercise real provider paths. Set
    ``VAG_DEMO_USE_FAKE_LLM=1`` only when you need an offline smoke run.
    """
    if os.environ.get("VAG_DEMO_USE_FAKE_LLM") == "1":
        return None

    load_dotenv()
    provider = os.environ.get("VAG_DEMO_PROVIDER", "ark")
    model_id = (
        os.environ.get("VAG_DEMO_MODEL")
        or os.environ.get("ARK_MODEL_NAME")
        or "deepseek-v3.2"
    )
    if provider == "ark" and not os.environ.get("ARK_API_KEY"):
        raise RuntimeError(
            "Set ARK_API_KEY in .env or environment, or run offline with "
            "VAG_DEMO_USE_FAKE_LLM=1."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY in .env or environment, or run offline with "
            "VAG_DEMO_USE_FAKE_LLM=1."
        )
    print(f"[model] provider={provider} id={model_id}")
    return PROVIDER_FACTORIES[provider](id=model_id)


def spawn_model(real_model):
    if real_model is not None:
        return real_model
    return fake_spawn_model()


def judge_model(real_model, decision: str):
    if real_model is not None:
        return real_model
    return fake_judge_model(decision)


def maintenance_repair_model(real_model):
    if real_model is not None:
        return real_model
    return fake_maintenance_model_repair()


def maintenance_discard_model(real_model):
    if real_model is not None:
        return real_model
    return fake_maintenance_model_discard()


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_provenance(skill_dir: Path) -> None:
    print("\nprovenance.jsonl:")
    for event in read_provenance_events(skill_dir):
        print(json.dumps(event, ensure_ascii=False, indent=2))


def reset_demo_dir() -> None:
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    GENERATED_SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def record_failures(manager: SkillEvolutionManager, skill_dir: Path, count: int = 2) -> None:
    for idx in range(count):
        manager.record_episode(
            skill_dir / "episodes.jsonl",
            outcome="failure",
            query=f"read the report by keyword, attempt {idx + 1}",
            tool_errors=1,
        )
    manager.update_meta_after_episode(skill_dir / "meta.json", "failure")
    for _ in range(count - 1):
        manager.update_meta_after_episode(skill_dir / "meta.json", "failure")


async def run_admission_and_promotion(
    manager: SkillEvolutionManager,
    critics: List,
    real_model,
) -> Path:
    print_section("1. Admission: generated skill must pass VaG before shadow install")
    admission_hooks = VaGLifecycleHooks(critics=critics, write_provenance=True)
    skill_name = await manager.maybe_spawn_skill(
        model=spawn_model(real_model),
        candidates=[{
            "title": "path-check-card",
            "content": (
                "Rule: check path before reading a file. "
                "Why: repeated failures came from guessed file paths. "
                "How to apply: list the parent directory, pick the exact "
                "filename, then read it."
            ),
            "repeat_count": 4,
            "type": "correction",
            "filename": "path_check.md",
        }],
        existing_skills=[],
        generated_skills_dir=GENERATED_SKILLS_DIR,
        hooks=admission_hooks,
    )
    if skill_name is None:
        raise RuntimeError(
            "The LLM did not install a shadow skill. Re-run with "
            "AGENTICA_LOG_LEVEL=debug or VAG_DEMO_USE_FAKE_LLM=1 for the "
            "deterministic offline smoke path."
        )
    skill_dir = GENERATED_SKILLS_DIR / str(skill_name)
    print(f"installed skill: {skill_name}")
    print(f"status after admission: {manager.read_meta(skill_dir / 'meta.json')['status']}")

    print_section("2. Promotion: shadow skill must pass gate before auto status")
    for idx in range(5):
        manager.record_episode(
            skill_dir / "episodes.jsonl",
            outcome="success",
            query=f"successful path-read episode {idx + 1}",
        )
    meta = manager.read_meta(skill_dir / "meta.json")
    meta.update({
        "total_episodes": 5,
        "success_count": 5,
        "failure_count": 0,
        "consecutive_failures": 0,
    })
    manager.write_meta(skill_dir / "meta.json", meta)

    promotion_hooks = VaGLifecycleHooks(critics=critics, write_provenance=True)
    decision = await manager.maybe_update_skill_state(
        model=judge_model(real_model, "promote"),
        skill_dir=skill_dir,
        checkpoint_interval=5,
        hooks=promotion_hooks,
    )
    print(f"promotion decision: {decision}")
    print(f"status after promotion: {manager.read_meta(skill_dir / 'meta.json')['status']}")
    print_provenance(skill_dir)
    return skill_dir


async def run_repair(
    manager: SkillEvolutionManager,
    critics: List,
    skill_dir: Path,
    real_model,
) -> None:
    print_section("3. Maintenance: repeated failures trigger repair when opt-in is enabled")
    record_failures(manager, skill_dir, count=2)

    repair_hooks = VaGLifecycleHooks(
        critics=critics,
        enable_maintenance=True,
        max_repair_attempts=3,
        write_provenance=True,
    )
    decision = await manager.maybe_update_skill_state(
        model=maintenance_repair_model(real_model),
        skill_dir=skill_dir,
        rollback_consecutive_failures=2,
        hooks=repair_hooks,
    )
    meta = manager.read_meta(skill_dir / "meta.json")
    print(f"maintenance decision: {decision}")
    print(f"status after repair: {meta['status']}")
    print(f"version after repair: {meta['version']}")
    print("skill file still present:", (skill_dir / "SKILL.md").exists())
    print_provenance(skill_dir)


async def run_discard(manager: SkillEvolutionManager, critics: List, real_model) -> None:
    print_section("4. Maintenance: LLM can discard an obsolete skill")
    obsolete_dir = GENERATED_SKILLS_DIR / "obsolete-tool-skill"
    obsolete_dir.mkdir(parents=True, exist_ok=True)
    (obsolete_dir / "SKILL.md").write_text(
        GOOD_SKILL_MD.replace(
            "name: check-path-before-read",
            "name: obsolete-tool-skill",
        ).replace(
            "Check that a path exists before reading it.",
            "Use the retired `legacy_read_file_v1` tool for all file reads.",
        ),
        encoding="utf-8",
    )
    manager.write_meta(obsolete_dir / "meta.json", {
        "skill_name": "obsolete-tool-skill",
        "status": "auto",
        "source_experience": "obsolete-tool-card",
        "version": 1,
        "total_episodes": 2,
        "success_count": 0,
        "failure_count": 2,
        "consecutive_failures": 2,
        "repair_attempts": 0,
    })
    record_failures(manager, obsolete_dir, count=2)

    discard_hooks = VaGLifecycleHooks(
        critics=critics,
        enable_maintenance=True,
        write_provenance=True,
    )
    decision = await manager.maybe_update_skill_state(
        model=maintenance_discard_model(real_model),
        skill_dir=obsolete_dir,
        rollback_consecutive_failures=2,
        hooks=discard_hooks,
    )
    meta = manager.read_meta(obsolete_dir / "meta.json")
    print(f"maintenance decision: {decision}")
    print(f"status after discard: {meta['status']}")
    print(f"SKILL.md exists after discard: {(obsolete_dir / 'SKILL.md').exists()}")
    print(f"SKILL.md.disabled exists: {(obsolete_dir / 'SKILL.md.disabled').exists()}")
    print_provenance(obsolete_dir)


async def main() -> None:
    reset_demo_dir()
    manager = SkillEvolutionManager()
    real_model = build_demo_model()
    critics = build_gate_critics(real_model)
    offline = real_model is None

    print_section("VaG Skill Lifecycle Demo")
    print(f"workspace: {DEMO_DIR}")
    print("SDK default lifecycle_hooks:", SkillUpgradeConfig().lifecycle_hooks)
    print("demo opts in to VaGLifecycleHooks for admission/promotion/repair/discard.")
    print("llm mode:", "offline fake responses" if offline else "real provider API")

    skill_dir = await run_admission_and_promotion(manager, critics, real_model)
    await run_repair(manager, critics, skill_dir, real_model)
    await run_discard(manager, critics, real_model)


if __name__ == "__main__":
    asyncio.run(main())
