# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill Tool - executes skills within the conversation.

This tool allows agents to invoke skills that provide specialized knowledge
and workflows for specific tasks.

Skills are modular packages that extend agent capabilities by providing
specialized knowledge, workflows, and tools. When a skill is invoked,
its instructions are loaded into the conversation context.

Usage:
    from agentica import Agent
    from agentica.tools.skill_tool import SkillTool
    from agentica.tools.shell_tool import ShellTool

    # Basic usage - skills loaded on-demand (not auto-loaded at startup)
    agent = Agent(
        name="Skill-Enabled Agent",
        tools=[SkillTool(), ShellTool()],
    )

    # Auto-load all skills from standard directories at startup
    skill_tool = SkillTool(auto_load=True)
    agent = Agent(
        name="Auto-Load Skill Agent",
        tools=[skill_tool, ShellTool()],
    )

    # With custom skill directories
    skill_tool = SkillTool(custom_skill_dirs=["./my-skills/web-research"])
    agent = Agent(
        name="Custom Skill Agent",
        tools=[skill_tool, ShellTool()],
    )
"""
import json
import time
from typing import List, Optional
from pathlib import Path

from agentica.config import AGENTICA_HOME
from agentica.tools.base import Tool
from agentica.skills import (
    Skill,
    SkillRegistry,
    get_skill_registry,
    load_skills,
    register_skill,
)
from agentica.skills.skill_loader import SkillLoader
from agentica.utils.log import logger


# When the registry holds more than this many skills, the system prompt
# only includes the FULL description for the top-N most-used / most-recent
# skills. The rest are listed as `name + trigger only`, and the agent
# pulls full details on demand via `get_skill_info(name)`.
_LAZY_THRESHOLD = 20

# Per-call score components for ranking skills in the system prompt.
# Recency wins over raw frequency to keep the visible set adaptive.
_USAGE_RECENCY_HALF_LIFE_DAYS = 14.0
_USAGE_RECENCY_WEIGHT = 0.7
_USAGE_FREQUENCY_WEIGHT = 0.3


def _skill_usage_state_path() -> Path:
    """Where skill usage counters are persisted (global, single JSON file).

    Lives directly under ~/.agentica/ alongside other single-file runtime
    caches (model_pricing_cache.json, agent_sessions.db, cli_history.txt).
    """
    return Path(AGENTICA_HOME).expanduser() / "skill_usage.json"


def _load_skill_usage() -> dict:
    path = _skill_usage_state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("skill usage state unreadable, resetting: %s", exc)
        return {}


def _save_skill_usage(state: dict) -> None:
    path = _skill_usage_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("failed to persist skill usage state: %s", exc)


def _bump_skill_usage(name: str) -> None:
    """Record one invocation of `name`. Best-effort; never raises."""
    state = _load_skill_usage()
    entry = state.get(name) or {"count": 0, "last_used": 0.0}
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_used"] = time.time()
    state[name] = entry
    _save_skill_usage(state)


def _score_skill_usage(usage: dict, name: str, now: float) -> float:
    """Composite ranking score: recency decay + log-frequency."""
    entry = usage.get(name)
    if not entry:
        return 0.0
    count = max(int(entry.get("count", 0)), 0)
    last_used = float(entry.get("last_used", 0.0))
    if last_used <= 0 or count <= 0:
        return 0.0
    age_days = max((now - last_used) / 86400.0, 0.0)
    recency = 0.5 ** (age_days / _USAGE_RECENCY_HALF_LIFE_DAYS)
    # Log so a single 100-call burst doesn't permanently freeze the top slot.
    import math
    frequency = math.log1p(count) / math.log1p(50)  # normalises around 50 calls
    frequency = min(frequency, 1.0)
    return _USAGE_RECENCY_WEIGHT * recency + _USAGE_FREQUENCY_WEIGHT * frequency


class SkillTool(Tool):
    """
    Tool for executing skills within the main conversation.

    Skills are modular packages that extend agent capabilities by providing
    specialized knowledge, workflows, and tools. When a skill is invoked,
    its instructions are loaded into the conversation context.

    Auto-loads skills from standard directories:
    - .claude/skills (project-level)
    - .agentica/skills (project-level)
    - ~/.claude/skills (user-level)
    - ~/.agentica/skills (user-level)

    Also supports custom skill directories via constructor.
    """

    _VISIBLE_GENERATED_STATUSES = {"shadow", "auto"}

    def __init__(
        self,
        custom_skill_dirs: Optional[List[str]] = None,
        auto_load: bool = False,
        name: str = "skill_tool",
    ):
        """
        Initialize the SkillTool.

        Args:
            custom_skill_dirs: Optional list of custom skill directory paths to load.
            auto_load: If True, automatically load skills from standard directories.
                       Default is False to avoid loading all skills on startup.
                       Set to True to auto-load all skills from standard directories.
            name: Name of the tool.
        """
        super().__init__(name=name)
        self._registry: Optional[SkillRegistry] = None
        self._custom_skill_dirs = custom_skill_dirs or []
        self._auto_load = auto_load
        self._initialized = False
        self._agent = None  # Set by Agent for runtime skill filtering
        self._init_name = name

        # Register tool functions
        self.register(self.list_skills)
        self.register(self.get_skill_info)

    def clone(self) -> "SkillTool":
        """Fresh instance so each agent owns its ``_agent`` slot.

        Reuses immutable config (custom_skill_dirs, auto_load, name). The
        cloned tool starts un-initialized and rebuilds its registry on demand.
        Preserves the source's exposed ``functions`` keys so a registry-side
        function filter survives Agent re-cloning.
        """
        from collections import OrderedDict
        new = SkillTool(
            custom_skill_dirs=list(self._custom_skill_dirs),
            auto_load=self._auto_load,
            name=self._init_name,
        )
        if set(new.functions) != set(self.functions):
            new.functions = OrderedDict(
                (name, new.functions[name])
                for name in self.functions
                if name in new.functions
            )
        return new

    def initialize(self) -> None:
        """Force skill registry to load now (instead of lazily on first use).

        Useful when the caller wants to inspect ``self.registry`` /
        ``list_skills()`` before the agent makes any tool call. Idempotent.
        """
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure skills are loaded before use."""
        if self._initialized:
            return

        # Auto-load from standard directories
        if self._auto_load:
            self._registry = load_skills()
        else:
            self._registry = get_skill_registry()

        self._load_custom_skill_dirs()

        self._initialized = True

    def _load_custom_skill_dirs(self) -> None:
        """Load custom skill directories during first initialization."""
        loader = SkillLoader()

        # Each entry can be either:
        # - A direct skill dir (contains SKILL.md) -> register it directly
        # - A parent dir with subdirectories (e.g., generated_skills/{slug}/SKILL.md)
        #   -> discover and register all visible generated sub-skills
        for skill_dir in self._custom_skill_dirs:
            skill_dir_path = Path(skill_dir).resolve()
            direct_md = skill_dir_path / "SKILL.md"
            if direct_md.exists():
                if (skill_dir_path / "meta.json").exists():
                    loaded = self._load_generated_skill(loader, direct_md)
                    if loaded and self._registry.register(loaded):
                        logger.info(f"Loaded generated skill: {loaded.name} from {direct_md}")
                else:
                    skill = register_skill(skill_dir)
                    if skill:
                        logger.info(f"Loaded custom skill: {skill.name} from {skill_dir}")
                continue

            if not skill_dir_path.is_dir():
                continue

            for skill_md_path in loader.discover_skills(skill_dir_path):
                loaded = self._load_generated_skill(loader, skill_md_path)
                if loaded and self._registry.register(loaded):
                    logger.info(f"Loaded generated skill: {loaded.name} from {skill_md_path}")

    @classmethod
    def _is_generated_skill_visible(cls, skill_md_path: Path) -> bool:
        """Check whether a generated skill should be visible at runtime."""
        meta_path = skill_md_path.parent / "meta.json"
        if not meta_path.exists():
            return True

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        status = meta.get("status")
        return status in cls._VISIBLE_GENERATED_STATUSES

    def _load_generated_skill(self, loader: SkillLoader, skill_md_path: Path) -> Optional[Skill]:
        """Load a generated skill only when its runtime status is visible."""
        if not self._is_generated_skill_visible(skill_md_path):
            return None
        return loader.load_skill(skill_md_path, "generated")

    def reload_generated_skills(self) -> int:
        """Re-scan custom skill dirs and sync generated skills.

        This performs a full sync for generated skills:
        - add newly visible skills
        - remove rolled back / draft skills
        - refresh revised skills in-place

        Returns:
            Number of visible generated skills after sync.
        """
        if self._registry is None:
            return 0

        loader = SkillLoader()
        generated_skills = {}
        for skill_dir in self._custom_skill_dirs:
            skill_dir_path = Path(skill_dir).resolve()
            if not skill_dir_path.is_dir():
                continue
            for skill_md_path in loader.discover_skills(skill_dir_path):
                loaded = self._load_generated_skill(loader, skill_md_path)
                if loaded is not None:
                    generated_skills[loaded.name] = loaded

        existing_generated_names = {
            skill.name for skill in self._registry.list_by_location("generated")
        }

        for skill_name in existing_generated_names:
            self._registry.remove(skill_name)

        for skill in generated_skills.values():
            self._registry.register(skill)

        return len(generated_skills)

    @property
    def registry(self) -> SkillRegistry:
        """Get the skill registry, loading skills if needed."""
        self._ensure_initialized()
        return self._registry

    def _get_enabled_skills(self) -> list:
        """Get list of enabled skills, respecting agent-level and query-level filtering."""
        all_skills = self.registry.list_all()
        if self._agent is None:
            return all_skills
        return [s for s in all_skills if self._agent._is_skill_enabled(s.name)]

    def _rank_skills_by_usage(self, skills: List[Skill]) -> List[Skill]:
        """Order skills by composite recency+frequency score, ties → name."""
        usage = _load_skill_usage()
        now = time.time()
        return sorted(
            skills,
            key=lambda s: (-_score_skill_usage(usage, s.name, now), s.name),
        )

    def list_skills(self) -> str:
        """
        List all available skills.

        Returns:
            Formatted string containing list of available skills with their descriptions
        """
        skills = self._get_enabled_skills()

        if not skills:
            return (
                "No skills available.\n\n"
                "Skills can be added to:\n"
                "- .claude/skills/ (project-level)\n"
                "- .agentica/skills/ (project-level)\n"
                "- ~/.claude/skills/ (user-level)\n"
                "- ~/.agentica/skills/ (user-level)"
            )

        ranked = self._rank_skills_by_usage(skills)
        result = f"Available Skills ({len(ranked)}):\n"
        result += "-" * 40 + "\n"
        for skill in ranked:
            result += f"- {skill.name}\n"
            result += f"  Description: {skill.description}\n"
            result += f"  Location: {skill.location}\n"
            result += f"  Path: {skill.path}\n\n"

        return result.strip()

    def get_skill_info(self, skill_name: str) -> str:
        """
        Get detailed information and full instructions for a specific skill.

        This loads the complete SKILL.md content for the requested skill.

        Args:
            skill_name: Name of the skill to get info for

        Returns:
            Full skill content including instructions, or error if not found
        """
        # Check if skill is enabled
        if self._agent is not None and not self._agent._is_skill_enabled(skill_name):
            raise ValueError(f"Skill '{skill_name}' is disabled.")

        skill_obj = self.registry.get(skill_name)

        if skill_obj is None:
            available = [s.name for s in self._get_enabled_skills()]
            raise ValueError(
                f"Skill '{skill_name}' not found.\n"
                f"Available skills: {', '.join(available[:50]) if available else 'None'}"
            )

        # Bump usage counter — this is how the system prompt decides which
        # skills earn a full description slot vs name-only.
        _bump_skill_usage(skill_obj.name)

        # Return full skill prompt with instructions
        result = f"=== Skill: {skill_obj.name} ===\n"
        result += f"Description: {skill_obj.description}\n"
        result += f"Location: {skill_obj.location}\n"
        result += f"Path: {skill_obj.path}\n"
        if skill_obj.license:
            result += f"License: {skill_obj.license}\n"
        if skill_obj.allowed_tools:
            result += f"Allowed Tools: {', '.join(skill_obj.allowed_tools)}\n"

        # Include full instructions from SKILL.md
        result += f"\n--- Instructions ---\n{skill_obj.content}\n"

        return result

    def add_skill_dir(self, skill_dir: str) -> Optional[Skill]:
        """
        Add a custom skill directory at runtime.

        Args:
            skill_dir: Path to the skill directory containing SKILL.md

        Returns:
            Skill instance if loaded successfully, None otherwise
        """
        self._ensure_initialized()
        skill = register_skill(skill_dir)
        if skill:
            logger.info(f"Added skill: {skill.name} from {skill_dir}")
        return skill

    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt for the skill tool.

        This prompt is injected into the agent's system message to guide
        the LLM on how to use skills effectively. Only includes enabled skills.

        Returns:
            System prompt string describing available skills
        """
        self._ensure_initialized()
        skills = self._get_enabled_skills()

        if not skills:
            return """# Skills

No skills are currently available.

If a matching skill is later installed, load it with `get_skill_info(skill_name)` before acting on the task.
"""

        ranked = self._rank_skills_by_usage(skills)
        total = len(ranked)

        # Below the threshold, every skill gets a full description.
        # At or above, only the top-N (by usage recency+frequency) keep their
        # description in the system prompt. The rest are listed by name +
        # optional trigger so the agent can `get_skill_info(name)` on demand.
        if total <= _LAZY_THRESHOLD:
            full_skills, lazy_skills = ranked, []
        else:
            full_skills = ranked[:_LAZY_THRESHOLD]
            lazy_skills = ranked[_LAZY_THRESHOLD:]

        full_lines = []
        for skill in full_skills:
            trigger_info = f" (trigger: `{skill.trigger}`)" if skill.trigger else ""
            full_lines.append(f"- **{skill.name}**{trigger_info}: {skill.description}")

        sections = [
            "# Skills",
            "",
            "Use a skill only when it clearly matches the current task.",
            "",
            f"## Top Skills ({len(full_skills)} of {total}, ranked by recent usage)",
            "\n".join(full_lines),
        ]

        if lazy_skills:
            lazy_lines = []
            for skill in lazy_skills:
                trigger_info = f" (`{skill.trigger}`)" if skill.trigger else ""
                lazy_lines.append(f"- {skill.name}{trigger_info}")
            sections.extend([
                "",
                f"## Other Skills ({len(lazy_skills)}, name only — call `get_skill_info(name)` to load full description before use)",
                "\n".join(lazy_lines),
            ])

        sections.extend([
            "",
            "## Skill Workflow",
            "- Load the matching skill with `get_skill_info(skill_name)` before giving task guidance.",
            "- Treat slash commands like `/<something>` as skill references and load the matching skill first.",
            "- Skills provide instructions, not executable actions.",
            "- Do not mention a skill without loading it.",
            "- Do not reload the same skill within the current turn.",
        ])

        return "\n".join(sections) + "\n"

    def __repr__(self) -> str:
        self._ensure_initialized()
        skill_count = len(self.registry)
        return f"<SkillTool skills={skill_count}>"
