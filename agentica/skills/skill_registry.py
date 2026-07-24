# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill Registry - manages loaded skills and provides lookup functionality.
"""
import re
from typing import Dict, Optional, List

from agentica.skills.skill import Skill


class SkillRegistry:
    """
    Registry for managing loaded skills.

    Skills are organized by name and can be looked up for execution.
    The registry also handles skill deduplication (project skills override user skills).

    Priority order: project > user > managed
    """

    # Priority mapping for location types
    LOCATION_PRIORITY = {
        "project": 0,
        "user": 1,
        "managed": 2,
        "generated": 3,
    }

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skills_by_location: Dict[str, List[Skill]] = {
            "project": [],
            "user": [],
            "managed": [],
            "generated": [],
        }

    def register(self, skill: Skill) -> bool:
        """
        Register a skill in the registry.

        Project skills take precedence over user skills.
        If a skill with the same name already exists from a higher priority location,
        the new skill is not registered.

        Args:
            skill: Skill instance to register

        Returns:
            True if the skill was registered, False if it was skipped
        """
        existing = self._skills.get(skill.name)

        if existing:
            existing_priority = self.LOCATION_PRIORITY.get(existing.location, 99)
            new_priority = self.LOCATION_PRIORITY.get(skill.location, 99)

            if new_priority >= existing_priority:
                # Skip - existing skill has higher or equal priority
                return False

        self._skills[skill.name] = skill
        if skill.location in self._skills_by_location:
            self._skills_by_location[skill.location].append(skill)
        return True

    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: Name of the skill

        Returns:
            Skill instance or None if not found
        """
        return self._skills.get(name)

    def exists(self, name: str) -> bool:
        """
        Check if a skill exists in the registry.

        Args:
            name: Name of the skill

        Returns:
            True if the skill exists
        """
        return name in self._skills

    def list_all(self) -> List[Skill]:
        """
        Get all registered skills.

        Returns:
            List of all skills
        """
        return list(self._skills.values())

    def list_by_location(self, location: str) -> List[Skill]:
        """
        Get skills by location type.

        Args:
            location: Location type (project, user, managed)

        Returns:
            List of skills from that location
        """
        return self._skills_by_location.get(location, [])

    def match_trigger(self, text: str) -> Optional[Skill]:
        """
        Find a skill that matches the given trigger text.

        First tries exact trigger prefix match, then falls back to
        keyword matching against when_to_use field.

        Only matches skills that are user_invocable=True.

        Args:
            text: User input text (e.g., "/commit fix bug")

        Returns:
            Matching Skill or None if no match found
        """
        text = text.strip()
        # 1. Exact trigger prefix match
        for skill in self._skills.values():
            if skill.user_invocable and skill.matches_trigger(text):
                return skill
        # 2. Keyword match from when_to_use
        for skill in self._skills.values():
            if skill.user_invocable and skill.matches_keywords(text):
                return skill
        return None

    def get_skill_by_trigger(self, trigger: str) -> Optional[Skill]:
        """
        Get a skill by its trigger command.

        Args:
            trigger: Trigger command (e.g., "/commit")

        Returns:
            Skill with matching trigger or None
        """
        for skill in self._skills.values():
            if skill.trigger == trigger:
                return skill
        return None

    def list_triggers(self) -> Dict[str, str]:
        """
        Get all registered trigger commands visible to the user.

        Skips hidden and non-user-invocable skills.

        Returns:
            Dict mapping trigger to skill name
        """
        triggers = {}
        for skill in self._skills.values():
            if skill.trigger and skill.user_invocable and not skill.is_hidden:
                triggers[skill.trigger] = skill.name
        return triggers

    def remove(self, name: str) -> bool:
        """
        Remove a skill from the registry.

        Args:
            name: Name of the skill to remove

        Returns:
            True if removed, False if not found
        """
        skill = self._skills.pop(name, None)
        if skill:
            location_list = self._skills_by_location.get(skill.location, [])
            if skill in location_list:
                location_list.remove(skill)
            return True
        return False

    def clear(self):
        """Clear all registered skills."""
        self._skills.clear()
        for location in self._skills_by_location:
            self._skills_by_location[location].clear()

    def generate_skills_prompt(self, char_budget: int = 10000) -> str:
        """
        Generate a prompt listing all available skills.

        Limits output to character budget to manage context window.

        Args:
            char_budget: Maximum characters for skills list

        Returns:
            Formatted skills prompt
        """
        skills = self.list_all()

        if not skills:
            return ""

        entries = []
        total_chars = 0

        for skill in skills:
            entry = skill.to_xml()
            if total_chars + len(entry) > char_budget:
                break
            entries.append(entry)
            total_chars += len(entry)

        if not entries:
            return ""

        skills_xml = "\n".join(entries)
        return f"""<available_skills>
{skills_xml}
</available_skills>"""

    def get_skill_instruction(self) -> str:
        """
        Get the instruction prompt for all registered skills.

        Returns:
            Formatted instruction string for skills
        """
        if not self._skills:
            return ""

        instruction = (
            "# Agent Skills\n"
            "The agent skills are a collection of folders of instructions, scripts, "
            "and resources that you can load dynamically to improve performance "
            "on specialized tasks. Each agent skill has a `SKILL.md` file in its "
            "folder that describes how to use the skill. If you want to use a "
            "skill, you MUST read its `SKILL.md` file carefully.\n\n"
        )

        skill_template = (
            "## {name}{trigger_info}\n"
            "{description}\n"
            'Check "{dir}/SKILL.md" for how to use this skill\n\n'
        )

        for skill in self._skills.values():
            trigger_info = f" (`{skill.trigger}`)" if skill.trigger else ""
            instruction += skill_template.format(
                name=skill.name,
                description=skill.description,
                dir=skill.path,
                trigger_info=trigger_info,
            )

        return instruction

    def get_skills_summary(self) -> str:
        """
        Get a brief summary of all available skills.

        Useful for including in system prompts without full details.

        Returns:
            Formatted summary string
        """
        if not self._skills:
            return ""

        lines = ["## Available Skills\n"]
        for name, skill in self._skills.items():
            trigger = f" (`{skill.trigger}`)" if skill.trigger else ""
            lines.append(f"- **{name}**{trigger}: {skill.description}")

        return "\n".join(lines)

    def auto_commands(self) -> Dict[str, "Skill"]:
        """Build a mapping of auto-generated slash commands to skills.

        For every user-invocable, non-hidden skill:
        - If it has an explicit ``trigger`` like ``/commit``, use that.
        - Otherwise, generate ``/slug`` from the skill name
          (e.g. "My Cool Skill" -> ``/my-cool-skill``).

        Returns:
            Dict mapping ``/slug`` -> Skill
        """
        cmds: Dict[str, "Skill"] = {}
        for skill in self._skills.values():
            if not skill.user_invocable or skill.is_hidden:
                continue
            if skill.trigger:
                slug = skill.trigger if skill.trigger.startswith("/") else f"/{skill.trigger}"
            else:
                slug = skill.name.lower().replace(" ", "-").replace("_", "-")
                slug = re.sub(r"[^a-z0-9\-]", "", slug)
                slug = re.sub(r"-+", "-", slug).strip("-")
                slug = f"/{slug}"
            if slug and slug != "/":
                cmds[slug] = skill
        return cmds

    def expand_invocation(self, text: str) -> Optional[str]:
        """Expand a ``/trigger [arguments]`` line into the skill's full prompt.

        Returns ``None`` when the text is not a skill invocation, so callers can
        pass the original message through untouched. Shared by the CLI, the SDK
        and the gateway so every surface frames the arguments identically.
        """
        if not text:
            return None
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        skill = self.auto_commands().get(parts[0].lower())
        if skill is None:
            return None
        return skill.render_invocation(parts[1] if len(parts) > 1 else "")

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={list(self._skills.keys())})"


# Global skill registry instance
_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """
    Get the global skill registry instance.

    Creates the registry if it doesn't exist.

    Returns:
        SkillRegistry instance
    """
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry


def reset_skill_registry():
    """Reset the global skill registry."""
    global _skill_registry
    _skill_registry = None
