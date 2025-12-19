# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill Registry - manages loaded skills and provides lookup functionality.
"""
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
    }

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skills_by_location: Dict[str, List[Skill]] = {
            "project": [],
            "user": [],
            "managed": [],
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
            "## {name}\n"
            "{description}\n"
            'Check "{dir}/SKILL.md" for how to use this skill\n\n'
        )

        for skill in self._skills.values():
            instruction += skill_template.format(
                name=skill.name,
                description=skill.description,
                dir=skill.path,
            )

        return instruction

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
