# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill Registry - manages loaded skills and provides lookup functionality.
"""
import re
from typing import Dict, Optional, List

from agentica.skills.catalog import render_skills_catalog
from agentica.skills.skill import Skill


class SkillRegistry:
    """
    Registry for managing loaded skills.

    Skills are organized by name and can be looked up for execution.
    The registry also handles skill deduplication (project skills override user skills).

    Priority order: project > user > managed > generated > bundled
    """

    # Priority mapping for location types. "bundled" ships inside the package
    # and sits last on purpose: a user who writes their own skill of the same
    # name is correcting the one we shipped, and must win.
    LOCATION_PRIORITY = {
        "project": 0,
        "user": 1,
        "managed": 2,
        "generated": 3,
        "bundled": 4,
    }

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skills_by_location: Dict[str, List[Skill]] = {
            "project": [],
            "user": [],
            "managed": [],
            "generated": [],
            "bundled": [],
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

        Matches skills that are user_invocable=True and whose ``trigger``
        prefixes the text. Discovery for the model uses ``description``
        (standard Agent Skills); this helper is only for explicit ``/slug``
        style invocations.

        Args:
            text: User input text (e.g., "/commit fix bug")

        Returns:
            Matching Skill or None if no match found
        """
        text = text.strip()
        for skill in self._skills.values():
            if skill.user_invocable and skill.matches_trigger(text):
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

    def get_skills_summary(self, context_window: Optional[int] = None) -> str:
        """Render the skills catalog for session guidance (no SkillTool workflow).

        The live agent path is ``SkillTool.get_system_prompt()``; this is the
        CLI fallback when the agent has no SkillTool, and a listing helper
        for examples. Same budget and description cap as the live catalog.
        """
        return render_skills_catalog(
            self.list_all(),
            context_window=context_window,
            include_workflow=False,
        )

    def auto_commands(self) -> Dict[str, "Skill"]:
        """Build a mapping of auto-generated slash commands to skills.

        For every user-invocable, non-hidden skill:
        - If it has an explicit ``trigger`` like ``/commit``, use that.
        - Otherwise, generate ``/slug`` from the skill name
          (e.g. "My Cool Skill" -> ``/my-cool-skill``).

        Keys are lowercased because every lookup site (``expand_invocation``,
        the CLI completer and its command dispatch) lowercases the user's
        input — a ``/MySkill`` trigger would otherwise be unreachable.

        Returns:
            Dict mapping ``/slug`` -> Skill
        """
        cmds: Dict[str, "Skill"] = {}
        for skill in self._skills.values():
            if not skill.user_invocable or skill.is_hidden:
                continue
            if skill.trigger:
                slug = skill.trigger if skill.trigger.startswith("/") else f"/{skill.trigger}"
                slug = slug.lower()
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
