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

    # Basic usage - auto-loads skills from standard directories
    agent = Agent(
        name="Skill-Enabled Agent",
        tools=[SkillTool(), ShellTool()],
    )

    # With custom skill directories
    skill_tool = SkillTool(custom_skill_dirs=["./my-skills/web-research"])
    agent = Agent(
        name="Custom Skill Agent",
        tools=[skill_tool, ShellTool()],
    )
"""
from typing import List, Optional

from agentica.tools.base import Tool
from agentica.skills import (
    Skill,
    SkillRegistry,
    get_skill_registry,
    load_skills,
    register_skill,
)
from agentica.utils.log import logger


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

    def __init__(
        self,
        custom_skill_dirs: Optional[List[str]] = None,
        auto_load: bool = True,
        name: str = "skill_tool",
    ):
        """
        Initialize the SkillTool.

        Args:
            custom_skill_dirs: Optional list of custom skill directory paths to load.
            auto_load: If True, automatically load skills from standard directories.
            name: Name of the tool.
        """
        super().__init__(name=name)
        self._registry: Optional[SkillRegistry] = None
        self._custom_skill_dirs = custom_skill_dirs or []
        self._auto_load = auto_load
        self._initialized = False

        # Register tool functions
        # Note: get_skill_prompt is removed as a tool function since skill prompts
        # are now injected into the system prompt via get_system_prompt()
        self.register(self.list_skills)
        self.register(self.get_skill_info)

    def _ensure_initialized(self):
        """Ensure skills are loaded before use."""
        if self._initialized:
            return

        # Auto-load from standard directories
        if self._auto_load:
            self._registry = load_skills()
        else:
            self._registry = get_skill_registry()

        # Load custom skill directories
        for skill_dir in self._custom_skill_dirs:
            skill = register_skill(skill_dir)
            if skill:
                logger.info(f"Loaded custom skill: {skill.name} from {skill_dir}")

        self._initialized = True

    @property
    def registry(self) -> SkillRegistry:
        """Get the skill registry, loading skills if needed."""
        self._ensure_initialized()
        return self._registry

    def list_skills(self) -> str:
        """
        List all available skills.

        Returns:
            Formatted string containing list of available skills with their descriptions
        """
        skills = self.registry.list_all()

        if not skills:
            return (
                "No skills available.\n\n"
                "Skills can be added to:\n"
                "- .claude/skills/ (project-level)\n"
                "- .agentica/skills/ (project-level)\n"
                "- ~/.claude/skills/ (user-level)\n"
                "- ~/.agentica/skills/ (user-level)"
            )

        result = f"Available Skills ({len(skills)}):\n"
        result += "-" * 40 + "\n"
        for skill in skills:
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
        skill_obj = self.registry.get(skill_name)

        if skill_obj is None:
            available = [s.name for s in self.registry.list_all()]
            return (
                f"Error: Skill '{skill_name}' not found.\n"
                f"Available skills: {', '.join(available[:50]) if available else 'None'}"
            )

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
        the LLM on how to use skills effectively. Only includes skill name
        and description - full instructions are loaded on demand.

        Returns:
            System prompt string describing available skills
        """
        self._ensure_initialized()
        skills = self.registry.list_all()

        if not skills:
            return """# Skills Tool

No skills are currently available. Skills can be added to:
- .claude/skills/ (project-level)
- .agentica/skills/ (project-level)
- ~/.claude/skills/ (user-level)
- ~/.agentica/skills/ (user-level)

Each skill directory should contain a SKILL.md file with:
- YAML frontmatter (name, description)
- Detailed usage instructions
"""

        # Build skill summary list (name + description only)
        skill_list = []
        for skill in skills:
            trigger_info = f" (trigger: `{skill.trigger}`)" if skill.trigger else ""
            skill_list.append(f"- **{skill.name}**{trigger_info}: {skill.description}")

        skills_summary = "\n".join(skill_list)

        return f"""# Skills

Skills provide specialized knowledge and workflows for specific tasks.

## Available Skills ({len(skills)}):
{skills_summary}

## How to Use Skills:
1. Use `get_skill_info(skill_name)` to get detailed instructions for a specific skill
2. Follow the skill's instructions to complete the task
3. Skills are NOT callable tools - they provide guidance on HOW to do tasks
"""

    def __repr__(self) -> str:
        self._ensure_initialized()
        skill_count = len(self.registry)
        return f"<SkillTool skills={skill_count}>"
