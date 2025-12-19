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
import json
from typing import Any, Dict, List, Optional

from agentica.tools.base import Tool
from agentica.skills import (
    Skill,
    SkillRegistry,
    get_skill_registry,
    load_skills,
    register_skill,
    list_skill_files,
    read_skill_file,
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
        self.register(self.execute_skill)
        self.register(self.list_skills)
        self.register(self.get_skill_info)
        self.register(self.list_skill_files)
        self.register(self.read_skill_file)

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

    def execute_skill(self, skill_name: str) -> str:
        """
        Execute a skill by loading its instructions into the conversation.

        When you invoke a skill, its prompt will expand and provide detailed
        instructions on how to complete the task.

        Args:
            skill_name: Name of the skill to execute (e.g., 'web-research', 'pdf-reader')

        Returns:
            The skill prompt and instructions, or error message if skill not found
        """
        # Check if skill exists
        skill_obj = self.registry.get(skill_name)

        if skill_obj is None:
            available = [s.name for s in self.registry.list_all()]
            return (
                f"Error: Unknown skill '{skill_name}'.\n"
                f"Available skills: {', '.join(available[:10]) if available else 'None'}\n"
                f"Use list_skills() to see all available skills."
            )

        # Get the skill prompt
        prompt = skill_obj.get_prompt()

        # Build response with skill content
        result = (
            f"=== Skill Activated: {skill_obj.name} ===\n"
            f"Description: {skill_obj.description}\n"
            f"Location: {skill_obj.location}\n"
            f"Path: {skill_obj.path}\n"
        )
        if skill_obj.allowed_tools:
            result += f"Allowed Tools: {', '.join(skill_obj.allowed_tools)}\n"
        result += f"\n--- Skill Instructions ---\n{prompt}"

        return result

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
        Get detailed information about a specific skill.

        Args:
            skill_name: Name of the skill to get info for

        Returns:
            Formatted string containing skill details or error if not found
        """
        skill_obj = self.registry.get(skill_name)

        if skill_obj is None:
            available = [s.name for s in self.registry.list_all()]
            return (
                f"Error: Skill '{skill_name}' not found.\n"
                f"Available skills: {', '.join(available[:50]) if available else 'None'}"
            )

        result = f"=== Skill: {skill_obj.name} ===\n"
        result += f"Description: {skill_obj.description}\n"
        result += f"Location: {skill_obj.location}\n"
        result += f"Path: {skill_obj.path}\n"
        if skill_obj.license:
            result += f"License: {skill_obj.license}\n"
        if skill_obj.allowed_tools:
            result += f"Allowed Tools: {', '.join(skill_obj.allowed_tools)}\n"
        if skill_obj.metadata:
            result += f"Metadata: {json.dumps(skill_obj.metadata, indent=2)}\n"

        return result

    def list_skill_files(self, directory: str) -> str:
        """
        List all files in a skill directory.

        This tool is specifically for exploring skill directories that contain
        resources like scripts, references, and assets bundled with the skill.
        
        Note: When used with DeepAgent which has built-in ls(), prefer using ls()
        for general file operations. Use this only for skill-specific exploration.

        Args:
            directory: Path to the skill directory to list

        Returns:
            Formatted string containing the list of files
        """
        return list_skill_files(directory)

    def read_skill_file(self, file_path: str) -> str:
        """
        Read the contents of a file in a skill directory.

        This tool is specifically for reading skill resources like SKILL.md,
        scripts, or reference documents bundled with the skill.
        
        Note: When used with DeepAgent which has built-in read_file(), prefer using
        read_file() for general file operations. Use this only for skill-specific files.

        Args:
            file_path: Path to the file to read

        Returns:
            Contents of the file, or error message if cannot be read
        """
        return read_skill_file(file_path)

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
        the LLM on how to use skills effectively.

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

        skills_xml = "\n".join(skill.to_xml() for skill in skills)

        return f"""# Skills Tool

When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

## How to use skills:
- Use execute_skill(skill_name) to load a skill's instructions
- The skill's prompt will provide detailed instructions on how to complete the task
- Use list_skills() to see all available skills
- Use get_skill_info(skill_name) to get details about a specific skill
- Use list_skill_files(directory) to explore skill directory resources
- Use read_skill_file(file_path) to read skill files (SKILL.md, scripts, references)

## Important:
- Only use skills listed in <available_skills> below
- Read the skill's instructions carefully before proceeding

<available_skills>
{skills_xml}
</available_skills>
"""

    def __repr__(self) -> str:
        self._ensure_initialized()
        skill_count = len(self.registry)
        return f"<SkillTool skills={skill_count}>"
