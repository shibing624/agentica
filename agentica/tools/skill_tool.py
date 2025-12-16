# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Skill Tool - A prompt-based skill system for agents.

Agent Skill is an approach proposed by Anthropic to improve agent capabilities on specific tasks.
Skills are not code-level extensions, but text instructions injected into the system prompt,
allowing the LLM to read and follow the instructions to complete tasks.

Reference: https://claude.com/blog/skills
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional, List

from agentica.tools.base import Tool
from agentica.utils.log import logger


@dataclass
class AgentSkill:
    """Data class representing an agent skill."""
    name: str
    description: str
    dir: str


class SkillTool(Tool):
    """
    Agent Skill Tool for managing and registering agent skills.

    Skills are prompt-based instructions that extend agent capabilities.
    Each skill is defined by a SKILL.md file containing YAML frontmatter
    with metadata (name, description) and detailed usage instructions.

    SkillTool inherits from Tool and includes built-in file reading capabilities,
    so it can be passed directly to Agent's tools list.

    Example SKILL.md format:
    ```markdown
    ---
    name: My Skill
    description: A skill for doing something useful.
    ---

    # My Skill

    ## Overview
    This skill helps you do something useful...

    ## Usage
    1. First, do this...
    2. Then, do that...
    ```

    Usage:
        skill_tool = SkillTool()
        skill_tool.register_skill("path/to/skill_directory")

        agent = Agent(
            instructions=[..., skill_tool.get_skill_prompt()],
            tools=[skill_tool, ShellTool(), RunPythonCodeTool()],
        )
    """

    # Default instruction template for skills
    DEFAULT_SKILL_INSTRUCTION = (
        "# Agent Skills\n"
        "The agent skills are a collection of folders of instructions, scripts, "
        "and resources that you can load dynamically to improve performance "
        "on specialized tasks. Each agent skill has a `SKILL.md` file in its "
        "folder that describes how to use the skill. If you want to use a "
        "skill, you MUST read its `SKILL.md` file carefully.\n\n"
    )

    # Default template for formatting each skill
    DEFAULT_SKILL_TEMPLATE = (
        "## {name}\n"
        "{description}\n"
        'Check "{dir}/SKILL.md" for how to use this skill\n'
    )

    def __init__(
        self,
        skill_instruction: Optional[str] = None,
        skill_template: Optional[str] = None,
    ):
        """
        Initialize the SkillTool.

        Args:
            skill_instruction: Custom instruction text that introduces skills to the agent.
                              Must explain what skills are and how to use them.
            skill_template: Custom template for formatting each skill's description.
                           Must contain {name}, {description}, and {dir} placeholders.
        """
        super().__init__(name="skill_tool")
        self.skills: Dict[str, AgentSkill] = {}
        self._skill_instruction = skill_instruction or self.DEFAULT_SKILL_INSTRUCTION
        self._skill_template = skill_template or self.DEFAULT_SKILL_TEMPLATE

        # Register functions for LLM to call
        self.register(self.list_skills)
        self.register(self.read_file)
        self.register(self.list_files)

    def register_skill(self, skill_dir: str) -> None:
        """
        Register an agent skill from a given directory.

        This function scans the directory, reads metadata from the SKILL.md file,
        and adds it to the skill registry. The SKILL.md file must contain YAML
        frontmatter with 'name' and 'description' fields.

        Args:
            skill_dir: Path to the skill directory containing SKILL.md.

        Raises:
            ValueError: If the directory doesn't exist or SKILL.md is missing/invalid.
            ImportError: If python-frontmatter package is not installed.

        Example:
            >>> skill_tool = SkillTool()
            >>> skill_tool.register_skill("./skills/web-research")
        """
        import frontmatter

        # Convert to absolute path
        skill_dir = os.path.abspath(skill_dir)

        # Validate skill directory exists
        if not os.path.isdir(skill_dir):
            raise ValueError(
                f"Skill directory '{skill_dir}' does not exist. "
                "Please provide a valid directory path."
            )

        # Check for SKILL.md file
        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md_path):
            raise ValueError(
                f"SKILL.md not found in '{skill_dir}'. "
                "Each skill directory must contain a SKILL.md file with YAML frontmatter."
            )

        # Parse SKILL.md frontmatter
        with open(skill_md_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        name = post.get("name")
        description = post.get("description")

        # Validate required fields
        if not name:
            raise ValueError(
                f"SKILL.md in '{skill_dir}' is missing the 'name' field in YAML frontmatter. "
                "Example format:\n---\nname: My Skill\ndescription: Skill description\n---"
            )

        if not description:
            raise ValueError(
                f"SKILL.md in '{skill_dir}' is missing the 'description' field in YAML frontmatter."
            )

        # Register the skill
        self.skills[name] = AgentSkill(
            name=name,
            description=description,
            dir=skill_dir,
        )
        logger.info(f"Registered skill: {name} from {skill_dir}")

    def register_skills(self, skill_dirs: List[str]) -> None:
        """
        Register multiple skills from a list of directories.

        Args:
            skill_dirs: List of paths to skill directories.

        Example:
            >>> skill_tool = SkillTool()
            >>> skill_tool.register_skills([
            ...     "./skills/web-research",
            ...     "./skills/code-analysis",
            ... ])
        """
        for skill_dir in skill_dirs:
            self.register_skill(skill_dir)

    def remove_skill(self, name: str) -> bool:
        """
        Remove a registered skill by name.

        Args:
            name: The name of the skill to remove.

        Returns:
            True if the skill was removed, False if it wasn't found.
        """
        if name in self.skills:
            del self.skills[name]
            logger.debug(f"Removed skill: {name}")
            return True
        return False

    def get_skill(self, name: str) -> Optional[AgentSkill]:
        """
        Get a registered skill by name.

        Args:
            name: The name of the skill to retrieve.

        Returns:
            The AgentSkill object if found, None otherwise.
        """
        return self.skills.get(name)

    def list_skills(self) -> str:
        """List all registered agent skills with their names, descriptions, and directories.

        Returns:
            str: A formatted string containing all registered skills information.
        """
        if not self.skills:
            return "No skills registered."

        result = ["Available Skills:"]
        for skill in self.skills.values():
            result.append(f"\n- {skill.name}")
            result.append(f"  Description: {skill.description}")
            result.append(f"  Directory: {skill.dir}")
            result.append(f"  SKILL.md: {skill.dir}/SKILL.md")
        return "\n".join(result)

    def read_file(self, file_path: str) -> str:
        """Read the contents of a file.

        Args:
            file_path: The path to the file to read.

        Returns:
            str: The contents of the file, or an error message if the file cannot be read.
        """
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.isfile(file_path):
                return f"Error: File '{file_path}' does not exist."

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Read file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"

    def list_files(self, directory: str) -> str:
        """List all files in a directory.

        Args:
            directory: The path to the directory to list.

        Returns:
            str: A formatted string containing the list of files in the directory.
        """
        try:
            directory = os.path.abspath(directory)
            if not os.path.isdir(directory):
                return f"Error: Directory '{directory}' does not exist."

            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(f"  [FILE] {item}")
                elif os.path.isdir(item_path):
                    files.append(f"  [DIR]  {item}/")

            if not files:
                return f"Directory '{directory}' is empty."

            logger.info(f"Listed files in: {directory}")
            return f"Contents of '{directory}':\n" + "\n".join(sorted(files))
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")
            return f"Error listing directory: {e}"

    def get_skill_prompt(self) -> Optional[str]:
        """
        Get the prompt text for all registered agent skills.

        This prompt should be added to the agent's instructions to make
        the skills available to the agent.

        Returns:
            A formatted string containing skill descriptions, or None if no skills registered.

        Example:
            >>> skill_tool = SkillTool()
            >>> skill_tool.register_skill("./skills/web-research")
            >>> agent = Agent(
            ...     instructions=[..., skill_tool.get_skill_prompt()],
            ...     tools=[skill_tool],
            ... )
        """
        if len(self.skills) == 0:
            return None

        skill_descriptions = [self._skill_instruction]

        for skill in self.skills.values():
            skill_descriptions.append(
                self._skill_template.format(
                    name=skill.name,
                    description=skill.description,
                    dir=skill.dir,
                )
            )

        return "\n".join(skill_descriptions)

    def __repr__(self) -> str:
        return f"<SkillTool skills={list(self.skills.keys())}>"

    def __str__(self) -> str:
        return self.__repr__()


if __name__ == "__main__":
    # Example usage
    import tempfile

    # Create a temporary skill directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "sample_skill")
        os.makedirs(skill_dir, exist_ok=True)

        # Create SKILL.md
        skill_md_content = """---
name: Sample Skill
description: A sample agent skill for demonstration purposes.
---

# Sample Skill

## Overview
This is a sample skill that demonstrates the skill system.

## Usage
1. Read this file to understand the skill
2. Follow the instructions below
3. Complete the task

## Instructions
- Do something useful
- Return the result
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(skill_md_content)

        # Test the SkillTool
        skill_tool = SkillTool()
        skill_tool.register_skill(skill_dir)

        print("Registered skills:", skill_tool.list_skills())
        print("\nSkill prompt:")
        print(skill_tool.get_skill_prompt())
