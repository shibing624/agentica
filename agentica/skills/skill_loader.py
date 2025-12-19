# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill Loader - discovers and loads skills from standard directories.

Skill directories can contain:
- SKILL.md: Required skill definition file with YAML frontmatter
- Additional files: scripts, resources, references, assets, etc.

The SkillLoader provides file operations (list_files, read_file) to allow
agents to explore and use skill resources.
"""
import os
from pathlib import Path
from typing import List, Optional

from agentica.skills.skill import Skill
from agentica.skills.skill_registry import SkillRegistry, get_skill_registry
from agentica.utils.string import truncate_if_too_long
from agentica.config import AGENTICA_SKILL_DIR
from agentica.utils.log import logger


class SkillLoader:
    """
    Discovers and loads skills from standard directories.

    Search paths (in priority order):
    1. .claude/skills (project-level)
    2. .agentica/skills (project-level)
    3. ~/.claude/skills (user-level)
    4. ~/.agentica/skills (user-level)

    Project-level skills override user-level skills with the same name.

    Skill directories can contain additional resources:
    - scripts/: Python or shell scripts
    - references/: Reference documents
    - assets/: Images, data files, etc.

    Use list_files() and read_file() to explore skill contents.
    """

    # Default skill directory names
    SKILL_DIRS = [
        ".claude/skills",
        ".agentica/skills",
    ]

    SKILL_FILE = "SKILL.md"

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the skill loader.

        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.home_dir = Path.home()

    def get_search_paths(self) -> List[tuple]:
        """
        Get all skill search paths with their location type.

        Returns:
            List of (path, location_type) tuples
        """
        paths = []

        # Project-level paths (higher priority)
        for skill_dir in self.SKILL_DIRS:
            project_path = self.project_root / skill_dir
            paths.append((project_path, "project"))

        # add AGENTICA_SKILL_DIR from config
        if AGENTICA_SKILL_DIR:
            skill_dir_path = Path(AGENTICA_SKILL_DIR)
            # skill_dir_path is dir, mkdir if not exists
            skill_dir_path.mkdir(parents=True, exist_ok=True)
            paths.append((skill_dir_path, "user"))

        # User-level paths (lower priority)
        for skill_dir in self.SKILL_DIRS:
            user_path = self.home_dir / skill_dir
            paths.append((user_path, "user"))
        
        return paths

    def discover_skills(self, skills_dir: Path) -> List[Path]:
        """
        Discover all skill directories within a skills directory.

        A skill directory must contain a SKILL.md file.

        Args:
            skills_dir: Directory containing skill subdirectories

        Returns:
            List of paths to SKILL.md files
        """
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []

        skill_files = []

        for item in skills_dir.iterdir():
            if item.is_dir():
                skill_md = item / self.SKILL_FILE
                if skill_md.exists():
                    skill_files.append(skill_md)
                else:
                    # Check for nested skill directories (e.g., document-skills/pdf)
                    for nested_item in item.iterdir():
                        if nested_item.is_dir():
                            nested_skill_md = nested_item / self.SKILL_FILE
                            if nested_skill_md.exists():
                                skill_files.append(nested_skill_md)

        return skill_files

    def load_skill(self, skill_md_path: Path, location: str) -> Optional[Skill]:
        """
        Load a single skill from its SKILL.md file.

        Args:
            skill_md_path: Path to the SKILL.md file
            location: Location type (project, user, managed)

        Returns:
            Skill instance or None if loading fails
        """
        try:
            skill = Skill.from_skill_md(skill_md_path, location)
            if skill:
                logger.debug(f"Loaded skill: {skill.name} from {skill_md_path}")
            else:
                logger.warning(f"Failed to parse skill: {skill_md_path}")
            return skill
        except Exception as e:
            logger.error(f"Error loading skill from {skill_md_path}: {e}")
            return None

    def load_skill_from_dir(self, skill_dir: str, location: str = "project") -> Optional[Skill]:
        """
        Load a single skill from a directory path.

        Args:
            skill_dir: Path to the skill directory containing SKILL.md
            location: Location type (project, user, managed)

        Returns:
            Skill instance or None if loading fails
        """
        skill_dir_path = Path(skill_dir).resolve()
        skill_md_path = skill_dir_path / self.SKILL_FILE

        if not skill_md_path.exists():
            logger.warning(f"SKILL.md not found in {skill_dir}")
            return None

        return self.load_skill(skill_md_path, location)

    def load_all(self, registry: Optional[SkillRegistry] = None) -> SkillRegistry:
        """
        Load all skills from all search paths into the registry.

        Args:
            registry: Optional registry to load into. Creates new if not provided.

        Returns:
            SkillRegistry containing all loaded skills
        """
        if registry is None:
            registry = get_skill_registry()

        for search_path, location in self.get_search_paths():
            skill_files = self.discover_skills(search_path)

            for skill_md_path in skill_files:
                skill = self.load_skill(skill_md_path, location)
                if skill:
                    registered = registry.register(skill)
                    if registered:
                        logger.info(f"Registered skill: {skill.name} ({location})")
                    else:
                        logger.debug(
                            f"Skipped skill {skill.name} - already registered from higher priority location"
                        )

        return registry

    def reload(self, registry: Optional[SkillRegistry] = None) -> SkillRegistry:
        """
        Reload all skills, clearing the existing registry first.

        Args:
            registry: Optional registry to reload. Uses global if not provided.

        Returns:
            SkillRegistry containing all reloaded skills
        """
        if registry is None:
            registry = get_skill_registry()

        registry.clear()
        return self.load_all(registry)

    @staticmethod
    def list_skill_files(directory: str) -> str:
        """
        List all files in a directory.

        This is useful for exploring skill directories that may contain
        additional resources like scripts, references, and assets.

        Args:
            directory: The path to the directory to list.

        Returns:
            A formatted string containing the list of files in the directory.
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

            result = f"Contents of '{directory}':\n" + "\n".join(sorted(files))
            logger.info(f"Listed files in: {directory}, found {len(files)} files")
            result = truncate_if_too_long(result)
            return result
        except Exception as e:
            logger.error(f"Error listing directory {directory}: {e}")
            return f"Error listing directory: {e}"

    @staticmethod
    def read_skill_file(file_path: str) -> str:
        """
        Read the contents of a file.

        This is useful for reading skill resources like SKILL.md,
        scripts, or reference documents.

        Args:
            file_path: The path to the file to read.

        Returns:
            The contents of the file, or an error message if the file cannot be read.
        """
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.isfile(file_path):
                return f"Error: File '{file_path}' does not exist."

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Read file: {file_path}, {len(content)} characters")
            content = truncate_if_too_long(content)
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"


def load_skills(project_root: Optional[Path] = None) -> SkillRegistry:
    """
    Convenience function to load all skills from standard directories.

    Args:
        project_root: Root directory of the project

    Returns:
        SkillRegistry containing all loaded skills
    """
    loader = SkillLoader(project_root)
    return loader.load_all()


def get_available_skills() -> List[Skill]:
    """
    Get list of all available skills.

    Loads skills if the registry is empty.

    Returns:
        List of available skills
    """
    registry = get_skill_registry()

    if len(registry) == 0:
        load_skills()

    return registry.list_all()


def register_skill(skill_dir: str, location: str = "project") -> Optional[Skill]:
    """
    Register a single skill from a directory.

    Args:
        skill_dir: Path to the skill directory containing SKILL.md
        location: Location type (project, user, managed)

    Returns:
        Skill instance if registered successfully, None otherwise
    """
    loader = SkillLoader()
    skill = loader.load_skill_from_dir(skill_dir, location)

    if skill:
        registry = get_skill_registry()
        if registry.register(skill):
            logger.info(f"Registered skill: {skill.name} from {skill_dir}")
            return skill
        else:
            logger.debug(f"Skill {skill.name} already registered from higher priority location")

    return None


def register_skills(skill_dirs: List[str], location: str = "project") -> List[Skill]:
    """
    Register multiple skills from directories.

    Args:
        skill_dirs: List of paths to skill directories
        location: Location type for all skills

    Returns:
        List of successfully registered skills
    """
    registered = []
    for skill_dir in skill_dirs:
        skill = register_skill(skill_dir, location)
        if skill:
            registered.append(skill)
    return registered


def list_skill_files(directory: str) -> str:
    """
    List all files in a skill directory.

    Convenience function for exploring skill directories.

    Args:
        directory: The path to the directory to list.

    Returns:
        A formatted string containing the list of files in the directory.
    """
    return SkillLoader.list_skill_files(directory)


def read_skill_file(file_path: str) -> str:
    """
    Read the contents of a skill file.

    Convenience function for reading skill resources.

    Args:
        file_path: The path to the file to read.

    Returns:
        The contents of the file, or an error message if the file cannot be read.
    """
    return SkillLoader.read_skill_file(file_path)
