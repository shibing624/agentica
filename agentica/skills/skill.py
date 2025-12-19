# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill data class representing a loaded skill.

Agent Skill is an approach proposed by Anthropic to improve agent capabilities on specific tasks.
Skills are not code-level extensions, but text instructions injected into the system prompt,
allowing the LLM to read and follow the instructions to complete tasks.

Reference: https://claude.com/blog/skills
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Skill:
    """
    Represents a loaded skill with its metadata and content.

    A skill is defined by a SKILL.md file containing YAML frontmatter
    with metadata (name, description) and detailed usage instructions.

    Example SKILL.md format:
    ```markdown
    ---
    name: My Skill
    description: A skill for doing something useful.
    license: MIT
    allowed-tools:
      - shell
      - python
    ---

    # My Skill

    ## Overview
    This skill helps you do something useful...

    ## Usage
    1. First, do this...
    2. Then, do that...
    ```
    """

    name: str
    description: str
    content: str  # The markdown body (instructions)
    path: Path  # Path to the skill directory

    # Optional metadata from frontmatter
    license: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source location type: project, user, managed
    location: str = "project"

    @classmethod
    def from_skill_md(cls, skill_md_path: Path, location: str = "project") -> Optional["Skill"]:
        """
        Parse a SKILL.md file and create a Skill instance.

        Args:
            skill_md_path: Path to the SKILL.md file
            location: Where the skill was found (project, user, managed)

        Returns:
            Skill instance or None if parsing fails
        """
        if not skill_md_path.exists():
            return None

        content = skill_md_path.read_text(encoding='utf-8')
        frontmatter, body = cls._parse_frontmatter(content)

        if not frontmatter:
            return None

        name = frontmatter.get('name')
        description = frontmatter.get('description')

        if not name or not description:
            return None

        return cls(
            name=name,
            description=description,
            content=body.strip(),
            path=skill_md_path.parent,
            license=frontmatter.get('license'),
            allowed_tools=frontmatter.get('allowed-tools', []) or [],
            metadata=frontmatter.get('metadata', {}) or {},
            location=location,
        )

    @staticmethod
    def _parse_frontmatter(content: str) -> tuple:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            content: Raw markdown content with potential YAML frontmatter

        Returns:
            Tuple of (frontmatter dict, body content)
        """
        # Match YAML frontmatter pattern: starts with ---, ends with ---
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return {}, content

        yaml_content = match.group(1)
        body = match.group(2)

        # Try yaml first, fallback to frontmatter package
        if yaml is not None:
            try:
                frontmatter_data = yaml.safe_load(yaml_content) or {}
                return frontmatter_data, body
            except yaml.YAMLError:
                pass

        # Fallback: try python-frontmatter package
        try:
            import frontmatter
            post = frontmatter.loads(content)
            return dict(post.metadata), post.content
        except ImportError:
            pass
        except Exception:
            pass

        return {}, content

    def get_prompt(self) -> str:
        """
        Get the full prompt content for this skill.

        Includes the skill location header for resolving relative paths
        to bundled resources (references/, scripts/, assets/).

        Returns:
            Full prompt string with base directory header
        """
        header = f"""Loading: {self.name}
Base directory: {self.path}

"""
        return header + self.content

    def to_xml(self) -> str:
        """
        Format skill as XML for inclusion in prompts.

        Returns:
            XML formatted skill entry
        """
        return f"""<skill>
<name>{self.name}</name>
<description>{self.description}</description>
<location>{self.location}</location>
</skill>"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert skill to dictionary representation.

        Returns:
            Dictionary with skill data
        """
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "path": str(self.path),
            "license": self.license,
            "allowed_tools": self.allowed_tools,
            "metadata": self.metadata,
            "location": self.location,
        }

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, location={self.location!r})"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
