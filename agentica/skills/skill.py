# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skill data class representing a loaded skill.

Agent Skill is an approach proposed by Anthropic to improve agent capabilities on specific tasks.
Skills are not code-level extensions, but text instructions injected into the system prompt,
allowing the LLM to read and follow the instructions to complete tasks.

Reference: https://claude.com/blog/skills
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from agentica.utils.log import logger

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
    trigger: /myskill
    argument-hint: "<file-path>"
    requires:
      - shell
      - python
    allowed-tools:
      - shell
      - python
    user-invocable: true
    is-hidden: false
    metadata:
      emoji: "🔧"
    ---

    # My Skill

    ## Overview
    This skill helps you do something useful...

    ## Usage
    1. First, do this...
    2. Then, do that...
    ```

    Attributes:
        name: Skill name (required)
        description: Skill description (required)
        content: The markdown body with instructions (lazy-loaded from SKILL.md)
        path: Path to the skill directory
        license: Optional license information
        trigger: Optional trigger command (e.g., /commit)
        argument_hint: Hint for trigger arguments (e.g. "<file-path>")
        when_to_use: Keywords describing when this skill is useful
        requires: List of required tools or commands
        allowed_tools: List of tools allowed for this skill
        metadata: Additional metadata from frontmatter
        user_invocable: If False, cannot be invoked via /trigger
        is_hidden: If True, hidden from typeahead and /skills listing
        location: Source location type (project, user, managed)
    """

    name: str
    description: str
    path: Path  # Path to the skill directory

    # Lazy-loaded content: body is read from SKILL.md on first .content access
    _content: Optional[str] = field(default=None, repr=False, init=False)
    _content_loaded: bool = field(default=False, repr=False, init=False)

    # Optional metadata from frontmatter
    license: Optional[str] = None
    trigger: Optional[str] = None  # Trigger command like /commit
    argument_hint: Optional[str] = None  # Hint for argument (e.g. "<file-path>")
    when_to_use: Optional[str] = None  # Keywords describing when this skill is useful
    requires: List[str] = field(default_factory=list)  # Required tools/commands
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Visibility and invocation control
    user_invocable: bool = True  # If False, skill cannot be invoked via /trigger directly
    is_hidden: bool = False  # If True, hidden from typeahead / /skills listing

    # Source location type: project, user, managed
    location: str = "project"

    @property
    def content(self) -> str:
        """Lazy-load the markdown body from SKILL.md on first access."""
        if not self._content_loaded:
            md_file = self.path / "SKILL.md"
            if md_file.exists():
                raw = md_file.read_text(encoding='utf-8')
                _, body = self._parse_frontmatter(raw.strip())
                self._content = body.strip()
            else:
                self._content = ""
            self._content_loaded = True
        return self._content or ""

    @content.setter
    def content(self, value: str) -> None:
        self._content = value
        self._content_loaded = True

    def invalidate_content(self) -> None:
        """Clear cached content so next .content access re-reads SKILL.md.

        Call this after the underlying SKILL.md file has been modified on disk.
        Only the body is reloaded; frontmatter fields (name, description, trigger,
        when_to_use, etc.) are NOT refreshed. To pick up frontmatter changes,
        re-parse via ``Skill.from_skill_md()`` and re-register.
        """
        self._content = None
        self._content_loaded = False

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
        frontmatter, _ = cls._parse_frontmatter(content.strip())

        if not frontmatter:
            logger.warning(
                f"Skill skipped: {skill_md_path} has no valid YAML frontmatter. "
                f"A SKILL.md must start with a '---' fenced block containing at least "
                f"'name' and 'description', e.g.:\n---\nname: my-skill\ndescription: ...\n---"
            )
            return None

        name = frontmatter.get('name')
        description = frontmatter.get('description')

        if not name or not description:
            missing = [k for k in ("name", "description") if not frontmatter.get(k)]
            logger.warning(
                f"Skill skipped: {skill_md_path} frontmatter is missing required "
                f"field(s): {', '.join(missing)}. Both 'name' and 'description' are required."
            )
            return None

        return cls(
            name=name,
            description=description,
            # content NOT passed -- lazy loaded via property on first access
            path=skill_md_path.parent,
            license=frontmatter.get('license'),
            trigger=frontmatter.get('trigger'),
            argument_hint=frontmatter.get('argument-hint'),
            when_to_use=frontmatter.get('when_to_use') or frontmatter.get('when-to-use'),
            requires=frontmatter.get('requires', []) or [],
            allowed_tools=frontmatter.get('allowed-tools', []) or [],
            metadata=frontmatter.get('metadata', {}) or {},
            user_invocable=frontmatter.get('user-invocable', True),
            is_hidden=frontmatter.get('is-hidden', False),
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

        # Fallback: tolerant line-based parse. Strict YAML rejects common
        # real-world frontmatter where a scalar value contains an unquoted
        # colon-with-space (e.g. ``description: Learn from experience: ...``),
        # which YAML misreads as a nested mapping. This keeps the whole tail of
        # each ``key: value`` line as the value so such skills still load.
        lenient = Skill._parse_frontmatter_lenient(yaml_content)
        if lenient and lenient.get("name"):
            return lenient, body

        return {}, content

    @staticmethod
    def _parse_frontmatter_lenient(yaml_content: str) -> Dict[str, Any]:
        """Best-effort fallback for frontmatter that is not strict YAML.

        Splits the frontmatter block into ``key: value`` lines and keeps the
        entire remainder of each line as the value, tolerating unquoted colons
        inside scalar values. Single-line JSON values (``metadata``,
        ``allowed-tools``, etc.) are decoded when possible. Only the common
        scalar coerctions (bool / null / int / float) are applied; multi-line
        or nested YAML structures are intentionally NOT supported here.
        """
        result: Dict[str, Any] = {}
        for line in yaml_content.splitlines():
            if not line.strip():
                continue
            m = re.match(r"^([A-Za-z0-9_-]+):\s?(.*)$", line)
            if not m:
                continue
            key = m.group(1)
            value = m.group(2).strip()
            # Strip a single layer of matching quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            # Decode single-line structured values (JSON).
            if value and value[0] in ("{", "["):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
            else:
                low = value.lower()
                if low in ("true", "false"):
                    value = low == "true"
                elif low in ("null", "~"):
                    value = None
                elif re.fullmatch(r"-?\d+", value):
                    value = int(value)
                elif re.fullmatch(r"-?\d+\.\d+", value):
                    value = float(value)
            result[key] = value
        return result

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

    def matches_keywords(self, text: str) -> bool:
        """Check if text matches this skill's when_to_use keywords.

        Args:
            text: User input text to check

        Returns:
            True if any keyword from when_to_use appears in text
        """
        if not self.when_to_use:
            return False
        text_lower = text.lower()
        keywords = [kw.strip().lower() for kw in re.split(r'[,;.\n]', self.when_to_use) if kw.strip()]
        return any(kw in text_lower for kw in keywords if len(kw) > 2)

    def to_xml(self) -> str:
        """
        Format skill as XML for inclusion in prompts.

        Returns:
            XML formatted skill entry
        """
        parts = [
            f"<name>{self.name}</name>",
            f"<description>{self.description}</description>",
            f"<location>{self.location}</location>",
        ]
        if self.when_to_use:
            parts.append(f"<when_to_use>{self.when_to_use}</when_to_use>")
        inner = "\n".join(parts)
        return f"<skill>\n{inner}\n</skill>"

    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """
        Convert skill to dictionary representation.

        Args:
            include_content: If True (default), access self.content which may
                trigger lazy loading from SKILL.md. Set False for discovery
                listings where only metadata is needed.

        Returns:
            Dictionary with skill data
        """
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "license": self.license,
            "trigger": self.trigger,
            "argument_hint": self.argument_hint,
            "when_to_use": self.when_to_use,
            "requires": self.requires,
            "allowed_tools": self.allowed_tools,
            "metadata": self.metadata,
            "user_invocable": self.user_invocable,
            "is_hidden": self.is_hidden,
            "location": self.location,
        }
        if include_content:
            d["content"] = self.content
        return d

    def matches_trigger(self, text: str) -> bool:
        """
        Check if the given text matches this skill's trigger.

        Args:
            text: User input text to check

        Returns:
            True if text starts with this skill's trigger command
        """
        if not self.trigger:
            return False
        return text.strip().startswith(self.trigger)

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, location={self.location!r})"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
