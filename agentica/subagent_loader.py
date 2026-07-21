# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Subagent Loader - discovers and loads custom subagent definitions
from ``.agentica/agents/*.md`` files (opencode-style file-based agents).

Each ``*.md`` file has a YAML frontmatter + Markdown body:

    ---
    description: Reviews code for quality and bugs
    allowed_tools: [read_file, ls, glob, grep]   # optional, None = inherit parent
    denied_tools: [task]                          # optional
    tool_call_limit: 10                           # optional
    ---
    You are a code review expert...

The file stem becomes the subagent name; the body becomes its system prompt.
``model`` overrides are intentionally NOT supported (see plan P1.2) - custom
subagents reuse the parent agent's auxiliary/main model.

Search paths (in priority order, higher wins on name collisions):
1. ``<cwd>/.agentica/agents/`` (project-level)
2. ``<AGENTICA_HOME>/agents/`` (user-level, defaults to ``~/.agentica/agents``)
3. ``AGENTICA_AGENT_DIR`` environment variable (if set)

The loader is fail-soft: a single malformed file (missing description, empty
body, bad YAML) only emits a ``logger.warning`` and is skipped so a broken
agent file can never block CLI startup.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.subagent import (
    register_custom_subagent,
    unregister_custom_subagent,
)
from agentica.utils.log import logger

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a core dependency
    yaml = None

# Subagent name must be filesystem-safe: letters, digits, hyphen, underscore.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Frontmatter delimiter: opening --- ... closing ---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def get_search_paths() -> List[str]:
    """Return the ordered list of agent search directories (as strings).

    Project-level first (highest priority), then user-level under
    ``AGENTICA_HOME``, then the optional ``AGENTICA_AGENT_DIR`` env override.
    Duplicate resolved paths are collapsed.
    """
    paths: List[str] = []
    seen: set = set()

    def add(p: str) -> None:
        resolved = str(Path(p).expanduser().resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        paths.append(p)

    # 1. Project-level (cwd)
    add(str(Path.cwd() / ".agentica" / "agents"))

    # 2. User-level (AGENTICA_HOME, defaults to ~/.agentica)
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    add(os.path.join(home, "agents"))

    # 3. Explicit env override
    env_dir = os.getenv("AGENTICA_AGENT_DIR")
    if env_dir:
        add(env_dir)

    return paths


def _parse_agent_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a single ``*.md`` agent file into a descriptor dict.

    Returns ``None`` (and logs a warning) when the file is malformed: missing
    frontmatter, missing ``description``, empty body, or unparseable YAML.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"Subagent loader: cannot read {path}: {e}")
        return None

    match = _FRONTMATTER_RE.match(content.strip())
    if not match:
        logger.warning(f"Subagent loader: no frontmatter block in {path}, skipped")
        return None

    yaml_text, body = match.group(1), match.group(2)
    if yaml is None:
        logger.warning("Subagent loader: pyyaml not installed, cannot parse frontmatter")
        return None

    try:
        frontmatter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Subagent loader: invalid YAML frontmatter in {path}: {e}")
        return None

    if not isinstance(frontmatter, dict):
        logger.warning(f"Subagent loader: frontmatter is not a mapping in {path}, skipped")
        return None

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        logger.warning(f"Subagent loader: missing 'description' in {path}, skipped")
        return None

    system_prompt = body.strip()
    if not system_prompt:
        logger.warning(f"Subagent loader: empty body (system_prompt) in {path}, skipped")
        return None

    def _as_str_list(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            out = [str(v) for v in value if v is not None]
            return out or None
        return None

    tool_call_limit_raw = frontmatter.get("tool_call_limit")
    tool_call_limit: Optional[int] = None
    if tool_call_limit_raw is not None:
        try:
            tool_call_limit = int(tool_call_limit_raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Subagent loader: invalid tool_call_limit in {path}, ignored"
            )

    return {
        "name": path.stem,
        "description": description.strip(),
        "system_prompt": system_prompt,
        "allowed_tools": _as_str_list(frontmatter.get("allowed_tools")),
        "denied_tools": _as_str_list(frontmatter.get("denied_tools")),
        "tool_call_limit": tool_call_limit,
        "path": str(path),
    }


def load_all_agents() -> int:
    """Scan all search paths and register every well-formed agent file.

    Returns the number of agents registered. Fail-soft per file (malformed
    files are skipped with a warning); an outer guard ensures any unexpected
    error cannot block CLI startup.
    """
    try:
        registered = 0
        seen_names: set = set()
        for search_dir in get_search_paths():
            directory = Path(search_dir).expanduser()
            if not directory.exists() or not directory.is_dir():
                continue
            try:
                files = sorted(directory.glob("*.md"))
            except OSError as e:
                logger.warning(f"Subagent loader: cannot list {directory}: {e}")
                continue
            for md_path in files:
                descriptor = _parse_agent_file(md_path)
                if descriptor is None:
                    continue
                name = descriptor["name"]
                key = name.lower()
                if key in seen_names:
                    # Higher-priority path already registered this name.
                    continue
                seen_names.add(key)
                register_custom_subagent(
                    name=name,
                    description=descriptor["description"],
                    system_prompt=descriptor["system_prompt"],
                    allowed_tools=descriptor["allowed_tools"],
                    denied_tools=descriptor["denied_tools"],
                    tool_call_limit=descriptor["tool_call_limit"],
                )
                registered += 1
        return registered
    except Exception as e:  # CLI startup tolerance boundary
        logger.warning(f"Subagent loader: load_all_agents failed: {e}")
        return 0


def list_defined_agents() -> List[Dict[str, Any]]:
    """Rescan disk and return descriptors for all well-formed agent files.

    Does not depend on the in-memory registry state, so it reflects the
    current filesystem regardless of what has been registered. Name
    collisions resolve to the higher-priority path (project over user).
    """
    result: List[Dict[str, Any]] = []
    seen_names: set = set()
    try:
        for search_dir in get_search_paths():
            directory = Path(search_dir).expanduser()
            if not directory.exists() or not directory.is_dir():
                continue
            try:
                files = sorted(directory.glob("*.md"))
            except OSError as e:
                logger.warning(f"Subagent loader: cannot list {directory}: {e}")
                continue
            for md_path in files:
                descriptor = _parse_agent_file(md_path)
                if descriptor is None:
                    continue
                key = descriptor["name"].lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                result.append(
                    {
                        "name": descriptor["name"],
                        "description": descriptor["description"],
                        "allowed_tools": descriptor["allowed_tools"],
                        "denied_tools": descriptor["denied_tools"],
                        "tool_call_limit": descriptor["tool_call_limit"],
                        "path": descriptor["path"],
                    }
                )
    except Exception as e:  # defensive: listing must never raise to callers
        logger.warning(f"Subagent loader: list_defined_agents failed: {e}")
    return result


def _resolve_target_dir(scope: str) -> Path:
    """Resolve the write directory for a new agent file given its scope."""
    if scope == "user":
        home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
        return Path(home) / "agents"
    # default: project-level
    return Path.cwd() / ".agentica" / "agents"


def create_agent_file(
    name: str,
    description: str,
    system_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    denied_tools: Optional[List[str]] = None,
    tool_call_limit: Optional[int] = None,
    scope: str = "project",
) -> str:
    """Write a new ``<name>.md`` agent file, register it, and return its path.

    ``name`` is validated to be filesystem-safe (alphanumeric, hyphen,
    underscore); path separators and ``..`` are rejected. ``scope`` selects
    the write directory: ``"project"`` (``<cwd>/.agentica/agents/``) or
    ``"user"`` (``<AGENTICA_HOME>/agents/``).
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid agent name {name!r}: only letters, digits, '-' and '_' are allowed"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")

    target_dir = _resolve_target_dir(scope)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{name}.md"

    frontmatter: Dict[str, Any] = {"description": description.strip()}
    if allowed_tools:
        frontmatter["allowed_tools"] = list(allowed_tools)
    if denied_tools:
        frontmatter["denied_tools"] = list(denied_tools)
    if tool_call_limit is not None:
        frontmatter["tool_call_limit"] = int(tool_call_limit)

    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    content = f"---\n{yaml_block}---\n{system_prompt.strip()}\n"
    file_path.write_text(content, encoding="utf-8")

    register_custom_subagent(
        name=name,
        description=description.strip(),
        system_prompt=system_prompt.strip(),
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        tool_call_limit=tool_call_limit,
    )
    logger.info(f"Created subagent file: {file_path}")
    return str(file_path)


def remove_agent_file(name: str) -> bool:
    """Delete the agent file for ``name`` from any search path and unregister it.

    Returns ``True`` if a file was found and removed, ``False`` otherwise.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return False
    target = f"{name}.md"
    for search_dir in get_search_paths():
        candidate = Path(search_dir).expanduser() / target
        if candidate.exists() and candidate.is_file():
            try:
                candidate.unlink()
            except OSError as e:
                logger.warning(f"Subagent loader: cannot delete {candidate}: {e}")
                return False
            unregister_custom_subagent(name)
            logger.info(f"Removed subagent file: {candidate}")
            return True
    return False
