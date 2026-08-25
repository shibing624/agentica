# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Load subagent definitions from Markdown files with YAML frontmatter.

Agent definitions use a Markdown body for the system prompt and YAML
frontmatter for runtime fields. Effective precedence is:

1. ``<cwd>/.agentica/agents/*.md``
2. ``~/.agentica/agents/*.md`` (or ``$AGENTICA_HOME/agents``)
3. ``$AGENTICA_AGENT_DIR/*.md``
4. package defaults in ``agentica/subagents/bundled/*.md``

Higher-priority files replace lower-priority definitions with the same stem.
Package defaults are the shipped source of truth; user and project directories
contain only additions or overrides, so upgrades do not copy stale defaults
into a user's home directory.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

from agentica.subagents.runtime import SubagentConfig, _replace_file_subagent_configs
from agentica.utils.log import logger

BUNDLED_SUBAGENT_DIR = Path(__file__).resolve().parent / "bundled"


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n(.*))?\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class AgentSearchLocation:
    path: Path
    source: str
    strict: bool = False


def get_search_locations() -> List[AgentSearchLocation]:
    """Return agent directories from highest to lowest precedence."""
    raw_locations = [
        AgentSearchLocation(Path.cwd() / ".agentica" / "agents", "project"),
        AgentSearchLocation(
            Path(os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))) / "agents",
            "user",
        ),
    ]
    env_dir = os.getenv("AGENTICA_AGENT_DIR")
    if env_dir:
        raw_locations.append(AgentSearchLocation(Path(env_dir).expanduser(), "environment"))
    raw_locations.append(AgentSearchLocation(BUNDLED_SUBAGENT_DIR, "package", strict=True))

    locations: List[AgentSearchLocation] = []
    seen: set[str] = set()
    for location in raw_locations:
        resolved = str(location.path.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        locations.append(location)
    return locations


def _warn_or_raise(path: Path, message: str, strict: bool) -> None:
    detail = f"Subagent loader: {message} in {path}"
    if strict:
        raise ValueError(detail)
    logger.warning(f"{detail}, skipped")


def _field_default(path: Path, message: str, strict: bool) -> None:
    detail = f"Subagent loader: {message} in {path}"
    if strict:
        raise ValueError(detail)
    logger.warning(f"{detail}; using default")


def _parse_agent_file(
    path: Path,
    *,
    source: str = "runtime",
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Parse one agent definition; package files fail loudly when invalid."""
    if not _NAME_RE.fullmatch(path.stem):
        _warn_or_raise(path, "invalid file stem", strict)
        return None
    if path.stem != path.stem.lower():
        _warn_or_raise(path, f"file name must be lowercase (got {path.stem!r})", strict)
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        if strict:
            raise
        logger.warning(f"Subagent loader: cannot read {path}: {exc}")
        return None

    stripped = content.strip()
    if not stripped.startswith("---"):
        _warn_or_raise(path, "missing YAML frontmatter", strict)
        return None
    match = _FRONTMATTER_RE.match(stripped)
    if not match:
        _warn_or_raise(path, "malformed YAML frontmatter block", strict)
        return None

    yaml_text, body = match.group(1), match.group(2) or ""
    try:
        frontmatter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        _warn_or_raise(path, f"invalid YAML frontmatter ({exc})", strict)
        return None

    if not isinstance(frontmatter, dict):
        _warn_or_raise(path, "frontmatter is not a mapping", strict)
        return None

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        _warn_or_raise(path, "missing description", strict)
        return None

    system_prompt = body.strip()
    if not system_prompt:
        _warn_or_raise(path, "empty Markdown body", strict)
        return None

    display_name = frontmatter.get("name", path.stem)
    if not isinstance(display_name, str) or not display_name.strip():
        _warn_or_raise(path, "invalid name", strict)
        return None

    def as_str_list(field: str) -> Optional[List[str]]:
        value = frontmatter.get(field)
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
        _field_default(path, f"invalid {field}", strict)
        return None

    def as_int(field: str, default: Optional[int], minimum: int) -> Optional[int]:
        value = frontmatter.get(field)
        if value is None:
            return default
        if isinstance(value, bool):
            _field_default(path, f"invalid {field}", strict)
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            _field_default(path, f"invalid {field}", strict)
            return default
        if parsed < minimum:
            _field_default(path, f"invalid {field}", strict)
            return default
        return parsed

    def as_bool(field: str, default: bool) -> bool:
        value = frontmatter.get(field)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        _field_default(path, f"invalid {field}", strict)
        return default

    model_tier = frontmatter.get("model_tier", "auxiliary")
    if model_tier not in ("auxiliary", "main"):
        _field_default(path, f"invalid model_tier {model_tier!r}", strict)
        model_tier = "auxiliary"

    execute_policy = frontmatter.get("execute_policy", "inherit")
    if execute_policy not in ("inherit", "read_only"):
        _field_default(path, f"invalid execute_policy {execute_policy!r}", strict)
        execute_policy = "inherit"

    return {
        "id": path.stem.lower(),
        "name": display_name.strip(),
        "description": description.strip(),
        "system_prompt": system_prompt,
        "allowed_tools": as_str_list("allowed_tools"),
        "denied_tools": as_str_list("denied_tools"),
        "tool_call_limit": as_int("tool_call_limit", None, 1),
        "max_turns": as_int("max_turns", 100, 1),
        "timeout": as_int("timeout", 1800, 0),
        "model_tier": model_tier,
        "execute_policy": execute_policy,
        "can_spawn_subagents": as_bool("can_spawn_subagents", False),
        "inherit_workspace": as_bool("inherit_workspace", False),
        "inherit_knowledge": as_bool("inherit_knowledge", False),
        "inherit_context": as_bool("inherit_context", False),
        "source": source,
        "path": str(path),
    }


def _descriptor_to_config(descriptor: Dict[str, Any]) -> SubagentConfig:
    return SubagentConfig(
        type=descriptor["id"],
        name=descriptor["name"],
        description=descriptor["description"],
        system_prompt=descriptor["system_prompt"],
        source=descriptor["source"],
        path=descriptor["path"],
        allowed_tools=descriptor["allowed_tools"],
        denied_tools=descriptor["denied_tools"],
        tool_call_limit=descriptor["tool_call_limit"],
        max_turns=descriptor["max_turns"],
        timeout=descriptor["timeout"],
        model_tier=descriptor["model_tier"],
        execute_policy=descriptor["execute_policy"],
        can_spawn_subagents=descriptor["can_spawn_subagents"],
        inherit_workspace=descriptor["inherit_workspace"],
        inherit_knowledge=descriptor["inherit_knowledge"],
        inherit_context=descriptor["inherit_context"],
    )


def _discover_agents() -> Tuple[Dict[str, SubagentConfig], List[Dict[str, Any]]]:
    """Discover effective definitions, applying high-priority overrides last."""
    configs: Dict[str, SubagentConfig] = {}
    descriptors: Dict[str, Dict[str, Any]] = {}
    locations = get_search_locations()

    for location in reversed(locations):
        directory = location.path.expanduser()
        if not directory.exists() or not directory.is_dir():
            if location.strict:
                raise FileNotFoundError(
                    f"Missing packaged agent directory: {directory}. "
                    "Reinstall agentica so agentica/subagents/bundled/*.md are shipped."
                )
            continue
        try:
            files = sorted(directory.glob("*.md"))
        except OSError:
            if location.strict:
                raise
            logger.warning(f"Subagent loader: cannot list {directory}")
            continue
        for md_path in files:
            descriptor = _parse_agent_file(
                md_path,
                source=location.source,
                strict=location.strict,
            )
            if descriptor is None:
                continue
            agent_id = descriptor["id"]
            configs[agent_id] = _descriptor_to_config(descriptor)
            descriptors[agent_id] = descriptor

    return configs, list(descriptors.values())


def load_all_agents() -> int:
    """Reload all file-backed agent definitions atomically."""
    configs, _ = _discover_agents()
    _replace_file_subagent_configs(configs)
    return len(configs)


def list_defined_agents() -> List[Dict[str, Any]]:
    """Return all effective on-disk definitions, including package defaults."""
    _, descriptors = _discover_agents()
    return descriptors


def _resolve_target_dir(scope: str) -> Path:
    """Resolve the write directory for a new user or project definition."""
    if scope == "user":
        home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
        return Path(home) / "agents"
    if scope == "project":
        return Path.cwd() / ".agentica" / "agents"
    raise ValueError("scope must be 'project' or 'user'")


def create_agent_file(
    name: str,
    description: str,
    system_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    denied_tools: Optional[List[str]] = None,
    tool_call_limit: Optional[int] = None,
    scope: str = "project",
    *,
    display_name: Optional[str] = None,
    model_tier: Literal["auxiliary", "main"] = "auxiliary",
    execute_policy: Literal["inherit", "read_only"] = "inherit",
    max_turns: int = 100,
    timeout: int = 1800,
    can_spawn_subagents: bool = False,
    inherit_context: bool = False,
    inherit_workspace: bool = False,
    inherit_knowledge: bool = False,
) -> str:
    """Write a complete agent definition, reload the registry, and return its path."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid agent name {name!r}: only letters, digits, '-' and '_' are allowed")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    if display_name is not None and (
        not isinstance(display_name, str) or not display_name.strip()
    ):
        raise ValueError("display_name must be a non-empty string when provided")
    for field, value in (
        ("allowed_tools", allowed_tools),
        ("denied_tools", denied_tools),
    ):
        if value is not None and (
            not isinstance(value, list)
            or not all(isinstance(item, str) for item in value)
        ):
            raise ValueError(f"{field} must be a list of strings or None")
    if model_tier not in ("auxiliary", "main"):
        raise ValueError("model_tier must be 'auxiliary' or 'main'")
    if execute_policy not in ("inherit", "read_only"):
        raise ValueError("execute_policy must be 'inherit' or 'read_only'")
    if isinstance(tool_call_limit, bool) or (
        tool_call_limit is not None
        and (not isinstance(tool_call_limit, int) or tool_call_limit < 1)
    ):
        raise ValueError("tool_call_limit must be a positive integer or None")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0:
        raise ValueError("timeout must be a non-negative integer")
    for field, value in (
        ("can_spawn_subagents", can_spawn_subagents),
        ("inherit_context", inherit_context),
        ("inherit_workspace", inherit_workspace),
        ("inherit_knowledge", inherit_knowledge),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")

    target_dir = _resolve_target_dir(scope)
    target_dir.mkdir(parents=True, exist_ok=True)
    agent_id = name.lower()
    file_path = target_dir / f"{agent_id}.md"

    frontmatter: Dict[str, Any] = {
        "description": description.strip(),
        "model_tier": model_tier,
        "execute_policy": execute_policy,
        "max_turns": max_turns,
        "timeout": timeout,
        "can_spawn_subagents": can_spawn_subagents,
        "inherit_context": inherit_context,
        "inherit_workspace": inherit_workspace,
        "inherit_knowledge": inherit_knowledge,
    }
    if display_name is not None:
        frontmatter["name"] = display_name.strip()
    if allowed_tools is not None:
        frontmatter["allowed_tools"] = list(allowed_tools)
    if denied_tools is not None:
        frontmatter["denied_tools"] = list(denied_tools)
    if tool_call_limit is not None:
        frontmatter["tool_call_limit"] = int(tool_call_limit)

    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    content = f"---\n{yaml_block}---\n{system_prompt.strip()}\n"
    file_path.write_text(content, encoding="utf-8")
    load_all_agents()
    logger.info(f"Created subagent file: {file_path}")
    return str(file_path)


def remove_agent_file(name: str) -> bool:
    """Delete the highest-priority user/project definition and reload."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return False
    agent_id = name.lower()
    for location in get_search_locations():
        if location.source == "package":
            continue
        directory = location.path.expanduser()
        if not directory.is_dir():
            continue
        candidate = directory / f"{agent_id}.md"
        if candidate.is_file():
            candidate.unlink()
            load_all_agents()
            logger.info(f"Removed subagent file: {candidate}")
            return True
    return False
