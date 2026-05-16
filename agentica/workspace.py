# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Workspace management for Agentica agents.
Inspired by OpenClaw's workspace concept.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import date, datetime

from agentica.config import (
    AGENTICA_WORKSPACE_DIR,
    AGENTICA_HOME,
    AGENTICA_MAX_MEMORY_CHARACTER_COUNT,
)
from agentica.security.redact import redact_sensitive_text
from agentica.utils.async_file import (
    async_read_text,
    async_write_text,
    extract_frontmatter_value,
    extract_frontmatter_int,
    strip_frontmatter,
)
from agentica.utils.log import logger


@dataclass
class WorkspaceConfig:
    """Workspace configuration.

    Attributes:
        agent_md: Agent instruction file name
        persona_md: Persona settings file name
        tools_md: Tool documentation file name
        user_md: User information file name
        memory_md: Long-term memory file name
        memory_dir: Daily memory directory name
        skills_dir: Skills directory name
        users_dir: User data directory name (for multi-user isolation)
    """
    agent_md: str = "AGENTS.md"
    persona_md: str = "PERSONA.md"
    tools_md: str = "TOOLS.md"
    user_md: str = "USER.md" # user infomation
    users_dir: str = "users" # for multi-user isolation
    memory_dir: str = "memory" # daily memory, under users/{user_id}/memory
    memory_md: str = "MEMORY.md" # user's long-term memory, under users/{user_id}/
    skills_dir: str = "skills" # each user's skills, under users/{user_id}/skills
    conversations_dir: str = "conversations" # conversation archive, under users/{user_id}/conversations
    reports_dir: str = "reports" # reports, under users/{user_id}/reports
    # Evidence-gate scratch space for unverified memory candidates (Phase 2)
    memory_candidates_dir: str = "memory_candidates"


class Workspace:
    """Agent Workspace.

    Workspace is the configuration and memory storage directory for Agent,
    supporting multi-user isolation. All user data is stored under users/ directory.

    Directory structure:
    - AGENTS.md: Agent instructions and constraints (globally shared)
    - PERSONA.md: Agent persona settings (globally shared)
    - TOOLS.md: Tool usage documentation (globally shared)
    - skills/: Custom skills directory (globally shared)
    - users/: User data directory (all users including default)
        - default/: Default user (when no user_id specified)
            - USER.md: User information
            - MEMORY.md: Long-term memory
            - memory/: Daily memory directory
        - {user_id}/: Other users
            - USER.md: User information
            - MEMORY.md: Long-term memory
            - memory/: Daily memory directory

    Default user mode:
        >>> workspace = Workspace("~/.agentica/workspace")  # user_id='default'
        >>> workspace.initialize()
        >>> await workspace.write_memory_entry("pref", "User prefers concise responses", "user")

    Custom user mode:
        >>> workspace = Workspace("~/.agentica/workspace", user_id="alice@example.com")
        >>> workspace.initialize()
        >>> await workspace.write_memory_entry("lang", "Alice likes Python", "user")

    Switch user:
        >>> workspace.set_user("bob@example.com")
        >>> await workspace.write_memory_entry("style", "Bob prefers detailed explanations", "user")
    """

    # Global config files (shared across all users)
    # Templates are intentionally minimal — boilerplate ("Friendly and
    # professional", default code-verification recipes) pollutes every
    # system prompt with zero behavioural signal. Customize the file on
    # disk when you actually have project-specific rules to add.
    DEFAULT_GLOBAL_FILES = {
        "AGENTS.md": """# Agent Instructions

<!-- Add project-specific agent rules here. -->
<!-- Empty file = no extra rules injected into the system prompt. -->
""",
        "PERSONA.md": "",
        "TOOLS.md": "",
    }

    # Default user file template
    DEFAULT_USER_MD = """# User Profile

## User ID
{user_id}

<!-- Optional: preferences, context, ongoing projects. -->
"""

    # Files whose body matches a default scaffold (just comments or blank
    # lines) are skipped entirely when assembling the system prompt —
    # there's no point telling the LLM "the user did not customize this".
    @staticmethod
    def _is_empty_template(content: str) -> bool:
        if not content:
            return True
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("<!--"):
                continue
            if stripped.startswith("- ") and ("Add" in stripped or "Example" in stripped):
                continue
            return False
        return True

    def __init__(
        self,
        path: Optional[str | Path] = None,
        config: Optional[WorkspaceConfig] = None,
        user_id: Optional[str] = None,
    ):
        """Initialize workspace.

        Args:
            path: Workspace path, defaults to AGENTICA_WORKSPACE_DIR (~/.agentica/workspace)
            config: Workspace configuration, defaults to WorkspaceConfig defaults
            user_id: User ID for multi-user isolation. Defaults to 'default' if not specified
        """
        if path is None:
            path = AGENTICA_WORKSPACE_DIR
        self.path = Path(path).expanduser().resolve()
        self.config = config or WorkspaceConfig()
        # Default to 'default' user if not specified
        self._user_id = user_id if user_id else "default"
        # Per-file locks for concurrent archive writes
        self._archive_locks: Dict[str, asyncio.Lock] = {}
        # Flag to avoid redundant _initialize_user_dir calls
        self._user_initialized: bool = False
        # Frozen snapshots for prompt cache stability (Hermes-style)
        self._context_snapshot: Optional[str] = None
        self._memory_snapshot: Optional[str] = None

    @property
    def user_id(self) -> str:
        """Get current user ID."""
        return self._user_id

    def set_user(self, user_id: Optional[str]):
        """Set current user ID.

        Args:
            user_id: User ID, defaults to 'default' if None
        """
        new_id = user_id if user_id else "default"
        if new_id != self._user_id:
            self._user_initialized = False
        self._user_id = new_id

    def _get_user_path(self) -> Path:
        """Get current user's data directory path.

        Returns:
            Path to users/{user_id}/ directory
        """
        # Sanitize user_id, replace unsafe characters
        safe_user_id = self._user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.path / self.config.users_dir / safe_user_id

    def _get_user_memory_dir(self) -> Path:
        """Get current user's daily memory directory."""
        return self._get_user_path() / self.config.memory_dir

    def _get_user_memory_md(self) -> Path:
        """Get current user's long-term memory file path."""
        return self._get_user_path() / self.config.memory_md

    def _get_user_md(self) -> Path:
        """Get current user's USER.md file path."""
        return self._get_user_path() / self.config.user_md

    # ── arch_v5.md §"Workspace Logical Partitioning" ──────────────────
    # New first-class folders for reports + archives + memory candidates.
    # Created on demand so existing workspaces don't need migration.

    def _get_user_reports_dir(self) -> Path:
        """Reports root for the current user (learning, runs, sessions, eval, ...).

        Lazily created to match the sibling helper (memory_candidates_dir)
        -- callers shouldn't have to choose between helpers based on whether
        they create the directory. RunJournal (P0 #3) and SessionArchive
        (P2 #7) both land under `reports/runs/` and `reports/sessions/`
        respectively; there is no separate top-level `archives/` partition.
        """
        path = self._get_user_path() / self.config.reports_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_user_learning_reports_dir(self) -> Path:
        """Folder where structured LearningReport markdown is persisted."""
        path = self._get_user_reports_dir() / "learning"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_user_memory_candidates_dir(self) -> Path:
        """Quarantine folder for memory entries lacking verified evidence."""
        path = self._get_user_path() / self.config.memory_candidates_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def initialize(self, force: bool = False) -> bool:
        """Initialize workspace.

        Creates workspace directory, global configuration files, and user data directory.

        Args:
            force: Whether to overwrite existing files

        Returns:
            Whether initialization was successful
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Create globally shared files (AGENTS.md, PERSONA.md, TOOLS.md)
        for filename, content in self.DEFAULT_GLOBAL_FILES.items():
            filepath = self.path / filename
            if not filepath.exists() or force:
                filepath.write_text(content, encoding="utf-8")

        # Create global directories
        (self.path / self.config.skills_dir).mkdir(exist_ok=True)
        (self.path / self.config.users_dir).mkdir(exist_ok=True)

        # Always create user directory (default or specified)
        self._initialize_user_dir()

        return True

    def _initialize_user_dir(self):
        """Initialize current user's data directory.

        Uses a cached flag to avoid redundant I/O on repeated calls.
        """
        if self._user_initialized:
            return

        user_path = self._get_user_path()
        user_path.mkdir(parents=True, exist_ok=True)

        # Create user's USER.md
        user_md = user_path / self.config.user_md
        if not user_md.exists():
            user_md.write_text(
                self.DEFAULT_USER_MD.format(user_id=self._user_id),
                encoding="utf-8"
            )

        # Create user's memory directory
        (user_path / self.config.memory_dir).mkdir(exist_ok=True)

        # Create user's conversations directory
        (user_path / self.config.conversations_dir).mkdir(exist_ok=True)

        self._user_initialized = True

    def exists(self) -> bool:
        """Check if workspace exists.

        Returns:
            Whether both workspace directory and AGENTS.md file exist
        """
        return self.path.exists() and (self.path / self.config.agent_md).exists()

    async def read_file_async(self, filename: str) -> Optional[str]:
        """Read workspace file asynchronously.

        Args:
            filename: File name (relative to workspace path)

        Returns:
            File content, or None if file doesn't exist or is empty
        """
        filepath = self.path / filename
        if filepath.exists() and filepath.is_file():
            content = (await async_read_text(filepath)).strip()
            return content if content else None
        return None

    def read_file(self, filename: str) -> Optional[str]:
        """Read workspace file (sync, for init-time use).

        Args:
            filename: File name (relative to workspace path)

        Returns:
            File content, or None if file doesn't exist or is empty
        """
        filepath = self.path / filename
        if filepath.exists() and filepath.is_file():
            content = filepath.read_text(encoding="utf-8").strip()
            return content if content else None
        return None

    def write_file(self, filename: str, content: str):
        """Write workspace file.

        Args:
            filename: File name (relative to workspace path)
            content: Content to write
        """
        filepath = self.path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    def append_file(self, filename: str, content: str):
        """Append content to workspace file.

        Args:
            filename: File name (relative to workspace path)
            content: Content to append
        """
        filepath = self.path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if filepath.exists():
            existing = filepath.read_text(encoding="utf-8").strip()

        new_content = f"{existing}\n\n{content}".strip() if existing else content
        filepath.write_text(new_content, encoding="utf-8")

    async def get_context_prompt(self) -> str:
        """Get workspace context (for injecting into System Prompt).

        Reads AGENTS.md, PERSONA.md, TOOLS.md file contents (globally shared),
        and user-specific USER.md file content.

        Also discovers AGENTS.md files along the directory chain from CWD up to
        the filesystem root (mirrors CC's multi-level CLAUDE.md merge).
        The merge order is: global ~/.agentica/AGENTS.md -> ancestor dirs
        (root first) -> CWD AGENTS.md -> workspace AGENTS.md.

        Returns:
            Merged context string
        """
        contents = []

        # 1. Prioritized AGENTS.md chain (global -> project -> local) with 40K budget
        chain_contents = self._load_agent_md_chain()
        if chain_contents:
            contents.append(f"<!-- Project AGENTS.md chain -->\n{chain_contents}")

        # 2. Workspace-level files (PERSONA.md, TOOLS.md) — skip empty scaffolds
        global_files = [
            self.config.persona_md,
            self.config.tools_md,
        ]
        for f in global_files:
            content = await self.read_file_async(f)
            if content and not self._is_empty_template(content):
                contents.append(f"<!-- {f} -->\n{content}")

        # 3. User-specific USER.md — skip empty scaffolds
        user_md_path = self._get_user_md()
        if user_md_path.exists():
            content = (await async_read_text(user_md_path)).strip()
            if content and not self._is_empty_template(content):
                contents.append(f"<!-- USER.md (user: {self._user_id}) -->\n{content}")

        return "\n\n---\n\n".join(contents) if contents else ""

    async def freeze_snapshots(self, query: str = "") -> None:
        """Freeze context + memory snapshots at session start.

        Once frozen, get_frozen_context() and get_frozen_memory() return the
        snapshot instead of re-reading from disk every turn. This keeps the
        system prompt prefix stable across turns, enabling LLM prompt cache
        hits (Hermes-style _system_prompt_snapshot pattern).

        Call once at session start. Memory tool writes update the live files
        on disk but do NOT mutate the frozen snapshot — the next session
        will pick up changes.
        """
        self._context_snapshot = await self.get_context_prompt()
        self._memory_snapshot = await self.get_relevant_memories(query=query)

    def get_frozen_context(self) -> Optional[str]:
        """Return frozen context snapshot, or None if not yet frozen."""
        return self._context_snapshot

    def get_frozen_memory(self) -> Optional[str]:
        """Return frozen memory snapshot, or None if not yet frozen."""
        return self._memory_snapshot

    # =========================================================================
    # Cross-product project config compatibility (Hermes-style)
    # =========================================================================

    # Project-level config files from other agent products, searched in CWD
    # and git root. First-match-wins per directory (like Hermes).
    # Only project-scoped files — we do NOT read ~/.claude/CLAUDE.md or
    # other HOME-level global configs (that's each product's own business).
    _PROJECT_CONFIG_NAMES: List[str] = [
        "AGENTS.md", "AGENT.md",       # Agentica / generic
        "CLAUDE.md", "claude.md",       # Claude Code
        ".cursorrules",                 # Cursor
    ]

    # Files whose full content is too noisy for the system prompt (developer
    # docs, architecture references). We surface only the path so the agent
    # can pull them on demand via read_file. AGENTS.md / AGENT.md stay as
    # full-content sources — those are agent behaviour rules, not docs.
    _PATH_ONLY_CONFIG_NAMES: frozenset = frozenset({
        "CLAUDE.md", "claude.md",
    })

    def _load_agent_md_chain(self) -> str:
        """Load prioritized AGENTS.md content with a 40K character budget."""
        sources = self._collect_agent_md_sources()
        if not sources:
            return ""

        selected = self._apply_agent_md_budget(sources, self.MAX_MEMORY_CHARACTER_COUNT)
        parts = [f"<!-- {path} -->\n{content}" for path, content in selected]
        return "\n\n---\n\n".join(parts) if parts else ""

    def _collect_agent_md_sources(self) -> List[Tuple[str, str]]:
        """Collect agent config sources from global, project, and workspace locations.

        Priority (lowest to highest):
        1. Global ~/.agentica/AGENTS.md
        2. Project directory chain (git root -> CWD), first-match-wins per dir
           Recognizes: AGENTS.md, CLAUDE.md, .cursorrules (cross-product compat)
        3. Workspace AGENTS.md
        """
        cwd = Path(os.getcwd())
        found: List[Tuple[str, str]] = []
        seen_paths: set[Path] = set()

        global_agent_md = Path(AGENTICA_HOME) / "AGENTS.md"
        if not global_agent_md.is_file():
            global_agent_md = Path(AGENTICA_HOME) / "AGENT.md"
        if global_agent_md.is_file():
            try:
                text = global_agent_md.read_text(encoding="utf-8").strip()
                if text and not self._is_empty_template(text):
                    resolved = global_agent_md.resolve()
                    found.append((str(global_agent_md), text))
                    seen_paths.add(resolved)
            except (OSError, UnicodeError) as exc:
                logger.debug("Skipping unreadable global agent file %s: %s", global_agent_md, exc)

        project_chain: List[Tuple[str, str]] = []
        visited = set()
        for dir_path in [cwd] + list(cwd.parents):
            resolved = dir_path.resolve()
            if resolved in visited:
                break
            visited.add(resolved)

            # First-match-wins per directory (Hermes-style priority)
            for name in self._PROJECT_CONFIG_NAMES:
                candidate = resolved / name
                if not candidate.is_file():
                    continue
                source_path = candidate.resolve()
                if source_path in seen_paths:
                    break
                if name in self._PATH_ONLY_CONFIG_NAMES:
                    # Don't inline developer docs; surface the path so the
                    # agent can pull them via read_file when relevant.
                    note = (
                        f"`{candidate}` is available for on-demand reading "
                        "(developer reference, not auto-loaded). "
                        "Use the read_file tool when its contents are needed."
                    )
                    project_chain.append((str(candidate), note))
                    seen_paths.add(source_path)
                    break
                try:
                    text = candidate.read_text(encoding="utf-8").strip()
                    if text:
                        project_chain.append((str(candidate), text))
                        seen_paths.add(source_path)
                except (OSError, UnicodeError) as exc:
                    logger.debug("Skipping unreadable project config %s: %s", candidate, exc)
                break  # first-match-wins: stop searching this directory

            if (resolved / ".git").exists():
                break

        project_chain.reverse()
        found.extend(project_chain)

        workspace_agent_md = self.path / self.config.agent_md
        if workspace_agent_md.is_file():
            try:
                workspace_content = workspace_agent_md.read_text(encoding="utf-8").strip()
                workspace_resolved = workspace_agent_md.resolve()
                if (
                    workspace_content
                    and workspace_resolved not in seen_paths
                    and not self._is_empty_template(workspace_content)
                ):
                    found.append((str(workspace_agent_md), workspace_content))
            except (OSError, UnicodeError) as exc:
                logger.debug("Skipping unreadable workspace agent file %s: %s", workspace_agent_md, exc)

        return found

    @staticmethod
    def _truncate_agent_md_content(content: str, max_chars: int) -> str:
        """Trim a single AGENTS.md file when it alone exceeds the remaining budget."""
        if max_chars <= 0:
            return ""
        if len(content) <= max_chars:
            return content
        if max_chars <= 32:
            return content[:max_chars]
        return content[: max_chars - 15].rstrip() + "\n\n[truncated]"

    def _apply_agent_md_budget(
        self,
        sources: List[Tuple[str, str]],
        max_chars: int,
    ) -> List[Tuple[str, str]]:
        """Apply a character budget while preserving the highest-priority AGENTS files."""
        selected_reversed: List[Tuple[str, str]] = []
        remaining = max_chars

        for path, content in reversed(sources):
            formatted = f"<!-- {path} -->\n{content}"
            if len(formatted) <= remaining:
                selected_reversed.append((path, content))
                remaining -= len(formatted)
                continue
            if not selected_reversed and remaining > 0:
                prefix_length = len(f"<!-- {path} -->\n")
                truncated = self._truncate_agent_md_content(content, remaining - prefix_length)
                if truncated:
                    selected_reversed.append((path, truncated))
                break
            break

        selected_reversed.reverse()
        return selected_reversed

    def get_git_context(self, max_status_lines: int = 30) -> Optional[str]:
        """Get git status context for system prompt injection.

        Returns branch, uncommitted changes, and recent commits.
        Returns None if not in a git repo or git is unavailable.
        """
        cwd = str(self.path)
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=cwd, capture_output=True, check=True, timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None

        parts = []
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if branch:
                parts.append(f"Git branch: {branch}")
        except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
            logger.debug("Failed to read git branch for %s: %s", cwd, exc)

        try:
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if status:
                lines = status.splitlines()
                if len(lines) > max_status_lines:
                    lines = lines[:max_status_lines] + [f"... ({len(lines) - max_status_lines} more)"]
                parts.append(f"Uncommitted changes:\n{chr(10).join(lines)}")
        except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
            logger.debug("Failed to read git status for %s: %s", cwd, exc)

        try:
            log = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if log:
                parts.append(f"Recent commits:\n{log}")
        except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
            logger.debug("Failed to read git log for %s: %s", cwd, exc)

        return "\n".join(parts) if parts else None

    # =========================================================================
    # Memory index constants (mirrors CC's MEMORY.md limits)
    # =========================================================================
    _MEMORY_INDEX_MAX_LINES: int = 200
    _MEMORY_INDEX_MAX_BYTES: int = 25_000

    # Injected after memory content to guard against stale references.
    _MEMORY_DRIFT_DEFENSE: str = (
        "Note: memories reflect the state at write time. "
        "If a memory references a specific file path, function, or flag, "
        "verify it still exists before recommending it."
    )
    _GLOBAL_AGENT_SYNC_START = "<!-- agentica:learned-preferences:start -->"
    _GLOBAL_AGENT_SYNC_END = "<!-- agentica:learned-preferences:end -->"
    _GLOBAL_AGENT_SYNC_HEADER = "## Learned Preferences"
    _GLOBAL_AGENT_SYNC_TYPES = {"user", "feedback"}
    MAX_MEMORY_CHARACTER_COUNT: int = AGENTICA_MAX_MEMORY_CHARACTER_COUNT
    _DURABLE_RULE_INCLUDE = re.compile(
        r"(?:\balways\b|\bnever\b|\bprefer\b|\bavoid\b|\bmust\b|\bshould\b|\buse\b|\bkeep\b|\brule\b|"
        r"\bstyle\b|\bpreference\b|\bformat\b|\bcommunication\b|"
        r"总是|永远|不要|避免|优先|必须|应该|尽量|禁止|风格|偏好|规则|格式|沟通)",
        re.IGNORECASE,
    )
    _DURABLE_RULE_EXCLUDE = re.compile(
        r"(?:```|https?://|\btraceback\b|\bstack trace\b|\berror code\b|\bcurrent task\b|"
        r"\bthis session\b|\btoday\b|\byesterday\b|\bcommit\b|"
        r"\brag\b|\bpipeline\b|\boracle\b|\bmrr\b|\bp@\d\b|\br@\d\b|\bf1\b|"
        r"\bprediction samples?\b|\btuning\b)",
        re.IGNORECASE,
    )

    def _get_global_agent_md_path(self) -> Path:
        """Return the user-global AGENTS.md path loaded into prompts."""
        global_home = Path(AGENTICA_HOME).expanduser()
        global_home.mkdir(parents=True, exist_ok=True)
        return global_home / "AGENTS.md"

    @staticmethod
    def _parse_frontmatter(content: str) -> Dict[str, str]:
        """Parse simple YAML frontmatter into a flat string dict."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, flags=re.DOTALL)
        if not match:
            return {}

        metadata: Dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    @staticmethod
    def _summarize_memory_for_global_agent(content: str, max_chars: int = 180) -> str:
        """Collapse a memory body into one readable line for global steering."""
        normalized = re.sub(r"\s+", " ", content).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _is_durable_global_preference(self, memory_type: str, metadata: Dict[str, str], content: str) -> bool:
        """Keep only concise, reusable rules in user-global AGENTS.md."""
        normalized = re.sub(r"\s+", " ", strip_frontmatter(content)).strip()
        if not normalized or len(normalized) > 240:
            return False
        combined = " ".join(
            part for part in [metadata.get("name", ""), metadata.get("description", ""), normalized] if part
        )
        if self._DURABLE_RULE_EXCLUDE.search(combined):
            return False
        if memory_type == "user":
            return True
        return bool(self._DURABLE_RULE_INCLUDE.search(combined))

    async def sync_memories_to_global_agent_md(self, limit: int = 30) -> str:
        """Compile user/feedback memories into ~/.agentica/AGENTS.md.

        This keeps long-lived preferences in the same global instruction chain
        that `get_context_prompt()` already loads on every run.
        """
        self._initialize_user_dir()
        memory_dir = self._get_user_memory_dir()

        synced_entries: List[str] = []
        if memory_dir.exists():
            memory_files = sorted(
                memory_dir.glob("*.md"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for memory_file in memory_files:
                raw = (await async_read_text(memory_file)).strip()
                if not raw:
                    continue

                metadata = self._parse_frontmatter(raw)
                memory_type = metadata.get("type", "project")
                if memory_type not in self._GLOBAL_AGENT_SYNC_TYPES:
                    continue
                if not self._is_durable_global_preference(memory_type, metadata, raw):
                    continue

                title = metadata.get("name", memory_file.stem)
                summary = self._summarize_memory_for_global_agent(strip_frontmatter(raw))
                if not summary:
                    continue

                synced_entries.append(f"- [{memory_type}] {title}: {summary}")
                if len(synced_entries) >= limit:
                    break

        # When there are no synced entries we still write an empty marker block
        # so subsequent runs can find/replace it, but we keep the block compact
        # (no placeholder bullet). The empty block is later stripped from prompt
        # injection by `_is_empty_template`.
        if synced_entries:
            block = "\n".join(
                [
                    self._GLOBAL_AGENT_SYNC_HEADER,
                    self._GLOBAL_AGENT_SYNC_START,
                    *synced_entries,
                    self._GLOBAL_AGENT_SYNC_END,
                ]
            )
        else:
            block = "\n".join(
                [
                    self._GLOBAL_AGENT_SYNC_HEADER,
                    self._GLOBAL_AGENT_SYNC_START,
                    self._GLOBAL_AGENT_SYNC_END,
                ]
            )

        global_agent_md = self._get_global_agent_md_path()
        existing = ""
        if global_agent_md.exists():
            existing = (await async_read_text(global_agent_md)).strip()

        if existing:
            pattern = (
                rf"{re.escape(self._GLOBAL_AGENT_SYNC_HEADER)}\n"
                rf"{re.escape(self._GLOBAL_AGENT_SYNC_START)}[\s\S]*?"
                rf"{re.escape(self._GLOBAL_AGENT_SYNC_END)}"
            )
            if re.search(pattern, existing):
                updated = re.sub(pattern, block, existing)
            else:
                updated = existing.rstrip() + "\n\n" + block
        else:
            updated = "# Agent Instructions\n\n" + block

        await async_write_text(global_agent_md, updated.strip() + "\n")
        return str(global_agent_md)

    async def get_relevant_memories(
        self,
        query: str = "",
        limit: int = 5,
        already_surfaced: Optional[set] = None,
    ) -> str:
        """Load MEMORY.md index, score entries against query, return top-k content.

        Implements CC-style relevance-based recall instead of dumping all memory:
        - Parses MEMORY.md as an index of entry links
        - Scores each entry against the query with keyword overlap
        - Loads only the top-k most relevant entry files
        - Appends a drift-defense note to guard against stale references

        Falls back to loading all entries when query is empty (same as before).

        Args:
            query: Current user query (used for relevance scoring)
            limit: Maximum number of memory entries to return
            already_surfaced: Set of filenames already shown this session (dedup)

        Returns:
            Formatted memory string ready for system prompt injection, or empty string.
        """
        self._initialize_user_dir()
        index_path = self._get_user_memory_md()
        memory_dir = self._get_user_memory_dir()

        if not index_path.exists() and not memory_dir.exists():
            return ""

        # --- Parse MEMORY.md index ---
        index_entries: List[Dict] = []
        if index_path.exists():
            index_content = (await async_read_text(index_path)).strip()
            if index_content:
                index_entries = self._parse_memory_index(index_content)

        # --- If no structured index exists, fall back to listing memory dir files ---
        if not index_entries and memory_dir.exists():
            for f in sorted(memory_dir.glob("*.md"), reverse=True):
                index_entries.append({
                    "title": f.stem,
                    "filename": f.name,
                    "hook": f.stem.replace("_", " "),
                })

        if not index_entries:
            return ""

        # --- Filter already-surfaced entries (avoid repeating in same session) ---
        if already_surfaced:
            index_entries = [e for e in index_entries if e["filename"] not in already_surfaced]

        if not index_entries:
            return ""

        # --- Score entries against query ---
        if query.strip():
            scored = self._score_memory_entries(query, index_entries)
        else:
            # No query: take the most recent entries (already sorted by recency from glob)
            scored = index_entries[:limit]

        top_entries = scored[:limit]

        # --- Load file content for selected entries ---
        parts = []
        for entry in top_entries:
            content_path = memory_dir / entry["filename"]
            if content_path.exists():
                raw = (await async_read_text(content_path)).strip()
                # Strip frontmatter (---...---) before injecting
                body = strip_frontmatter(raw)
                if body:
                    parts.append(f"### {entry['title']}\n\n{body}")
                    # Write back to already_surfaced for session-level dedup
                    if already_surfaced is not None:
                        already_surfaced.add(entry["filename"])

        if not parts:
            return ""

        result = "\n\n".join(parts)
        result += f"\n\n*{self._MEMORY_DRIFT_DEFENSE}*"
        return result

    # ── arch_v5.md §"Evidence Gate" ───────────────────────────────────
    # Sources allowed to write directly into the canonical memory folder.
    # Anything else lands in `memory_candidates/` until promoted.
    _MEMORY_TRUSTED_SOURCES = {"verified", "manual", "user_confirmed"}

    async def write_memory_entry(
        self,
        title: str,
        content: str,
        memory_type: str = "project",
        description: str = "",
        sync_to_global_agent_md: bool = False,
        *,
        source: str = "verified",
        evidence_refs: Optional[List[str]] = None,
    ) -> str:
        """Write a typed memory entry as an individual file and update MEMORY.md index.

        Each entry gets its own .md file under users/{user_id}/memory/ with a
        YAML frontmatter header (name, description, type). The MEMORY.md index
        is updated with a single-line reference to the new file.

        The description field is the key relevance signal — it should contain
        searchable keywords that identify when this memory is relevant.

        Evidence gate (arch_v5.md Phase 2):
            Entries with `source` outside `_MEMORY_TRUSTED_SOURCES` are written
            to `memory_candidates/` instead of the canonical memory folder, so
            unverified LLM-extracted content cannot pollute long-term memory.
            Trusted sources include `verified` (default), `manual`, and
            `user_confirmed`. Pass `source="auto_extract"` (or any other value)
            to route writes through the candidate quarantine.

        Args:
            title: Short display name for the memory
            content: Full memory content (why + how to apply)
            memory_type: One of "user", "feedback", "project", "reference"
            description: One-line hook for relevance scoring (defaults to title)
            sync_to_global_agent_md: If True, recompile synced memories into
                ~/.agentica/AGENTS.md after this write. Ignored for candidates.
            source: Provenance string — controls which folder the entry lands in.
            evidence_refs: Optional list of supporting references (file paths,
                URLs, run_ids). Persisted in the frontmatter so reviewers can
                trace WHY this memory was written.

        Returns:
            Absolute path to the written memory file.
        """
        self._initialize_user_dir()

        is_trusted = source in self._MEMORY_TRUSTED_SOURCES
        if is_trusted:
            target_dir = self._get_user_memory_dir()
        else:
            target_dir = self._get_user_memory_candidates_dir()
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r"[^\w\-]", "_", title.lower())[:50].strip("_")
        filename = f"{memory_type}_{safe_title}.md"
        filepath = target_dir / filename

        hook = description or title
        evidence_lines = ""
        if evidence_refs:
            cleaned = [str(r) for r in evidence_refs if r]
            if cleaned:
                # JSON encode so values containing YAML-special chars (`#`, `:`,
                # `[`, ...) survive parsing. JSON arrays of strings are valid YAML
                # flow sequences, so the frontmatter remains parseable by both.
                evidence_lines = (
                    f"evidence_refs: {json.dumps(cleaned, ensure_ascii=False)}\n"
                )
        frontmatter = (
            f"---\nname: {title}\n"
            f"description: {hook}\n"
            f"type: {memory_type}\n"
            f"source: {source}\n"
            f"{evidence_lines}"
            f"---\n\n"
        )
        await async_write_text(filepath, frontmatter + content)

        if is_trusted:
            # M7 fix: a verified write supersedes any quarantined candidate of
            # the same name. Drop the stale candidate so the workspace doesn't
            # accumulate ghost duplicates after a manual promotion.
            candidate_dup = self._get_user_memory_candidates_dir() / filename
            if candidate_dup.exists() and candidate_dup != filepath:
                try:
                    candidate_dup.unlink()
                except OSError as e:
                    logger.warning(
                        f"failed to drop superseded candidate {candidate_dup}: {e}"
                    )
            await self._update_memory_index(
                index_path=self._get_user_memory_md(),
                filename=filename,
                title=title,
                hook=hook,
            )
            if sync_to_global_agent_md and memory_type in self._GLOBAL_AGENT_SYNC_TYPES:
                await self.sync_memories_to_global_agent_md()
        else:
            logger.debug(
                f"memory entry quarantined to candidates (source={source}): {filepath}"
            )

        return str(filepath)

    # ── arch_v5.md §"Evidence Gate" — candidate review API ────────────
    # Memory candidates accumulate when LLM-extracted entries fail the
    # evidence gate. These helpers let an operator (or a future review UI)
    # list, promote, or reject them so the quarantine doesn't grow forever.

    def list_memory_candidates(self) -> List[Dict[str, Any]]:
        """List all quarantined memory candidate files for the current user.

        Returns one dict per candidate:
            {filename, path, name, type, source, mtime, evidence_refs}
        Frontmatter is parsed best-effort: malformed candidates still appear
        but with empty metadata so reviewers can see and clean them up.
        """
        cand_dir = self._get_user_memory_candidates_dir()
        out: List[Dict[str, Any]] = []
        for p in sorted(cand_dir.glob("*.md")):
            try:
                raw = p.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(f"unable to read memory candidate {p}: {e}")
                continue
            name = extract_frontmatter_value(raw, "name") or p.stem
            mtype = extract_frontmatter_value(raw, "type") or ""
            source = extract_frontmatter_value(raw, "source") or ""
            ev_raw = extract_frontmatter_value(raw, "evidence_refs") or ""
            evidence: List[str] = []
            ev_raw = ev_raw.strip()
            if ev_raw:
                try:
                    parsed = json.loads(ev_raw)
                    if isinstance(parsed, list):
                        evidence = [str(x) for x in parsed]
                except (ValueError, TypeError):
                    pass
            out.append({
                "filename": p.name,
                "path": str(p),
                "name": name,
                "type": mtype,
                "source": source,
                "mtime": p.stat().st_mtime,
                "evidence_refs": evidence,
            })
        return out

    async def promote_memory_candidate(
        self,
        filename: str,
        sync_to_global_agent_md: bool = False,
    ) -> Optional[str]:
        """Promote a quarantined candidate into the canonical memory folder.

        Reads the candidate's body and frontmatter, then re-writes via
        `write_memory_entry(source="user_confirmed")`. The original candidate
        file is removed by `write_memory_entry` (M7 cleanup) so each entry
        ends up in exactly one place.

        Args:
            filename: bare filename inside `memory_candidates/` (e.g.
                `feedback_python_style.md`).
            sync_to_global_agent_md: forward to the underlying writer.

        Returns:
            Absolute path to the canonical entry, or None if the candidate
            doesn't exist or its body is empty.
        """
        cand_path = self._get_user_memory_candidates_dir() / filename
        if not cand_path.exists():
            return None

        raw = await async_read_text(cand_path)
        title = extract_frontmatter_value(raw, "name") or cand_path.stem
        mtype = extract_frontmatter_value(raw, "type") or "project"
        description = extract_frontmatter_value(raw, "description") or ""
        ev_raw = (extract_frontmatter_value(raw, "evidence_refs") or "").strip()
        evidence: Optional[List[str]] = None
        if ev_raw:
            try:
                parsed = json.loads(ev_raw)
                if isinstance(parsed, list):
                    evidence = [str(x) for x in parsed]
            except (ValueError, TypeError):
                evidence = None

        body = strip_frontmatter(raw).strip()
        if not body:
            logger.warning(f"refusing to promote empty candidate: {cand_path}")
            return None

        return await self.write_memory_entry(
            title=title,
            content=body,
            memory_type=mtype,
            description=description,
            sync_to_global_agent_md=sync_to_global_agent_md,
            source="user_confirmed",
            evidence_refs=evidence,
        )

    def reject_memory_candidate(self, filename: str) -> bool:
        """Permanently delete a quarantined candidate. Idempotent.

        Returns True if a file was deleted, False if it didn't exist. Other
        OS errors raise -- a failed delete on a present file is a real bug
        the operator needs to see.
        """
        cand_path = self._get_user_memory_candidates_dir() / filename
        if not cand_path.exists():
            return False
        cand_path.unlink()
        return True

    async def _update_memory_index(
        self,
        index_path: Path,
        filename: str,
        title: str,
        hook: str,
    ) -> None:
        """Append or update an entry in MEMORY.md index, enforcing size limits.

        Format: `- [Title](memory/filename.md) — one-line hook`
        Limits: 200 lines / 25KB (CC convention). Oldest entries are evicted.
        """
        new_entry = f"- [{title}](memory/{filename}) — {hook[:100]}"

        existing = ""
        if index_path.exists():
            existing = (await async_read_text(index_path)).strip()

        lines = [l for l in existing.splitlines() if l.strip()] if existing else []

        # Remove existing entry for this file (update case)
        lines = [l for l in lines if f"(memory/{filename})" not in l]
        lines.append(new_entry)

        # Enforce hard limits: evict oldest entries from the front
        while len(lines) > self._MEMORY_INDEX_MAX_LINES:
            lines.pop(0)

        content = "\n".join(lines)
        while len(content.encode("utf-8")) > self._MEMORY_INDEX_MAX_BYTES:
            if not lines:
                break
            lines.pop(0)
            content = "\n".join(lines)

        await async_write_text(index_path, content)

    def _parse_memory_index(self, index_content: str) -> List[Dict]:
        """Parse MEMORY.md index lines into entry dicts.

        Expected format: `- [Title](memory/filename.md) — one-line hook`
        """
        entries = []
        for line in index_content.splitlines():
            m = re.match(r"-\s+\[(.+?)\]\(memory/(.+?)\)\s*[—\-]\s*(.+)", line)
            if m:
                entries.append({
                    "title": m.group(1).strip(),
                    "filename": m.group(2).strip(),
                    "hook": m.group(3).strip(),
                })
        return entries

    @staticmethod
    def compute_relevance_score(query_lower: str, text_lower: str) -> float:
        """Compute relevance score using hybrid word + character bigram matching.

        Supports both English (word-level) and CJK (character bigram) queries.

        Args:
            query_lower: Lowercased query string
            text_lower: Lowercased text to match against

        Returns:
            Relevance score (0.0 = no match, higher = better match)
        """
        word_tokens = set(query_lower.split())
        char_bigrams: set = set()
        for i in range(len(query_lower) - 1):
            bigram = query_lower[i:i + 2].strip()
            if bigram:
                char_bigrams.add(bigram)

        if not word_tokens and not char_bigrams:
            return 0.0

        score = 0.0
        if word_tokens:
            word_hits = sum(1.0 for w in word_tokens if w in text_lower)
            score += word_hits / len(word_tokens)
        if char_bigrams:
            ngram_hits = sum(1.0 for ng in char_bigrams if ng in text_lower)
            score += 0.5 * ngram_hits / len(char_bigrams)
        return score

    def _score_memory_entries(self, query: str, entries: List[Dict]) -> List[Dict]:
        """Score memory entries by token overlap with query.

        Returns entries sorted by score descending. Entries with score=0 are
        included at the end (ensures fallback when no token matches).
        """
        query_lower = query.lower()
        scored = []
        for entry in entries:
            text = f"{entry['title']} {entry['hook']}".lower()
            score = self.compute_relevance_score(query_lower, text)
            scored.append({**entry, "_score": score})

        scored.sort(key=lambda x: -x["_score"])
        return scored

    async def write_memory(self, content: str, to_daily: bool = True):
        """Write memory content. Delegates to write_memory_entry() for indexed storage.

        For backward compatibility. New code should use write_memory_entry() directly.

        Args:
            content: Memory content
            to_daily: Ignored (kept for API compatibility). All entries go to memory/ dir.
        """
        # Derive a title from the first 50 chars of content
        title = content[:50].strip().replace("\n", " ")
        if not title:
            title = "untitled"
        await self.write_memory_entry(
            title=title,
            content=content,
            memory_type="project",
            description=title,
        )

    async def save_memory(self, content: str, long_term: bool = False):
        """Save memory (alias for write_memory, kept for backward compatibility).

        Args:
            content: Memory content
            long_term: Ignored (kept for API compatibility).
        """
        await self.write_memory(content)

    def get_skills_dir(self) -> Path:
        """Get skills directory path.

        Returns:
            Absolute path to skills directory
        """
        return self.path / self.config.skills_dir

    def list_files(self) -> Dict[str, bool]:
        """List workspace global file status.

        Returns:
            Dictionary with file names as keys and existence status as values.
            Note: Only lists globally shared files, not user-specific files.
        """
        # Only list globally shared config files
        files = [
            self.config.agent_md,
            self.config.persona_md,
            self.config.tools_md,
        ]
        return {f: (self.path / f).exists() for f in files}

    def get_all_memory_files(self) -> List[Path]:
        """Get all memory file paths for current user.

        Returns:
            List of all memory file paths
        """
        files = []

        # Long-term memory
        memory_md = self._get_user_memory_md()
        if memory_md.exists():
            files.append(memory_md)

        # Daily memory
        memory_dir = self._get_user_memory_dir()
        if memory_dir.exists():
            files.extend(sorted(memory_dir.glob("*.md"), reverse=True))

        return files

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> List[Dict]:
        """Search memory with hybrid word + character n-gram matching.

        Uses a combination of word-level matching (for English and space-delimited
        languages) and character bigram matching (for CJK languages like Chinese)
        to support multilingual queries.

        Args:
            query: Search query (supports English, Chinese, and mixed)
            limit: Maximum number of results
            min_score: Minimum match score threshold

        Returns:
            List of matching memories, each containing content, file_path, score
        """
        query_lower = query.lower()
        if not query_lower.strip():
            return []

        results = []
        for file_path in self.get_all_memory_files():
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            score = self.compute_relevance_score(query_lower, content.lower())

            if score >= min_score:
                results.append({
                    "content": content,
                    "file_path": str(file_path.relative_to(self.path)),
                    "score": round(score, 4),
                })

        results.sort(key=lambda x: -x["score"])
        return results[:limit]

    def clear_daily_memory(self, keep_days: int = 7):
        """Clear old daily memory files (date-pattern only).

        Only deletes files matching YYYY-MM-DD.md pattern. Typed memory entry
        files (e.g. user_role.md, project_deploy.md) are never deleted.

        Args:
            keep_days: Number of most recent date files to keep
        """
        memory_dir = self._get_user_memory_dir()
        if not memory_dir.exists():
            return

        # Only match date-pattern files: YYYY-MM-DD.md
        date_files = sorted(
            [f for f in memory_dir.glob("*.md") if re.match(r"\d{4}-\d{2}-\d{2}\.md$", f.name)],
            reverse=True,
        )
        for f in date_files[keep_days:]:
            f.unlink()

    # =========================================================================
    # Conversation Archive
    # =========================================================================

    def _get_user_conversations_dir(self) -> Path:
        """Get current user's conversation archive directory."""
        return self._get_user_path() / self.config.conversations_dir

    def _get_archive_lock(self, filepath: Path) -> asyncio.Lock:
        """Get or create a per-file asyncio.Lock for serializing archive writes."""
        key = str(filepath)
        # Use setdefault for atomic get-or-create (W-01 fix)
        return self._archive_locks.setdefault(key, asyncio.Lock())

    async def archive_conversation(self, messages: List[Dict], session_id: Optional[str] = None) -> str:
        """Archive a conversation to daily Markdown file.

        Messages are appended to users/{user_id}/conversations/YYYY-MM-DD.md.
        Uses per-file locking to prevent concurrent write-write races.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            session_id: Optional session identifier for grouping

        Returns:
            Path to the archive file
        """
        self._initialize_user_dir()
        conv_dir = self._get_user_conversations_dir()
        conv_dir.mkdir(parents=True, exist_ok=True)

        today = date.today().isoformat()
        filepath = conv_dir / f"{today}.md"

        now = datetime.now().strftime("%H:%M:%S")
        header = f"\n\n---\n\n### {now}"
        if session_id:
            header += f" (session: {session_id})"
        header += "\n\n"

        lines = [header]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue
            content = redact_sensitive_text(content)
            # Truncate very long messages in archive
            if len(content) > 2000:
                content = content[:2000] + "\n...[truncated]"
            lines.append(f"**{role}**: {content}\n\n")

        archive_text = "".join(lines)

        # Use per-file lock to serialize concurrent writes
        lock = self._get_archive_lock(filepath)
        async with lock:
            existing = ""
            if filepath.exists():
                existing = (await async_read_text(filepath)).strip()
            new_content = f"{existing}{archive_text}".strip() if existing else archive_text.strip()
            await async_write_text(filepath, new_content)

        return str(filepath)

    def search_conversations(
        self,
        query: str,
        limit: int = 10,
        max_files: Optional[int] = None,
    ) -> List[Dict]:
        """Search conversation archive by keyword.

        Args:
            query: Search query (keyword matching)
            limit: Maximum number of matching blocks to return
            max_files: Only search the most recent N archive files (None = search all)

        Returns:
            List of matching conversation blocks with date, content, score
        """
        conv_dir = self._get_user_conversations_dir()
        if not conv_dir.exists():
            return []

        files = sorted(conv_dir.glob("*.md"), reverse=True)
        if max_files is not None:
            files = files[:max_files]

        query_lower = query.lower()
        query_words = query_lower.split()
        results = []

        for filepath in files:
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue

            # Split into conversation blocks by ---
            blocks = content.split("---")
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                block_lower = block.lower()
                score = sum(1.0 for w in query_words if w in block_lower) / max(len(query_words), 1)
                if score > 0:
                    results.append({
                        "date": filepath.stem,
                        "content": block[:500] + ("..." if len(block) > 500 else ""),
                        "file_path": str(filepath.relative_to(self.path)),
                        "score": score,
                    })

        results.sort(key=lambda x: -x["score"])
        return results[:limit]

    def get_conversation_files(self, max_files: Optional[int] = None) -> List[Path]:
        """Get conversation archive files for current user.

        Args:
            max_files: Only return the most recent N files (None = return all)

        Returns:
            List of conversation file paths, newest first
        """
        conv_dir = self._get_user_conversations_dir()
        if not conv_dir.exists():
            return []
        files = sorted(conv_dir.glob("*.md"), reverse=True)
        if max_files is not None:
            files = files[:max_files]
        return files

    # =========================================================================
    # Experience System (self-evolution) — delegates to experience package
    # =========================================================================

    _EXPERIENCE_INDEX_FILE = "EXPERIENCE.md"
    _EXPERIENCE_DIR = "experiences"

    def _get_user_experience_dir(self) -> Path:
        """Get current user's experience directory."""
        return self._get_user_path() / self._EXPERIENCE_DIR

    def _get_user_experience_md(self) -> Path:
        """Get current user's experience index file path."""
        return self._get_user_path() / self._EXPERIENCE_INDEX_FILE

    def _get_user_generated_skills_dir(self) -> Path:
        """Get current user's generated skills directory."""
        return self._get_user_path() / "generated_skills"

    def get_experience_event_store(self):
        """Get the ExperienceEventStore for the current user.

        Returns:
            ExperienceEventStore instance pointing at users/{user_id}/experiences/.
        """
        from agentica.experience.event_store import ExperienceEventStore
        self._initialize_user_dir()
        return ExperienceEventStore(self._get_user_experience_dir())

    def get_compiled_experience_store(self):
        """Get the CompiledExperienceStore for the current user.

        Returns:
            CompiledExperienceStore instance with relevance scorer from Workspace.
        """
        from agentica.experience.compiled_store import CompiledExperienceStore
        self._initialize_user_dir()
        return CompiledExperienceStore(
            exp_dir=self._get_user_experience_dir(),
            index_path=self._get_user_experience_md(),
            relevance_scorer=self.compute_relevance_score,
        )

    # ── Backward-compatible delegation methods ────────────────────────────

    async def get_relevant_experiences(
        self,
        query: str = "",
        limit: int = 5,
    ) -> str:
        """Retrieve top-k experiences for system prompt injection.

        Delegates to CompiledExperienceStore.

        Args:
            query: Current user query for relevance scoring
            limit: Maximum number of experiences to return

        Returns:
            Formatted markdown string, or empty string.
        """
        store = self.get_compiled_experience_store()
        return await store.get_relevant(query=query, limit=limit)

    # Frontmatter helpers delegate to shared utils
    _extract_frontmatter_value = staticmethod(extract_frontmatter_value)
    _extract_frontmatter_int = staticmethod(extract_frontmatter_int)

    def __repr__(self) -> str:
        return f"Workspace(path={self.path}, exists={self.exists()}, user_id={self._user_id})"

    def __str__(self) -> str:
        return str(self.path)

    def list_users(self) -> List[str]:
        """List all registered user IDs.

        Returns:
            List of user IDs
        """
        users_dir = self.path / self.config.users_dir
        if not users_dir.exists():
            return []

        users = []
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                users.append(user_dir.name)
        return sorted(users)

    def get_user_info(self, user_id: Optional[str] = None) -> Dict:
        """Get user information summary.

        Args:
            user_id: User ID, uses current user if not specified

        Returns:
            User info dictionary containing user_id, memory_count, last_activity, etc.
        """
        target_user = user_id or self._user_id
        old_user = self._user_id

        try:
            self._user_id = target_user

            memory_files = self.get_all_memory_files()
            memory_count = len(memory_files)

            last_activity = None
            if memory_files:
                # Get modification time of latest memory file
                latest_file = memory_files[0]
                if latest_file.exists():
                    mtime = latest_file.stat().st_mtime
                    last_activity = datetime.fromtimestamp(mtime).isoformat()

            return {
                "user_id": target_user,
                "memory_count": memory_count,
                "last_activity": last_activity,
                "user_path": str(self._get_user_path()),
            }
        finally:
            self._user_id = old_user

    def delete_user(self, user_id: str, confirm: bool = False) -> bool:
        """Delete user data.

        Args:
            user_id: User ID to delete
            confirm: Must be set to True to execute deletion

        Returns:
            Whether deletion was successful
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete user data")

        if not user_id:
            raise ValueError("user_id cannot be empty")

        safe_user_id = user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        user_path = self.path / self.config.users_dir / safe_user_id

        if not user_path.exists():
            return False

        shutil.rmtree(user_path)
        return True
