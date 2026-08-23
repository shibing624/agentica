# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Workspace management for Agentica agents.
Inspired by OpenClaw's workspace concept.
"""
import asyncio
import threading
import json
import os
import re
import shutil
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
        agent_md: Per-user instruction file name (``users/{id}/AGENTS.md``)
        memory_md: Long-term memory file name
        memory_dir: Daily memory directory name
        skills_dir: Skills directory name
        users_dir: User data directory name (for multi-user isolation)
    """
    agent_md: str = "AGENTS.md"
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
    - skills/: Custom skills directory (globally shared)
    - users/: User data directory (all users including default)
        - default/: Default user (when no user_id specified)
            - AGENTS.md: this user's own standing instructions
            - MEMORY.md: Long-term memory
            - memory/: Daily memory directory
        - {user_id}/: Other users
            - AGENTS.md: this user's own standing instructions
            - MEMORY.md: Long-term memory
            - memory/: Daily memory directory

    There is no workspace-root ``AGENTS.md``. Standing rules live in
    ``users/{user_id}/AGENTS.md``; project rules live in the repo-root chain.

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

    # Sentinel for the "single-user CLI / local install" mode. The default
    # user's real rules still live under users/default/AGENTS.md, but we expose
    # ~/.agentica/AGENTS.md as a symlink for mainstream coding-agent
    # compatibility. Other user_id values never use that home-global alias, so
    # tenant-A preferences do not leak into tenant-B's prompts.
    DEFAULT_USER_ID = "default"

    @staticmethod
    def sanitize_user_id(user_id: Optional[str]) -> str:
        """Sanitize a user_id into a path-safe segment.

        Single source of truth for the "(user_id) -> directory segment" mapping.
        Used by every component that builds a per-user filesystem path
        (Workspace itself, tool_result_storage, etc.) so two callers can never
        diverge on the encoding of e.g. ``"alice/bob"``.

        Returns ``DEFAULT_USER_ID`` for None / empty / whitespace-only input.
        """
        uid = (user_id or "").strip() if user_id else ""
        if not uid:
            return Workspace.DEFAULT_USER_ID
        return uid.replace("/", "_").replace("\\", "_").replace("..", "_")

    # Scaffold for users/{user_id}/AGENTS.md — this user's own instructions,
    # hand-edited or appended to by the agent with its ordinary file tools.
    # Intentionally empty of behavioural rules: boilerplate in the scaffold
    # would pollute every system prompt with zero signal.
    DEFAULT_USER_AGENT_MD = """# User Instructions ({user_id})

<!-- Who this user is, how they want you to work, standing rules. -->
<!-- Loaded into the system prompt of every session of this user. -->
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
        self._user_id = user_id if user_id else self.DEFAULT_USER_ID
        # Per-file locks for concurrent archive writes
        self._archive_locks: Dict[str, asyncio.Lock] = {}
        # Guards the (_user_id, _user_initialized) tuple so it flips
        # atomically when set_user() is called. Does NOT protect against
        # caller-side races (write captures old user_id then set_user
        # changes it before the write reaches the filesystem) — see the
        # set_user() docstring for the multi-tenant guidance.
        self._user_switch_lock = threading.Lock()
        # Flag to avoid redundant _initialize_user_dir calls
        self._user_initialized: bool = False
        # Frozen snapshots for prompt cache stability (Hermes-style)
        self._context_snapshot: Optional[str] = None
        self._memory_snapshot: Optional[str] = None
        self._experience_snapshot: Optional[str] = None

    @property
    def user_id(self) -> str:
        """Get current user ID."""
        return self._user_id

    def set_user(self, user_id: Optional[str]):
        """Set current user ID.

        Multi-tenant safety: ``set_user`` mutates shared instance state
        (``_user_id`` + ``_user_initialized``). Two concurrent requests
        switching users on the SAME Workspace instance can interleave and
        cause writes to land in the wrong user's directory. In SaaS-style
        deployments **instantiate ``Workspace(user_id=...)`` per request
        instead of sharing one Workspace across users**. The internal lock
        below only guarantees that ``_user_id`` and ``_user_initialized``
        flip together — it cannot protect a write that already captured
        the old user_id.

        Args:
            user_id: User ID, defaults to 'default' if None
        """
        new_id = user_id if user_id else self.DEFAULT_USER_ID
        with self._user_switch_lock:
            if new_id != self._user_id:
                self._user_initialized = False
            self._user_id = new_id

    def _get_user_path(self) -> Path:
        """Get current user's data directory path.

        Returns:
            Path to users/{user_id}/ directory
        """
        safe_user_id = self.sanitize_user_id(self._user_id)
        return self.path / self.config.users_dir / safe_user_id

    def _get_user_memory_dir(self) -> Path:
        """Get current user's daily memory directory."""
        return self._get_user_path() / self.config.memory_dir

    def _get_user_memory_md(self) -> Path:
        """Get current user's long-term memory file path."""
        return self._get_user_path() / self.config.memory_md

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

    def initialize(self) -> bool:
        """Initialize workspace directories and this user's scaffold files.

        Never overwrites an existing file: the only file written here is a
        missing ``users/{id}/AGENTS.md`` scaffold. There used to be a ``force``
        flag for rewriting the workspace-root ``AGENTS.md`` template; that file
        is gone, so the flag governed nothing.

        Returns:
            Whether initialization was successful
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Create global directories (no workspace-root AGENTS.md — that role is
        # users/{id}/AGENTS.md for standing rules and the project chain for
        # repo rules).
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

        # Create this user's own AGENTS.md. user_agent_md_path() also reconciles
        # the default user's ~/.agentica/AGENTS.md compatibility symlink, so it
        # must run before the scaffold is written — otherwise a legacy file can
        # be stranded behind the freshly-created target.
        user_agent_md = self.user_agent_md_path()
        if not user_agent_md.exists():
            user_agent_md.write_text(
                self.DEFAULT_USER_AGENT_MD.format(user_id=self._user_id),
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
            Whether the workspace directory and its ``users/`` tree exist.
        """
        return self.path.exists() and (self.path / self.config.users_dir).is_dir()

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

        Everything comes from one kind of file, AGENTS.md, read from two kinds
        of place: this user's own (``users/{user_id}/AGENTS.md``) and the
        project chain from CWD up to the repo root (also recognizing other
        products' CLAUDE.md / .cursorrules). There is no workspace-root
        AGENTS.md.

        Returns:
            Merged context string
        """
        chain_contents = self._load_agent_md_chain()
        if not chain_contents:
            return ""
        snapshot_note = (
            "Note: This AGENTS.md context is a session-start snapshot for "
            "prompt-cache stability. If these rules change during this session, "
            "use the read_file tool on the file paths shown below when the latest "
            "content is needed; new sessions load the latest content automatically."
        )
        return f"<!-- AGENTS.md chain -->\n{snapshot_note}\n\n{chain_contents}"

    async def freeze_snapshots(self, query: str = "") -> None:
        """Freeze context + memory + experience snapshots at session start.

        Once frozen, get_frozen_context() / get_frozen_memory() /
        get_frozen_experiences() return the snapshot instead of re-reading from
        disk every turn. This keeps the system prompt prefix stable across
        turns, enabling LLM prompt cache hits (Hermes-style
        _system_prompt_snapshot pattern).

        Experiences are frozen for the same reason and are the one that used to
        break it: the capture hooks write new cards *during* the session (tool
        errors, user corrections, the batched judge), so retrieval returned a
        different set mid-conversation and re-priced every request behind it.
        The way back to the current set is ``experience_index_path``, which the
        prompt names — one file read, only when the agent actually wants it.

        Call once at session start. Memory / experience writes update the live
        files on disk but do NOT mutate the frozen snapshot — the next session
        will pick up changes. ``query`` is only used for experience recall;
        the memory snapshot is the MEMORY.md index (hooks), not topic bodies.
        """
        self._context_snapshot = await self.get_context_prompt()
        self._memory_snapshot = await self.get_memory_index_prompt()
        self._experience_snapshot = await self.get_relevant_experiences(query=query)

    def get_frozen_context(self) -> Optional[str]:
        """Return frozen context snapshot, or None if not yet frozen."""
        return self._context_snapshot

    def get_frozen_memory(self) -> Optional[str]:
        """Return frozen memory snapshot, or None if not yet frozen."""
        return self._memory_snapshot

    def get_frozen_experiences(self) -> Optional[str]:
        """Return frozen experience snapshot, or None if not yet frozen.

        An empty string means "frozen, and there was nothing to inject" — the
        caller must not read that as "not frozen" and fall back to a live read.
        """
        return self._experience_snapshot

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
        """Collect AGENTS.md sources: this user first, then the project chain.

        Order is budget priority (earlier entries keep the character budget):
        1. This user's own ``users/{user_id}/AGENTS.md``
        2. Project directory chain (git root -> CWD), first-match-wins per dir
           Recognizes: AGENTS.md, CLAUDE.md, .cursorrules (cross-product compat)

        A workspace-root ``AGENTS.md`` is intentionally not read: that file
        used to inject shared scaffolding into every tenant's prompt, and
        standing rules already have a home under ``users/``.
        """
        cwd = Path(os.getcwd())
        found: List[Tuple[str, str]] = []
        seen_paths: set[Path] = set()

        # Single source of truth for "this user's own AGENTS.md".
        user_agent_md = self.user_agent_md_path()
        if user_agent_md.is_file():
            try:
                text = user_agent_md.read_text(encoding="utf-8").strip()
                if text and not self._is_empty_template(text):
                    found.append((str(user_agent_md), text))
                    seen_paths.add(user_agent_md.resolve())
            except (OSError, UnicodeError) as exc:
                logger.debug("Skipping unreadable user agent file %s: %s", user_agent_md, exc)

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
        """Apply a character budget, keeping earlier (higher-priority) sources first.

        ``_collect_agent_md_sources`` puts the user's AGENTS.md first, so a
        tight budget truncates or drops the project chain — never the user's
        standing rules — unless the user file alone exceeds the budget.
        """
        selected: List[Tuple[str, str]] = []
        remaining = max_chars

        for path, content in sources:
            formatted = f"<!-- {path} -->\n{content}"
            if len(formatted) <= remaining:
                selected.append((path, content))
                remaining -= len(formatted)
                continue
            if not selected and remaining > 0:
                prefix_length = len(f"<!-- {path} -->\n")
                truncated = self._truncate_agent_md_content(content, remaining - prefix_length)
                if truncated:
                    selected.append((path, truncated))
                break
            break

        return selected

    # =========================================================================
    # Memory index constants (mirrors CC's MEMORY.md limits)
    # =========================================================================
    _MEMORY_INDEX_MAX_LINES: int = 200
    _MEMORY_INDEX_MAX_BYTES: int = 25_000
    # Prompt copy is tighter than the on-disk index. Full file: memory_index_path().
    _MEMORY_INJECT_MAX_LINES: int = 60
    _MEMORY_INJECT_MAX_BYTES: int = 4_000

    # Injected after memory content to guard against stale references.
    _MEMORY_DRIFT_DEFENSE: str = (
        "Note: memories reflect the state at write time. "
        "If a memory references a specific file path, function, or flag, "
        "verify it still exists before recommending it."
    )
    MAX_MEMORY_CHARACTER_COUNT: int = AGENTICA_MAX_MEMORY_CHARACTER_COUNT
    def user_agent_md_path(self) -> Path:
        """Return this user's own AGENTS.md — ``users/{user_id}/AGENTS.md``.

        There is exactly one canonical layout, and the CLI is not an exception
        to it: it is simply the ``default`` user. For compatibility with coding
        agents that discover user-level instructions at ``~/.agentica/AGENTS.md``,
        the default user also gets that path as a symlink to the canonical file.

        Public because prompt text names this file: the agent is told it may
        append durable instructions to it, and a hardcoded path in a prompt
        would send one tenant's write into a file every other tenant reads.
        """
        user_dir = self._get_user_path()
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / self.config.agent_md
        self._ensure_home_agent_md_symlink(path)
        return path

    def memory_index_path(self) -> Path:
        """Return this user's MEMORY.md index — ``users/{user_id}/MEMORY.md``."""
        self._initialize_user_dir()
        return self._get_user_memory_md()

    def list_memory_index_entries(self) -> List[Dict]:
        """Parse MEMORY.md into ``{title, filename, hook}`` dicts (settings page)."""
        path = self.memory_index_path()
        if not path.is_file():
            return []
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return self._parse_memory_index(content)

    def read_user_agents_md(self) -> str:
        """Return this user's AGENTS.md body (empty string if the file is missing)."""
        path = self.user_agent_md_path()
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_user_agents_md(self, content: str) -> Path:
        """Replace this user's AGENTS.md. Clears the frozen context snapshot.

        The Runner only re-reads AGENTS.md when ``get_frozen_context()`` is
        None. The workspace instance is shared across gateway sessions, so
        leaving the snapshot in place would keep serving the pre-edit file
        until process restart.
        """
        if len(content) > self.MAX_MEMORY_CHARACTER_COUNT:
            raise ValueError(
                f"AGENTS.md exceeds {self.MAX_MEMORY_CHARACTER_COUNT} characters"
            )
        path = self.user_agent_md_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        self._context_snapshot = None
        return path

    def _ensure_home_agent_md_symlink(self, target: Path) -> None:
        """Expose ``~/.agentica/AGENTS.md`` as the default user's alias.

        The canonical file stays under ``users/default/AGENTS.md`` so the
        multi-user layout remains uniform. The home-level path is created only
        for the configured default workspace; custom SDK workspaces should not
        silently repoint the user's real home alias.
        """
        default_workspace = Path(AGENTICA_WORKSPACE_DIR).expanduser().resolve()
        if self._user_id != self.DEFAULT_USER_ID or self.path != default_workspace:
            return
        alias = Path(AGENTICA_HOME).expanduser() / self.config.agent_md
        if alias == target:
            return
        alias.parent.mkdir(parents=True, exist_ok=True)

        try:
            if alias.is_symlink():
                if alias.resolve(strict=False) == target.resolve(strict=False):
                    return
                alias.unlink()
            elif alias.exists():
                self._fold_home_agent_md_file(alias, target)
                if alias.exists():
                    alias.unlink()

            alias.symlink_to(target)
        except (OSError, UnicodeError) as exc:
            logger.warning("Could not link %s to %s: %s", alias, target, exc)

    def _fold_home_agent_md_file(self, alias: Path, target: Path) -> None:
        """Preserve a pre-existing home AGENTS.md before replacing it with a symlink."""
        if not target.exists():
            alias.replace(target)
            logger.info("Moved %s to %s and kept %s as a compatibility symlink", alias, target, alias)
            return

        alias_text = alias.read_text(encoding="utf-8")
        if self._is_empty_template(alias_text):
            return

        target_text = target.read_text(encoding="utf-8")
        if alias_text.strip() in target_text:
            return

        target_body = target_text.rstrip()
        alias_body = alias_text.rstrip()
        merged = f"{target_body}\n\n{alias_body}\n" if target_body else f"{alias_body}\n"
        target.write_text(merged, encoding="utf-8")
        logger.info("Merged existing %s into %s before creating compatibility symlink", alias, target)

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

    async def get_memory_index_prompt(self) -> str:
        """MEMORY.md index for the system prompt: title, hook, relative path.

        Topic-file bodies stay on disk. The prompt copy is capped well below
        the on-disk 200-line / 25KB index so a large store cannot dominate
        every turn; the absolute path of the full file is always named.
        Newest entries are kept when truncating.
        """
        self._initialize_user_dir()
        index_path = self.memory_index_path()
        if not index_path.is_file():
            return ""
        content = (await async_read_text(index_path)).strip()
        if not content:
            return ""

        lines = [line for line in content.splitlines() if line.strip()]
        truncated = False
        if len(lines) > self._MEMORY_INJECT_MAX_LINES:
            lines = lines[-self._MEMORY_INJECT_MAX_LINES:]
            truncated = True
        body = "\n".join(lines)
        while len(body.encode("utf-8")) > self._MEMORY_INJECT_MAX_BYTES:
            if not lines:
                break
            lines.pop(0)
            truncated = True
            body = "\n".join(lines)
        if not body:
            return ""

        abs_path = str(index_path.resolve())
        note = (
            "Note: This MEMORY.md index is a session-start snapshot for "
            "prompt-cache stability. Topic files are not injected — use "
            "read_file or search_memory when a hook matches. "
            f"Full index: {abs_path}"
        )
        if truncated:
            note += (
                f" Index truncated to {self._MEMORY_INJECT_MAX_LINES} lines / "
                f"{self._MEMORY_INJECT_MAX_BYTES} bytes for the system prompt."
            )
        return f"{note}\n\n{body}"

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
    ) -> Optional[str]:
        """Promote a quarantined candidate into the canonical memory folder.

        Reads the candidate's body and frontmatter, then re-writes via
        `write_memory_entry(source="user_confirmed")`. The original candidate
        file is removed by `write_memory_entry` (M7 cleanup) so each entry
        ends up in exactly one place.

        Args:
            filename: bare filename inside `memory_candidates/` (e.g.
                `feedback_python_style.md`).

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
        """List standing-instruction file status for the current user.

        Returns:
            Dictionary with file names as keys and existence status as values.
            Reports this user's ``AGENTS.md`` (not a workspace-root copy).
        """
        return {self.config.agent_md: self.user_agent_md_path().is_file()}

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

    @property
    def experience_index_path(self) -> Path:
        """Path to the current user's EXPERIENCE.md index.

        Public because the system prompt names it: the injected experiences are
        a session-start snapshot, and this is where the agent reads the current
        set from.
        """
        return self._get_user_experience_md()

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

        safe_user_id = self.sanitize_user_id(user_id)
        user_path = self.path / self.config.users_dir / safe_user_id

        if not user_path.exists():
            return False

        shutil.rmtree(user_path)
        return True
