# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Workspace management for Agentica agents.
Inspired by OpenClaw's workspace concept.
"""
import asyncio
import functools
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import date

from agentica.config import AGENTICA_WORKSPACE_DIR


async def _async_read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read text file in executor to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(path.read_text, encoding=encoding))


async def _async_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text file in executor to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(path.write_text, content, encoding=encoding))


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
    agent_md: str = "AGENT.md"
    persona_md: str = "PERSONA.md"
    tools_md: str = "TOOLS.md"
    user_md: str = "USER.md" # user infomation
    users_dir: str = "users" # for multi-user isolation
    memory_dir: str = "memory" # daily memory, under users/{user_id}/memory
    memory_md: str = "MEMORY.md" # user's long-term memory, under users/{user_id}/
    skills_dir: str = "skills" # each user's skills, under users/{user_id}/skills


class Workspace:
    """Agent Workspace.

    Workspace is the configuration and memory storage directory for Agent,
    supporting multi-user isolation. All user data is stored under users/ directory.

    Directory structure:
    - AGENT.md: Agent instructions and constraints (globally shared)
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
        >>> workspace.save_memory("User prefers concise responses")  # Stored in users/default/

    Custom user mode:
        >>> workspace = Workspace("~/.agentica/workspace", user_id="alice@example.com")
        >>> workspace.initialize()
        >>> workspace.save_memory("Alice likes Python")  # Stored in users/alice@example.com/

    Switch user:
        >>> workspace.set_user("bob@example.com")
        >>> workspace.save_memory("Bob prefers detailed explanations")
    """

    # Global config files (shared across all users)
    DEFAULT_GLOBAL_FILES = {
        "AGENT.md": """# Agent Instructions

You are a helpful AI assistant.

## Guidelines
1. Be concise and accurate
2. Use tools when needed
3. Store important information in memory
4. Follow user preferences in USER.md

## Code Verification

**VERY IMPORTANT**: After completing code changes, you MUST verify your work:

1. **Find Commands**: Check project files for validation commands:
   - README.md, package.json, pyproject.toml, Makefile

2. **Execute Validation**: Use shell tool to run:
   - Lint: `npm run lint`, `ruff check .`, etc.
   - Type check: `npm run typecheck`, `mypy .`, etc.
   - Test: `npm test`, `pytest`, etc.

3. **Fix Issues**: If validation fails, fix and re-run until passing.

## Build/Lint/Test Commands

<!-- Add project-specific commands here -->
<!-- Example:
- Build: `npm run build`
- Lint: `npm run lint`
- Test: `npm test`
- Single test: `npm test -- --grep "test name"`
-->
""",
        "PERSONA.md": """# Persona

## Personality
- Friendly and professional
- Direct and honest
- Proactive in helping

## Communication Style
- Clear and concise
- Use examples when explaining
- Ask clarifying questions when needed
""",
        "TOOLS.md": """# Tool Usage Guidelines

## File Operations
- Always use absolute paths
- Read files before editing
- Create backups for important changes

## Shell Commands
- Prefer safe, non-destructive commands
- Explain what commands will do
""",
    }
    
    # Default user file template
    DEFAULT_USER_MD = """# User Profile

## User ID
{user_id}

## Preferences
- Language: Chinese or English
- Style: Concise

## Context
<!-- User's background, projects, etc. -->
"""

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

    @property
    def user_id(self) -> str:
        """Get current user ID."""
        return self._user_id

    def set_user(self, user_id: Optional[str]):
        """Set current user ID.

        Args:
            user_id: User ID, defaults to 'default' if None
        """
        self._user_id = user_id if user_id else "default"

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

    def initialize(self, force: bool = False) -> bool:
        """Initialize workspace.

        Creates workspace directory, global configuration files, and user data directory.

        Args:
            force: Whether to overwrite existing files

        Returns:
            Whether initialization was successful
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # Create globally shared files (AGENT.md, PERSONA.md, TOOLS.md)
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
        """Initialize current user's data directory."""
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

    def exists(self) -> bool:
        """Check if workspace exists.

        Returns:
            Whether both workspace directory and AGENT.md file exist
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
            content = (await _async_read_text(filepath)).strip()
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

        Reads AGENT.md, PERSONA.md, TOOLS.md file contents (globally shared),
        and user-specific USER.md file content.

        Returns:
            Merged context string
        """
        contents = []

        # Read globally shared files
        global_files = [
            self.config.agent_md,
            self.config.persona_md,
            self.config.tools_md,
        ]
        for f in global_files:
            content = await self.read_file_async(f)
            if content:
                contents.append(f"<!-- {f} -->\n{content}")

        # Read user-specific USER.md (always from users/{user_id}/)
        user_md_path = self._get_user_md()
        if user_md_path.exists():
            content = (await _async_read_text(user_md_path)).strip()
            if content:
                contents.append(f"<!-- USER.md (user: {self._user_id}) -->\n{content}")

        return "\n\n---\n\n".join(contents) if contents else ""

    async def get_memory_prompt(self, days: int = 2) -> str:
        """Get recent memory (for injecting into context).

        Reads user-specific MEMORY.md long-term memory and recent daily memories.

        Args:
            days: Number of recent days to read

        Returns:
            Memory content string
        """
        contents = []

        # Read long-term memory (user-specific, from users/{user_id}/)
        long_term_path = self._get_user_memory_md()
        if long_term_path.exists():
            long_term = (await _async_read_text(long_term_path)).strip()
            if long_term:
                contents.append(f"## Long-term Memory (user: {self._user_id})\n\n{long_term}")

        # Read daily memory (user-specific, from users/{user_id}/memory/)
        memory_dir = self._get_user_memory_dir()
        if memory_dir.exists():
            files = sorted(memory_dir.glob("*.md"), reverse=True)[:days]
            for f in files:
                content = (await _async_read_text(f)).strip()
                if content:
                    contents.append(f"## {f.stem}\n\n{content}")

        return "\n\n".join(contents) if contents else ""

    async def write_memory(self, content: str, to_daily: bool = True):
        """Write memory.

        Writes content to current user's memory file (users/{user_id}/).

        Args:
            content: Memory content
            to_daily: True to write to daily memory, False to write to long-term memory
        """
        # Ensure user directory exists
        self._initialize_user_dir()

        if to_daily:
            today = date.today().isoformat()
            memory_dir = self._get_user_memory_dir()
            memory_dir.mkdir(parents=True, exist_ok=True)
            filepath = memory_dir / f"{today}.md"

            # Append content
            existing = ""
            if filepath.exists():
                existing = (await _async_read_text(filepath)).strip()
            new_content = f"{existing}\n\n{content}".strip() if existing else content
            await _async_write_text(filepath, new_content)
        else:
            memory_md = self._get_user_memory_md()
            memory_md.parent.mkdir(parents=True, exist_ok=True)

            # Append content
            existing = ""
            if memory_md.exists():
                existing = (await _async_read_text(memory_md)).strip()
            new_content = f"{existing}\n\n{content}".strip() if existing else content
            await _async_write_text(memory_md, new_content)

    async def save_memory(self, content: str, long_term: bool = False):
        """Save memory (alias for write_memory with more semantic naming).

        Args:
            content: Memory content
            long_term: True to write to long-term memory, False to write to daily memory (default)
        """
        await self.write_memory(content, to_daily=not long_term)

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
        min_score: float = 0.3,
    ) -> List[Dict]:
        """Search memory (simple keyword search).

        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum match score

        Returns:
            List of matching memories, each containing content, file_path, score
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        for file_path in self.get_all_memory_files():
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            content_lower = content.lower()

            # Calculate simple match score
            score = 0.0
            for word in query_words:
                if word in content_lower:
                    score += 1.0 / len(query_words)

            if score >= min_score:
                results.append({
                    "content": content,
                    "file_path": str(file_path.relative_to(self.path)),
                    "score": score,
                })

        # Sort by score
        results.sort(key=lambda x: -x["score"])
        return results[:limit]

    def clear_daily_memory(self, keep_days: int = 7):
        """Clear old daily memories.

        Clears old daily memory files for current user.

        Args:
            keep_days: Number of recent days to keep
        """
        memory_dir = self._get_user_memory_dir()
        if not memory_dir.exists():
            return

        files = sorted(memory_dir.glob("*.md"), reverse=True)
        for f in files[keep_days:]:
            f.unlink()

    def create_memory_search(self):
        """Create workspace memory search instance.

        Returns:
            WorkspaceMemorySearch instance for vector and keyword search

        Example:
            >>> workspace = Workspace()
            >>> search = workspace.create_memory_search()
            >>> search.index()
            >>> results = search.search("Python programming")
        """
        from agentica.memory import WorkspaceMemorySearch
        return WorkspaceMemorySearch(workspace_path=str(self.path))

    def search_memory_hybrid(
        self,
        query: str,
        limit: int = 5,
        embedder=None,
    ) -> List[Dict]:
        """Hybrid memory search (vector + keyword).

        Uses combination of vector similarity and keyword matching to search memories.

        Args:
            query: Search query
            limit: Maximum number of results
            embedder: Optional embedding model instance, uses OpenAIEmb if not provided

        Returns:
            List of matching memories

        Example:
            >>> workspace = Workspace()
            >>> workspace.initialize()
            >>> workspace.write_memory("Python is great for AI")
            >>> results = workspace.search_memory_hybrid("artificial intelligence")
        """
        from agentica.memory import WorkspaceMemorySearch

        search = WorkspaceMemorySearch(workspace_path=str(self.path))
        search.index()

        results = search.search_hybrid(query, limit=limit, embedder=embedder)

        # Convert MemoryChunk to dict for consistency with search_memory
        return [
            {
                "content": r.content,
                "file_path": r.file_path,
                "score": r.score,
            }
            for r in results
        ]

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
                    import datetime
                    mtime = latest_file.stat().st_mtime
                    last_activity = datetime.datetime.fromtimestamp(mtime).isoformat()

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

        import shutil
        shutil.rmtree(user_path)
        return True
